#!/usr/bin/env python3
"""Fetch real Sentinel-2 L2A data from Microsoft Planetary Computer.

Downloads multi-temporal scenes for a California Central Valley farm field,
converts to the numpy band-dict format expected by agent/scene.py, and
fetches USDA CropScape ground truth labels.

No authentication required — Planetary Computer's STAC API is free.

Usage:
    uv run python data/fetch_sentinel2.py
    uv run python data/fetch_sentinel2.py --lat 36.75 --lon -120.24 --size 2000
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import planetary_computer
import pystac_client
import rasterio
from rasterio.transform import from_bounds as affine_from_bounds
from rasterio.warp import reproject, transform_bounds, Resampling

SCENES_DIR = Path("data/scenes")
SCENES_DIR.mkdir(parents=True, exist_ok=True)

# Sentinel-2 bands we use — asset keys in Planetary Computer match band IDs
BAND_IDS = ["B02", "B03", "B04", "B08", "B8A", "B11", "B12"]

# Default: almond orchards near Madera, CA — Central Valley
DEFAULT_LAT = 36.944
DEFAULT_LON = -120.108
DEFAULT_SIZE_M = 2000  # 2km x 2km AOI

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

# Landsat Collection 2 Level 2 surface temperature band
# atmos corrected
LANDSAT_COLLECTION = "landsat-c2-l2"
LANDSAT_THERMAL_ASSET = "lwir11"  # TIRS Band 10 (10.9 µm)
# raw pixel values used to convert the digital numbers (DN) to Kelvin
LANDSAT_ST_SCALE = 0.00341802  # DN to Kelvin scale factor
LANDSAT_ST_OFFSET = 149.0  # DN to Kelvin offset


def meters_to_degrees(meters: float, lat: float) -> tuple[float, float]:
    """Approximate conversion from meters to lat/lon degrees."""
    lat_deg = meters / 111_320
    lon_deg = meters / (111_320 * np.cos(np.radians(lat)))
    return lat_deg, lon_deg


def make_bbox(lat: float, lon: float, size_m: float) -> list[float]:
    """Create a bounding box [west, south, east, north] around a center point."""
    dlat, dlon = meters_to_degrees(size_m / 2, lat)
    return [lon - dlon, lat - dlat, lon + dlon, lat + dlat]


def search_scenes(bbox: list[float], date_range: str, max_items: int = 10):
    """Search Planetary Computer for Sentinel-2 L2A scenes."""
    catalog = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": 15}},
        max_items=max_items,
        sortby=[{"field": "datetime", "direction": "asc"}],
    )
    items = list(search.items())
    print(f"Found {len(items)} raw results with <15% cloud cover")

    # Deduplicate: keep only one item per date (the one with lowest cloud cover)
    by_date: dict[str, list] = {}
    for item in items:
        date_key = item.datetime.strftime("%Y-%m-%d")
        by_date.setdefault(date_key, []).append(item)
    deduped = []
    for date_key in sorted(by_date):
        # Pick the item with lowest cloud cover that actually covers our AOI
        candidates = sorted(by_date[date_key], key=lambda it: it.properties.get("eo:cloud_cover", 100))
        deduped.append(candidates[0])

    print(f"  {len(deduped)} unique dates after deduplication")
    return deduped


def _target_transform(bbox: list[float], shape: tuple[int, int]):
    """Return a north-up Affine transform for a WGS84 bbox and pixel grid.

    Every scene and baseline band is reprojected onto this identical grid,
    guaranteeing pixel-level co-registration across different Sentinel-2 passes.
    """
    return affine_from_bounds(*bbox, shape[1], shape[0])


def read_band_from_item(item, band_key: str, bbox: list[float], target_shape: tuple[int, int]) -> np.ndarray:
    """Read a single band from a STAC item, reprojected to a north-up WGS84 grid.

    All bands are warped onto an identical target grid defined by (bbox, target_shape)
    so that cross-scene comparisons (e.g. baseline change detection) are pixel-aligned.
    """
    href = item.assets[band_key].href
    dst_transform = _target_transform(bbox, target_shape)
    dst_crs = "EPSG:4326"

    with rasterio.open(href) as src:
        dst = np.zeros(target_shape, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
        )
    # Convert to reflectance (Sentinel-2 L2A stores as uint16 * 10000)
    return dst / 10_000.0


def fetch_scene(item, bbox: list[float], target_size: int = 200) -> dict[str, np.ndarray]:
    """Fetch all bands for a single STAC item, returning a band dict."""
    target_shape = (target_size, target_size)
    bands = {}
    for band_key in BAND_IDS:
        sys.stdout.write(f"  {band_key}...")
        sys.stdout.flush()
        bands[band_key] = read_band_from_item(item, band_key, bbox, target_shape)
        sys.stdout.write(" ok\n")
    return bands


def fetch_cropscape_labels(bbox: list[float], year: int) -> dict:
    """Fetch USDA CropScape crop type labels for the AOI.

    Uses WMS GetMap to fetch a small CDL raster chip and returns the
    dominant crop code with its name from the USDA CDL legend.
    """
    import httpx
    from collections import Counter
    from io import BytesIO

    # Common CDL crop codes — covers Central Valley agriculture + double crops
    CDL_NAMES = {
        0: "Background", 1: "Corn", 2: "Cotton", 3: "Rice", 4: "Sorghum",
        5: "Soybeans", 6: "Sunflower", 10: "Peanuts", 12: "Sweet Corn",
        13: "Pop or Orn Corn", 14: "Mint",
        21: "Barley", 22: "Durum Wheat", 23: "Spring Wheat",
        24: "Winter Wheat", 25: "Other Small Grains", 26: "Dbl Crop WinWht/Soybeans",
        27: "Rye", 28: "Oats", 29: "Millet",
        30: "Speltz", 31: "Canola", 32: "Flaxseed", 33: "Safflower",
        34: "Rape Seed", 35: "Mustard", 36: "Alfalfa",
        37: "Other Hay/Non Alfalfa", 38: "Camelina",
        39: "Buckwheat", 41: "Sugarbeets", 42: "Dry Beans",
        43: "Potatoes", 44: "Other Crops", 45: "Sugarcane",
        46: "Sweet Potatoes", 47: "Misc Vegs & Fruits",
        48: "Watermelons", 49: "Onions", 50: "Cucumbers",
        51: "Chick Peas", 52: "Lentils", 53: "Peas",
        54: "Tomatoes", 55: "Caneberries", 56: "Hops",
        57: "Herbs", 58: "Clover/Wildflowers", 59: "Sod/Grass Seed",
        60: "Switchgrass", 61: "Fallow/Idle Cropland",
        63: "Forest", 64: "Shrubland", 65: "Barren",
        66: "Cherries", 67: "Peaches", 68: "Apples", 69: "Grapes",
        70: "Christmas Trees", 71: "Other Tree Crops", 72: "Citrus",
        74: "Pecans", 75: "Almonds", 76: "Walnuts", 77: "Pears",
        92: "Aquaculture", 111: "Open Water", 112: "Perennial Ice/Snow",
        121: "Developed/Open Space", 122: "Developed/Low Intensity",
        123: "Developed/Medium Intensity", 124: "Developed/High Intensity",
        131: "Barren", 141: "Deciduous Forest", 142: "Evergreen Forest",
        143: "Mixed Forest", 152: "Shrubland",
        171: "Grassland/Pasture", 176: "Grassland/Pasture",
        190: "Woody Wetlands", 195: "Herbaceous Wetlands",
        204: "Pistachios", 205: "Triticale", 206: "Carrots",
        207: "Asparagus", 208: "Garlic", 209: "Cantaloupes",
        210: "Prunes", 211: "Olives", 212: "Oranges",
        213: "Honeydew Melons", 214: "Broccoli", 216: "Peppers",
        217: "Pomegranates", 218: "Nectarines", 219: "Greens",
        220: "Plums", 221: "Strawberries", 222: "Squash",
        223: "Apricots", 224: "Vetch", 225: "Dbl Crop WinWht/Corn",
        226: "Dbl Crop Oats/Corn", 227: "Lettuce",
        228: "Dbl Crop Triticale/Corn", 229: "Pumpkins",
        230: "Dbl Crop Lettuce/Durum Wht", 231: "Dbl Crop Lettuce/Cantaloupe",
        232: "Dbl Crop Lettuce/Cotton", 233: "Dbl Crop Lettuce/Barley",
        234: "Dbl Crop Durum Wht/Sorghum", 235: "Dbl Crop Barley/Sorghum",
        236: "Dbl Crop WinWht/Sorghum", 237: "Dbl Crop Barley/Corn",
        238: "Dbl Crop WinWht/Cotton", 239: "Dbl Crop Soybeans/Cotton",
        240: "Dbl Crop Soybeans/Oats", 241: "Dbl Crop Corn/Soybeans",
        242: "Blueberries", 243: "Cabbage", 244: "Cauliflower",
        245: "Celery", 246: "Radishes", 247: "Turnips",
        248: "Eggplants", 249: "Gourds", 250: "Cranberries",
        254: "Dbl Crop Barley/Soybeans",
    }

    # Use full AOI bbox at 200x200 resolution to match scene grid
    wms_bbox = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"

    # Try requested year first, then fall back to previous years
    for try_year in [year, year - 1, year - 2]:
        layer = f"cdl_{try_year}"
        url = (
            "https://nassgeodata.gmu.edu/CropScapeService/wms_cdlall.cgi"
            f"?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&LAYERS={layer}"
            f"&SRS=EPSG:4326&BBOX={wms_bbox}&WIDTH=200&HEIGHT=200&FORMAT=image/tiff"
        )
        try:
            resp = httpx.get(url, timeout=30)
            if resp.status_code != 200 or "image/tiff" not in resp.headers.get("content-type", ""):
                continue
            data = np.frombuffer(resp.content, dtype=np.uint8)
            # Parse TIFF with rasterio
            with rasterio.open(BytesIO(resp.content)) as src:
                arr = src.read(1)
            # Find dominant non-zero crop code
            codes = arr.flatten()
            codes = codes[codes > 0]
            if len(codes) == 0:
                continue
            dominant_code = Counter(codes.tolist()).most_common(1)[0][0]
            crop_name = CDL_NAMES.get(dominant_code, f"Unknown (code {dominant_code})")
            all_codes = Counter(codes.tolist()).most_common()
            print(f"  CDL {try_year}: dominant crop = {crop_name} (code {dominant_code})")
            return {
                "crop_code": int(dominant_code),
                "crop_name": crop_name,
                "all_codes": {CDL_NAMES.get(c, f"code_{c}"): n for c, n in all_codes},
                "source": "USDA CDL via WMS",
                "year": try_year,
            }
        except Exception as e:
            print(f"  Warning: CDL WMS query for {try_year} failed ({e})")
            continue

    print("  Warning: Could not fetch CDL labels for any recent year")
    return {"error": "CDL WMS unavailable", "source": "USDA CDL via WMS", "year": year}


def build_timeseries_metadata(items, bbox: list[float], target_size: int = 200) -> dict:
    """Build NDVI timeseries from multiple scenes for metadata."""
    dates = []
    ndvi_center_values = []
    ndvi_corner_values = []

    center_r, center_c = target_size // 2, target_size // 2
    corner_r, corner_c = target_size // 4, target_size // 4

    for item in items:
        date_str = item.datetime.strftime("%Y-%m-%d")
        try:
            nir = read_band_from_item(item, "B08", bbox, (target_size, target_size))
            red = read_band_from_item(item, "B04", bbox, (target_size, target_size))
            ndvi = (nir - red) / (nir + red + 1e-8)
            dates.append(date_str)
            ndvi_center_values.append(round(float(ndvi[center_r, center_c]), 3))
            ndvi_corner_values.append(round(float(ndvi[corner_r, corner_c]), 3))
        except Exception as e:
            print(f"  Skipping {date_str} for timeseries: {e}")
            continue

    return {
        "dates": dates,
        "point_a": {"row": center_r, "col": center_c, "ndvi": ndvi_center_values},
        "point_b": {"row": corner_r, "col": corner_c, "ndvi": ndvi_corner_values},
    }


def search_landsat(bbox: list[float], target_date: str, max_days: int = 10) -> list:
    """Search for Landsat scenes near a target date for thermal data."""
    from datetime import timedelta
    target_dt = datetime.strptime(target_date, "%Y-%m-%d")
    start = (target_dt - timedelta(days=max_days)).strftime("%Y-%m-%d")
    end = (target_dt + timedelta(days=max_days)).strftime("%Y-%m-%d")
    catalog = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
    search = catalog.search(
        collections=[LANDSAT_COLLECTION],
        bbox=bbox,
        datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": 30}},
        max_items=10,
    )
    items = sorted(
        search.items(),
        key=lambda it: abs((it.datetime.replace(tzinfo=None) - target_dt).days),
    )
    return items


def fetch_landsat_thermal(item, bbox: list[float], target_shape: tuple[int, int]) -> np.ndarray:
    """Fetch Landsat surface temperature, reproject to match Sentinel-2 grid, return in Kelvin."""
    href = item.assets[LANDSAT_THERMAL_ASSET].href
    dst_transform = _target_transform(bbox, target_shape)
    dst_crs = "EPSG:4326"
    with rasterio.open(href) as src:
        dst = np.zeros(target_shape, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
        )
    # Convert DN to Kelvin
    lst_kelvin = dst * LANDSAT_ST_SCALE + LANDSAT_ST_OFFSET
    # Mask invalid pixels (nodata = 0 in source → offset value in output)
    lst_kelvin[dst == 0] = np.nan
    return lst_kelvin


def main():
    parser = argparse.ArgumentParser(description="Fetch real Sentinel-2 data from Planetary Computer")
    parser.add_argument("--lat", type=float, default=DEFAULT_LAT, help="Center latitude")
    parser.add_argument("--lon", type=float, default=DEFAULT_LON, help="Center longitude")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE_M, help="AOI size in meters")
    parser.add_argument("--target-pixels", type=int, default=200, help="Output array size (NxN)")
    parser.add_argument("--current-date", default="2024-07-15", help="Target date for current scene (YYYY-MM-DD)")
    parser.add_argument("--baseline-date", default="2024-06-01", help="Target date for baseline scene (YYYY-MM-DD)")
    parser.add_argument("--scene-id", default="central_valley", help="Scene ID prefix for output files")
    args = parser.parse_args()

    bbox = make_bbox(args.lat, args.lon, args.size)
    print(f"AOI: {bbox}")
    print(f"Center: ({args.lat}, {args.lon}), Size: {args.size}m")

    # --- Search for scenes around the current date ---
    current_dt = datetime.strptime(args.current_date, "%Y-%m-%d")
    current_range = f"{current_dt.year}-04-01/{current_dt.year}-09-30"
    print(f"\nSearching for scenes in {current_range}...")
    items = search_scenes(bbox, current_range, max_items=20)

    if len(items) < 2:
        print("ERROR: Need at least 2 scenes for current + baseline. Try a wider date range or lower cloud threshold.")
        sys.exit(1)

    # Pick the scene closest to current_date for "current"
    items_sorted = sorted(items, key=lambda it: abs((it.datetime.replace(tzinfo=None) - current_dt).days))
    current_item = items_sorted[0]
    print(f"\nCurrent scene: {current_item.datetime.strftime('%Y-%m-%d')} (cloud: {current_item.properties.get('eo:cloud_cover', '?')}%)")

    # Pick the scene closest to baseline_date for "baseline" (must be a different date)
    baseline_dt = datetime.strptime(args.baseline_date, "%Y-%m-%d")
    current_date_str = current_item.datetime.strftime("%Y-%m-%d")
    baseline_candidates = [
        it for it in items
        if it.datetime.strftime("%Y-%m-%d") != current_date_str
    ]
    baseline_candidates.sort(key=lambda it: abs((it.datetime.replace(tzinfo=None) - baseline_dt).days))
    baseline_item = baseline_candidates[0] if baseline_candidates else None

    # --- Fetch current scene bands ---
    print(f"\nFetching current scene bands...")
    current_bands = fetch_scene(current_item, bbox, args.target_pixels)
    np.save(SCENES_DIR / f"{args.scene_id}_bands.npy", current_bands)
    print(f"Saved: {args.scene_id}_bands.npy")

    # --- Fetch baseline scene bands ---
    has_baseline = False
    if baseline_item:
        print(f"\nBaseline scene: {baseline_item.datetime.strftime('%Y-%m-%d')} (cloud: {baseline_item.properties.get('eo:cloud_cover', '?')}%)")
        print("Fetching baseline bands...")
        baseline_bands = fetch_scene(baseline_item, bbox, args.target_pixels)
        np.save(SCENES_DIR / f"{args.scene_id}_baseline_bands.npy", baseline_bands)
        print(f"Saved: {args.scene_id}_baseline_bands.npy")
        has_baseline = True

    # --- Fetch Landsat thermal (surface temperature) ---
    print(f"\nSearching for Landsat thermal data near {current_item.datetime.strftime('%Y-%m-%d')}...")
    landsat_items = search_landsat(bbox, current_item.datetime.strftime("%Y-%m-%d"))
    landsat_item = None
    landsat_date = None
    if landsat_items:
        landsat_item = landsat_items[0]
        landsat_date = landsat_item.datetime.strftime("%Y-%m-%d")
        day_offset = abs((landsat_item.datetime.replace(tzinfo=None) - current_item.datetime.replace(tzinfo=None)).days)
        print(f"  Best match: {landsat_item.id} ({landsat_date}, {day_offset}d offset, cloud={landsat_item.properties.get('eo:cloud_cover', '?')}%)")
        print("  Fetching thermal band (lwir11)...")
        lst = fetch_landsat_thermal(landsat_item, bbox, (args.target_pixels, args.target_pixels))
        valid_frac = (~np.isnan(lst)).mean()
        print(f"  LST range: {np.nanmin(lst):.1f}–{np.nanmax(lst):.1f} K, valid pixels: {valid_frac:.1%}")
        # Save thermal band into the main bands file
        current_bands["LST"] = lst
        np.save(SCENES_DIR / f"{args.scene_id}_bands.npy", current_bands)
        print(f"  Added LST band to {args.scene_id}_bands.npy")
    else:
        print("  No Landsat scenes found within ±10 days")

    # --- Build timeseries from all available scenes ---
    print(f"\nBuilding NDVI timeseries from {len(items)} scenes...")
    timeseries = build_timeseries_metadata(items, bbox, args.target_pixels)

    # Remap timeseries to use stressed_point/healthy_point naming for compatibility
    # with agent/scene.py's get_timeseries_ndvi
    ts_compat = {"dates": timeseries["dates"]}
    if len(timeseries["point_a"]["ndvi"]) > 0 and len(timeseries["point_b"]["ndvi"]) > 0:
        mean_a = np.mean(timeseries["point_a"]["ndvi"])
        mean_b = np.mean(timeseries["point_b"]["ndvi"])
        if mean_a < mean_b:
            ts_compat["stressed_point"] = timeseries["point_a"]
            ts_compat["healthy_point"] = timeseries["point_b"]
        else:
            ts_compat["stressed_point"] = timeseries["point_b"]
            ts_compat["healthy_point"] = timeseries["point_a"]
    else:
        ts_compat["representative_point"] = timeseries["point_a"]

    # --- CropScape ground truth ---
    print("\nFetching USDA CropScape ground truth...")
    cropscape = fetch_cropscape_labels(bbox, current_dt.year)

    # --- Save metadata ---
    meta = {
        "scene_id": args.scene_id,
        "date": current_item.datetime.strftime("%Y-%m-%d"),
        "region": f"Central Valley, CA ({args.lat:.3f}, {args.lon:.3f})",
        "bbox_wgs84": bbox,
        "center_lat": args.lat,
        "center_lon": args.lon,
        "aoi_size_m": args.size,
        "bands": BAND_IDS + (["LST"] if landsat_item else []),
        "shape": [args.target_pixels, args.target_pixels],
        "pixel_size_m": round(args.size / args.target_pixels, 1),
        "source": "Sentinel-2 L2A via Microsoft Planetary Computer",
        "current_scene_id": current_item.id,
        "has_baseline": has_baseline,
        "baseline_date": baseline_item.datetime.strftime("%Y-%m-%d") if baseline_item else None,
        "baseline_scene_id": baseline_item.id if baseline_item else None,
        "timeseries": ts_compat,
        "ground_truth": cropscape,
        "landsat_thermal": {
            "available": landsat_item is not None,
            "scene_id": landsat_item.id if landsat_item else None,
            "date": landsat_date,
            "band_key": "LST",
            "unit": "Kelvin",
            "source": "Landsat Collection 2 Level 2 (lwir11 / TIRS Band 10)",
        },
    }
    meta_path = SCENES_DIR / f"{args.scene_id}_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Saved: {meta_path}")

    print(f"\nDone! Scene '{args.scene_id}' ready for analysis.")
    print(f"  Current:  {current_item.datetime.strftime('%Y-%m-%d')}")
    if baseline_item:
        print(f"  Baseline: {baseline_item.datetime.strftime('%Y-%m-%d')}")
    print(f"  Timeseries: {len(timeseries['dates'])} dates")
    print(f"\nRun the agent:")
    print(f"  uv run python -m agent.loop --scene {args.scene_id}")


if __name__ == "__main__":
    main()
