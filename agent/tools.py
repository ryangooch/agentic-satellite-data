# agent/tools.py
"""Tool implementations for the crop health agent.

Design rules:
- Each tool is a plain Python function.
- Tools never call other tools. Shared band math lives in `_compute_index_array`.
- Return dataclasses with success/error fields instead of raising exceptions.
- Return a human-readable `summary` string for the agent's context window.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from agent.models import (
    BoundingBox,
    NDVIResult,
    NDWIResult,
    EVIResult,
    CWSIResult,
    TimeseriesResult,
    AnomalyResult,
    DiffResult,
)
from agent import scene as _scene
from agent import rag as _rag

_IMAGES_DIR = Path("data/images")
_IMAGES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Internal helper — shared band math (keeps tools flat)
# ---------------------------------------------------------------------------


def _compute_index_array(index: str, region: BoundingBox) -> np.ndarray:
    """Compute a spectral index array directly from bands. Not a public tool."""
    nir = _scene.get_band("B08", region)
    red = _scene.get_band("B04", region)
    green = _scene.get_band("B03", region)
    swir = _scene.get_band("B11", region)
    blue = _scene.get_band("B02", region)
    if index == "ndvi":
        return (nir - red) / (nir + red + 1e-8)
    if index == "ndwi":
        return (green - swir) / (green + swir + 1e-8)
    if index == "evi":
        denom = nir + 6 * red - 7.5 * blue + 1
        return np.clip(2.5 * (nir - red) / (denom + 1e-8), -1.0, 1.0)
    raise ValueError(f"Unknown index: {index!r}. Choose from ndvi, ndwi, evi.")


def _save_image(
    arr: np.ndarray, title: str, filename: str, vmin: float = -1, vmax: float = 1
) -> str:
    path = _IMAGES_DIR / filename
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(arr, cmap="RdYlGn", vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax)
    ax.set_title(title)
    fig.savefig(path, dpi=72, bbox_inches="tight")
    plt.close(fig)
    return str(path)


# ---------------------------------------------------------------------------
# Public tools
# ---------------------------------------------------------------------------


def compute_ndvi(region: BoundingBox) -> NDVIResult:
    """Compute NDVI = (NIR - Red) / (NIR + Red) for a region.

    Returns mean, std, fraction of pixels below 0.3 (stressed), and a rendered image path.
    """
    ndvi = _compute_index_array("ndvi", region)
    mean_val = float(ndvi.mean())
    std_val = float(ndvi.std())
    low_frac = float((ndvi < 0.3).mean())
    image_path = _save_image(ndvi, f"NDVI (mean={mean_val:.3f})", "ndvi.png", vmin=-0.2, vmax=1.0)
    alert = (
        "⚠️ Significant stressed area detected."
        if low_frac > 0.1
        else "Vegetation appears generally healthy."
    )
    summary = (
        f"NDVI for rows {region.row_min}–{region.row_max}, "
        f"cols {region.col_min}–{region.col_max}: "
        f"mean={mean_val:.3f}, std={std_val:.3f}. "
        f"Pixels below 0.3 (stressed): {low_frac:.1%}. {alert}"
    )
    return NDVIResult(
        success=True,
        ndvi_array=ndvi,
        mean=mean_val,
        std=std_val,
        low_fraction=low_frac,
        image_path=image_path,
        summary=summary,
    )


def compute_ndwi(region: BoundingBox) -> NDWIResult:
    """Compute NDWI = (Green - SWIR) / (Green + SWIR) using B03 and B11.

    Positive: canopy moisture present. Negative: water stress or dry soil.
    """
    ndwi = _compute_index_array("ndwi", region)
    mean_val = float(ndwi.mean())
    std_val = float(ndwi.std())
    neg_frac = float((ndwi < 0).mean())
    image_path = _save_image(ndwi, f"NDWI (mean={mean_val:.3f})", "ndwi.png", vmin=-0.6, vmax=0.6)
    if mean_val < -0.1:
        stress_msg = "water-stressed (dry canopy)"
    elif mean_val > 0.1:
        stress_msg = "well-watered"
    else:
        stress_msg = "moderate moisture"
    summary = (
        f"NDWI for rows {region.row_min}–{region.row_max}, "
        f"cols {region.col_min}–{region.col_max}: "
        f"mean={mean_val:.3f}, std={std_val:.3f}. "
        f"stress_msg: {stress_msg}. Negative-NDWI pixels: {neg_frac:.1%}."
    )
    return NDWIResult(
        success=True,
        ndwi_array=ndwi,
        mean=mean_val,
        std=std_val,
        negative_fraction=neg_frac,
        image_path=image_path,
        summary=summary,
    )


def compute_evi(region: BoundingBox) -> EVIResult:
    """Compute EVI = 2.5*(NIR-Red)/(NIR+6*Red-7.5*Blue+1) using B08, B04, B02.

    Less saturation than NDVI in dense canopy. Use to cross-check NDVI findings.
    """
    evi = _compute_index_array("evi", region)
    mean_val = float(evi.mean())
    std_val = float(evi.std())
    low_frac = float((evi < 0.2).mean())
    image_path = _save_image(evi, f"EVI (mean={mean_val:.3f})", "evi.png", vmin=0.0, vmax=0.8)
    summary = (
        f"EVI for rows {region.row_min}–{region.row_max}, "
        f"cols {region.col_min}–{region.col_max}: "
        f"mean={mean_val:.3f}, std={std_val:.3f}. "
        f"Pixels below 0.2: {low_frac:.1%}. "
        "EVI is less saturated than NDVI in dense canopy — use to verify NDVI findings."
    )
    return EVIResult(
        success=True,
        evi_array=evi,
        mean=mean_val,
        std=std_val,
        low_fraction=low_frac,
        image_path=image_path,
        summary=summary,
    )


# Crop-specific VPD baselines (kPa) for empirical CWSI
# Based on Idso et al. (1981) and Jackson et al. (1981)
_CWSI_BASELINES = {
    "alfalfa": {"vpd_lower": 0.9, "vpd_upper": 4.0},
    "almond": {"vpd_lower": 1.0, "vpd_upper": 4.5},
    "corn": {"vpd_lower": 0.8, "vpd_upper": 3.5},
    "cotton": {"vpd_lower": 1.2, "vpd_upper": 5.0},
    "grape": {"vpd_lower": 0.9, "vpd_upper": 4.0},
    "tomato": {"vpd_lower": 0.8, "vpd_upper": 3.8},
}


def _thermal_sharpen(lst_coarse: np.ndarray, ndvi_fine: np.ndarray, block: int = 10) -> np.ndarray:
    """Sharpen coarse-resolution LST using fine-resolution NDVI (TsHARP method).

    Fits a linear NDVI–LST relationship at coarse resolution, then applies
    the residual correction at fine resolution.  Based on Agam et al. (2007).
    """
    from scipy.ndimage import uniform_filter
    h, w = ndvi_fine.shape
    # Aggregate NDVI to coarse resolution to match LST
    ndvi_coarse = uniform_filter(ndvi_fine, size=block, mode='nearest')
    # Fit linear regression: LST = a * NDVI + b (at coarse scale)
    valid = ~np.isnan(lst_coarse) & ~np.isnan(ndvi_coarse)
    if valid.sum() < 10:
        return lst_coarse  # not enough data, return as-is
    a, b = np.polyfit(ndvi_coarse[valid], lst_coarse[valid], 1)
    # Predict LST at fine resolution using NDVI
    lst_predicted_fine = a * ndvi_fine + b
    lst_predicted_coarse = a * ndvi_coarse + b
    # Residual at coarse resolution
    residual = np.where(valid, lst_coarse - lst_predicted_coarse, 0.0)
    # Add residual back to fine prediction
    lst_sharp = lst_predicted_fine + residual
    return lst_sharp


def compute_cwsi(
    region: BoundingBox, air_temp_f: float, vpd_kpa: float, crop_type: str = "alfalfa"
) -> CWSIResult:
    """Compute CWSI (Crop Water Stress Index) using thermal data when available.

    When Landsat LST is available, uses the empirical CWSI method (Jackson et al. 1981)
    with NDVI-based thermal sharpening (TsHARP, Agam et al. 2007) to downscale
    100m thermal data to 10m resolution.

    Falls back to VPD+NDVI proxy when no thermal band is present.

    CWSI ranges 0-1: 0 = no stress, 1 = maximum stress.
    Values > 0.5 indicate significant water stress.
    """
    crop = crop_type.lower()
    if crop not in _CWSI_BASELINES:
        error_msg = f"Unknown crop type {crop_type!r}. Choose from: {', '.join(_CWSI_BASELINES)}."
        return CWSIResult(
            success=False,
            error_message=error_msg,
        )

    bl = _CWSI_BASELINES[crop]
    ndvi = _compute_index_array("ndvi", region)

    # Try thermal-based CWSI first
    try:
        lst_raw = _scene.get_band("LST", region)
        has_thermal = not np.all(np.isnan(lst_raw))
    except KeyError:
        has_thermal = False

    if has_thermal:
        # Thermal sharpening: downscale 100m LST using 10m NDVI
        lst = _thermal_sharpen(lst_raw, ndvi, block=10)
        # Air temperature in Kelvin
        ta_k = (air_temp_f - 32) * 5 / 9 + 273.15
        # Canopy-air temperature differential
        dt = lst - ta_k
        # Crop-specific non-water-stressed baseline (NWSB):
        #   dT_lower = a + b*VPD  (well-watered: canopy cooler than air via transpiration)
        #   dT_upper = no transpiration (canopy heats toward soil equilibrium)
        # Empirical parameters from Idso et al. (1981), Jackson et al. (1981)
        dt_lower = 1.5 - 2.0 * vpd_kpa  # well-watered canopy (almond NWSB)
        dt_upper = 15.0  # fully stressed in semi-arid conditions
        cwsi = np.clip((dt - dt_lower) / (dt_upper - dt_lower), 0, 1)
        method = "thermal (Landsat LST + TsHARP)"
    else:
        # Fallback: VPD + NDVI proxy
        base_cwsi = max(
            0.0, min(1.0, (vpd_kpa - bl["vpd_lower"]) / (bl["vpd_upper"] - bl["vpd_lower"]))
        )
        ndvi_norm = np.clip((ndvi - 0.1) / (0.8 - 0.1), 0, 1)
        cwsi = np.clip(base_cwsi + (1 - ndvi_norm) * 0.3 - ndvi_norm * 0.15, 0, 1)
        method = "VPD+NDVI proxy (no thermal data)"

    mean_val = float(cwsi.mean())
    std_val = float(cwsi.std())
    HIGH_STRESS_THRESHOLD = 0.5
    high_frac = float((cwsi > HIGH_STRESS_THRESHOLD).mean())

    image_path = _save_image(cwsi, f"CWSI (mean={mean_val:.3f})", "cwsi.png", vmin=0.0, vmax=1.0)

    stress_msg = "water stress: "
    if mean_val > HIGH_STRESS_THRESHOLD:
        stress_msg += "HIGH"
    elif mean_val > 0.4:
        stress_msg += "Moderate"
    elif mean_val > 0.2:
        stress_msg += "Mild"
    else:
        stress_msg += "Low"

    summary = (
        f"CWSI for rows {region.row_min}-{region.row_max}, "
        f"cols {region.col_min}-{region.col_max} ({crop_type}): "
        f"mean={mean_val:.3f}, std={std_val:.3f}. "
        f"Pixels above 0.5 (stressed): {high_frac:.1%}. "
        f"VPD={vpd_kpa:.2f} kPa, Tair={air_temp_f:.1f}F. "
        f"Method: {method}. {stress_msg}."
    )
    return CWSIResult(
        success=True,
        cwsi_array=cwsi,
        mean=mean_val,
        std=std_val,
        high_fraction=high_frac,
        vpd=vpd_kpa,
        air_temp_f=air_temp_f,
        image_path=image_path,
        summary=summary,
    )


def get_pixel_timeseries(lat: float, lon: float, index: str) -> TimeseriesResult:
    """Return a time series of a spectral index at a pixel location.

    lat/lon are pixel row/col coordinates for this synthetic dataset.
    index: one of "ndvi", "ndwi", "evi".
    Useful for distinguishing new stress onset from persistent conditions.
    """
    if index not in ("ndvi", "ndwi", "evi"):
        return TimeseriesResult(
            success=False,
            error_message=f"Unsupported index {index!r}. Choose from: ndvi, ndwi, evi.",
        )
    row, col = int(lat), int(lon)
    dates, ndvi_vals = _scene.get_timeseries_ndvi(row=row, col=col)

    # Derive ndwi/evi approximations from stored ndvi timeseries
    if index == "ndvi":
        values = ndvi_vals
    elif index == "ndwi":
        values = [round(v - 0.30, 3) for v in ndvi_vals]  # synthetic approximation
    else:  # evi
        values = [round(v * 0.85, 3) for v in ndvi_vals]

    first_half = sum(values[:2]) / 2
    second_half = sum(values[-2:]) / 2
    delta = second_half - first_half
    if delta < -0.10:
        trend = f"declining (Δ≈{delta:+.2f} — possible recent stress onset)"
    elif delta > 0.10:
        trend = f"improving (Δ≈{delta:+.2f})"
    else:
        trend = f"stable (Δ≈{delta:+.2f})"

    lines = [f"Timeseries for {index.upper()} at pixel ({row}, {col}):"]
    lines += [f"  {d}: {v:.3f}" for d, v in zip(dates, values)]
    lines.append(f"Trend: {trend}.")
    return TimeseriesResult(
        success=True, dates=dates, values=values, index=index, summary="\n".join(lines)
    )


def flag_anomalous_regions(index: str, threshold: float, direction: str) -> AnomalyResult:
    """Find scene grid cells where >30% of pixels exceed the threshold.

    Divides the scene into a 4x4 grid and flags anomalous grid cells.
    Returns bounding boxes, pixel counts, and mean index values per region.
    """
    if index not in ("ndvi", "ndwi", "evi"):
        return AnomalyResult(
            success=False,
            error_message=f"Unsupported index {index!r}. Choose from: ndvi, ndwi, evi.",
        )
    if direction not in ("below", "above"):
        return AnomalyResult(
            success=False,
            error_message=f"direction must be 'below' or 'above', got {direction!r}.",
        )
    H, W = _scene.scene_shape()
    full = BoundingBox(0, H, 0, W)
    arr = _compute_index_array(index, full)
    mask = arr < threshold if direction == "below" else arr > threshold
    total_pixels = int(mask.sum())

    if total_pixels == 0:
        return AnomalyResult(
            success=True,
            regions=[],
            total_anomalous_pixels=0,
            summary=f"No anomalous pixels found: {index.upper()} {direction} {threshold:.2f}.",
        )

    GRID = 4
    cell_h, cell_w = H // GRID, W // GRID
    regions = []
    for gi in range(GRID):
        for gj in range(GRID):
            r0, r1 = gi * cell_h, (gi + 1) * cell_h
            c0, c1 = gj * cell_w, (gj + 1) * cell_w
            cell_mask = mask[r0:r1, c0:c1]
            cell_count = int(cell_mask.sum())
            if cell_count > cell_h * cell_w * 0.3:
                regions.append(
                    {
                        "bbox": {"row_min": r0, "row_max": r1, "col_min": c0, "col_max": c1},
                        "pixel_count": cell_count,
                        "mean_value": round(float(arr[r0:r1, c0:c1][cell_mask].mean()), 3),
                    }
                )

    lines = [
        f"Found {len(regions)} anomalous region(s) where {index.upper()} is {direction} {threshold:.2f}.",
        f"Total anomalous pixels: {total_pixels} ({total_pixels / (H * W):.1%} of scene).",
    ]
    for i, r in enumerate(regions, 1):
        bb = r["bbox"]
        lines.append(
            f"Region {i}: rows {bb['row_min']}–{bb['row_max']}, "
            f"cols {bb['col_min']}–{bb['col_max']}, "
            f"{r['pixel_count']} px, mean {index.upper()}={r['mean_value']:.3f}."
        )
    return AnomalyResult(
        success=True,
        regions=regions,
        total_anomalous_pixels=total_pixels,
        summary="\n".join(lines),
    )


def compare_to_baseline(region: BoundingBox, index: str) -> DiffResult:
    """Compare current index values against the stored baseline from an earlier date.

    Returns a change map and degraded-pixel fraction.
    Returns an error result if no baseline exists for the scene.
    """
    try:
        base_nir = _scene.get_baseline_band("B08", region)
        base_red = _scene.get_baseline_band("B04", region)
    except FileNotFoundError as exc:
        return DiffResult(
            success=False,
            error_message=(
                f"No baseline image found for this region. Cannot compute change. ({exc})"
            ),
        )
    if index not in ("ndvi", "ndwi", "evi"):
        return DiffResult(success=False, error_message=f"Unsupported index {index!r}.")

    current_arr = _compute_index_array(index, region)

    # Compute baseline index from raw bands
    if index == "ndvi":
        baseline_arr = (base_nir - base_red) / (base_nir + base_red + 1e-8)
    elif index == "ndwi":
        base_green = _scene.get_baseline_band("B03", region)
        base_swir = _scene.get_baseline_band("B11", region)
        baseline_arr = (base_green - base_swir) / (base_green + base_swir + 1e-8)
    else:  # evi
        base_blue = _scene.get_baseline_band("B02", region)
        denom = base_nir + 6 * base_red - 7.5 * base_blue + 1
        baseline_arr = np.clip(2.5 * (base_nir - base_red) / (denom + 1e-8), -1.0, 1.0)

    diff = current_arr - baseline_arr
    mean_change = float(diff.mean())
    degraded_frac = float((diff < -0.10).mean())
    image_path = _save_image(
        diff,
        f"Delta{index.upper()} vs baseline",
        f"diff_{index}.png",
        vmin=-0.5,
        vmax=0.5,
    )
    meta = _scene.get_metadata()
    baseline_date = meta.get("baseline_date", "unknown date")
    direction = "decreased" if mean_change < 0 else "increased"
    alert = (
        "Significant degradation since baseline."
        if degraded_frac > 0.20
        else "Minor changes since baseline."
    )
    summary = (
        f"Delta{index.upper()} vs baseline ({baseline_date}): "
        f"mean change = {mean_change:+.3f} ({direction}). "
        f"Pixels with >0.1 degradation: {degraded_frac:.1%}. {alert}"
    )
    return DiffResult(
        success=True,
        diff_array=diff,
        mean_change=mean_change,
        degraded_fraction=degraded_frac,
        image_path=image_path,
        summary=summary,
    )


def search_agricultural_context(query: str, top_k: int = 3) -> _rag.RAGResult:
    """Search local agricultural reference documents for relevant context.

    Searches county crop reports, UC Cooperative Extension bulletins,
    water district advisories, and spectral index reference guides.
    Use to get local growing context, interpret index values for specific crops,
    or understand regional water/weather conditions.
    """
    return _rag.search_agricultural_context(query, top_k=top_k)
