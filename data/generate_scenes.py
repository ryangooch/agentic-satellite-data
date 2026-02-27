# data/generate_scenes.py
"""Generate synthetic Sentinel-2-like scenes for the agentic satellite lecture.

Scene A: Clear water-stress anomaly in NW quadrant. Used for scripted walkthrough.
Scene B: Mild uniform stress with no spatial anomaly. Used for live demo.

Run: python data/generate_scenes.py
"""
import json
import numpy as np
from pathlib import Path

RNG = np.random.default_rng(42)
H, W = 200, 200
SCENES_DIR = Path("data/scenes")
SCENES_DIR.mkdir(parents=True, exist_ok=True)


def _healthy_bands(h=H, w=W) -> dict:
    """Return reflectance arrays for healthy irrigated crops."""
    return {
        "B02": RNG.uniform(0.03, 0.07, (h, w)).astype(np.float32),  # Blue
        "B03": RNG.uniform(0.05, 0.09, (h, w)).astype(np.float32),  # Green
        "B04": RNG.uniform(0.04, 0.09, (h, w)).astype(np.float32),  # Red
        "B08": RNG.uniform(0.42, 0.60, (h, w)).astype(np.float32),  # NIR
        "B8A": RNG.uniform(0.40, 0.58, (h, w)).astype(np.float32),  # Red-edge
        "B11": RNG.uniform(0.08, 0.18, (h, w)).astype(np.float32),  # SWIR
        "B12": RNG.uniform(0.05, 0.14, (h, w)).astype(np.float32),  # SWIR2
    }


def _apply_water_stress(bands: dict, rows: slice, cols: slice) -> None:
    """Modify bands in-place to simulate water stress.

    Water stress = lower NIR (canopy wilts) + higher SWIR (less leaf water).
    Result: NDVI drops to 0.15–0.35, NDWI = (Green-SWIR)/(Green+SWIR) goes negative.
    """
    h = rows.stop - rows.start
    w = cols.stop - cols.start
    bands["B08"][rows, cols] = RNG.uniform(0.18, 0.32, (h, w)).astype(np.float32)
    bands["B04"][rows, cols] = RNG.uniform(0.10, 0.18, (h, w)).astype(np.float32)
    bands["B11"][rows, cols] = RNG.uniform(0.26, 0.38, (h, w)).astype(np.float32)
    bands["B12"][rows, cols] = RNG.uniform(0.18, 0.28, (h, w)).astype(np.float32)


def generate_scene_a() -> None:
    # Current: NW quadrant is water-stressed
    bands = _healthy_bands()
    _apply_water_stress(bands, slice(0, 100), slice(0, 100))
    np.save(SCENES_DIR / "scene_a_bands.npy", bands)

    # Baseline (30 days ago): all healthy
    np.save(SCENES_DIR / "scene_a_baseline_bands.npy", _healthy_bands())

    # Timeseries: 5 dates, ~60-day window
    # Stressed point (row=50, col=50) shows sharp decline in last 2 dates
    dates = ["2023-05-15", "2023-06-01", "2023-06-15", "2023-07-01", "2023-07-15"]
    nir_s = [0.55, 0.52, 0.48, 0.28, 0.22]
    red_s = [0.07, 0.07, 0.09, 0.14, 0.16]
    ndvi_stressed = [round((n - r) / (n + r), 3) for n, r in zip(nir_s, red_s)]

    nir_h = [0.54, 0.55, 0.57, 0.56, 0.55]
    red_h = [0.07, 0.07, 0.07, 0.07, 0.07]
    ndvi_healthy = [round((n - r) / (n + r), 3) for n, r in zip(nir_h, red_h)]

    meta = {
        "scene_id": "scene_a",
        "date": "2023-07-15",
        "region": "Central Valley, CA (synthetic)",
        "bands": ["B02", "B03", "B04", "B08", "B8A", "B11", "B12"],
        "shape": [200, 200],
        "pixel_size_m": 10,
        "has_baseline": True,
        "baseline_date": "2023-06-15",
        "timeseries": {
            "dates": dates,
            "stressed_point": {"row": 50, "col": 50, "ndvi": ndvi_stressed},
            "healthy_point": {"row": 150, "col": 150, "ndvi": ndvi_healthy},
        },
    }
    (SCENES_DIR / "scene_a_metadata.json").write_text(json.dumps(meta, indent=2))
    print("Scene A generated.")


def generate_scene_b() -> None:
    # Mild, spatially uniform NDVI depression — ambiguous
    bands = {
        "B02": RNG.uniform(0.03, 0.07, (H, W)).astype(np.float32),
        "B03": RNG.uniform(0.05, 0.09, (H, W)).astype(np.float32),
        "B04": RNG.uniform(0.09, 0.14, (H, W)).astype(np.float32),  # slightly elevated red
        "B08": RNG.uniform(0.33, 0.48, (H, W)).astype(np.float32),  # slightly depressed NIR
        "B8A": RNG.uniform(0.31, 0.46, (H, W)).astype(np.float32),
        "B11": RNG.uniform(0.12, 0.22, (H, W)).astype(np.float32),  # mildly elevated SWIR
        "B12": RNG.uniform(0.08, 0.17, (H, W)).astype(np.float32),
    }
    np.save(SCENES_DIR / "scene_b_bands.npy", bands)
    # No baseline file for scene B — compare_to_baseline must fail gracefully

    dates = ["2023-05-15", "2023-06-01", "2023-06-15", "2023-07-01", "2023-07-15"]
    ndvi_vals = [0.45, 0.46, 0.44, 0.43, 0.44]  # stable mild depression

    meta = {
        "scene_id": "scene_b",
        "date": "2023-07-15",
        "region": "Central Valley, CA (synthetic)",
        "bands": ["B02", "B03", "B04", "B08", "B8A", "B11", "B12"],
        "shape": [200, 200],
        "pixel_size_m": 10,
        "has_baseline": False,
        "timeseries": {
            "dates": dates,
            "representative_point": {"row": 100, "col": 100, "ndvi": ndvi_vals},
        },
    }
    (SCENES_DIR / "scene_b_metadata.json").write_text(json.dumps(meta, indent=2))
    print("Scene B generated.")


if __name__ == "__main__":
    generate_scene_a()
    generate_scene_b()
    print("All scenes saved to data/scenes/")
