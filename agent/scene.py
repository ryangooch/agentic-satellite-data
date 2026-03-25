# agent/scene.py
"""Scene loader — manages the currently active scene's band data and metadata.

Call `load_scene("scene_a")` before using any tool.
Tools access bands via `get_band()` rather than reading disk directly.
"""
import json
import numpy as np
from pathlib import Path
from typing import Optional

from agent.models import BoundingBox

_CURRENT_SCENE: dict = {}
_CURRENT_META: dict = {}
_CURRENT_BASELINE: dict = {}
_CURRENT_CDL: Optional[np.ndarray] = None
_CROP_MASK: Optional[np.ndarray] = None  # True = include pixel
_SCENES_DIR = Path("data/scenes")

# USDA CDL crop codes (subset covering Central Valley agriculture)
CDL_CODES = {
    "corn": 1, "cotton": 2, "rice": 3, "sorghum": 4,
    "barley": 21, "winter wheat": 24, "alfalfa": 36,
    "other hay": 37, "tomatoes": 54, "grapes": 69,
    "almonds": 75, "walnuts": 76, "pistachios": 204,
    "olives": 211, "prunes": 210, "cherries": 66,
    "peaches": 67, "pomegranates": 217, "citrus": 72,
}


def load_scene(scene_id: str) -> None:
    """Load a scene into module-level state. Must be called before using tools."""
    # globals aren't super great practice in Python but we do it here to simplify the notebooks
    global _CURRENT_SCENE, _CURRENT_META, _CURRENT_BASELINE, _CURRENT_CDL, _CROP_MASK
    bands_path = _SCENES_DIR / f"{scene_id}_bands.npy"
    meta_path = _SCENES_DIR / f"{scene_id}_metadata.json"
    if not bands_path.exists():
        raise FileNotFoundError(f"Scene bands not found: {bands_path}")
    _CURRENT_SCENE = np.load(bands_path, allow_pickle=True).item()
    _CURRENT_META = json.loads(meta_path.read_text())
    baseline_path = _SCENES_DIR / f"{scene_id}_baseline_bands.npy"
    _CURRENT_BASELINE = (
        np.load(baseline_path, allow_pickle=True).item()
        if baseline_path.exists() else {}
    )
    # Load CDL if available
    cdl_path = _SCENES_DIR / f"{scene_id}_cdl_2025.npy"
    _CURRENT_CDL = np.load(cdl_path) if cdl_path.exists() else None
    _CROP_MASK = None  # reset on scene load


def set_crop_mask(crop_name: str) -> int:
    """Activate a crop mask so get_band() returns NaN for non-crop pixels.

    Returns the number of pixels matching the crop.
    Raises ValueError if crop name is unknown or no CDL data is loaded.
    """
    global _CROP_MASK
    if _CURRENT_CDL is None:
        raise ValueError("No CDL data available for this scene.")
    key = crop_name.lower()
    if key not in CDL_CODES:
        raise ValueError(
            f"Unknown crop {crop_name!r}. Available: {', '.join(sorted(CDL_CODES))}"
        )
    _CROP_MASK = _CURRENT_CDL == CDL_CODES[key]
    count = int(_CROP_MASK.sum())
    if count == 0:
        _CROP_MASK = None
        raise ValueError(f"No pixels found for crop {crop_name!r} in this scene.")
    return count


def clear_crop_mask() -> None:
    """Remove the active crop mask so all pixels are included."""
    global _CROP_MASK
    _CROP_MASK = None


def get_crop_mask(region: Optional[BoundingBox] = None) -> Optional[np.ndarray]:
    """Return the current crop mask (True=include), optionally cropped to region."""
    if _CROP_MASK is None:
        return None
    mask = _CROP_MASK
    if region is not None:
        shape = mask.shape
        region = _clamp_region(region, shape)
        mask = mask[region.row_min:region.row_max, region.col_min:region.col_max]
    return mask


def _clamp_region(region: BoundingBox, shape: tuple) -> BoundingBox:
    """Clamp a bounding box to valid array bounds."""
    h, w = shape
    return BoundingBox(
        row_min=max(0, min(region.row_min, h)),
        row_max=max(0, min(region.row_max, h)),
        col_min=max(0, min(region.col_min, w)),
        col_max=max(0, min(region.col_max, w)),
    )


def get_band(band_name: str, region: Optional[BoundingBox] = None) -> np.ndarray:
    """Return a band array, optionally cropped to a BoundingBox."""
    if band_name not in _CURRENT_SCENE:
        raise KeyError(
            f"Band {band_name!r} not in current scene. "
            f"Available: {sorted(_CURRENT_SCENE.keys())}"
        )
    arr = _CURRENT_SCENE[band_name]
    if region is not None:
        region = _clamp_region(region, arr.shape)
        arr = arr[region.row_min:region.row_max, region.col_min:region.col_max]
    if _CROP_MASK is not None:
        mask = get_crop_mask(region)
        arr = arr.astype(float, copy=True)
        arr[~mask] = np.nan
    return arr


def get_baseline_band(band_name: str, region: Optional[BoundingBox] = None) -> np.ndarray:
    """Return a baseline band array. Raises FileNotFoundError if no baseline exists."""
    if not _CURRENT_BASELINE:
        raise FileNotFoundError(
            "No baseline available for the current scene. "
            f"Scene '{_CURRENT_META.get('scene_id')}' has has_baseline=False."
        )
    if band_name not in _CURRENT_BASELINE:
        raise KeyError(f"Band {band_name!r} not in baseline scene.")
    arr = _CURRENT_BASELINE[band_name]
    if region is not None:
        region = _clamp_region(region, arr.shape)
        arr = arr[region.row_min:region.row_max, region.col_min:region.col_max]
    if _CROP_MASK is not None:
        mask = get_crop_mask(region)
        arr = arr.astype(float, copy=True)
        arr[~mask] = np.nan
    return arr


def get_metadata() -> dict:
    """Return the current scene metadata dict."""
    return _CURRENT_META


def get_timeseries_ndvi(row: int, col: int):
    """Return (dates, ndvi_values) timeseries for a pixel.

    Uses the nearest stored point from scene metadata — either the
    stressed_point, healthy_point, or representative_point.
    """
    ts = _CURRENT_META.get("timeseries", {})
    dates = ts.get("dates", [])
    if "stressed_point" in ts:
        sp = ts["stressed_point"]
        hp = ts.get("healthy_point", sp)
        dist_s = abs(row - sp["row"]) + abs(col - sp["col"])
        dist_h = abs(row - hp["row"]) + abs(col - hp["col"])
        ndvi_vals = sp["ndvi"] if dist_s <= dist_h else hp["ndvi"]
    elif "representative_point" in ts:
        ndvi_vals = ts["representative_point"]["ndvi"]
    else:
        ndvi_vals = [0.5] * len(dates)
    return dates, ndvi_vals


def scene_shape() -> tuple:
    """Return (height, width) of the current scene."""
    if not _CURRENT_SCENE:
        raise RuntimeError("No scene loaded. Call load_scene() first.")
    return next(iter(_CURRENT_SCENE.values())).shape
