# Agentic Satellite Lecture: Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a complete, self-contained lecture repository demonstrating agentic AI applied to Sentinel-2 satellite crop health assessment — synthetic scenes, pure-Python tools, a ~80-line agent loop, and four teaching notebooks.

**Architecture:** Synthetic Sentinel-2-like NumPy scenes; flat tool functions (no tool calls other tools); Anthropic Messages API agent loop with no framework; pytest for unit tests; mock mode replays pre-recorded JSON traces for offline demos.

**Tech Stack:** Python 3.11+, `anthropic>=0.40`, `numpy>=1.26`, `matplotlib>=3.8`, `jupyter`, `pytest>=8.0`

---

## Directory Structure (final state)

```
agentic-satellite-data/
├── README.md
├── Makefile
├── requirements.txt
├── EXERCISES.md
├── agent/
│   ├── __init__.py
│   ├── types.py        # BoundingBox + Result dataclasses
│   ├── scene.py        # Scene loader (module-level state)
│   ├── tools.py        # 6 tool functions
│   ├── schemas.py      # JSON schemas for Anthropic API
│   ├── loop.py         # ~80-line agent loop
│   └── mock.py         # Replay pre-recorded traces
├── data/
│   ├── generate_scenes.py
│   ├── download_data.py
│   ├── mock_responses/scene_a.json
│   └── scenes/         # Generated: scene_a_bands.npy, etc.
├── docs/plans/         # This file
├── notebooks/
│   ├── 00_data_exploration.ipynb
│   ├── 01_tools.ipynb
│   ├── 02_agent_loop.ipynb
│   └── 03_failure_modes.ipynb
├── slides/             # Placeholder dir
└── tests/
    ├── conftest.py
    ├── test_types.py
    ├── test_scene.py
    ├── test_tools.py
    ├── test_schemas.py
    ├── test_loop.py
    └── test_mock.py
```

---

## Key Design Decisions

- **Synthetic data only** — No real Sentinel-2 download required to run. `data/generate_scenes.py` creates deterministic `.npy` scenes. `data/download_data.py` is a bonus script for students.
- **Flat tools** — Tools never call each other. A private `_compute_index_array(index, region)` helper in `tools.py` handles shared band math.
- **Module-level scene state** — `agent/scene.py` holds `_CURRENT_SCENE` as a module global. Tests call `load_scene("scene_a")` in a fixture.
- **Model** — Use `claude-sonnet-4-6`. Never smaller; tool-call quality degrades noticeably.
- **Mock mode** — `agent/mock.py` replays `data/mock_responses/scene_a.json`. Real tool calls are executed (so output is live), but model responses are canned.

---

## Phase 1: Foundation

### Task 1: Project Scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `Makefile`
- Create: `agent/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `slides/.gitkeep`
- Modify: `README.md`

**Step 1: Create directories and placeholder files**

```bash
mkdir -p agent tests data/scenes data/mock_responses notebooks slides docs/plans
touch agent/__init__.py tests/__init__.py slides/.gitkeep
```

**Step 2: Write `requirements.txt`**

```
anthropic>=0.40.0
numpy>=1.26.0
matplotlib>=3.8.0
jupyter>=1.0.0
nbconvert>=7.0.0
pytest>=8.0.0
pytest-cov>=5.0.0
```

**Step 3: Write `Makefile`**

```makefile
.PHONY: setup generate-data test demo download-data

setup:
	pip install -r requirements.txt

generate-data:
	python data/generate_scenes.py

test: generate-data
	pytest tests/ -v

demo: generate-data
	python -m agent.loop --scene scene_a --mock

download-data:
	python data/download_data.py --region "37.5,-120.5,37.7,-120.3" --date "2023-07-15"
```

**Step 4: Write `tests/conftest.py`**

```python
# tests/conftest.py
import subprocess
from pathlib import Path
import pytest


@pytest.fixture(scope="session", autouse=True)
def ensure_scenes_exist():
    """Generate synthetic scenes if they don't exist yet."""
    if not Path("data/scenes/scene_a_bands.npy").exists():
        subprocess.run(["python", "data/generate_scenes.py"], check=True)
```

**Step 5: Replace `README.md`**

```markdown
# Agentic AI for Satellite Crop Health Assessment

A 75-minute lecture for senior ECE undergraduates demonstrating agentic AI
applied to Sentinel-2 satellite imagery.

## Quick Start

```bash
pip install -r requirements.txt
make demo          # Run Scene A in mock mode — no API key needed
```

## Notebooks

- `00_data_exploration.ipynb` — Band visualization, orientation
- `01_tools.ipynb` — Tools as plain Python functions
- `02_agent_loop.ipynb` — **Start here** — main teaching notebook
- `03_failure_modes.ipynb` — Where agents break (take-home)

## Live API Runs

Set `ANTHROPIC_API_KEY` in your environment, then:

```bash
python -m agent.loop --scene scene_a
python -m agent.loop --scene scene_b
```

Estimated cost: $0.10–0.30 per run with `claude-sonnet-4-6`.

## Running Tests

```bash
make test
```

See `EXERCISES.md` for take-home assignments.
```

**Step 6: Commit**

```bash
git add requirements.txt Makefile agent/__init__.py tests/__init__.py \
        tests/conftest.py slides/.gitkeep README.md docs/
git commit -m "feat: project scaffolding - dirs, requirements, Makefile"
```

---

### Task 2: Data Types

**Files:**
- Create: `agent/types.py`
- Create: `tests/test_types.py`

**Step 1: Write the failing test**

```python
# tests/test_types.py
import numpy as np
from agent.types import (
    BoundingBox, NDVIResult, NDWIResult, EVIResult,
    TimeseriesResult, AnomalyResult, DiffResult,
)


def test_bounding_box_properties():
    bb = BoundingBox(row_min=10, row_max=110, col_min=5, col_max=55)
    assert bb.height == 100
    assert bb.width == 50


def test_bounding_box_to_dict():
    bb = BoundingBox(0, 100, 0, 100)
    d = bb.to_dict()
    assert d == {"row_min": 0, "row_max": 100, "col_min": 0, "col_max": 100}


def test_ndvi_result_success():
    arr = np.zeros((10, 10))
    r = NDVIResult(success=True, ndvi_array=arr, mean=0.5, std=0.1,
                   low_fraction=0.2, summary="NDVI mean: 0.5")
    assert r.success is True
    assert r.error_message is None


def test_ndvi_result_failure():
    r = NDVIResult(success=False, error_message="Missing NIR band")
    assert r.success is False
    assert r.ndvi_array is None


def test_all_result_types_have_required_fields():
    for cls in (NDVIResult, NDWIResult, EVIResult, AnomalyResult, DiffResult):
        r = cls(success=False, error_message="test error")
        assert hasattr(r, "success")
        assert hasattr(r, "error_message")
        assert hasattr(r, "summary")
```

**Step 2: Run to confirm failure**

```bash
pytest tests/test_types.py -v
```
Expected: `ModuleNotFoundError: No module named 'agent.types'`

**Step 3: Write `agent/types.py`**

```python
# agent/types.py
from dataclasses import dataclass
from typing import Optional, List
import numpy as np


@dataclass
class BoundingBox:
    row_min: int
    row_max: int
    col_min: int
    col_max: int

    @property
    def height(self) -> int:
        return self.row_max - self.row_min

    @property
    def width(self) -> int:
        return self.col_max - self.col_min

    def to_dict(self) -> dict:
        return {
            "row_min": self.row_min, "row_max": self.row_max,
            "col_min": self.col_min, "col_max": self.col_max,
        }


@dataclass
class NDVIResult:
    success: bool
    ndvi_array: Optional[np.ndarray] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    low_fraction: Optional[float] = None   # fraction of pixels below 0.3
    image_path: Optional[str] = None
    summary: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class NDWIResult:
    success: bool
    ndwi_array: Optional[np.ndarray] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    negative_fraction: Optional[float] = None  # fraction with NDWI < 0
    image_path: Optional[str] = None
    summary: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class EVIResult:
    success: bool
    evi_array: Optional[np.ndarray] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    low_fraction: Optional[float] = None
    image_path: Optional[str] = None
    summary: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class TimeseriesResult:
    success: bool
    dates: Optional[List[str]] = None
    values: Optional[List[float]] = None
    index: Optional[str] = None
    summary: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class AnomalyResult:
    success: bool
    # Each entry: {"bbox": {row_min,row_max,col_min,col_max}, "pixel_count": int, "mean_value": float}
    regions: Optional[List[dict]] = None
    total_anomalous_pixels: Optional[int] = None
    summary: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class DiffResult:
    success: bool
    diff_array: Optional[np.ndarray] = None
    mean_change: Optional[float] = None
    degraded_fraction: Optional[float] = None
    image_path: Optional[str] = None
    summary: Optional[str] = None
    error_message: Optional[str] = None
```

**Step 4: Run to verify passing**

```bash
pytest tests/test_types.py -v
```
Expected: 5 tests PASS

**Step 5: Commit**

```bash
git add agent/types.py tests/test_types.py
git commit -m "feat: add data types - BoundingBox and result dataclasses"
```

---

### Task 3: Synthetic Scene Generation

**Files:**
- Create: `data/generate_scenes.py`
- Create: `tests/test_generate_scenes.py`

**Scene A (scripted):** 200×200 px, 7 bands. NW quadrant (rows 0–100, cols 0–100) has water stress: depressed NIR, elevated SWIR → NDVI 0.15–0.35, NDWI strongly negative. The rest is healthy: NDVI 0.60–0.80. A baseline from 30 days prior (all healthy) is stored. Timeseries shows 5 dates over 60 days with a sharp NDVI decline in the last two dates at the stressed point.

**Scene B (ambiguous):** 200×200 px. Mild uniform NDVI depression (0.40–0.55) with no strong spatial anomaly. NDWI neutral. No baseline file (so `compare_to_baseline` fails gracefully). Timeseries is stable — not a recent onset.

**Step 1: Write the failing test**

```python
# tests/test_generate_scenes.py
import json
import numpy as np
from pathlib import Path


def test_scene_a_files_exist():
    d = Path("data/scenes")
    assert (d / "scene_a_bands.npy").exists()
    assert (d / "scene_a_metadata.json").exists()
    assert (d / "scene_a_baseline_bands.npy").exists()


def test_scene_a_bands_shape_and_dtype():
    data = np.load("data/scenes/scene_a_bands.npy", allow_pickle=True).item()
    assert set(data.keys()) == {"B02", "B03", "B04", "B08", "B8A", "B11", "B12"}
    for band in data.values():
        assert band.shape == (200, 200)
        assert band.dtype == np.float32


def test_scene_a_ndvi_anomaly_in_nw():
    data = np.load("data/scenes/scene_a_bands.npy", allow_pickle=True).item()
    nir, red = data["B08"], data["B04"]
    ndvi = (nir - red) / (nir + red + 1e-8)
    ndvi_nw = ndvi[:100, :100].mean()
    ndvi_se = ndvi[100:, 100:].mean()
    assert ndvi_nw < 0.40, f"NW NDVI {ndvi_nw:.3f} should be below 0.40"
    assert ndvi_se > 0.55, f"SE NDVI {ndvi_se:.3f} should be above 0.55"


def test_scene_a_timeseries_has_five_dates():
    meta = json.loads(Path("data/scenes/scene_a_metadata.json").read_text())
    assert len(meta["timeseries"]["dates"]) == 5


def test_scene_b_files_exist():
    d = Path("data/scenes")
    assert (d / "scene_b_bands.npy").exists()
    assert (d / "scene_b_metadata.json").exists()


def test_scene_b_no_baseline():
    assert not Path("data/scenes/scene_b_baseline_bands.npy").exists()


def test_scene_b_ndvi_mild_uniform():
    data = np.load("data/scenes/scene_b_bands.npy", allow_pickle=True).item()
    nir, red = data["B08"], data["B04"]
    ndvi = (nir - red) / (nir + red + 1e-8)
    # Mild depression, no quadrant more than 0.2 below overall mean
    overall = ndvi.mean()
    assert 0.35 < overall < 0.60, f"Scene B mean NDVI {overall:.3f} not in mild range"
    nw_mean = ndvi[:100, :100].mean()
    se_mean = ndvi[100:, 100:].mean()
    assert abs(nw_mean - se_mean) < 0.20, "Scene B should have no strong spatial gradient"
```

**Step 2: Run to confirm failure**

```bash
pytest tests/test_generate_scenes.py -v
```
Expected: FAIL (files don't exist yet)

**Step 3: Write `data/generate_scenes.py`**

```python
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
```

**Step 4: Run the script then verify tests**

```bash
python data/generate_scenes.py
pytest tests/test_generate_scenes.py -v
```
Expected: Script prints "Scene A generated. Scene B generated." then all 7 tests PASS

**Step 5: Commit**

```bash
git add data/generate_scenes.py tests/test_generate_scenes.py
git commit -m "feat: synthetic scene generation (Scene A water-stress, Scene B ambiguous)"
```

---

## Phase 2: Tools

### Task 4: Scene Loader

**Files:**
- Create: `agent/scene.py`
- Create: `tests/test_scene.py`

**Step 1: Write the failing test**

```python
# tests/test_scene.py
import pytest
import numpy as np
from agent.scene import load_scene, get_band, get_baseline_band, get_metadata, get_timeseries_ndvi, scene_shape
from agent.types import BoundingBox


@pytest.fixture(autouse=True)
def reset_to_scene_a():
    load_scene("scene_a")


def test_get_band_returns_full_array():
    nir = get_band("B08")
    assert isinstance(nir, np.ndarray)
    assert nir.shape == (200, 200)


def test_get_band_with_region():
    bb = BoundingBox(0, 50, 0, 50)
    nir = get_band("B08", region=bb)
    assert nir.shape == (50, 50)


def test_get_band_invalid_raises():
    with pytest.raises(KeyError):
        get_band("B99")


def test_get_metadata():
    meta = get_metadata()
    assert meta["scene_id"] == "scene_a"
    assert meta["has_baseline"] is True


def test_get_timeseries_stressed_point_declines():
    dates, values = get_timeseries_ndvi(row=50, col=50)
    assert len(dates) == 5
    assert values[-1] < values[0], "Stressed point should show NDVI decline"


def test_get_baseline_band_scene_a():
    arr = get_baseline_band("B08")
    assert arr.shape == (200, 200)


def test_get_baseline_band_scene_b_raises():
    load_scene("scene_b")
    with pytest.raises(FileNotFoundError):
        get_baseline_band("B08")


def test_scene_shape():
    assert scene_shape() == (200, 200)
```

**Step 2: Run to confirm failure**

```bash
pytest tests/test_scene.py -v
```
Expected: `ModuleNotFoundError: No module named 'agent.scene'`

**Step 3: Write `agent/scene.py`**

```python
# agent/scene.py
"""Scene loader — manages the currently active scene's band data and metadata.

Call `load_scene("scene_a")` before using any tool.
Tools access bands via `get_band()` rather than reading disk directly.
"""
import json
import numpy as np
from pathlib import Path
from agent.types import BoundingBox

_CURRENT_SCENE: dict = {}
_CURRENT_META: dict = {}
_CURRENT_BASELINE: dict = {}
_SCENES_DIR = Path("data/scenes")


def load_scene(scene_id: str) -> None:
    """Load a scene into module-level state. Must be called before using tools."""
    global _CURRENT_SCENE, _CURRENT_META, _CURRENT_BASELINE
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


def get_band(band_name: str, region: BoundingBox = None) -> np.ndarray:
    """Return a band array, optionally cropped to a BoundingBox."""
    if band_name not in _CURRENT_SCENE:
        raise KeyError(
            f"Band {band_name!r} not in current scene. "
            f"Available: {sorted(_CURRENT_SCENE.keys())}"
        )
    arr = _CURRENT_SCENE[band_name]
    if region is not None:
        arr = arr[region.row_min:region.row_max, region.col_min:region.col_max]
    return arr


def get_baseline_band(band_name: str, region: BoundingBox = None) -> np.ndarray:
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
        arr = arr[region.row_min:region.row_max, region.col_min:region.col_max]
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
```

**Step 4: Run to verify passing**

```bash
pytest tests/test_scene.py -v
```
Expected: All 8 tests PASS

**Step 5: Commit**

```bash
git add agent/scene.py tests/test_scene.py
git commit -m "feat: scene loader with band access, timeseries, and baseline support"
```

---

### Task 5: Tool — `compute_ndvi`

**Files:**
- Create: `agent/tools.py` (initial version, ndvi only)
- Create: `tests/test_tools.py` (initial version)

**Step 1: Write failing test**

```python
# tests/test_tools.py
import pytest
import numpy as np
from agent.scene import load_scene
from agent.types import BoundingBox
from agent.tools import compute_ndvi


@pytest.fixture(autouse=True)
def load_a():
    load_scene("scene_a")


class TestComputeNDVI:
    def test_success_and_shape(self):
        result = compute_ndvi(BoundingBox(0, 200, 0, 200))
        assert result.success is True
        assert result.ndvi_array.shape == (200, 200)

    def test_ndvi_values_in_range(self):
        result = compute_ndvi(BoundingBox(0, 200, 0, 200))
        assert result.ndvi_array.min() >= -1.0
        assert result.ndvi_array.max() <= 1.0

    def test_nw_quadrant_lower_than_se(self):
        result = compute_ndvi(BoundingBox(0, 200, 0, 200))
        nw = result.ndvi_array[:100, :100].mean()
        se = result.ndvi_array[100:, 100:].mean()
        assert nw < se, f"NW ({nw:.3f}) should be lower than SE ({se:.3f})"

    def test_stats_populated(self):
        result = compute_ndvi(BoundingBox(0, 200, 0, 200))
        assert result.mean is not None
        assert result.std is not None
        assert result.low_fraction is not None
        assert 0.0 <= result.low_fraction <= 1.0

    def test_summary_is_string_with_ndvi(self):
        result = compute_ndvi(BoundingBox(0, 200, 0, 200))
        assert isinstance(result.summary, str)
        assert "NDVI" in result.summary
```

**Step 2: Run to confirm failure**

```bash
pytest tests/test_tools.py::TestComputeNDVI -v
```
Expected: `ModuleNotFoundError: No module named 'agent.tools'`

**Step 3: Write initial `agent/tools.py`**

```python
# agent/tools.py
"""Tool implementations for the crop health agent.

Design rules (also taught in lecture):
- Each tool is a plain Python function — no framework.
- Tools NEVER call other tools. Shared band math lives in `_compute_index_array`.
- Return dataclasses with success/error fields — never raise exceptions.
- Return a human-readable `summary` string for the agent's context window.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from agent.types import (
    BoundingBox, NDVIResult, NDWIResult, EVIResult,
    TimeseriesResult, AnomalyResult, DiffResult,
)
from agent import scene as _scene

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


def _save_image(arr: np.ndarray, title: str, filename: str,
                vmin: float = -1, vmax: float = 1) -> str:
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
    image_path = _save_image(
        ndvi, f"NDVI (mean={mean_val:.3f})", "ndvi.png", vmin=-0.2, vmax=1.0
    )
    alert = "⚠️ Significant stressed area detected." if low_frac > 0.1 else "Vegetation appears generally healthy."
    summary = (
        f"NDVI for rows {region.row_min}–{region.row_max}, "
        f"cols {region.col_min}–{region.col_max}: "
        f"mean={mean_val:.3f}, std={std_val:.3f}. "
        f"Pixels below 0.3 (stressed): {low_frac:.1%}. {alert}"
    )
    return NDVIResult(
        success=True, ndvi_array=ndvi, mean=mean_val, std=std_val,
        low_fraction=low_frac, image_path=image_path, summary=summary,
    )
```

**Step 4: Run to verify passing**

```bash
pytest tests/test_tools.py::TestComputeNDVI -v
```
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add agent/tools.py tests/test_tools.py
git commit -m "feat: add compute_ndvi tool with image rendering"
```

---

### Task 6: Tool — `compute_ndwi`

**Files:**
- Modify: `agent/tools.py`
- Modify: `tests/test_tools.py`

NDWI = (Green − SWIR) / (Green + SWIR) using B03 and B11. Positive = moist canopy. Negative = water stress.

**Step 1: Add failing test class to `tests/test_tools.py`**

```python
from agent.tools import compute_ndwi   # add to imports at top

class TestComputeNDWI:
    def test_success_and_shape(self):
        result = compute_ndwi(BoundingBox(0, 200, 0, 200))
        assert result.success is True
        assert result.ndwi_array.shape == (200, 200)

    def test_nw_more_negative_than_se(self):
        """Water-stressed NW quadrant should have more negative NDWI."""
        result = compute_ndwi(BoundingBox(0, 200, 0, 200))
        nw = result.ndwi_array[:100, :100].mean()
        se = result.ndwi_array[100:, 100:].mean()
        assert nw < se, f"NW NDWI ({nw:.3f}) should be lower than SE ({se:.3f})"

    def test_summary_mentions_ndwi(self):
        result = compute_ndwi(BoundingBox(0, 200, 0, 200))
        assert "NDWI" in result.summary
```

**Step 2: Run to confirm failure**

```bash
pytest tests/test_tools.py::TestComputeNDWI -v
```
Expected: `ImportError`

**Step 3: Add `compute_ndwi` to `agent/tools.py`**

```python
def compute_ndwi(region: BoundingBox) -> NDWIResult:
    """Compute NDWI = (Green - SWIR) / (Green + SWIR) using B03 and B11.

    Positive: canopy moisture present. Negative: water stress or dry soil.
    """
    ndwi = _compute_index_array("ndwi", region)
    mean_val = float(ndwi.mean())
    std_val = float(ndwi.std())
    neg_frac = float((ndwi < 0).mean())
    image_path = _save_image(
        ndwi, f"NDWI (mean={mean_val:.3f})", "ndwi.png", vmin=-0.6, vmax=0.6
    )
    if mean_val < -0.1:
        status = "water-stressed (dry canopy)"
    elif mean_val > 0.1:
        status = "well-watered"
    else:
        status = "moderate moisture"
    summary = (
        f"NDWI for rows {region.row_min}–{region.row_max}, "
        f"cols {region.col_min}–{region.col_max}: "
        f"mean={mean_val:.3f}, std={std_val:.3f}. "
        f"Status: {status}. Negative-NDWI pixels: {neg_frac:.1%}."
    )
    return NDWIResult(
        success=True, ndwi_array=ndwi, mean=mean_val, std=std_val,
        negative_fraction=neg_frac, image_path=image_path, summary=summary,
    )
```

**Step 4: Run to verify passing**

```bash
pytest tests/test_tools.py::TestComputeNDWI -v
```
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add agent/tools.py tests/test_tools.py
git commit -m "feat: add compute_ndwi tool"
```

---

### Task 7: Tool — `compute_evi`

**Files:**
- Modify: `agent/tools.py`
- Modify: `tests/test_tools.py`

EVI = 2.5 × (NIR − Red) / (NIR + 6×Red − 7.5×Blue + 1). Less saturation than NDVI in dense canopy.

**Step 1: Add failing test class**

```python
from agent.tools import compute_evi   # add to imports

class TestComputeEVI:
    def test_success_and_shape(self):
        result = compute_evi(BoundingBox(0, 200, 0, 200))
        assert result.success is True
        assert result.evi_array.shape == (200, 200)

    def test_evi_clipped_range(self):
        result = compute_evi(BoundingBox(0, 200, 0, 200))
        assert result.evi_array.min() >= -1.0
        assert result.evi_array.max() <= 1.0

    def test_summary_mentions_evi(self):
        result = compute_evi(BoundingBox(0, 200, 0, 200))
        assert "EVI" in result.summary
```

**Step 2: Run to confirm failure**

```bash
pytest tests/test_tools.py::TestComputeEVI -v
```
Expected: `ImportError`

**Step 3: Add `compute_evi` to `agent/tools.py`**

```python
def compute_evi(region: BoundingBox) -> EVIResult:
    """Compute EVI = 2.5*(NIR-Red)/(NIR+6*Red-7.5*Blue+1) using B08, B04, B02.

    Less saturation than NDVI in dense canopy. Use to cross-check NDVI findings.
    """
    evi = _compute_index_array("evi", region)
    mean_val = float(evi.mean())
    std_val = float(evi.std())
    low_frac = float((evi < 0.2).mean())
    image_path = _save_image(
        evi, f"EVI (mean={mean_val:.3f})", "evi.png", vmin=0.0, vmax=0.8
    )
    summary = (
        f"EVI for rows {region.row_min}–{region.row_max}, "
        f"cols {region.col_min}–{region.col_max}: "
        f"mean={mean_val:.3f}, std={std_val:.3f}. "
        f"Pixels below 0.2: {low_frac:.1%}. "
        "EVI is less saturated than NDVI in dense canopy — use to verify NDVI findings."
    )
    return EVIResult(
        success=True, evi_array=evi, mean=mean_val, std=std_val,
        low_fraction=low_frac, image_path=image_path, summary=summary,
    )
```

**Step 4: Run to verify passing**

```bash
pytest tests/test_tools.py::TestComputeEVI -v
```
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add agent/tools.py tests/test_tools.py
git commit -m "feat: add compute_evi tool"
```

---

### Task 8: Tool — `get_pixel_timeseries`

**Files:**
- Modify: `agent/tools.py`
- Modify: `tests/test_tools.py`

Note: `lat`/`lon` parameters match the API schema name from the lecture plan, but map to pixel row/col for synthetic data.

**Step 1: Add failing test class**

```python
from agent.tools import get_pixel_timeseries   # add to imports

class TestGetPixelTimeseries:
    def test_success_returns_five_dates(self):
        result = get_pixel_timeseries(lat=50, lon=50, index="ndvi")
        assert result.success is True
        assert len(result.dates) == 5
        assert len(result.values) == 5

    def test_stressed_point_shows_decline(self):
        """Row 50, col 50 is in the stressed NW quadrant."""
        result = get_pixel_timeseries(lat=50, lon=50, index="ndvi")
        assert result.values[-1] < result.values[0], "Stressed point should show NDVI decline"

    def test_unsupported_index_returns_error(self):
        result = get_pixel_timeseries(lat=50, lon=50, index="fakeidx")
        assert result.success is False
        assert result.error_message is not None

    def test_summary_mentions_trend(self):
        result = get_pixel_timeseries(lat=50, lon=50, index="ndvi")
        low = result.summary.lower()
        assert "trend" in low or "decline" in low or "stable" in low
```

**Step 2: Run to confirm failure**

```bash
pytest tests/test_tools.py::TestGetPixelTimeseries -v
```

**Step 3: Add `get_pixel_timeseries` to `agent/tools.py`**

```python
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
    return TimeseriesResult(success=True, dates=dates, values=values, index=index,
                            summary="\n".join(lines))
```

**Step 4: Run to verify passing**

```bash
pytest tests/test_tools.py::TestGetPixelTimeseries -v
```
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add agent/tools.py tests/test_tools.py
git commit -m "feat: add get_pixel_timeseries tool"
```

---

### Task 9: Tool — `flag_anomalous_regions`

**Files:**
- Modify: `agent/tools.py`
- Modify: `tests/test_tools.py`

Uses a 4×4 grid of 50×50 cells — flags cells where >30% of pixels exceed the threshold. Keeps tools flat (uses `_compute_index_array` directly).

**Step 1: Add failing test class**

```python
from agent.tools import flag_anomalous_regions   # add to imports

class TestFlagAnomalousRegions:
    def test_finds_stressed_nw_region(self):
        result = flag_anomalous_regions(index="ndvi", threshold=0.4, direction="below")
        assert result.success is True
        assert result.total_anomalous_pixels > 0
        assert len(result.regions) > 0

    def test_regions_have_bbox_and_count(self):
        result = flag_anomalous_regions(index="ndvi", threshold=0.4, direction="below")
        for r in result.regions:
            assert "bbox" in r and "pixel_count" in r
            bb = r["bbox"]
            assert all(k in bb for k in ("row_min", "row_max", "col_min", "col_max"))

    def test_primary_anomaly_in_nw(self):
        result = flag_anomalous_regions(index="ndvi", threshold=0.4, direction="below")
        largest = max(result.regions, key=lambda r: r["pixel_count"])
        bb = largest["bbox"]
        center_r = (bb["row_min"] + bb["row_max"]) / 2
        center_c = (bb["col_min"] + bb["col_max"]) / 2
        assert center_r < 110 and center_c < 110, "Largest anomaly should be in NW quadrant"

    def test_unsupported_index_returns_error(self):
        result = flag_anomalous_regions(index="fakeidx", threshold=0.5, direction="below")
        assert result.success is False

    def test_impossible_threshold_returns_empty(self):
        result = flag_anomalous_regions(index="ndvi", threshold=-0.99, direction="below")
        assert result.success is True
        assert result.total_anomalous_pixels == 0
```

**Step 2: Run to confirm failure**

```bash
pytest tests/test_tools.py::TestFlagAnomalousRegions -v
```

**Step 3: Add `flag_anomalous_regions` to `agent/tools.py`**

```python
def flag_anomalous_regions(index: str, threshold: float, direction: str) -> AnomalyResult:
    """Find scene grid cells where >30% of pixels exceed the threshold.

    Divides the scene into a 4×4 grid and flags anomalous grid cells.
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
            success=True, regions=[], total_anomalous_pixels=0,
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
                regions.append({
                    "bbox": {"row_min": r0, "row_max": r1, "col_min": c0, "col_max": c1},
                    "pixel_count": cell_count,
                    "mean_value": round(float(arr[r0:r1, c0:c1][cell_mask].mean()), 3),
                })

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
        success=True, regions=regions, total_anomalous_pixels=total_pixels,
        summary="\n".join(lines),
    )
```

**Step 4: Run to verify passing**

```bash
pytest tests/test_tools.py::TestFlagAnomalousRegions -v
```
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add agent/tools.py tests/test_tools.py
git commit -m "feat: add flag_anomalous_regions tool with 4x4 grid detection"
```

---

### Task 10: Tool — `compare_to_baseline`

**Files:**
- Modify: `agent/tools.py`
- Modify: `tests/test_tools.py`

Key teaching moment: this tool returns an error result (not an exception) when no baseline exists, so the agent can reason about the failure.

**Step 1: Add failing test class**

```python
from agent.tools import compare_to_baseline   # add to imports

class TestCompareToBaseline:
    def test_scene_a_returns_diff(self):
        result = compare_to_baseline(BoundingBox(0, 200, 0, 200), index="ndvi")
        assert result.success is True
        assert result.diff_array is not None
        assert result.mean_change is not None

    def test_nw_quadrant_shows_degradation(self):
        result = compare_to_baseline(BoundingBox(0, 100, 0, 100), index="ndvi")
        assert result.mean_change < -0.10, "NW quadrant should show NDVI degradation"

    def test_scene_b_no_baseline_returns_error_not_exception(self):
        load_scene("scene_b")
        result = compare_to_baseline(BoundingBox(0, 200, 0, 200), index="ndvi")
        assert result.success is False
        assert "baseline" in result.error_message.lower()

    def test_summary_mentions_change(self):
        load_scene("scene_a")
        result = compare_to_baseline(BoundingBox(0, 200, 0, 200), index="ndvi")
        assert "change" in result.summary.lower() or "Δ" in result.summary
```

**Step 2: Run to confirm failure**

```bash
pytest tests/test_tools.py::TestCompareToBaseline -v
```

**Step 3: Add `compare_to_baseline` to `agent/tools.py`**

```python
def compare_to_baseline(region: BoundingBox, index: str) -> DiffResult:
    """Compare current index values against the stored baseline from an earlier date.

    Returns a change map and degraded-pixel fraction.
    Returns an error result (does NOT raise) if no baseline exists for the scene.
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
        return DiffResult(success=False,
                          error_message=f"Unsupported index {index!r}.")

    current_arr = _compute_index_array(index, region)

    # Compute baseline index from raw bands (no tool call — tools stay flat)
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
        diff, f"Δ{index.upper()} vs baseline", f"diff_{index}.png",
        vmin=-0.5, vmax=0.5,
    )
    meta = _scene.get_metadata()
    baseline_date = meta.get("baseline_date", "unknown date")
    direction = "decreased" if mean_change < 0 else "increased"
    alert = (
        "⚠️ Significant degradation since baseline."
        if degraded_frac > 0.20 else "Minor changes since baseline."
    )
    summary = (
        f"Δ{index.upper()} vs baseline ({baseline_date}): "
        f"mean change = {mean_change:+.3f} ({direction}). "
        f"Pixels with >0.1 degradation: {degraded_frac:.1%}. {alert}"
    )
    return DiffResult(
        success=True, diff_array=diff, mean_change=mean_change,
        degraded_fraction=degraded_frac, image_path=image_path, summary=summary,
    )
```

**Step 4: Run the full tools test suite**

```bash
pytest tests/test_tools.py -v
```
Expected: All tests PASS

**Step 5: Commit**

```bash
git add agent/tools.py tests/test_tools.py
git commit -m "feat: add compare_to_baseline tool with graceful error on missing baseline"
```

---

## Phase 3: Agent Infrastructure

### Task 11: JSON Schemas

**Files:**
- Create: `agent/schemas.py`
- Create: `tests/test_schemas.py`

**Step 1: Write the failing test**

```python
# tests/test_schemas.py
from agent.schemas import TOOL_SCHEMAS


def test_schema_count():
    assert len(TOOL_SCHEMAS) == 6


def test_each_schema_has_required_keys():
    for s in TOOL_SCHEMAS:
        assert "name" in s
        assert "description" in s
        assert "input_schema" in s
        assert s["input_schema"]["type"] == "object"


def test_tool_names():
    names = {s["name"] for s in TOOL_SCHEMAS}
    expected = {
        "compute_ndvi", "compute_ndwi", "compute_evi",
        "get_pixel_timeseries", "flag_anomalous_regions", "compare_to_baseline",
    }
    assert names == expected


def test_region_schema_present_where_expected():
    for name in ("compute_ndvi", "compute_ndwi", "compute_evi", "compare_to_baseline"):
        s = next(x for x in TOOL_SCHEMAS if x["name"] == name)
        props = s["input_schema"]["properties"]
        assert "region" in props
        region_keys = props["region"]["properties"].keys()
        assert all(k in region_keys for k in ("row_min", "row_max", "col_min", "col_max"))


def test_flag_anomalous_regions_schema():
    s = next(x for x in TOOL_SCHEMAS if x["name"] == "flag_anomalous_regions")
    props = s["input_schema"]["properties"]
    assert "index" in props and "threshold" in props and "direction" in props


def test_get_pixel_timeseries_schema():
    s = next(x for x in TOOL_SCHEMAS if x["name"] == "get_pixel_timeseries")
    props = s["input_schema"]["properties"]
    assert "lat" in props and "lon" in props and "index" in props
```

**Step 2: Run to confirm failure**

```bash
pytest tests/test_schemas.py -v
```
Expected: `ModuleNotFoundError`

**Step 3: Write `agent/schemas.py`**

```python
# agent/schemas.py
"""JSON schemas for all tools, in the format required by the Anthropic Messages API.

Passed directly to the `tools=` parameter of `client.messages.create()`.
No framework required — these are plain Python dicts.
"""

_REGION = {
    "type": "object",
    "description": "Bounding box in pixel coordinates (integers).",
    "properties": {
        "row_min": {"type": "integer", "description": "Start row (inclusive)"},
        "row_max": {"type": "integer", "description": "End row (exclusive)"},
        "col_min": {"type": "integer", "description": "Start column (inclusive)"},
        "col_max": {"type": "integer", "description": "End column (exclusive)"},
    },
    "required": ["row_min", "row_max", "col_min", "col_max"],
}

TOOL_SCHEMAS = [
    {
        "name": "compute_ndvi",
        "description": (
            "Compute NDVI (Normalized Difference Vegetation Index) for a region. "
            "NDVI = (NIR - Red) / (NIR + Red). Values near 1 = dense healthy vegetation; "
            "below 0.3 = stress, sparse vegetation, or bare soil. "
            "Returns mean, std, stressed-pixel fraction, and an image path."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"region": _REGION},
            "required": ["region"],
        },
    },
    {
        "name": "compute_ndwi",
        "description": (
            "Compute NDWI (Normalized Difference Water Index) for a region. "
            "NDWI = (Green - SWIR) / (Green + SWIR). Positive = moist canopy; "
            "negative = water stress or dry soil. "
            "Use to distinguish water stress from nutrient or pest stress."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"region": _REGION},
            "required": ["region"],
        },
    },
    {
        "name": "compute_evi",
        "description": (
            "Compute EVI (Enhanced Vegetation Index) for a region. "
            "EVI = 2.5*(NIR-Red)/(NIR+6*Red-7.5*Blue+1). "
            "Less saturated than NDVI in dense canopy. "
            "Use to cross-check NDVI findings or in high-biomass areas."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"region": _REGION},
            "required": ["region"],
        },
    },
    {
        "name": "get_pixel_timeseries",
        "description": (
            "Return a time series of a spectral index at a specific pixel. "
            "Useful for determining if an anomaly is recent (new stress onset) or "
            "persistent (chronic condition or expected crop phenology). "
            "Note: lat/lon are pixel row/col coordinates for this dataset."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "lat": {"type": "number", "description": "Pixel row coordinate"},
                "lon": {"type": "number", "description": "Pixel column coordinate"},
                "index": {
                    "type": "string",
                    "enum": ["ndvi", "ndwi", "evi"],
                    "description": "Spectral index to retrieve",
                },
            },
            "required": ["lat", "lon", "index"],
        },
    },
    {
        "name": "flag_anomalous_regions",
        "description": (
            "Find regions where a spectral index exceeds a threshold. "
            "Returns bounding boxes, pixel counts, and mean values per anomalous region. "
            "Use after a broad index assessment to pinpoint specific areas of concern."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "index": {
                    "type": "string",
                    "enum": ["ndvi", "ndwi", "evi"],
                    "description": "Spectral index to threshold",
                },
                "threshold": {
                    "type": "number",
                    "description": "Threshold value (e.g., 0.3 for NDVI stress)",
                },
                "direction": {
                    "type": "string",
                    "enum": ["below", "above"],
                    "description": "Flag pixels below or above the threshold",
                },
            },
            "required": ["index", "threshold", "direction"],
        },
    },
    {
        "name": "compare_to_baseline",
        "description": (
            "Compare current index values to a stored baseline from an earlier date. "
            "Returns a change map and fraction of pixels with significant degradation. "
            "IMPORTANT: This tool will return an error if no baseline exists for the "
            "current scene. Handle this gracefully and adapt your strategy."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "region": _REGION,
                "index": {
                    "type": "string",
                    "enum": ["ndvi", "ndwi", "evi"],
                    "description": "Index to compare against baseline",
                },
            },
            "required": ["region", "index"],
        },
    },
]
```

**Step 4: Run to verify passing**

```bash
pytest tests/test_schemas.py -v
```
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add agent/schemas.py tests/test_schemas.py
git commit -m "feat: add JSON schemas for all 6 tools"
```

---

### Task 12: Agent Loop

**Files:**
- Create: `agent/loop.py`
- Create: `tests/test_loop.py`

**Step 1: Write the failing test**

```python
# tests/test_loop.py
import pytest
from unittest.mock import MagicMock, patch
from agent.loop import run_agent, dispatch_tool
from agent.scene import load_scene
from agent.types import BoundingBox


@pytest.fixture(autouse=True)
def load_a():
    load_scene("scene_a")


def test_dispatch_tool_compute_ndvi():
    result = dispatch_tool(
        "compute_ndvi",
        {"region": {"row_min": 0, "row_max": 50, "col_min": 0, "col_max": 50}},
    )
    assert result.success is True


def test_dispatch_tool_unknown_raises():
    with pytest.raises(KeyError):
        dispatch_tool("nonexistent_tool", {})


def test_run_agent_end_turn_immediately(mock_client_factory):
    mock_client = mock_client_factory([
        {"stop_reason": "end_turn",
         "content": [{"type": "text", "text": "No issues found."}]},
    ])
    result = run_agent("Analyze.", client=mock_client, verbose=False)
    assert mock_client.messages.create.call_count == 1


def test_run_agent_tool_call_then_end_turn(mock_client_factory):
    mock_client = mock_client_factory([
        {
            "stop_reason": "tool_use",
            "content": [{
                "type": "tool_use", "id": "id1", "name": "compute_ndvi",
                "input": {"region": {"row_min": 0, "row_max": 50, "col_min": 0, "col_max": 50}},
            }],
        },
        {"stop_reason": "end_turn",
         "content": [{"type": "text", "text": "Stress detected in NW."}]},
    ])
    run_agent("Analyze.", client=mock_client, verbose=False)
    assert mock_client.messages.create.call_count == 2


def test_run_agent_respects_max_iterations(mock_client_factory):
    # Always returns tool_use — loop should stop at max_iterations
    mock_client = mock_client_factory([{
        "stop_reason": "tool_use",
        "content": [{
            "type": "tool_use", "id": "id1", "name": "compute_ndvi",
            "input": {"region": {"row_min": 0, "row_max": 50, "col_min": 0, "col_max": 50}},
        }],
    }] * 10)
    run_agent("Analyze.", client=mock_client, max_iterations=3, verbose=False)
    assert mock_client.messages.create.call_count == 3


@pytest.fixture
def mock_client_factory():
    """Factory that builds a mock Anthropic client from a list of response dicts."""
    def _make(responses):
        client = MagicMock()
        mock_responses = []
        for resp in responses:
            mr = MagicMock()
            mr.stop_reason = resp["stop_reason"]
            content_blocks = []
            for block in resp["content"]:
                mb = MagicMock()
                mb.type = block["type"]
                if block["type"] == "text":
                    mb.text = block["text"]
                elif block["type"] == "tool_use":
                    mb.id = block["id"]
                    mb.name = block["name"]
                    mb.input = block["input"]
                content_blocks.append(mb)
            mr.content = content_blocks
            mock_responses.append(mr)
        client.messages.create.side_effect = mock_responses
        return client
    return _make
```

**Step 2: Run to confirm failure**

```bash
pytest tests/test_loop.py -v
```
Expected: `ModuleNotFoundError: No module named 'agent.loop'`

**Step 3: Write `agent/loop.py`**

```python
# agent/loop.py
"""The agent loop — the core teaching artifact.

This is the complete agentic pattern in ~80 lines:
  1. Call the model with messages + tool schemas
  2. If stop_reason == "tool_use": execute tools, append results, repeat
  3. If stop_reason == "end_turn": done

No framework. No magic.
"""
import json
import os
from typing import Optional

import anthropic

from agent.schemas import TOOL_SCHEMAS
from agent.types import BoundingBox
from agent import tools as _tools

MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are an autonomous crop health analyst working with satellite imagery. \
Your job is to assess crop health in a given region and produce a diagnostic report.

You have access to tools that compute spectral indices (NDVI, NDWI, EVI), retrieve \
timeseries data, flag anomalous regions, and compare current conditions to historical baselines.

**Your workflow**:
1. Start with a broad assessment (e.g., compute NDVI for the region)
2. If you detect anomalies or low values, investigate further using additional indices or timeseries
3. Explain your reasoning in plain text before each tool call. Describe what you observed and why \
you're choosing the next tool.
4. Only call tools when warranted by your observations. You do not need to use all available tools.
5. If tools return errors, adapt your strategy and explain what happened.
6. When you have sufficient information, produce a final diagnostic report with:
   - Summary of findings
   - Specific locations of concern (if any)
   - Likely cause of stress (water, nutrients, pests) with confidence level
   - Recommended follow-up actions
7. Express uncertainty when appropriate. If indices give conflicting signals, say so.

**Important**: Be concise in your reasoning (2-3 sentences per tool call). \
The goal is clarity, not verbosity."""

_TOOL_MAP = {
    "compute_ndvi": _tools.compute_ndvi,
    "compute_ndwi": _tools.compute_ndwi,
    "compute_evi": _tools.compute_evi,
    "get_pixel_timeseries": _tools.get_pixel_timeseries,
    "flag_anomalous_regions": _tools.flag_anomalous_regions,
    "compare_to_baseline": _tools.compare_to_baseline,
}


def dispatch_tool(name: str, input_dict: dict):
    """Convert API dict input to typed Python args and call the tool function."""
    if name not in _TOOL_MAP:
        raise KeyError(f"Unknown tool: {name!r}. Available: {list(_TOOL_MAP)}")
    kwargs = dict(input_dict)
    if "region" in kwargs:
        kwargs["region"] = BoundingBox(**kwargs["region"])
    return _TOOL_MAP[name](**kwargs)


def run_agent(
    user_request: str,
    client: Optional[anthropic.Anthropic] = None,
    temperature: float = 0,
    max_iterations: int = 20,
    verbose: bool = True,
) -> str:
    """Run the agent loop. Returns the final text response."""
    if client is None:
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    messages = [{"role": "user", "content": user_request}]
    final_text = ""

    for _ in range(max_iterations):
        response = client.messages.create(
            model=MODEL,
            system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            messages=messages,
            temperature=temperature,
            max_tokens=4096,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    final_text += block.text
            if verbose:
                print("\n" + "=" * 60)
                print("FINAL REPORT")
                print("=" * 60)
                print(final_text)
            return final_text

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if hasattr(block, "type") and block.type == "tool_use":
                    if verbose:
                        print(f"\n[TOOL CALL] {block.name}")
                        print(f"[INPUT] {json.dumps(block.input, indent=2)}")
                    result = dispatch_tool(block.name, block.input)
                    if verbose:
                        print(f"[RESULT] {result.summary}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result.summary,
                    })
            messages.append({"role": "user", "content": tool_results})

    if verbose:
        print(f"\n⚠️  Warning: Hit max_iterations ({max_iterations}) without end_turn.")
    return final_text


if __name__ == "__main__":
    import argparse
    from agent.scene import load_scene

    parser = argparse.ArgumentParser(description="Run the crop health agent")
    parser.add_argument("--scene", choices=["scene_a", "scene_b"], default="scene_a")
    parser.add_argument("--mock", action="store_true", help="Use mock mode (no API calls)")
    parser.add_argument("--temperature", type=float, default=0)
    args = parser.parse_args()

    load_scene(args.scene)

    if args.mock:
        from agent.mock import run_mock_agent
        run_mock_agent(args.scene)
    else:
        run_agent(
            "Assess crop health for the entire scene. "
            "Identify any areas of concern and determine likely causes.",
            temperature=args.temperature,
        )
```

**Step 4: Run to verify passing**

```bash
pytest tests/test_loop.py -v
```
Expected: All 5 tests PASS

**Step 5: Run the full test suite**

```bash
pytest tests/ -v
```
Expected: All tests PASS

**Step 6: Commit**

```bash
git add agent/loop.py tests/test_loop.py
git commit -m "feat: add agent loop with tool dispatch and max_iterations guard"
```

---

### Task 13: Mock Mode

**Files:**
- Create: `agent/mock.py`
- Create: `data/mock_responses/scene_a.json`
- Create: `tests/test_mock.py`

**Step 1: Write the failing test**

```python
# tests/test_mock.py
import json
import pytest
from pathlib import Path
from agent.scene import load_scene


@pytest.fixture(autouse=True)
def load_a():
    load_scene("scene_a")


def test_mock_responses_file_exists():
    assert Path("data/mock_responses/scene_a.json").exists()


def test_mock_responses_structure():
    data = json.loads(Path("data/mock_responses/scene_a.json").read_text())
    assert "turns" in data
    assert len(data["turns"]) >= 4
    for turn in data["turns"]:
        assert "stop_reason" in turn
        assert "content" in turn
        assert turn["stop_reason"] in ("tool_use", "end_turn")


def test_run_mock_agent_completes(capsys):
    from agent.mock import run_mock_agent
    run_mock_agent("scene_a")
    captured = capsys.readouterr()
    assert "FINAL REPORT" in captured.out


def test_run_mock_agent_missing_scene_raises():
    from agent.mock import run_mock_agent
    with pytest.raises(FileNotFoundError):
        run_mock_agent("nonexistent_scene")
```

**Step 2: Run to confirm failure**

```bash
pytest tests/test_mock.py -v
```

**Step 3: Create `data/mock_responses/scene_a.json`**

```json
{
  "scene_id": "scene_a",
  "description": "Pre-recorded agent trace for Scene A: water stress in NW quadrant",
  "turns": [
    {
      "stop_reason": "tool_use",
      "content": [
        {
          "type": "text",
          "text": "I'll begin with a broad NDVI assessment to understand the overall vegetation health before looking at specific anomalies."
        },
        {
          "type": "tool_use",
          "id": "mock_01",
          "name": "compute_ndvi",
          "input": {"region": {"row_min": 0, "row_max": 200, "col_min": 0, "col_max": 200}}
        }
      ]
    },
    {
      "stop_reason": "tool_use",
      "content": [
        {
          "type": "text",
          "text": "The NDVI shows a significant stressed area (>25% of pixels below 0.3). I'll flag the anomalous regions to get precise bounding boxes."
        },
        {
          "type": "tool_use",
          "id": "mock_02",
          "name": "flag_anomalous_regions",
          "input": {"index": "ndvi", "threshold": 0.4, "direction": "below"}
        }
      ]
    },
    {
      "stop_reason": "tool_use",
      "content": [
        {
          "type": "text",
          "text": "The anomaly is concentrated in the northwest quadrant (rows 0-100, cols 0-100). I'll compute NDWI for this region to determine if this is water stress or another form of stress."
        },
        {
          "type": "tool_use",
          "id": "mock_03",
          "name": "compute_ndwi",
          "input": {"region": {"row_min": 0, "row_max": 100, "col_min": 0, "col_max": 100}}
        }
      ]
    },
    {
      "stop_reason": "tool_use",
      "content": [
        {
          "type": "text",
          "text": "NDWI is strongly negative in the NW quadrant, confirming water stress rather than nutrient or pest issues. I'll check the timeseries to determine if this is a recent onset or a persistent condition."
        },
        {
          "type": "tool_use",
          "id": "mock_04",
          "name": "get_pixel_timeseries",
          "input": {"lat": 50, "lon": 50, "index": "ndvi"}
        }
      ]
    },
    {
      "stop_reason": "tool_use",
      "content": [
        {
          "type": "text",
          "text": "The timeseries shows a sharp NDVI decline in the last 2-3 weeks — recent onset. I'll compare against the baseline to quantify the change."
        },
        {
          "type": "tool_use",
          "id": "mock_05",
          "name": "compare_to_baseline",
          "input": {
            "region": {"row_min": 0, "row_max": 100, "col_min": 0, "col_max": 100},
            "index": "ndvi"
          }
        }
      ]
    },
    {
      "stop_reason": "end_turn",
      "content": [
        {
          "type": "text",
          "text": "## Crop Health Diagnostic Report\n\n**Summary of findings:**\nSignificant vegetation stress in the northwest quadrant (rows 0–100, cols 0–100), affecting ~25% of the total scene area.\n\n**Specific locations of concern:**\n- Northwest quadrant: mean NDVI dropped from ~0.65 (baseline) to ~0.25 (current), a Δ ≈ −0.40.\n\n**Likely cause: Water stress (HIGH confidence)**\n- NDWI is strongly negative (< −0.2) in the affected area — reduced canopy water content.\n- Timeseries shows a sharp NDVI decline over the past 2–3 weeks, consistent with irrigation failure or drought onset, not seasonal phenology.\n- Baseline comparison confirms the NW quadrant was healthy 30 days ago.\n\n**Recommended follow-up actions:**\n1. Inspect irrigation infrastructure in the northwest quadrant immediately.\n2. Check field water meters and irrigation schedules for the past month.\n3. Consider emergency irrigation if crops are at a critical growth stage.\n4. Re-assess in 7–10 days after intervention to monitor recovery."
        }
      ]
    }
  ]
}
```

**Step 4: Write `agent/mock.py`**

```python
# agent/mock.py
"""Mock mode: replay pre-recorded agent traces without calling the Anthropic API.

Real tool calls ARE executed (results are live), but model responses are canned.
Useful for: offline demos, testing, cost control, reproducible lecture demonstrations.

Pre-recorded responses: data/mock_responses/<scene_id>.json
"""
import json
from pathlib import Path
from agent.loop import dispatch_tool

_MOCK_DIR = Path("data/mock_responses")


def run_mock_agent(scene_id: str) -> str:
    """Replay a pre-recorded agent trace, executing real tool calls."""
    mock_path = _MOCK_DIR / f"{scene_id}.json"
    if not mock_path.exists():
        raise FileNotFoundError(
            f"No mock responses found for {scene_id!r}. Expected: {mock_path}"
        )

    data = json.loads(mock_path.read_text())
    print(f"\n[MOCK MODE] Replaying pre-recorded trace for {scene_id}")
    print("=" * 60)

    final_text = ""
    for turn in data["turns"]:
        for block in turn["content"]:
            if block["type"] == "text":
                print(f"\n[AGENT REASONING]\n{block['text']}")
            elif block["type"] == "tool_use":
                print(f"\n[TOOL CALL] {block['name']}")
                print(f"[INPUT] {json.dumps(block['input'], indent=2)}")
                result = dispatch_tool(block["name"], block["input"])
                print(f"[RESULT] {result.summary}")

        if turn["stop_reason"] == "end_turn":
            for block in turn["content"]:
                if block["type"] == "text":
                    final_text = block["text"]
            print("\n" + "=" * 60)
            print("FINAL REPORT")
            print("=" * 60)
            print(final_text)

    return final_text
```

**Step 5: Run tests**

```bash
pytest tests/test_mock.py -v
```
Expected: All 4 tests PASS

**Step 6: Test the full mock demo end-to-end**

```bash
python -m agent.loop --scene scene_a --mock
```
Expected: Full trace printed, ending with the diagnostic report.

**Step 7: Commit**

```bash
git add agent/mock.py data/mock_responses/scene_a.json tests/test_mock.py
git commit -m "feat: mock mode with pre-recorded Scene A trace"
```

---

## Phase 4: Notebooks

### Task 14: Notebook 00 — Data Exploration

**Files:**
- Create: `notebooks/00_data_exploration.ipynb`

Create this notebook using `nbformat`. Each cell is listed with its type and content.

**Step 1: Write a script to generate the notebook**

Create a temporary script `notebooks/build_nb00.py` (delete after use):

```python
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.cells = [
    nbf.v4.new_markdown_cell(
        "# Notebook 00: Data Exploration\n\n"
        "Orient yourself to the Sentinel-2-like synthetic scenes before we introduce the agent.\n\n"
        "**What you'll learn:** What the band data looks like, how NDVI is computed from scratch, "
        "and why the ambiguity in Scene B motivates an adaptive pipeline."
    ),
    nbf.v4.new_code_cell(
        "import sys\nsys.path.insert(0, '..')\n\n"
        "import numpy as np\nimport matplotlib.pyplot as plt\n"
        "from agent.scene import load_scene, get_band, get_metadata\n"
        "from agent.types import BoundingBox\n\n"
        "load_scene('scene_a')\n"
        "meta = get_metadata()\n"
        "print(f\"Scene: {meta['scene_id']} | Date: {meta['date']} | \"\n"
        "      f\"Shape: {meta['shape']} | Bands: {meta['bands']}\")"
    ),
    nbf.v4.new_markdown_cell(
        "## Band Overview\n\n"
        "| Band | Name | Wavelength | Notes |\n"
        "|------|------|------------|-------|\n"
        "| B02 | Blue | 490 nm | Atmospheric scattering baseline |\n"
        "| B03 | Green | 560 nm | Used in NDWI |\n"
        "| B04 | Red | 665 nm | Chlorophyll absorption |\n"
        "| B08 | NIR | 842 nm | Strong vegetation reflection |\n"
        "| B8A | Red-edge | 865 nm | Sensitive to chlorophyll content |\n"
        "| B11 | SWIR | 1610 nm | Leaf water content |\n"
        "| B12 | SWIR2 | 2190 nm | Soil and dry vegetation |"
    ),
    nbf.v4.new_code_cell(
        "# Plot all 7 bands as a grid\n"
        "bands = ['B02', 'B03', 'B04', 'B08', 'B8A', 'B11', 'B12']\n"
        "names = ['Blue', 'Green', 'Red', 'NIR', 'Red-edge', 'SWIR', 'SWIR2']\n\n"
        "fig, axes = plt.subplots(2, 4, figsize=(16, 8))\n"
        "axes = axes.flatten()\n"
        "for i, (band, name) in enumerate(zip(bands, names)):\n"
        "    arr = get_band(band)\n"
        "    im = axes[i].imshow(arr, cmap='gray',\n"
        "                        vmin=np.percentile(arr, 2),\n"
        "                        vmax=np.percentile(arr, 98))\n"
        "    axes[i].set_title(f'{band} ({name})')\n"
        "    axes[i].axis('off')\n"
        "    plt.colorbar(im, ax=axes[i], fraction=0.046)\n"
        "axes[-1].axis('off')\n"
        "plt.suptitle('Scene A — All Bands (gray scale)', fontsize=14)\n"
        "plt.tight_layout()\nplt.show()"
    ),
    nbf.v4.new_markdown_cell(
        "## Computing NDVI by Hand\n\n"
        "NDVI = (NIR − Red) / (NIR + Red)\n\n"
        "- NIR high, Red low → healthy green vegetation → NDVI near 1\n"
        "- NIR low, Red elevated → stressed or senescent → NDVI near 0 or negative\n\n"
        "Let's compute it directly in numpy — no tools yet:"
    ),
    nbf.v4.new_code_cell(
        "nir = get_band('B08')\nred = get_band('B04')\n"
        "ndvi = (nir - red) / (nir + red + 1e-8)  # epsilon avoids division by zero\n\n"
        "fig, axes = plt.subplots(1, 3, figsize=(15, 5))\n"
        "axes[0].imshow(nir, cmap='gray'); axes[0].set_title('NIR (B08)'); axes[0].axis('off')\n"
        "axes[1].imshow(red, cmap='gray'); axes[1].set_title('Red (B04)'); axes[1].axis('off')\n"
        "im = axes[2].imshow(ndvi, cmap='RdYlGn', vmin=-0.2, vmax=1.0)\n"
        "axes[2].set_title('NDVI'); axes[2].axis('off')\n"
        "plt.colorbar(im, ax=axes[2])\n"
        "plt.suptitle('NDVI = (NIR - Red) / (NIR + Red)', fontsize=13)\n"
        "plt.tight_layout(); plt.show()\n\n"
        "print(f'NW quadrant mean NDVI: {ndvi[:100, :100].mean():.3f}')\n"
        "print(f'SE quadrant mean NDVI: {ndvi[100:, 100:].mean():.3f}')"
    ),
    nbf.v4.new_markdown_cell(
        "## The Ambiguity Problem — Scene B\n\n"
        "Scene A has a clear anomaly. Scene B is harder. "
        "A fixed pipeline (\"always run NDVI → threshold → alert\") would struggle here. "
        "This is why we need an adaptive agent."
    ),
    nbf.v4.new_code_cell(
        "load_scene('scene_b')\n"
        "nir_b = get_band('B08'); red_b = get_band('B04')\n"
        "ndvi_b = (nir_b - red_b) / (nir_b + red_b + 1e-8)\n\n"
        "fig, axes = plt.subplots(1, 2, figsize=(12, 5))\n"
        "im0 = axes[0].imshow(ndvi, cmap='RdYlGn', vmin=-0.2, vmax=1.0)\n"
        "axes[0].set_title(f'Scene A NDVI (mean={ndvi.mean():.3f})')\n"
        "axes[0].axis('off'); plt.colorbar(im0, ax=axes[0])\n"
        "im1 = axes[1].imshow(ndvi_b, cmap='RdYlGn', vmin=-0.2, vmax=1.0)\n"
        "axes[1].set_title(f'Scene B NDVI (mean={ndvi_b.mean():.3f})')\n"
        "axes[1].axis('off'); plt.colorbar(im1, ax=axes[1])\n"
        "plt.suptitle('Same threshold, different diagnostic paths', fontsize=13)\n"
        "plt.tight_layout(); plt.show()\n\n"
        "print('Scene A: clear spatial anomaly → investigate NW quadrant')\n"
        "print('Scene B: mild uniform depression → is this phenological? early stress?')"
    ),
]

nbf.write(nb, 'notebooks/00_data_exploration.ipynb')
print("Created notebooks/00_data_exploration.ipynb")
```

**Step 2: Run the build script**

```bash
cd /path/to/repo
pip install nbformat
python notebooks/build_nb00.py
rm notebooks/build_nb00.py
```

**Step 3: Verify the notebook runs**

```bash
jupyter nbconvert --to notebook --execute notebooks/00_data_exploration.ipynb \
    --output /tmp/nb00_check.ipynb --ExecutePreprocessor.timeout=60
echo "Notebook 00 executed successfully"
rm /tmp/nb00_check.ipynb
```
Expected: Runs without errors

**Step 4: Commit**

```bash
git add notebooks/00_data_exploration.ipynb
git commit -m "feat: add notebook 00 - data exploration and band visualization"
```

---

### Task 15: Notebook 01 — Tools in Isolation

**Files:**
- Create: `notebooks/01_tools.ipynb`

**Purpose:** Show each tool as a plain Python function. Students see the source, the schema, and the output. Core message: "tools are just functions."

**Step 1: Create using nbformat**

Create `notebooks/build_nb01.py`:

```python
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.cells = [
    nbf.v4.new_markdown_cell(
        "# Notebook 01: Tools in Isolation\n\n"
        "**Core message:** Tools are plain Python functions. There is no magic.\n\n"
        "We'll inspect the source, the JSON schema, call each tool manually, "
        "and see how error handling works."
    ),
    nbf.v4.new_code_cell(
        "import sys; sys.path.insert(0, '..')\n"
        "import inspect\n"
        "from agent.scene import load_scene\n"
        "from agent.types import BoundingBox\n"
        "from agent import tools\nfrom agent.schemas import TOOL_SCHEMAS\n\n"
        "load_scene('scene_a')\nprint('Scene A loaded.')"
    ),
    nbf.v4.new_markdown_cell(
        "## One Tool in Detail: `compute_ndvi`\n\n"
        "First, let's read the source:"
    ),
    nbf.v4.new_code_cell("print(inspect.getsource(tools.compute_ndvi))"),
    nbf.v4.new_code_cell(
        "# Call it directly — no agent, no API\n"
        "bb = BoundingBox(row_min=0, row_max=200, col_min=0, col_max=200)\n"
        "result = tools.compute_ndvi(bb)\n\n"
        "print(f'success: {result.success}')\n"
        "print(f'mean:    {result.mean:.3f}')\n"
        "print(f'std:     {result.std:.3f}')\n"
        "print(f'low_fraction: {result.low_fraction:.1%}')\n"
        "print(f'\\nSUMMARY STRING (this is what the agent sees):')\n"
        "print(result.summary)"
    ),
    nbf.v4.new_markdown_cell(
        "## One Schema in Detail\n\n"
        "This JSON dict is what we pass to `tools=` in the API call. "
        "It's how the model knows what parameters to send."
    ),
    nbf.v4.new_code_cell(
        "import json\n"
        "ndvi_schema = next(s for s in TOOL_SCHEMAS if s['name'] == 'compute_ndvi')\n"
        "print(json.dumps(ndvi_schema, indent=2))"
    ),
    nbf.v4.new_markdown_cell(
        "## All 6 Tools\n\n"
        "Call each tool on a small region and print its summary:"
    ),
    nbf.v4.new_code_cell(
        "bb_full = BoundingBox(0, 200, 0, 200)\n"
        "bb_nw   = BoundingBox(0, 100, 0, 100)\n\n"
        "for fn, args in [\n"
        "    (tools.compute_ndvi,          {'region': bb_full}),\n"
        "    (tools.compute_ndwi,          {'region': bb_nw}),\n"
        "    (tools.compute_evi,           {'region': bb_full}),\n"
        "    (tools.get_pixel_timeseries,  {'lat': 50, 'lon': 50, 'index': 'ndvi'}),\n"
        "    (tools.flag_anomalous_regions,{'index': 'ndvi', 'threshold': 0.4, 'direction': 'below'}),\n"
        "    (tools.compare_to_baseline,   {'region': bb_nw, 'index': 'ndvi'}),\n"
        "]:\n"
        "    result = fn(**args)\n"
        "    print(f'=== {fn.__name__} ===')\n"
        "    print(result.summary)\n"
        "    print()"
    ),
    nbf.v4.new_markdown_cell(
        "## Error Handling: What Happens With No Baseline?\n\n"
        "This is **the key design decision**: tools return error results, they don't raise exceptions. "
        "This lets the agent reason about failures and adapt."
    ),
    nbf.v4.new_code_cell(
        "load_scene('scene_b')  # Scene B has no baseline\n"
        "result = tools.compare_to_baseline(BoundingBox(0, 200, 0, 200), index='ndvi')\n\n"
        "print(f'success: {result.success}')\n"
        "print(f'error:   {result.error_message}')\n"
        "print('\\n→ The agent receives this error message and can adapt.')"
    ),
    nbf.v4.new_markdown_cell(
        "## How Tool Dispatch Works\n\n"
        "The `dispatch_tool` function is just a dict lookup. No magic."
    ),
    nbf.v4.new_code_cell(
        "from agent.loop import dispatch_tool\nprint(inspect.getsource(dispatch_tool))"
    ),
]

nbf.write(nb, 'notebooks/01_tools.ipynb')
print("Created notebooks/01_tools.ipynb")
```

**Step 2: Run and verify**

```bash
python notebooks/build_nb01.py
rm notebooks/build_nb01.py
jupyter nbconvert --to notebook --execute notebooks/01_tools.ipynb \
    --output /tmp/nb01_check.ipynb --ExecutePreprocessor.timeout=60
rm /tmp/nb01_check.ipynb
```

**Step 3: Commit**

```bash
git add notebooks/01_tools.ipynb
git commit -m "feat: add notebook 01 - tools in isolation with schema walkthrough"
```

---

### Task 16: Notebook 02 — Agent Loop (Main Teaching Notebook)

**Files:**
- Create: `notebooks/02_agent_loop.ipynb`

**Purpose:** The core lecture notebook. Follows the 47-minute lecture section outline from `lecture_plan.md`.

**Step 1: Create using nbformat**

Create `notebooks/build_nb02.py`:

```python
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.cells = [
    nbf.v4.new_markdown_cell(
        "# Notebook 02: The Agent Loop\n\n"
        "**Learning objectives:**\n"
        "- Understand what makes a system 'agentic' vs. a standard ML pipeline\n"
        "- See how tool calling works at the API level (no framework)\n"
        "- Read the agent's reasoning trace and understand each decision\n"
        "- Observe how the agent adapts to a different scene\n\n"
        "**Prerequisites:** Run Notebook 00 and 01 first (or at least 00)."
    ),
    nbf.v4.new_code_cell(
        "import sys; sys.path.insert(0, '..')\n"
        "import os, inspect\n"
        "from agent.scene import load_scene\n"
        "from agent.types import BoundingBox\n"
        "from agent import tools\n"
        "from agent.schemas import TOOL_SCHEMAS\n"
        "from agent.loop import run_agent, SYSTEM_PROMPT\n"
        "from agent.mock import run_mock_agent\n\n"
        "load_scene('scene_a')\n"
        "print('Scene A loaded. Ready.')"
    ),
    nbf.v4.new_markdown_cell(
        "## Step 1: One Tool Example\n\n"
        "Before the agent, let's confirm: tools are just functions."
    ),
    nbf.v4.new_code_cell(
        "bb = BoundingBox(0, 200, 0, 200)\n"
        "result = tools.compute_ndvi(bb)\n"
        "print(result.summary)"
    ),
    nbf.v4.new_markdown_cell(
        "## Step 2: One Schema Example\n\n"
        "This is how the model learns what `compute_ndvi` accepts:"
    ),
    nbf.v4.new_code_cell(
        "import json\n"
        "ndvi_schema = next(s for s in TOOL_SCHEMAS if s['name'] == 'compute_ndvi')\n"
        "print(json.dumps(ndvi_schema, indent=2))"
    ),
    nbf.v4.new_markdown_cell(
        "## Step 3: The System Prompt\n\n"
        "Read this carefully — every instruction shapes agent behavior.\n\n"
        "Note especially:\n"
        "- *\"Explain your reasoning before each tool call\"* → creates the trace we learn from\n"
        "- *\"Only call tools when warranted\"* → prevents exhaustive tool use\n"
        "- *\"Express uncertainty when appropriate\"* → prevents overconfident reports"
    ),
    nbf.v4.new_code_cell("print(SYSTEM_PROMPT)"),
    nbf.v4.new_markdown_cell(
        "## Step 4: The Loop\n\n"
        "The entire agent loop is ~50 lines. Read it:"
    ),
    nbf.v4.new_code_cell(
        "from agent.loop import run_agent\n"
        "print(inspect.getsource(run_agent))"
    ),
    nbf.v4.new_markdown_cell(
        "## Step 5: Scripted Run — Scene A\n\n"
        "We use **mock mode** here so the trace is identical every time. "
        "The model responses are pre-recorded, but **all tool calls are real** (live numpy computation).\n\n"
        "Watch each `[TOOL CALL]` block:\n"
        "1. What did the agent observe that prompted this call?\n"
        "2. What new information did it learn?\n"
        "3. What might it do next?"
    ),
    nbf.v4.new_code_cell(
        "load_scene('scene_a')\n"
        "run_mock_agent('scene_a')"
    ),
    nbf.v4.new_markdown_cell(
        "### Discussion Questions\n\n"
        "1. After `compute_ndvi`: Why did the agent choose `flag_anomalous_regions` next rather than immediately calling `compute_ndwi`?\n"
        "2. After `flag_anomalous_regions`: The agent now calls `compute_ndwi` on the *anomalous sub-region*, not the full scene. Why is this more efficient?\n"
        "3. After `compute_ndwi`: The NDWI is negative. Could this mean water stress, bare soil, or something else? What rules this out?\n"
        "4. After `get_pixel_timeseries`: Why does recent onset matter for diagnosis?\n"
        "5. The agent called `compare_to_baseline` last, not first. Was this the right order?"
    ),
    nbf.v4.new_markdown_cell(
        "## Step 6: Live Run — Scene B\n\n"
        "Scene B is ambiguous: mild uniform NDVI depression, no strong spatial anomaly, no baseline.\n\n"
        "**Watch:** Does the agent take a different path than it did for Scene A?\n\n"
        "⚠️ **Requires `ANTHROPIC_API_KEY`** in your environment. "
        "Estimated cost: ~$0.10–0.20 for this run.\n\n"
        "If you don't have an API key, skip this cell — the take-home exercises cover this."
    ),
    nbf.v4.new_code_cell(
        "# LIVE API CALL — requires ANTHROPIC_API_KEY\n"
        "if 'ANTHROPIC_API_KEY' in os.environ:\n"
        "    load_scene('scene_b')\n"
        "    run_agent(\n"
        "        'Assess crop health for the entire scene. '\n"
        "        'Identify any areas of concern and determine likely causes.',\n"
        "        temperature=0,\n"
        "    )\n"
        "else:\n"
        "    print('ANTHROPIC_API_KEY not set — skipping live run.')\n"
        "    print('Set your key and re-run this cell to see the live agent in action.')"
    ),
]

nbf.write(nb, 'notebooks/02_agent_loop.ipynb')
print("Created notebooks/02_agent_loop.ipynb")
```

**Step 2: Run and verify (mock cells only)**

```bash
python notebooks/build_nb02.py
rm notebooks/build_nb02.py
# Execute only the cells that don't require an API key
# (cell with live run is guarded by ANTHROPIC_API_KEY check)
ANTHROPIC_API_KEY="" jupyter nbconvert --to notebook --execute notebooks/02_agent_loop.ipynb \
    --output /tmp/nb02_check.ipynb --ExecutePreprocessor.timeout=120
rm /tmp/nb02_check.ipynb
```

**Step 3: Commit**

```bash
git add notebooks/02_agent_loop.ipynb
git commit -m "feat: add notebook 02 - main agent loop teaching notebook"
```

---

### Task 17: Notebook 03 — Failure Modes (Take-Home)

**Files:**
- Create: `notebooks/03_failure_modes.ipynb`

**Purpose:** Take-home notebook. 5 failure modes from `lecture_plan.md`, each with a broken cell, explanation, fix, and reflection question.

**Step 1: Create using nbformat**

Create `notebooks/build_nb03.py` with these 5 sections:

**Section 1: Bad tool description** — Rename schema parameter `region` → `area`, show agent misuse.

**Section 2: Exception vs. error result** — Show a `buggy_compare` that raises, breaking the loop. Compare to the good version.

**Section 3: Contradictory indices** — Patch `_compute_index_array` to return saturated NDVI but normal EVI; show agent behavior with/without "check for contradictions" instruction.

**Section 4: Infinite loop risk** — Show `max_iterations` guard, then discuss what happens without it.

**Section 5: Overconfident report** — Compare a minimal system prompt vs. the full one.

The build script should create ~30 cells. Key insight: each section needs:
- A `# BROKEN:` code cell showing the failure
- A markdown explanation
- A `# FIXED:` code cell showing the solution
- A markdown reflection question

```python
import nbformat as nbf

nb = nbf.v4.new_notebook()
# ... (build all cells as described above)
nbf.write(nb, 'notebooks/03_failure_modes.ipynb')
```

For brevity, the full cell content follows the pattern of sections 1-5. Implement each section based on the failure modes described in `lecture_plan.md` lines 262–295. Each cell is 10–20 lines of code.

**Step 2: Run and verify**

```bash
python notebooks/build_nb03.py
rm notebooks/build_nb03.py
```

**Step 3: Commit**

```bash
git add notebooks/03_failure_modes.ipynb
git commit -m "feat: add notebook 03 - failure modes take-home notebook"
```

---

## Phase 5: Student Materials

### Task 18: EXERCISES.md and Download Script

**Files:**
- Create: `EXERCISES.md`
- Create: `data/download_data.py`

**Step 1: Create `EXERCISES.md`**

```markdown
# Take-Home Exercises

## Prerequisites

Clone the repo, run `make setup`, then `make demo` to verify everything works.

## Exercise 1: Add a New Tool (Easy)

**Goal:** Understand tool definition and integration.

Implement `compute_savi(region: BoundingBox, L: float = 0.5) -> SAVIResult`:

```
SAVI = ((NIR - Red) / (NIR + Red + L)) * (1 + L)
```

1. Add `SAVIResult` dataclass to `agent/types.py`
2. Implement `compute_savi` in `agent/tools.py`
3. Write the JSON schema in `agent/schemas.py`
4. Add it to `_TOOL_MAP` in `agent/loop.py`
5. Write a test in `tests/test_tools.py`
6. Run the agent and check if it uses SAVI when appropriate

**Reflection:** Did the agent know when to use SAVI vs. NDVI? What in the schema description would make it more likely to choose correctly?

---

## Exercise 2: Modify the System Prompt (Medium)

**Goal:** Understand how instructions shape agent behavior.

Create 3 variants of `SYSTEM_PROMPT` in `agent/loop.py`:

1. **Minimal** — Just the role, no workflow instructions
2. **Directive** — Add "Always call NDVI before any other index"
3. **Original** — The current prompt

Run Scene A with each. Compare:
- Tool call sequences (order and which tools are called)
- Quality of reasoning text
- Handling of the water stress ambiguity

**Reflection:** Which prompt is most useful? Which produces the most instructive trace?

---

## Exercise 3: Error-Handling Tool (Medium)

**Goal:** Understand tool error handling and agent adaptation.

Modify `compare_to_baseline` to fail 50% of the time (use `random.random() < 0.5`).

Then add a new tool:
```python
check_baseline_exists(region: BoundingBox) -> BaselineCheckResult
# Returns whether a baseline exists for the current scene
```

Modify the system prompt to suggest checking for baselines before comparing.

**Reflection:** What are the tradeoffs between preventing errors (defensive tools) vs. recovering from errors (error results)?

---

## Exercise 4: Multi-Scene Comparison (Advanced)

**Goal:** Extend the agent to handle more complex workflows.

New user request: *"Compare crop health in Scene A vs. Scene B and identify which region has more severe stress."*

Options:
- Add a `switch_scene(scene_id: str)` tool
- Or: modify all tools to accept an optional `scene_id` parameter

**Reflection:** Does the agent develop a coherent comparison strategy? What breaks? What would you change?

---

## Exercise 5: Simple Evaluation (Advanced)

**Goal:** Understand evaluation challenges for agentic systems.

Ground truth: Scene A has water stress in the NW quadrant. Scene B has no localized anomaly.

Build an evaluator that checks:
1. Did the agent identify the correct issue (or correctly report no issue)?
2. Did it call reasonable tools in a sensible order?
3. Did it express appropriate confidence?

Run the agent 5 times on each scene (`temperature=0.5` for variety). Measure consistency.

**Reflection:** Is it harder to evaluate correctness of the final report, or quality of the reasoning process? Why does this distinction matter for production systems?
```

**Step 2: Create `data/download_data.py`**

```python
#!/usr/bin/env python3
"""Optional script to download real Sentinel-2 scenes from Copernicus Data Space.

NOTE: For lecture purposes, synthetic scenes are already in data/scenes/ —
this script is for students who want to experiment with real satellite data.

Requirements: pip install sentinelsat
API registration: https://dataspace.copernicus.eu/
"""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Download Sentinel-2 L2A scenes from Copernicus Data Space"
    )
    parser.add_argument("--region", required=True,
                        help="Bounding box: 'lat_min,lon_min,lat_max,lon_max'")
    parser.add_argument("--date", required=True,
                        help="Target date: YYYY-MM-DD")
    parser.add_argument("--output-dir", default="data/scenes/real",
                        help="Output directory for downloaded scenes")
    args = parser.parse_args()

    try:
        from sentinelsat import SentinelAPI
    except ImportError:
        print("Install sentinelsat: pip install sentinelsat")
        sys.exit(1)

    lat_min, lon_min, lat_max, lon_max = [float(x) for x in args.region.split(",")]
    print(f"Searching for Sentinel-2 L2A scenes near {args.region} on {args.date}...")
    print("Note: You will need Copernicus Data Space credentials.")
    print("Register at: https://dataspace.copernicus.eu/")
    print("\nFor lecture use, the synthetic scenes in data/scenes/ are sufficient.")
    print("This script is provided for students who want to work with real data.")


if __name__ == "__main__":
    main()
```

**Step 3: Commit**

```bash
git add EXERCISES.md data/download_data.py
git commit -m "docs: add EXERCISES.md and optional Sentinel-2 download script"
```

---

### Task 19: Final Verification and Cleanup

**Step 1: Run the complete test suite**

```bash
pytest tests/ -v --tb=short
```
Expected: All tests PASS with no warnings.

**Step 2: Run the mock demo end-to-end**

```bash
make demo
```
Expected: Full agent trace, ending with the diagnostic report.

**Step 3: Verify notebook execution**

```bash
for nb in notebooks/00_data_exploration.ipynb notebooks/01_tools.ipynb; do
    jupyter nbconvert --to notebook --execute "$nb" \
        --output /tmp/nb_check.ipynb --ExecutePreprocessor.timeout=120
    rm /tmp/nb_check.ipynb
    echo "✓ $nb OK"
done
```
Expected: Both notebooks execute without errors.

**Step 4: Check repo size**

```bash
du -sh data/scenes/ data/mock_responses/
du -sh .
```
Expected: scenes < 50MB, total repo < 200MB.

**Step 5: Final commit**

```bash
git status
git add -A
git commit -m "feat: complete lecture repository - all notebooks, tools, agent loop, mock mode"
```

---

## Summary

| Phase | Tasks | Key Deliverables |
|-------|-------|-----------------|
| Foundation | 1–3 | Scaffolding, types, synthetic scenes |
| Tools | 4–10 | Scene loader, 6 tested tool functions |
| Agent Infrastructure | 11–13 | Schemas, loop, mock mode |
| Notebooks | 14–17 | 4 teaching notebooks |
| Student Materials | 18–19 | EXERCISES.md, download script, final check |

**Total estimated implementation time:** ~4–6 hours for a focused session.

**Before lecture:** Run `make test && make demo` on the lecture machine. Have `make demo` output pre-saved as a fallback.
