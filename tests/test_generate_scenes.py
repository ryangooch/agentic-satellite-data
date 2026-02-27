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
