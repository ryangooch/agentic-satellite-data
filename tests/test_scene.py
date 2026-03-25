# tests/test_scene.py
import pytest
import numpy as np
from agent.scene import load_scene, get_band, get_baseline_band, get_metadata, get_timeseries_ndvi, scene_shape
from agent.models import BoundingBox


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
