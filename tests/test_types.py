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
