# tests/test_tools.py
import pytest
import numpy as np
from agent.scene import load_scene
from agent.types import BoundingBox
from agent.tools import (
    compute_ndvi, compute_ndwi, compute_evi,
    get_pixel_timeseries, flag_anomalous_regions, compare_to_baseline,
)


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
