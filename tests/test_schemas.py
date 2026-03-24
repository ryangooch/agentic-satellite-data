# tests/test_schemas.py
from agent.schemas import TOOL_SCHEMAS


def test_schema_count():
    assert len(TOOL_SCHEMAS) == 8


def test_each_schema_has_required_keys():
    for s in TOOL_SCHEMAS:
        assert "name" in s
        assert "description" in s
        assert "input_schema" in s
        assert s["input_schema"]["type"] == "object"


def test_tool_names():
    names = {s["name"] for s in TOOL_SCHEMAS}
    expected = {
        "compute_ndvi", "compute_ndwi", "compute_evi", "compute_cwsi",
        "get_pixel_timeseries", "flag_anomalous_regions", "compare_to_baseline",
        "search_agricultural_context",
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
