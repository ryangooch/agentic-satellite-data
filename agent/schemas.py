# agent/schemas.py
"""JSON schemas for all tools, in the format required by the Anthropic Messages API.

Passed directly to the `tools=` parameter of `client.messages.create()`.
We use plain Python dicts for simplicity, but in general you'll use other libraries for this,
pydantic, or langgraph, etc.
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
