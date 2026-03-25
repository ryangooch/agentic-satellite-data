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
class CWSIResult:
    success: bool
    cwsi_array: Optional[np.ndarray] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    high_fraction: Optional[float] = None  # fraction of pixels > 0.5 (stressed)
    vpd: Optional[float] = None            # vapor pressure deficit (kPa)
    air_temp_f: Optional[float] = None     # air temperature (°F)
    image_path: Optional[str] = None
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
