"""Compute CWSI for a scene.

Usage: python compute_cwsi.py <scene_id> <air_temp_f> <vpd_kpa> [crop_type]
"""
import sys
from _common import init_scene
from agent.tools import compute_cwsi

scene_id, meta, bbox = init_scene()
air_temp_f = float(sys.argv[2])
vpd_kpa = float(sys.argv[3])
crop_type = sys.argv[4] if len(sys.argv) > 4 else "alfalfa"

result = compute_cwsi(bbox, air_temp_f=air_temp_f, vpd_kpa=vpd_kpa, crop_type=crop_type)
print(result.summary)
print(f"Image: {result.image_path}")
