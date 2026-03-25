"""Compute a spectral index (ndvi, evi) for a scene.

Usage: python compute_index.py <scene_id> <index>
  index: ndvi (default) or evi
"""
import sys
from _common import init_scene
from agent.tools import compute_ndvi, compute_evi

scene_id, meta, bbox = init_scene()
index = sys.argv[2] if len(sys.argv) > 2 else "ndvi"

print(f'Scene: {meta["scene_id"]} -- {meta["region"]}')
print(f'Date: {meta["date"]} | Size: {meta["shape"][0]}x{meta["shape"][1]} | Pixel: {meta.get("pixel_size_m", 10)}m')
print()

if index == "evi":
    result = compute_evi(bbox)
else:
    result = compute_ndvi(bbox)

print(result.summary)
print(f"Image: {result.image_path}")
