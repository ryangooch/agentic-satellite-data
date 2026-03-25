"""Compare current scene to baseline.

Usage: python compare_baseline.py <scene_id> [index]
"""
import sys
from _common import init_scene
from agent.tools import compare_to_baseline

_, meta, bbox = init_scene()
index = sys.argv[2] if len(sys.argv) > 2 else "ndvi"

result = compare_to_baseline(bbox, index)
print(result.summary)
if result.image_path:
    print(f"Image: {result.image_path}")
