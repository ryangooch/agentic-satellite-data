"""Flag anomalous regions in a scene.

Usage: python flag_anomalies.py <scene_id> [index] [threshold] [direction]
"""
import sys
from _common import init_scene
from agent.tools import flag_anomalous_regions

init_scene()
index = sys.argv[2] if len(sys.argv) > 2 else "ndvi"
threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
direction = sys.argv[4] if len(sys.argv) > 4 else "below"

result = flag_anomalous_regions(index, threshold, direction)
print(result.summary)
