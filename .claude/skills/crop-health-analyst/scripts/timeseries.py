"""Get pixel timeseries for stressed and healthy areas.

Usage: python timeseries.py <scene_id> [row1,col1] [row2,col2]
"""
import sys
from _common import init_scene
from agent.tools import get_pixel_timeseries

init_scene()

def parse_coord(arg, default):
    parts = arg.split(",")
    return int(parts[0]), int(parts[1])

r1, c1 = parse_coord(sys.argv[2], "50,50") if len(sys.argv) > 2 else (50, 50)
r2, c2 = parse_coord(sys.argv[3], "150,150") if len(sys.argv) > 3 else (150, 150)

print(get_pixel_timeseries(r1, c1, "ndvi").summary)
print()
print(get_pixel_timeseries(r2, c2, "ndvi").summary)
