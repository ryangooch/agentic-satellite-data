"""Shared helpers for skill scripts."""
import json
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path so `agent` package is importable
_project_root = os.getcwd()
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from agent.scene import load_scene, set_crop_mask, clear_crop_mask  # noqa: E402
from agent.models import BoundingBox  # noqa: E402


def _pop_flag(flag):
    """Remove --flag value from sys.argv and return the value, or None."""
    try:
        idx = sys.argv.index(flag)
        val = sys.argv[idx + 1]
        del sys.argv[idx:idx + 2]
        return val
    except (ValueError, IndexError):
        return None


def init_scene(default="central_valley"):
    """Load scene from argv[1] (or default), return (scene_id, metadata, full_bbox).

    Supports --crop <name> flag to filter analysis to a specific crop type.
    """
    crop_filter = _pop_flag("--crop")
    scene_id = sys.argv[1] if len(sys.argv) > 1 else default
    load_scene(scene_id)
    meta = json.loads(Path(f"data/scenes/{scene_id}_metadata.json").read_text())
    H, W = meta["shape"]
    bbox = BoundingBox(0, H, 0, W)
    if crop_filter:
        n_pixels = set_crop_mask(crop_filter)
        total = H * W
        print(f"Crop filter: {crop_filter} — {n_pixels:,} pixels ({n_pixels/total:.1%} of scene)")
        print()
    return scene_id, meta, bbox
