"""Shared helpers for skill scripts."""
import json
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path so `agent` package is importable
_project_root = os.getcwd()
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from agent.scene import load_scene  # noqa: E402
from agent.models import BoundingBox  # noqa: E402


def init_scene(default="central_valley"):
    """Load scene from argv[1] (or default), return (scene_id, metadata, full_bbox)."""
    scene_id = sys.argv[1] if len(sys.argv) > 1 else default
    load_scene(scene_id)
    meta = json.loads(Path(f"data/scenes/{scene_id}_metadata.json").read_text())
    H, W = meta["shape"]
    bbox = BoundingBox(0, H, 0, W)
    return scene_id, meta, bbox
