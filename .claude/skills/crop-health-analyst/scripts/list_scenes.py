"""List available scenes."""
import json
from pathlib import Path

for f in sorted(Path("data/scenes").glob("*_metadata.json")):
    m = json.loads(f.read_text())
    print(f'{m["scene_id"]}: {m["region"]} -- {m["date"]} (baseline: {m.get("has_baseline", False)})')
