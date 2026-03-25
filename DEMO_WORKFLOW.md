# Live Demo Workflow

Step-by-step guide for running the crop health analysis demo in Claude Code.

## Before the Lecture

### 1. Fetch real satellite data

```bash
uv run python data/fetch_sentinel2.py
```

This downloads real Sentinel-2 L2A imagery from Microsoft Planetary Computer
for almond orchards near Madera, CA. Takes ~2 minutes. No API key needed.

Output:
- `data/scenes/central_valley_bands.npy` — current scene (7 spectral bands)
- `data/scenes/central_valley_baseline_bands.npy` — baseline from an earlier date
- `data/scenes/central_valley_metadata.json` — coordinates, dates, timeseries

You can customize the location and dates:

```bash
uv run python data/fetch_sentinel2.py \
  --lat 36.75 --lon -120.24 \
  --current-date 2024-07-15 \
  --baseline-date 2024-06-01 \
  --size 2000
```

### 2. Verify the data works

```bash
uv run python -c "
from agent.scene import load_scene
from agent.tools import compute_ndvi
from agent.types import BoundingBox
load_scene('central_valley')
r = compute_ndvi(BoundingBox(0, 200, 0, 200))
print(r.summary)
"
```

You should see real NDVI statistics and an image saved to `data/images/ndvi.png`.

### 3. Verify tests pass

```bash
make test
```

## During the Lecture

### Opening Claude Code

```bash
cd /path/to/agentic-satellite-data
claude
```

On first launch, Claude Code will detect the weather MCP server in `.mcp.json`
and ask you to approve it. Say yes — this connects the Open-Meteo weather API.

### Running the Demo

Type one of:

```
/crop-health-analyst central_valley
```

or simply ask in natural language:

```
Analyze crop health for the central valley scene. Identify areas of concern
and determine likely causes.
```

## Specific Commands of Interest

```bash
# alfalfa-only plot updates, using Skill code
uv run python .claude/skills/crop-health-analyst/scripts/compute_cwsi.py central_valley 95 2.5 alfalfa --crop alfalfa

uv run python .claude/skills/crop-health-analyst/scripts/compute_index.py central_valley ndvi --crop alfalfa

uv run python .claude/skills/crop-health-analyst/scripts/flag_anomalies.py central_valley ndvi 0.3 below --crop alfalfa

uv run python .claude/skills/crop-health-analyst/scripts/compute_index.py central_valley ndvi --crop alfalfa
```