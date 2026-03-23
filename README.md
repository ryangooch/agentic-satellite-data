# Agentic AI for Satellite Crop Health Assessment

A 75-minute lecture for senior ECE undergraduates demonstrating agentic AI
applied to real Sentinel-2 satellite imagery from California's Central Valley.

## Quick Start

```bash
uv sync
make demo          # Run Scene A in mock mode — no API key needed
```

## What's In This Repo

This project teaches three modern agentic patterns through a single domain
(satellite crop health analysis):

1. **Tool-calling agent loop** — the foundational pattern (`agent/loop.py`)
2. **Claude Code Agent Skill** — wraps the analysis workflow for the CLI (`.claude/skills/`)
3. **MCP server** — live weather data integration (`mcp_servers/weather.py`)

### Architecture

```
┌─────────────────────────────────────────────────────┐
│  Claude Code CLI                                    │
│                                                     │
│  /crop-health-analyst                               │
│    ├── Spectral tools (NDVI, NDWI, EVI, anomalies) │
│    ├── RAG search (county ag reports, UCCE guides)  │
│    └── MCP weather server (Open-Meteo)              │
│                                                     │
│  Real Sentinel-2 data (Planetary Computer)          │
└─────────────────────────────────────────────────────┘
```

## Data

### Synthetic scenes (included)

Two pre-built scenes for offline use and testing:

- **Scene A** — Clear water-stress anomaly in NW quadrant (scripted walkthrough)
- **Scene B** — Mild uniform stress, no spatial anomaly (ambiguous case)

```bash
make generate-data   # Regenerate synthetic scenes
```

### Real Sentinel-2 data

Download real satellite imagery from Microsoft Planetary Computer (free, no auth):

```bash
uv run python data/fetch_sentinel2.py
```

This fetches multi-temporal Sentinel-2 L2A scenes for almond orchards near
Madera, CA — including a current scene, a baseline from a different date,
and an NDVI timeseries across 6+ cloud-free dates.

See [DEMO_WORKFLOW.md](DEMO_WORKFLOW.md) for the full live demo workflow.

## Notebooks

- `00_data_exploration.ipynb` — Band visualization, orientation
- `01_tools.ipynb` — Tools as plain Python functions
- `02_agent_loop.ipynb` — The agent loop pattern (foundational)
- `03_failure_modes.ipynb` — Where agents break (take-home)

## Live Demo (Claude Code CLI)

The primary demo runs in Claude Code, not Jupyter. This gives students the
streaming, interactive experience that makes the agentic pattern click.

**See [DEMO_WORKFLOW.md](DEMO_WORKFLOW.md) for step-by-step instructions.**

Quick version:

```bash
# 1. Fetch real data (do this before the lecture)
uv run python data/fetch_sentinel2.py

# 2. Open Claude Code in this project directory
claude

# 3. Approve the weather MCP server when prompted

# 4. Run the analysis
/crop-health-analyst central_valley
```

## API-Based Agent Loop (Alternative)

Set `ANTHROPIC_API_KEY` in `.env`, then:

```bash
uv run python -m agent.loop --scene central_valley   # Real data
uv run python -m agent.loop --scene scene_a           # Synthetic (stress demo)
uv run python -m agent.loop --scene scene_b           # Synthetic (ambiguous)
```

Estimated cost: $0.10–0.30 per run with `claude-sonnet-4-6`.

## Running Tests

```bash
make test
```

See `EXERCISES.md` for take-home assignments.
