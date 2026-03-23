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

### What Happens

Claude Code will follow the skill's 7-step workflow:

1. **Load scene & compute NDVI** — shows the real satellite image with field boundaries
2. **Compute NDWI** — checks for water stress specifically
3. **Compute EVI** — cross-checks NDVI in dense canopy areas
4. **Flag anomalous regions** — identifies specific grid cells with stress
5. **Search agricultural context** — RAG retrieves relevant passages from local
   ag reports (Madera County crop report, UCCE almond irrigation guide, etc.)
6. **Check weather** — calls the MCP weather server for historical conditions
   (temperature, precipitation, ET0, soil moisture) and correlates with observations
7. **Produce diagnostic report** — synthesizes all findings into a structured report

### Points to Highlight

**During Step 1-3 (Spectral Indices):**
- "These are real Sentinel-2 pixels, not synthetic data"
- "Notice Claude decides which tools to call and in what order"
- "The images show actual farm parcels — you can see field boundaries"

**During Step 5 (RAG):**
- "The agent is searching local agricultural documents for context"
- "It knows that NDVI 0.55-0.75 is healthy for almonds because it retrieved
  the UCCE extension guide"

**During Step 6 (Weather MCP):**
- "This is a live API call to Open-Meteo — real weather data"
- "The agent correlates the satellite observations with actual weather events"
- "This is the MCP pattern — a long-running server providing tools"

**During Step 7 (Report):**
- "Notice it synthesizes spectral data + local context + weather into a diagnosis"
- "It expresses uncertainty when signals conflict"
- "This is the agentic pattern: autonomous reasoning with tools"

### Skill vs MCP — The Two Patterns

Use this moment to teach the distinction:

| | Agent Skill | MCP Server |
|---|---|---|
| **What** | Instructions + bundled code | Long-running tool server |
| **Where** | `.claude/skills/SKILL.md` | `.mcp.json` + server process |
| **When** | On-demand, per invocation | Always available |
| **Example** | Crop analysis workflow | Weather API |
| **Best for** | Procedural workflows, local tools | External APIs, stateful services |

The skill *orchestrates* the analysis. The MCP server *provides* weather data.
They work together — the skill's instructions tell Claude to query the MCP
server at the right moment.

## Fallback: Mock Mode (No API Key)

If you don't have an Anthropic API key or want a guaranteed reproducible demo:

```bash
make demo
```

This runs Scene A with pre-recorded agent responses. Real tool calls execute
(live spectral index computation), but model responses are canned.

## Fallback: API-Based Loop

If Claude Code isn't available, the API-based agent loop still works:

```bash
# Set your API key
export ANTHROPIC_API_KEY=sk-ant-api03-...

# Run on real data
uv run python -m agent.loop --scene central_valley

# Run on synthetic data (Scene A has a clear anomaly)
uv run python -m agent.loop --scene scene_a
```

## Troubleshooting

**MCP weather server not connecting:**
- Check `.mcp.json` exists in the project root
- Restart Claude Code (`/quit` then `claude`)
- Run `/mcp` to check server status

**Sentinel-2 download fails:**
- Planetary Computer is free but occasionally has outages
- The synthetic scenes (`scene_a`, `scene_b`) always work as fallback
- Try a different date range if cloud cover filters too aggressively

**RAG returns no results:**
- Check that `data/rag_documents/` contains `.md` files
- The TF-IDF index builds on first query — make sure scipy is installed (`uv sync`)
