# Agentic AI for Satellite Crop Health Assessment

A 75-minute lecture for senior ECE undergraduates demonstrating agentic AI
applied to Sentinel-2 satellite imagery.

## Quick Start

```bash
pip install -r requirements.txt
make demo          # Run Scene A in mock mode — no API key needed
```

## Notebooks

- `00_data_exploration.ipynb` — Band visualization, orientation
- `01_tools.ipynb` — Tools as plain Python functions
- `02_agent_loop.ipynb` — **Start here** — main teaching notebook
- `03_failure_modes.ipynb` — Where agents break (take-home)

## Live API Runs

Set `ANTHROPIC_API_KEY` in your environment, then:

```bash
python -m agent.loop --scene scene_a
python -m agent.loop --scene scene_b
```

Estimated cost: $0.10–0.30 per run with `claude-sonnet-4-6`.

## Running Tests

```bash
make test
```

See `EXERCISES.md` for take-home assignments.
