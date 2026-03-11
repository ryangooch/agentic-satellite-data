# Agentic AI for Satellite Crop Health Assessment

A 75-minute lecture for senior ECE undergraduates demonstrating agentic AI
applied to Sentinel-2 satellite imagery.

## Quick Start

```bash
uv sync
make demo          # Run Scene A in mock mode — no API key needed
```

## Notebooks

- `00_data_exploration.ipynb` — Band visualization, orientation
- `01_tools.ipynb` — Tools as plain Python functions
- `02_agent_loop.ipynb` — **Start here** — main teaching notebook
- `03_failure_modes.ipynb` — Where agents break (take-home)

## Live API Runs

### Obtaining your Anthropic API Key

Sign in to `platform.claude.com` with your credentials, then create your key there.

Put the key in the `.env` file as such:

```
ANTHROPIC_API_KEY=sk-ant-api03-...
```

### Running the Loop
Set `ANTHROPIC_API_KEY` in your environment, then:

```bash
uv run python -m agent.loop --scene scene_a
uv run python -m agent.loop --scene scene_b
```

Estimated cost: $0.10–0.30 per run with `claude-sonnet-4-6`.

Sonnet is excellent at most reasoning and coding-related tasks while being much cheaper than Opus,
so it's a good idea to start with Sonnet (or even Haiku) to see if it works for your needs before
going to the most expensive and highest quality model.


## Running Tests

```bash
make test
```

See `EXERCISES.md` for take-home assignments.
