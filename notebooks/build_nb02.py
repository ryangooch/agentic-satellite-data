import sys
sys.path.insert(0, '.')
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.cells = [
    nbf.v4.new_markdown_cell(
        "# Notebook 02: The Agent Loop\n\n"
        "**Learning objectives:**\n"
        "- Understand what makes a system 'agentic' vs. a standard ML pipeline\n"
        "- See how tool calling works at the API level (no framework)\n"
        "- Read the agent's reasoning trace and understand each decision\n"
        "- Observe how the agent adapts to a different scene\n\n"
        "**Prerequisites:** Run Notebook 00 and 01 first (or at least 00)."
    ),
    nbf.v4.new_code_cell(
        "import sys, os\n"
        "from pathlib import Path\n"
        "# Move to project root so relative data paths resolve correctly\n"
        "_here = Path(os.getcwd())\n"
        "if not (_here / 'data' / 'scenes').exists() and (_here.parent / 'data' / 'scenes').exists():\n"
        "    os.chdir(_here.parent)\n"
        "sys.path.insert(0, str(Path(os.getcwd())))\n\n"
        "import inspect\n"
        "from agent.scene import load_scene\n"
        "from agent.types import BoundingBox\n"
        "from agent import tools\n"
        "from agent.schemas import TOOL_SCHEMAS\n"
        "from agent.loop import run_agent, SYSTEM_PROMPT\n"
        "from agent.mock import run_mock_agent\n\n"
        "load_scene('scene_a')\n"
        "print('Scene A loaded. Ready.')"
    ),
    nbf.v4.new_markdown_cell(
        "## Step 1: One Tool Example\n\n"
        "Before the agent, let's confirm: tools are just functions."
    ),
    nbf.v4.new_code_cell(
        "bb = BoundingBox(0, 200, 0, 200)\n"
        "result = tools.compute_ndvi(bb)\n"
        "print(result.summary)"
    ),
    nbf.v4.new_markdown_cell(
        "## Step 2: One Schema Example\n\n"
        "This is how the model learns what `compute_ndvi` accepts:"
    ),
    nbf.v4.new_code_cell(
        "import json\n"
        "ndvi_schema = next(s for s in TOOL_SCHEMAS if s['name'] == 'compute_ndvi')\n"
        "print(json.dumps(ndvi_schema, indent=2))"
    ),
    nbf.v4.new_markdown_cell(
        "## Step 3: The System Prompt\n\n"
        "Read this carefully — every instruction shapes agent behavior.\n\n"
        "Note especially:\n"
        "- *\"Explain your reasoning before each tool call\"* → creates the trace we learn from\n"
        "- *\"Only call tools when warranted\"* → prevents exhaustive tool use\n"
        "- *\"Express uncertainty when appropriate\"* → prevents overconfident reports"
    ),
    nbf.v4.new_code_cell("print(SYSTEM_PROMPT)"),
    nbf.v4.new_markdown_cell(
        "## Step 4: The Loop\n\n"
        "The entire agent loop is ~50 lines. Read it:"
    ),
    nbf.v4.new_code_cell(
        "from agent.loop import run_agent\n"
        "print(inspect.getsource(run_agent))"
    ),
    nbf.v4.new_markdown_cell(
        "## Step 5: Scripted Run — Scene A\n\n"
        "We use **mock mode** here so the trace is identical every time. "
        "The model responses are pre-recorded, but **all tool calls are real** (live numpy computation).\n\n"
        "Watch each `[TOOL CALL]` block:\n"
        "1. What did the agent observe that prompted this call?\n"
        "2. What new information did it learn?\n"
        "3. What might it do next?"
    ),
    nbf.v4.new_code_cell(
        "load_scene('scene_a')\n"
        "run_mock_agent('scene_a')"
    ),
    nbf.v4.new_markdown_cell(
        "### Discussion Questions\n\n"
        "1. After `compute_ndvi`: Why did the agent choose `flag_anomalous_regions` next rather than immediately calling `compute_ndwi`?\n"
        "2. After `flag_anomalous_regions`: The agent now calls `compute_ndwi` on the *anomalous sub-region*, not the full scene. Why is this more efficient?\n"
        "3. After `compute_ndwi`: The NDWI is negative. Could this mean water stress, bare soil, or something else? What rules this out?\n"
        "4. After `get_pixel_timeseries`: Why does recent onset matter for diagnosis?\n"
        "5. The agent called `compare_to_baseline` last, not first. Was this the right order?"
    ),
    nbf.v4.new_markdown_cell(
        "## Step 6: Live Run — Scene B\n\n"
        "Scene B is ambiguous: mild uniform NDVI depression, no strong spatial anomaly, no baseline.\n\n"
        "**Watch:** Does the agent take a different path than it did for Scene A?\n\n"
        "Warning: **Requires `ANTHROPIC_API_KEY`** in your environment. "
        "Estimated cost: ~$0.10–0.20 for this run.\n\n"
        "If you don't have an API key, skip this cell — the take-home exercises cover this."
    ),
    nbf.v4.new_code_cell(
        "# LIVE API CALL — requires ANTHROPIC_API_KEY\n"
        "if 'ANTHROPIC_API_KEY' in os.environ:\n"
        "    load_scene('scene_b')\n"
        "    run_agent(\n"
        "        'Assess crop health for the entire scene. '\n"
        "        'Identify any areas of concern and determine likely causes.',\n"
        "        temperature=0,\n"
        "    )\n"
        "else:\n"
        "    print('ANTHROPIC_API_KEY not set — skipping live run.')\n"
        "    print('Set your key and re-run this cell to see the live agent in action.')"
    ),
]

nbf.write(nb, 'notebooks/02_agent_loop.ipynb')
print("Created notebooks/02_agent_loop.ipynb")
