import sys
sys.path.insert(0, '.')
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.cells = [
    # --- Header ---
    nbf.v4.new_markdown_cell(
        "# Notebook 03: Failure Modes (Take-Home)\n\n"
        "This notebook covers 5 ways agentic systems break — and how to fix them.\n\n"
        "**Each section:** broken cell → explanation → fix → reflection question.\n\n"
        "Work through these after lecture to deepen your understanding of agentic design."
    ),
    nbf.v4.new_code_cell(
        "import sys, os\n"
        "from pathlib import Path\n"
        "# Move to project root so relative data paths resolve correctly\n"
        "_here = Path(os.getcwd())\n"
        "if not (_here / 'data' / 'scenes').exists() and (_here.parent / 'data' / 'scenes').exists():\n"
        "    os.chdir(_here.parent)\n"
        "sys.path.insert(0, str(Path(os.getcwd())))\n\n"
        "import json\n"
        "from agent.scene import load_scene\n"
        "from agent.types import BoundingBox\n"
        "from agent import tools\n"
        "from agent.schemas import TOOL_SCHEMAS\n"
        "from agent.loop import run_agent, dispatch_tool, SYSTEM_PROMPT\n"
        "from agent.mock import run_mock_agent\n"
        "from unittest.mock import MagicMock\n\n"
        "load_scene('scene_a')\n"
        "print('Setup complete.')"
    ),

    # --- Section 1: Bad tool description ---
    nbf.v4.new_markdown_cell(
        "---\n\n## Failure 1: Bad Tool Description\n\n"
        "**Lesson:** Tool descriptions must be unambiguous. If a parameter name or description "
        "is unclear, the model may call the tool incorrectly or not at all."
    ),
    nbf.v4.new_code_cell(
        "# BROKEN: A schema with an ambiguous parameter name\n"
        "broken_schema = {\n"
        "    'name': 'compute_ndvi',\n"
        "    'description': 'Compute vegetation index.',  # Vague — no guidance on when to use it\n"
        "    'input_schema': {\n"
        "        'type': 'object',\n"
        "        'properties': {\n"
        "            'area': {  # Renamed from 'region' — model will likely send 'area' not 'region'\n"
        "                'type': 'object',\n"
        "                'properties': {\n"
        "                    'row_min': {'type': 'integer'},\n"
        "                    'row_max': {'type': 'integer'},\n"
        "                    'col_min': {'type': 'integer'},\n"
        "                    'col_max': {'type': 'integer'},\n"
        "                },\n"
        "            }\n"
        "        },\n"
        "        'required': ['area'],\n"
        "    },\n"
        "}\n"
        "print('Broken schema has parameter named: area')\n"
        "print('Good schema has parameter named:', list(next(s for s in TOOL_SCHEMAS if s['name'] == 'compute_ndvi')['input_schema']['properties'].keys()))\n"
        "print()\n"
        "print('If the model is told the parameter is called \"area\" but dispatch_tool')\n"
        "print('expects \"region\", the call will fail with a TypeError.')"
    ),
    nbf.v4.new_markdown_cell(
        "**Why it breaks:** `dispatch_tool` calls `tools.compute_ndvi(region=...)`. "
        "If the model sends `area=...` (as the schema dictates), Python raises `TypeError: unexpected keyword argument 'area'`.\n\n"
        "**The fix:** Schema parameter names must exactly match the Python function parameter names. "
        "Descriptions should tell the model *when* to call the tool, not just *what* it does."
    ),
    nbf.v4.new_code_cell(
        "# FIXED: Show the correct schema for context\n"
        "good_schema = next(s for s in TOOL_SCHEMAS if s['name'] == 'compute_ndvi')\n"
        "print(json.dumps(good_schema, indent=2))\n"
        "print()\n"
        "print('Key fixes:')\n"
        "print('  1. Parameter named \"region\" (matches Python function argument)')\n"
        "print('  2. Detailed description tells model WHEN to use it and what output to expect')"
    ),
    nbf.v4.new_markdown_cell(
        "**Reflection question:** Look at the schema for `get_pixel_timeseries`. "
        "It uses `lat` and `lon` as parameter names even though these are actually pixel row/col coordinates. "
        "Is this a good naming choice? What are the tradeoffs? How might you fix it without breaking existing prompts?"
    ),

    # --- Section 2: Exception vs. Error Result ---
    nbf.v4.new_markdown_cell(
        "---\n\n## Failure 2: Exception vs. Error Result\n\n"
        "**Lesson:** Tools that raise exceptions break the agent loop. Tools that return error results "
        "let the agent reason about the failure and adapt."
    ),
    nbf.v4.new_code_cell(
        "# BROKEN: A version of compare_to_baseline that raises instead of returning error\n"
        "def buggy_compare_to_baseline(region: BoundingBox, index: str):\n"
        "    from agent import scene as _scene\n"
        "    # This raises FileNotFoundError instead of returning a DiffResult\n"
        "    base_nir = _scene.get_baseline_band('B08', region)  # raises for scene_b!\n"
        "    return 'would compute diff here'\n\n"
        "load_scene('scene_b')  # Scene B has no baseline\n"
        "print('Calling buggy version on scene_b:')\n"
        "try:\n"
        "    buggy_compare_to_baseline(BoundingBox(0, 200, 0, 200), 'ndvi')\n"
        "except FileNotFoundError as e:\n"
        "    print(f'EXCEPTION RAISED: {e}')\n"
        "    print()\n"
        "    print('The agent loop receives an unhandled exception.')\n"
        "    print('The loop crashes — no final report, no graceful degradation.')"
    ),
    nbf.v4.new_code_cell(
        "# FIXED: The actual implementation returns an error result\n"
        "load_scene('scene_b')\n"
        "result = tools.compare_to_baseline(BoundingBox(0, 200, 0, 200), index='ndvi')\n\n"
        "print(f'success: {result.success}')\n"
        "print(f'error_message: {result.error_message}')\n"
        "print()\n"
        "print('The agent receives this message in the tool_result block.')\n"
        "print('It can then say: \"No baseline available — I will rely on NDVI and NDWI instead.\"')"
    ),
    nbf.v4.new_markdown_cell(
        "**Reflection question:** Imagine a tool that calls an external weather API. "
        "Sometimes the API is unavailable (network error). "
        "Should this tool raise an exception or return an error result? "
        "What information should the error message contain to help the agent decide what to do next?"
    ),

    # --- Section 3: Contradictory Indices ---
    nbf.v4.new_markdown_cell(
        "---\n\n## Failure 3: Contradictory Indices\n\n"
        "**Lesson:** Spectral indices can give conflicting signals. "
        "Without an explicit instruction to check for contradictions, the agent may confidently report "
        "based on only one index."
    ),
    nbf.v4.new_code_cell(
        "# Demonstrate contradictory indices using Scene B (mild uniform depression)\n"
        "load_scene('scene_b')\n"
        "ndvi_result = tools.compute_ndvi(BoundingBox(0, 200, 0, 200))\n"
        "evi_result = tools.compute_evi(BoundingBox(0, 200, 0, 200))\n\n"
        "print('NDVI result:')\n"
        "print(ndvi_result.summary)\n"
        "print()\n"
        "print('EVI result:')\n"
        "print(evi_result.summary)\n"
        "print()\n"
        "# Check if signals agree\n"
        "ndvi_stressed = ndvi_result.low_fraction > 0.3\n"
        "evi_stressed = evi_result.low_fraction > 0.3\n"
        "if ndvi_stressed != evi_stressed:\n"
        "    print('Contradiction: NDVI and EVI disagree on stress level!')\n"
        "    print(f'   NDVI low-fraction: {ndvi_result.low_fraction:.1%}')\n"
        "    print(f'   EVI  low-fraction: {evi_result.low_fraction:.1%}')\n"
        "    print('   Scene B is designed to be ambiguous — this is expected.')\n"
        "else:\n"
        "    print('Indices agree on stress level.')"
    ),
    nbf.v4.new_markdown_cell(
        "**Why this matters:** Without a system prompt instruction to 'check for contradictions', "
        "the agent might report based on whichever index it happened to call first.\n\n"
        "**The fix:** Add an explicit instruction like:\n"
        "```\n"
        "If NDVI and EVI give contradictory signals (e.g., one suggests stress, the other does not),\n"
        "explicitly note the contradiction in your report and express lower confidence.\n"
        "```"
    ),
    nbf.v4.new_code_cell(
        "# Show the 'express uncertainty' instruction from the real system prompt\n"
        "uncertainty_line = [line for line in SYSTEM_PROMPT.split('\\n') if 'uncertainty' in line.lower() or 'conflicting' in line.lower()]\n"
        "print('System prompt instructions about uncertainty:')\n"
        "for line in uncertainty_line:\n"
        "    print(f'  {line.strip()}')\n"
        "print()\n"
        "print('This is why the \"express uncertainty\" instruction is in the system prompt.')"
    ),
    nbf.v4.new_markdown_cell(
        "**Reflection question:** The current system prompt says 'Express uncertainty when appropriate. "
        "If indices give conflicting signals, say so.' Is this specific enough? "
        "Write a stronger version of this instruction that tells the agent exactly what to do "
        "when NDVI and EVI disagree by more than 0.1."
    ),

    # --- Section 4: Infinite Loop Risk ---
    nbf.v4.new_markdown_cell(
        "---\n\n## Failure 4: Infinite Loop Risk\n\n"
        "**Lesson:** Without a `max_iterations` guard, a misbehaving tool or prompt "
        "could cause the agent to loop forever, burning API credits."
    ),
    nbf.v4.new_code_cell(
        "# Demonstrate: what happens if stop_reason is never 'end_turn'?\n"
        "# We'll build a mock client that always returns tool_use\n\n"
        "def make_always_tool_use_client():\n"
        "    client = MagicMock()\n"
        "    def always_tool_use(*args, **kwargs):\n"
        "        mr = MagicMock()\n"
        "        mr.stop_reason = 'tool_use'\n"
        "        block = MagicMock()\n"
        "        block.type = 'tool_use'\n"
        "        block.id = 'mock_id'\n"
        "        block.name = 'compute_ndvi'\n"
        "        block.input = {'region': {'row_min': 0, 'row_max': 50, 'col_min': 0, 'col_max': 50}}\n"
        "        mr.content = [block]\n"
        "        return mr\n"
        "    client.messages.create.side_effect = always_tool_use\n"
        "    return client\n\n"
        "load_scene('scene_a')\n"
        "client = make_always_tool_use_client()\n"
        "print('Running agent with max_iterations=3 and a stuck client...')\n"
        "run_agent('Analyze.', client=client, max_iterations=3, verbose=True)\n"
        "print(f'\\nAPI calls made: {client.messages.create.call_count} (stopped at max_iterations=3)')"
    ),
    nbf.v4.new_markdown_cell(
        "**The guard that saves us:** In `agent/loop.py`, the outer `for _ in range(max_iterations)` "
        "loop ensures we never exceed `max_iterations` API calls, even if the model never returns `end_turn`.\n\n"
        "**What could cause infinite loops in practice?**\n"
        "- A tool that always returns an error, causing the agent to retry forever\n"
        "- A system prompt that says 'keep calling tools until you are certain' with no exit condition\n"
        "- A model bug where `stop_reason` is never set to `end_turn`"
    ),
    nbf.v4.new_code_cell(
        "# Show the guard in the actual loop code\n"
        "import inspect\n"
        "from agent.loop import run_agent\n"
        "source = inspect.getsource(run_agent)\n"
        "# Find the max_iterations line\n"
        "for i, line in enumerate(source.split('\\n')):\n"
        "    if 'max_iterations' in line:\n"
        "        print(f'Line {i+1}: {line}')\n"
        "print()\n"
        "print('The for loop ensures we stop. The warning at the end tells you when it triggers.')"
    ),
    nbf.v4.new_markdown_cell(
        "**Reflection question:** The current `max_iterations` default is 20. "
        "Is this the right value? What factors would you consider when choosing it for a production system? "
        "What would you log when the guard triggers to help with debugging?"
    ),

    # --- Section 5: Overconfident Report ---
    nbf.v4.new_markdown_cell(
        "---\n\n## Failure 5: Overconfident Report\n\n"
        "**Lesson:** Without explicit uncertainty instructions, agents tend to produce "
        "confident-sounding reports even when the data is ambiguous."
    ),
    nbf.v4.new_code_cell(
        "# Compare minimal vs full system prompt behavior with Scene B (ambiguous)\n"
        "MINIMAL_PROMPT = \"\"\"You are a crop health analyst. Assess the scene and produce a report.\"\"\"\n\n"
        "print('=== MINIMAL SYSTEM PROMPT ===')\n"
        "print(MINIMAL_PROMPT)\n"
        "print()\n"
        "print('=== FULL SYSTEM PROMPT (key uncertainty instructions) ===')\n"
        "for line in SYSTEM_PROMPT.split('\\n'):\n"
        "    if any(kw in line.lower() for kw in ['uncertainty', 'conflicting', 'uncertain', 'appropriate']):\n"
        "        print(f'  {line.strip()}')"
    ),
    nbf.v4.new_code_cell(
        "# Run mock agent to show the trace structure\n"
        "# (Full comparison would need live API — we show the structural difference)\n"
        "print('The minimal prompt omits:')\n"
        "print('  - Instruction to explain reasoning before each tool call')\n"
        "print('  - Instruction to only call tools when warranted')\n"
        "print('  - Instruction to express uncertainty when appropriate')\n"
        "print('  - Specific report structure requirements')\n"
        "print()\n"
        "print('Expected behavior difference with Scene B (ambiguous):')\n"
        "print('  Minimal prompt -> confident report (\"water stress detected\") without caveats')\n"
        "print('  Full prompt    -> hedged report (\"possible mild stress, confidence LOW,\"')\n"
        "print('                   \"recommend further investigation\")')\n"
        "print()\n"
        "print('Try it yourself with ANTHROPIC_API_KEY set:')\n"
        "print('  run_agent(request, system_prompt=MINIMAL_PROMPT)  # compare to default')"
    ),
    nbf.v4.new_markdown_cell(
        "**Reflection question:** Overconfident reports are particularly dangerous in high-stakes domains "
        "like agriculture, medicine, or infrastructure. "
        "Beyond system prompt instructions, what other mechanisms could you use to prevent overconfidence? "
        "Consider: output validation, human-in-the-loop triggers, confidence calibration, ensemble methods."
    ),

    # --- Summary ---
    nbf.v4.new_markdown_cell(
        "---\n\n## Summary: The 5 Failure Modes\n\n"
        "| # | Failure | Fix |\n"
        "|---|---------|-----|\n"
        "| 1 | Bad tool description | Match parameter names to Python; write clear, specific descriptions |\n"
        "| 2 | Exception vs. error result | Tools return error dicts; the loop never sees raw exceptions |\n"
        "| 3 | Contradictory indices | Explicit system prompt instruction to flag and explain contradictions |\n"
        "| 4 | Infinite loop | `max_iterations` guard; log when triggered |\n"
        "| 5 | Overconfident report | Uncertainty instructions in system prompt; structural report requirements |\n\n"
        "**The meta-lesson:** The agent loop itself is simple. Most failures come from the periphery — "
        "tool design, schema quality, and system prompt instructions. "
        "This is why each of these components deserves careful testing in isolation (Notebook 01) "
        "before combining them into an agent (Notebook 02)."
    ),
]

nbf.write(nb, 'notebooks/03_failure_modes.ipynb')
print("Created notebooks/03_failure_modes.ipynb")
