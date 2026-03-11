# agent/loop.py
"""The agent loop — the core teaching artifact.

This is the complete agentic pattern in ~80 lines:
  1. Call the model with messages + tool schemas
  2. If stop_reason == "tool_use": execute tools, append results, repeat
  3. If stop_reason == "end_turn": done

No framework. No magic.
"""
import json
import os
from typing import Optional

import anthropic

from agent.schemas import TOOL_SCHEMAS
from agent.types import BoundingBox
from agent import tools as _tools

MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are an autonomous crop health analyst working with satellite imagery. \
Your job is to assess crop health in a given region and produce a diagnostic report.

You have access to tools that compute spectral indices (NDVI, NDWI, EVI), retrieve \
timeseries data, flag anomalous regions, and compare current conditions to historical baselines.

**Your workflow**:
1. Start with a broad assessment (e.g., compute NDVI for the region)
2. If you detect anomalies or low values, investigate further using additional indices or timeseries
3. Explain your reasoning in plain text before each tool call. Describe what you observed and why \
you're choosing the next tool.
4. Only call tools when warranted by your observations. You do not need to use all available tools.
5. If tools return errors, adapt your strategy and explain what happened.
6. When you have sufficient information, produce a final diagnostic report with:
   - Summary of findings
   - Specific locations of concern (if any)
   - Likely cause of stress (water, nutrients, pests) with confidence level
   - Recommended follow-up actions
7. Express uncertainty when appropriate. If indices give conflicting signals, say so.

**Important**: Be concise in your reasoning (2-3 sentences per tool call). \
The goal is clarity, not verbosity."""

_TOOL_MAP = {
    "compute_ndvi": _tools.compute_ndvi,
    "compute_ndwi": _tools.compute_ndwi,
    "compute_evi": _tools.compute_evi,
    "get_pixel_timeseries": _tools.get_pixel_timeseries,
    "flag_anomalous_regions": _tools.flag_anomalous_regions,
    "compare_to_baseline": _tools.compare_to_baseline,
}


def dispatch_tool(name: str, input_dict: dict):
    """Convert API dict input to typed Python args and call the tool function."""
    if name not in _TOOL_MAP:
        raise KeyError(f"Unknown tool: {name!r}. Available: {list(_TOOL_MAP)}")
    kwargs = dict(input_dict)
    if "region" in kwargs:
        kwargs["region"] = BoundingBox(**kwargs["region"])
    return _TOOL_MAP[name](**kwargs)


def run_agent(
    user_request: str,
    client: Optional[anthropic.Anthropic] = None,
    temperature: float = 0,
    max_iterations: int = 20,
    verbose: bool = True,
) -> str:
    """Run the agent loop. Returns the final text response."""
    if client is None:
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    messages = [{"role": "user", "content": user_request}]
    final_text = ""

    for _ in range(max_iterations):
        response = client.messages.create(
            model=MODEL,
            system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            cache_control={"type": "ephemeral"},
            messages=messages,
            temperature=temperature,
            max_tokens=4096,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    final_text += block.text
            if verbose:
                print("\n" + "=" * 60)
                print("FINAL REPORT")
                print("=" * 60)
                print(final_text)
            return final_text

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if hasattr(block, "type") and block.type == "tool_use":
                    if verbose:
                        print(f"\n[TOOL CALL] {block.name}")
                        print(f"[INPUT] {json.dumps(block.input, indent=2)}")
                    result = dispatch_tool(block.name, block.input)
                    if verbose:
                        print(f"[RESULT] {result.summary}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result.summary,
                    })
            messages.append({"role": "user", "content": tool_results})

    if verbose:
        print(f"\nWarning: Hit max_iterations ({max_iterations}) without end_turn.")
    return final_text


if __name__ == "__main__":
    import argparse
    from agent.scene import load_scene

    parser = argparse.ArgumentParser(description="Run the crop health agent")
    parser.add_argument("--scene", choices=["scene_a", "scene_b"], default="scene_a")
    parser.add_argument("--mock", action="store_true", help="Use mock mode (no API calls)")
    parser.add_argument("--temperature", type=float, default=0)
    args = parser.parse_args()

    load_scene(args.scene)

    if args.mock:
        from agent.mock import run_mock_agent
        run_mock_agent(args.scene)
    else:
        run_agent(
            "Assess crop health for the entire scene. "
            "Identify any areas of concern and determine likely causes.",
            temperature=args.temperature,
        )
