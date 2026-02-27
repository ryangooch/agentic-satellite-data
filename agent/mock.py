# agent/mock.py
"""Mock mode: replay pre-recorded agent traces without calling the Anthropic API.

Real tool calls ARE executed (results are live), but model responses are canned.
Useful for: offline demos, testing, cost control, reproducible lecture demonstrations.

Pre-recorded responses: data/mock_responses/<scene_id>.json
"""
import json
from pathlib import Path
from agent.loop import dispatch_tool

_MOCK_DIR = Path("data/mock_responses")


def run_mock_agent(scene_id: str) -> str:
    """Replay a pre-recorded agent trace, executing real tool calls."""
    mock_path = _MOCK_DIR / f"{scene_id}.json"
    if not mock_path.exists():
        raise FileNotFoundError(
            f"No mock responses found for {scene_id!r}. Expected: {mock_path}"
        )

    data = json.loads(mock_path.read_text())
    print(f"\n[MOCK MODE] Replaying pre-recorded trace for {scene_id}")
    print("=" * 60)

    final_text = ""
    for turn in data["turns"]:
        for block in turn["content"]:
            if block["type"] == "text":
                print(f"\n[AGENT REASONING]\n{block['text']}")
            elif block["type"] == "tool_use":
                print(f"\n[TOOL CALL] {block['name']}")
                print(f"[INPUT] {json.dumps(block['input'], indent=2)}")
                result = dispatch_tool(block["name"], block["input"])
                print(f"[RESULT] {result.summary}")

        if turn["stop_reason"] == "end_turn":
            for block in turn["content"]:
                if block["type"] == "text":
                    final_text = block["text"]
            print("\n" + "=" * 60)
            print("FINAL REPORT")
            print("=" * 60)
            print(final_text)

    return final_text
