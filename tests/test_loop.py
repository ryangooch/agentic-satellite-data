# tests/test_loop.py
import pytest
from unittest.mock import MagicMock, patch
from agent.loop import run_agent, dispatch_tool
from agent.scene import load_scene
from agent.types import BoundingBox


@pytest.fixture(autouse=True)
def load_a():
    load_scene("scene_a")


def test_dispatch_tool_compute_ndvi():
    result = dispatch_tool(
        "compute_ndvi",
        {"region": {"row_min": 0, "row_max": 50, "col_min": 0, "col_max": 50}},
    )
    assert result.success is True


def test_dispatch_tool_unknown_raises():
    with pytest.raises(KeyError):
        dispatch_tool("nonexistent_tool", {})


def test_run_agent_end_turn_immediately(mock_client_factory):
    mock_client = mock_client_factory([
        {"stop_reason": "end_turn",
         "content": [{"type": "text", "text": "No issues found."}]},
    ])
    result = run_agent("Analyze.", client=mock_client, verbose=False)
    assert mock_client.messages.create.call_count == 1


def test_run_agent_tool_call_then_end_turn(mock_client_factory):
    mock_client = mock_client_factory([
        {
            "stop_reason": "tool_use",
            "content": [{
                "type": "tool_use", "id": "id1", "name": "compute_ndvi",
                "input": {"region": {"row_min": 0, "row_max": 50, "col_min": 0, "col_max": 50}},
            }],
        },
        {"stop_reason": "end_turn",
         "content": [{"type": "text", "text": "Stress detected in NW."}]},
    ])
    run_agent("Analyze.", client=mock_client, verbose=False)
    assert mock_client.messages.create.call_count == 2


def test_run_agent_respects_max_iterations(mock_client_factory):
    # Always returns tool_use — loop should stop at max_iterations
    mock_client = mock_client_factory([{
        "stop_reason": "tool_use",
        "content": [{
            "type": "tool_use", "id": "id1", "name": "compute_ndvi",
            "input": {"region": {"row_min": 0, "row_max": 50, "col_min": 0, "col_max": 50}},
        }],
    }] * 10)
    run_agent("Analyze.", client=mock_client, max_iterations=3, verbose=False)
    assert mock_client.messages.create.call_count == 3


@pytest.fixture
def mock_client_factory():
    """Factory that builds a mock Anthropic client from a list of response dicts."""
    def _make(responses):
        client = MagicMock()
        mock_responses = []
        for resp in responses:
            mr = MagicMock()
            mr.stop_reason = resp["stop_reason"]
            content_blocks = []
            for block in resp["content"]:
                mb = MagicMock()
                mb.type = block["type"]
                if block["type"] == "text":
                    mb.text = block["text"]
                elif block["type"] == "tool_use":
                    mb.id = block["id"]
                    mb.name = block["name"]
                    mb.input = block["input"]
                content_blocks.append(mb)
            mr.content = content_blocks
            mock_responses.append(mr)
        client.messages.create.side_effect = mock_responses
        return client
    return _make
