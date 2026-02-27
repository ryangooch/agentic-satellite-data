# tests/test_mock.py
import json
import pytest
from pathlib import Path
from agent.scene import load_scene


@pytest.fixture(autouse=True)
def load_a():
    load_scene("scene_a")


def test_mock_responses_file_exists():
    assert Path("data/mock_responses/scene_a.json").exists()


def test_mock_responses_structure():
    data = json.loads(Path("data/mock_responses/scene_a.json").read_text())
    assert "turns" in data
    assert len(data["turns"]) >= 4
    for turn in data["turns"]:
        assert "stop_reason" in turn
        assert "content" in turn
        assert turn["stop_reason"] in ("tool_use", "end_turn")


def test_run_mock_agent_completes(capsys):
    from agent.mock import run_mock_agent
    run_mock_agent("scene_a")
    captured = capsys.readouterr()
    assert "FINAL REPORT" in captured.out


def test_run_mock_agent_missing_scene_raises():
    from agent.mock import run_mock_agent
    with pytest.raises(FileNotFoundError):
        run_mock_agent("nonexistent_scene")
