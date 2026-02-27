# tests/conftest.py
import subprocess
from pathlib import Path
import pytest


@pytest.fixture(scope="session", autouse=True)
def ensure_scenes_exist():
    """Generate synthetic scenes if they don't exist yet."""
    if not Path("data/scenes/scene_a_bands.npy").exists():
        subprocess.run(["python", "data/generate_scenes.py"], check=True)
