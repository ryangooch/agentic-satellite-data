.PHONY: setup generate-data test demo fetch-data

setup:
	uv sync

generate-data:
	uv run python data/generate_scenes.py

fetch-data:
	uv run python data/fetch_sentinel2.py

test: generate-data
	uv run pytest tests/ -v

demo: generate-data
	uv run python -m agent.loop --scene scene_a --mock
