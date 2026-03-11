.PHONY: setup generate-data test demo download-data

setup:
	uv sync

generate-data:
	uv run python data/generate_scenes.py

test: generate-data
	uv run pytest tests/ -v

demo: generate-data
	uv run python -m agent.loop --scene scene_a --mock

download-data:
	uv run python data/download_data.py --region "37,-85.5,38,-84.5" --date "2023-07-15"
