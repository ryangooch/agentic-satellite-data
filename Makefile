.PHONY: setup generate-data test demo download-data

setup:
	pip install -r requirements.txt

generate-data:
	python data/generate_scenes.py

test: generate-data
	pytest tests/ -v

demo: generate-data
	python -m agent.loop --scene scene_a --mock

download-data:
	python data/download_data.py --region "37.5,-120.5,37.7,-120.3" --date "2023-07-15"
