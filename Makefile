.PHONY: setup test run

setup:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

test:
	. .venv/bin/activate && pytest -q

run:
	. .venv/bin/activate && python -m src.laliga_pipeline.cli --data_dir ./data --out_dir ./artifacts
