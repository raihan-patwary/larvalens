# Run "make help" for the list of targets. Uses uv for the environment.
.PHONY: help install lint format test data figures support

help:
	@echo "install  set up the environment"
	@echo "lint     ruff check and format check"
	@echo "format   ruff format"
	@echo "test     run pytest"
	@echo "data     download, verify and clean the small Zenodo files"
	@echo "figures  rebuild every figure from saved results"
	@echo "support  update the private Kaggle dataset larvalens-support"

install:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

test:
	uv run pytest -q

data:
	uv run python -m larvalens data

figures:
	uv run python -m larvalens figures

support:
	uv run python scripts/build_support.py
	kaggle datasets version -p build/support -m "update" -r zip
