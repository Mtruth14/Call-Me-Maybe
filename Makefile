.PHONY: install run debug clean lint lint-strict

FUNCTIONS_DEFINITION ?= data/input/functions_definition.json
INPUT ?= data/input/function_calling_tests.json
OUTPUT ?= data/output/function_calling_results.json

install:
	uv sync

run:
	uv run python -m src \
		--functions_definition $(FUNCTIONS_DEFINITION) \
		--input $(INPUT) \
		--output $(OUTPUT)

debug:
	uv run python -m pdb -m src \
		--functions_definition $(FUNCTIONS_DEFINITION) \
		--input $(INPUT) \
		--output $(OUTPUT)

clean:
	find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} +
	rm -rf .mypy_cache .pytest_cache data/output

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports \
		--disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict
