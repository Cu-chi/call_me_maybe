SRC_DIR = src
MYPY_FLAGS = --warn-return-any --warn-unused-ignores --ignore-missing-imports \
--disallow-untyped-defs --check-untyped-defs
VENV = .venv
LIB_LLM_SDK = llm_sdk

install:
	uv sync

run:
	uv run python -m $(SRC_DIR) $(ARGS)

debug:
	uv run python -m pdb -m $(SRC_DIR) $(ARGS)

clean:
	@rm -rf $$(find . -type d -name "__pycache__") $$(find . -type d -name ".mypy_cache")

lint:
	uv run python -m flake8 . --extend-exclude $(VENV),$(LIB_LLM_SDK)
	uv run python -m mypy . $(MYPY_FLAGS)

lint-strict:
	uv run python -m flake8 . --extend-exclude $(VENV),$(LIB_LLM_SDK)
	uv run python -m mypy . --strict

test:
	uv run python src/test.py

.PHONY: install run debug clean lint lint-strict test
