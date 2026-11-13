.PHONY: check
check:
	uv run ruff format --check .
	uv run ruff check .
	uv run mypy worm
	uv run pytest -q
