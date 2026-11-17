.PHONY: check docker-check
check:
	uv run ruff format --check .
	uv run ruff check .
	uv run mypy worm
	uv run pytest -q

# Run the same check in the CPU-only Linux image.
docker-check:
	docker build -t navigating-worm .
	docker run --rm navigating-worm
