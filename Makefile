.PHONY: check docker-check
check:
	uv run --locked ruff format --check .
	uv run --locked ruff check .
	uv run --locked mypy worm
	uv run --locked pytest -q

# Run the same check in the CPU-only Linux image.
docker-check:
	docker build -t navigating-worm .
	docker run --rm navigating-worm
