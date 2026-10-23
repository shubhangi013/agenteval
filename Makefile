.PHONY: install test typecheck lint bench up down

install:
	uv sync --extra dev

test:
	uv run pytest

typecheck:
	uv run mypy --strict src/agenteval

lint:
	uv run ruff check src tests scripts

up:
	docker compose up --build --scale worker=4

down:
	docker compose down -v

bench:
	bash scripts/bench.sh
