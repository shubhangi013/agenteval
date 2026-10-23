#!/usr/bin/env bash
set -euo pipefail

cleanup() {
  docker compose down -v --remove-orphans
}
trap cleanup EXIT

uv sync --extra dev
docker compose up -d --build --scale worker=1
uv run python scripts/benchmark.py
