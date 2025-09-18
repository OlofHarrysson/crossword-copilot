SHELL := /bin/bash

install_dependencies:
	uv sync
	uv run pre-commit install --hook-type pre-push --hook-type post-checkout --hook-type pre-commit

run_precommit:
	uv run pre-commit run --all-files

start_api:
	uv run python src/mycode/api/start_server.py
