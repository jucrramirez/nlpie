.PHONY: all ci lint format test test-python test-rust audit

all: ci

# Run complete CI suite locally
ci:
	./scripts/run_ci.sh

# Code quality and linting (Python + Rust)
lint:
	uv run ruff check python/ tests/
	uv run ruff format --check python/ tests/
	uv run mypy python/nlpie tests/
	uv run pylint python/nlpie
	cargo fmt --check
	cargo clippy --all-targets --features extension-module -- -D warnings

# Auto-format Python and Rust code
format:
	uv run ruff format python/ tests/
	uv run ruff check --fix python/ tests/
	cargo fmt

# Run all tests
test: test-rust test-python

test-python:
	uv run pytest tests/ -v

test-rust:
	cargo test

# Security audit
audit:
	uv run --with pip-audit pip-audit
	cargo audit
