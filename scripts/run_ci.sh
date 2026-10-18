#!/usr/bin/env bash
set -euo pipefail

# Local CI test runner for NLPie
# Runs the exact same code quality, linting, formatting, and test checks as GitHub Actions.

echo "=========================================="
echo " Running NLPie Local CI Suite"
echo "=========================================="

# Ensure cargo is in PATH
if [ -f "$HOME/.cargo/env" ]; then
    # shellcheck source=/dev/null
    source "$HOME/.cargo/env"
fi

echo ""
echo "--- [1/8] Ruff format check ---"
uv run ruff format --check python/ tests/

echo ""
echo "--- [2/8] Ruff lint check ---"
uv run ruff check python/ tests/

echo ""
echo "--- [3/8] Mypy type check ---"
uv run mypy python/nlpie tests/

echo ""
echo "--- [4/8] Pylint standards check ---"
uv run pylint python/nlpie

echo ""
echo "--- [5/8] Rust fmt check ---"
cargo fmt --check

echo ""
echo "--- [6/8] Rust clippy check ---"
cargo clippy --all-targets --features extension-module -- -D warnings

echo ""
echo "--- [7/8] Cargo audit (security) ---"
if command -v cargo-audit &> /dev/null; then
    cargo audit
else
    echo "cargo-audit not installed; skipping audit (install via 'cargo install cargo-audit')"
fi

echo ""
echo "--- [8/8] Tests (Rust + Python) ---"
cargo test
uv run pytest tests/ -v

echo ""
echo "=========================================="
echo " ✅ All CI checks passed successfully!"
echo "=========================================="
