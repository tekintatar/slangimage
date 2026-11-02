#!/usr/bin/env bash
# One command for "is this change ready?": whitespace, lint, types and tests.
# Usage: scripts/check.sh [--no-gpu] [extra pytest arguments...]
#   --no-gpu            skip tests that need a GPU (what GitHub CI runs)
#   SLANGIMAGE_BACKEND  metal | cuda | vulkan selects the test backend (default: auto)
#   PYTHON              interpreter with the dev extra installed (default: python3)
set -euo pipefail

cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
MARK=()
if [[ "${1:-}" == "--no-gpu" ]]; then
    MARK=(-m "not gpu")
    shift
fi

if git rev-parse --git-dir > /dev/null 2>&1; then
    echo "== whitespace (git diff --check)"
    git diff --check HEAD
fi
echo "== ruff"
"$PY" -m ruff check src/ tests/ scripts/ examples/ benchmarks/ playground/
echo "== mypy"
"$PY" -m mypy
echo "== pytest"
"$PY" -m pytest -q ${MARK[@]+"${MARK[@]}"} "$@"
echo "== all checks passed"
