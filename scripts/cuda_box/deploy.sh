#!/usr/bin/env bash
# Runs on the LOCAL machine. Syncs the repo to a remote Linux NVIDIA GPU host
# (Ubuntu 24.04, driver installed, reachable over SSH) and (re)runs setup.sh.
# Usage: scripts/cuda_box/deploy.sh [host]
#        SLANGIMAGE_CUDA_HOST=gpu-host scripts/cuda_box/deploy.sh
set -euo pipefail

HOST="${1:-${SLANGIMAGE_CUDA_HOST:-}}"
if [[ -z "$HOST" ]]; then
    echo "Set SLANGIMAGE_CUDA_HOST or pass an SSH host as the first argument." >&2
    exit 2
fi
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

# Never sync anything git ignores (local notes, outputs, environments): honor
# .gitignore files and, in a clone, the repository's local exclude list.
FILTERS=(--filter=':- .gitignore')
if [[ -f "$REPO_ROOT/.git/info/exclude" ]]; then
    FILTERS+=(--exclude-from="$REPO_ROOT/.git/info/exclude")
fi

echo "== rsync -> $HOST:~/slangimage =="
rsync -az --delete "${FILTERS[@]}" \
    --exclude '.git' \
    --exclude '.env' \
    --exclude '__pycache__' \
    --exclude '*.egg-info' \
    --exclude '.pytest_cache' \
    --exclude 'test_images' \
    --exclude 'examples/out' \
    --exclude 'dist' \
    --exclude '.venv' \
    --exclude '.claude' \
    --exclude 'CLAUDE.local.md' \
    --exclude '.DS_Store' \
    "$REPO_ROOT/" "$HOST:slangimage/"

echo "== remote setup =="
ssh "$HOST" 'bash slangimage/scripts/cuda_box/setup.sh'

echo "== deploy done. run tests with: scripts/cuda_box/run_tests.sh =="
