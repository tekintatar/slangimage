#!/usr/bin/env bash
# Runs on the LOCAL machine. Executes the standard benchmark suite on the
# remote GPU host (prepared by deploy.sh) for CUDA + Vulkan, copies the result
# JSONs back into benchmarks/, and regenerates docs/BENCHMARKS.md with a local
# interpreter that has slangimage installed (PYTHON, default: python3 on PATH,
# e.g. your activated virtual environment).
# Usage: SLANGIMAGE_CUDA_HOST=gpu-host scripts/cuda_box/run_bench.sh [backend ...]
#        Backends default to: cuda vulkan
set -euo pipefail

HOST="${SLANGIMAGE_CUDA_HOST:-}"
if [[ -z "$HOST" ]]; then
    echo "Set SLANGIMAGE_CUDA_HOST to the SSH host for the GPU box." >&2
    exit 2
fi
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BACKENDS=("$@")
[ ${#BACKENDS[@]} -eq 0 ] && BACKENDS=(cuda vulkan)

for backend in "${BACKENDS[@]}"; do
    case "$backend" in
        auto|cuda|vulkan|metal) ;;
        *) echo "Unsupported backend: $backend" >&2; exit 2 ;;
    esac
done

for backend in "${BACKENDS[@]}"; do
    echo "==================== bench: $backend ===================="
    # The backend is validated above and intentionally expanded on the client.
    # shellcheck disable=SC2029
    ssh "$HOST" "cd slangimage && CUDA_PATH=~/cuda-shim LD_LIBRARY_PATH=~/cuda-shim/lib64 \
        ~/venvs/slangimage/bin/python -m slangimage.bench run --backend $backend --out benchmarks/results_$backend.json"
    scp -q "$HOST:slangimage/benchmarks/results_$backend.json" "$REPO_ROOT/benchmarks/"
done

echo "== regenerating docs/BENCHMARKS.md =="
"${PYTHON:-python3}" -m slangimage.bench report "$REPO_ROOT"/benchmarks/results_*.json -o "$REPO_ROOT/docs/BENCHMARKS.md"
