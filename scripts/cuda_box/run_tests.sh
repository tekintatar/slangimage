#!/usr/bin/env bash
# Runs on the LOCAL machine. Executes the test suite on the remote GPU host
# (prepared by deploy.sh) once per requested backend.
# Usage: SLANGIMAGE_CUDA_HOST=gpu-host scripts/cuda_box/run_tests.sh [backend ...]
#        Backends default to: cuda vulkan
set -euo pipefail

HOST="${SLANGIMAGE_CUDA_HOST:-}"
if [[ -z "$HOST" ]]; then
    echo "Set SLANGIMAGE_CUDA_HOST to the SSH host for the GPU box." >&2
    exit 2
fi
BACKENDS=("${@:-cuda vulkan}")
[ $# -eq 0 ] && BACKENDS=(cuda vulkan)

for backend in "${BACKENDS[@]}"; do
    case "$backend" in
        auto|cuda|vulkan|metal) ;;
        *) echo "Unsupported backend: $backend" >&2; exit 2 ;;
    esac
done

# Run every backend, record each status, exit nonzero if ANY failed — so this
# can serve as a release gate (previously every run swallowed failures with
# `|| true` and always reported success).
fail=0
for backend in "${BACKENDS[@]}"; do
    echo "==================== requested texture backend: $backend ===================="
    echo "Results separate selected-backend texture + host, AUTO-device, and CUDA autograd tests."
    # The backend is validated above and intentionally expanded on the client.
    # shellcheck disable=SC2029
    if ! ssh "$HOST" "cd slangimage && CUDA_PATH=~/cuda-shim LD_LIBRARY_PATH=~/cuda-shim/lib64 SLANGIMAGE_BACKEND=$backend ~/venvs/slangimage/bin/pytest tests/ -q -ra"; then
        echo "!! backend $backend FAILED"
        fail=1
    fi
done
[ "$fail" -eq 0 ] && echo "== all backends passed ==" || echo "== one or more backends FAILED =="
exit "$fail"
