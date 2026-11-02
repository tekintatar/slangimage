#!/usr/bin/env bash
# Runs ON the remote GPU host (invoked by deploy.sh). Idempotent — safe to
# re-run after every deploy.
# Assumes: Ubuntu 24.04, NVIDIA driver already installed, repo synced to ~/slangimage.
#
# CUDA version policy: everything on the SAME major as torch's build
# (torch 2.13 = cu130): NVRTC/cudart 13 from pip wheels for the slang
# runtime, nvcc 13.0 (compiler package only, pinned) from the NVIDIA apt
# repo for building the slangpy-torch extension.
set -euo pipefail

REPO=~/slangimage
VENV=~/venvs/slangimage
CUDA_HOME_13=/usr/local/cuda-13.0

echo "== apt packages =="
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    python3-venv python3-pip \
    libvulkan1 vulkan-tools

echo "== nvcc 13.0 (NVIDIA repo, compiler + cudart-dev only) =="
if [ ! -x "$CUDA_HOME_13/bin/nvcc" ]; then
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb
    sudo dpkg -i /tmp/cuda-keyring.deb > /dev/null
    sudo apt-get update -qq
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cuda-nvcc-13-0 cuda-cudart-dev-13-0
fi
"$CUDA_HOME_13/bin/nvcc" --version | tail -1

echo "== build and inspect release artifacts =="
BUILD_ROOT=$(mktemp -d)
trap 'rm -rf "$BUILD_ROOT"' EXIT
python3 -m venv "$BUILD_ROOT/venv"
"$BUILD_ROOT/venv/bin/pip" install -q --upgrade pip build==1.4.0 twine==7.0.0
rm -rf "$REPO/dist"
PIP_CONSTRAINT="$REPO/build-constraints.txt" \
    "$BUILD_ROOT/venv/bin/python" -m build "$REPO"
"$BUILD_ROOT/venv/bin/twine" check "$REPO"/dist/*
"$BUILD_ROOT/venv/bin/python" "$REPO/scripts/check_artifacts.py" \
    "$REPO/dist" --source-root "$REPO"
WHEEL=$(realpath "$REPO"/dist/*.whl)

echo "== clean artifact environment =="
rm -rf "$VENV"
python3 -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
# SlangPy 0.43.0/0.43.1 bundle Slang 2026.12, whose generated backward hangs
# when runtime-dependent control flow encloses a differentiable tensor load
# (slangpy#1167; docs/UPSTREAM_ISSUES.md). Keep this host on the validated
# 0.42.0 stack until a released wheel passes that reproducer.
"$VENV/bin/pip" install -q slangpy==0.42.0 "${WHEEL}[dev,bench,playground]"

echo "== torch 2.13 (cu130) + matching torchvision 0.28 =="
# Pin the validated stack. An unbounded install pulled torch 2.14 as soon as
# it was published, ahead of the validated slangpy/autodiff compatibility
# matrix.
"$VENV/bin/pip" install -q \
    torch==2.13.0 \
    torchvision==0.28.0 \
    kornia \
    nvidia-ml-py

echo "== CUDA shim (NVRTC 13 + headers from torch's bundled cu13 libs) =="
# slang-rhi needs libnvrtc + cuda_runtime.h at runtime; located via CUDA_PATH
# (device layer) and the dynamic loader (slang pass-through). torch cu130
# ships these under site-packages/nvidia/cu13 (standalone cu13 wheels do
# not exist as of 2026-07). Env for CUDA runs (exported by run_tests.sh /
# run_bench.sh):
#   CUDA_PATH=~/cuda-shim  LD_LIBRARY_PATH=~/cuda-shim/lib64
SP=$(ls -d "$VENV"/lib/python3*/site-packages)
rm -rf ~/cuda-shim && mkdir -p ~/cuda-shim/lib64
ln -sfn "$SP/nvidia/cu13/include" ~/cuda-shim/include
ln -sf "$SP"/nvidia/cu13/lib/libnvrtc* ~/cuda-shim/lib64/
ln -sf "$SP"/nvidia/cu13/lib/libcudart* ~/cuda-shim/lib64/
ln -sf "$SP/nvidia/cu13/lib/libnvrtc.so.13" ~/cuda-shim/lib64/libnvrtc.so

echo "== slangpy-torch extension (differentiable ops) =="
CUDA_HOME="$CUDA_HOME_13" "$VENV/bin/pip" install -q slangpy-torch --no-build-isolation
# Resolve the wheel's documented torch extra after the bridge is built with
# the pinned CUDA toolchain; this confirms the artifact metadata itself.
CUDA_HOME="$CUDA_HOME_13" "$VENV/bin/pip" install -q "${WHEEL}[torch]" --no-build-isolation
"$VENV/bin/pip" check

echo "== environment report =="
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
"$VENV/bin/python" -c "import slangpy, torch; print('slangpy', slangpy.__version__, '| torch', torch.__version__, '| torch cuda', torch.version.cuda)"
"$VENV/bin/python" -c "import torch, slangpy_torch; print('slangpy-torch: OK')"
vulkaninfo --summary 2>/dev/null | grep -m1 deviceName || echo "vulkan: no ICD visible"
ls ~/cuda-shim/lib64/libnvrtc.so ~/cuda-shim/include/cuda_runtime.h > /dev/null && echo "cuda shim: OK (cu13)"

echo "== installed artifact smoke (outside checkout) =="
(
    cd /tmp
    CUDA_PATH=~/cuda-shim \
    LD_LIBRARY_PATH=~/cuda-shim/lib64 \
    "$VENV/bin/python" "$REPO/scripts/validate_installed_artifact.py" \
        --source-root "$REPO" --gpu-smoke --backend cuda
)

echo "== setup done =="
