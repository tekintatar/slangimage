# Installing the PyTorch extra

`slangimage.diff` and `slangimage.nn` need the `torch` extra, which is
supported on Linux with an NVIDIA GPU and driver 580 or newer (`nvidia-smi`
shows "CUDA Version: 13.0"). The README's
[PyTorch extra](../README.md#pytorch-extra-slangimagetorch) section has the
standard three-step install; this page explains why it takes three steps and
covers an install without sudo and common failures.

Both recipes below were run verbatim in fresh virtual environments on Ubuntu
24.04 with an NVIDIA L40S (driver 580.126.16), Python 3.10, 3.12 and 3.13,
PyTorch 2.13.0+cu130, SlangPy 0.42.0 and slangpy-torch 0.7.0; the CUDA autograd
tests passed with each.

## Why `pip install "slangimage[torch]"` alone cannot work

1. The extra pins `slangpy[torch]==0.42.0`, which requires
   `slangpy-torch==0.7.0`. That package is published as source only, and its
   build imports torch to compile a PyTorch C++ extension. pip's default build
   isolation hides the environment's torch, so the install stops with
   `ERROR: PyTorch not found!` and `metadata-generation-failed` — even when
   torch 2.13 is already installed. `uv pip install` fails the same way.
2. With `--no-build-isolation`, torch still has to be installed *before* the
   bridge is built, so torch must come first, in a separate command.
3. The bridge has no `.cu` sources, so **nvcc is not needed**. The build needs
   a C++ compiler, the Python headers (`Python.h`), the CUDA runtime headers
   **including `crt/`** (`cuda_runtime_api.h` includes `crt/host_defines.h`)
   and a linkable `libcudart.so`, found through `CUDA_HOME`, `CUDA_PATH` or
   `/usr/local/cuda`.
4. At run time, SlangPy compiles the kernels with NVRTC, as for the texture
   operators, so NVRTC must be installed and no newer than the driver's CUDA
   version. PyTorch 2.13 from PyPI is the CUDA 13.0 build.

## With sudo (Ubuntu 24.04, NVIDIA's CUDA apt repository)

No environment variables are needed.

```bash
# Once per machine; skip the first two lines if `apt-cache policy cuda-crt-13-0`
# already shows a candidate.
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt-get update
sudo apt-get install -y --no-install-recommends build-essential python3-dev python3-venv \
    cuda-nvrtc-dev-13-0 cuda-cudart-dev-13-0 cuda-crt-13-0
sudo ldconfig   # refresh the shared-library cache so NVRTC is found

# Per project:
python3.12 -m venv .venv && . .venv/bin/activate
pip install "torch>=2.13,<2.14"                                          # 1. torch first
pip install --no-build-isolation --no-cache-dir "slangpy-torch==0.7.0"   # 2. build the bridge
pip install "slangimage[torch]"                                          # 3. nothing left to build
python -c "import slangimage.diff as sid; print(sid.available())"        # True
```

`cuda-crt-13-0` is an 83 kB header package; without it the bridge build stops
at `fatal error: crt/host_defines.h: No such file or directory`. Building the
bridge takes about 40 seconds; the step 1 download is about 2.7 GB and a
finished environment about 5 GB.

## Without sudo (CUDA headers from pip wheels)

You still need a C++ compiler and the Python headers (with Ubuntu's system
Python, `build-essential` and `python3-dev`; uv-managed and python.org
interpreters ship their own headers).

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install "torch>=2.13,<2.14" "cuda-toolkit[crt]==13.0.*"   # crt/ headers are not in torch's CUDA wheels
CU=$(python -c 'import os, nvidia; print(os.path.join(list(nvidia.__path__)[0], "cu13"))')
ln -sf libcudart.so.13 "$CU/lib/libcudart.so"   # link name the bridge build needs (-lcudart)
ln -sf libnvrtc.so.13 "$CU/lib/libnvrtc.so"     # file name SlangPy's CUDA backend looks for
export CUDA_PATH="$CU" LD_LIBRARY_PATH="$CU/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
pip install --no-build-isolation --no-cache-dir "slangpy-torch==0.7.0"
pip install "slangimage[torch]"
python -c "import slangimage.diff as sid; print(sid.available())"   # True
```

Keep `CUDA_PATH` and `LD_LIBRARY_PATH` exported in every shell that uses
SlangImage, for example by appending the `export` line to `.venv/bin/activate`.
`CUDA_PATH` serves both the bridge build and SlangPy's run-time compiler.
Without `libnvrtc.so` and `LD_LIBRARY_PATH`, `import` still works, but
`sid.available()` is `False` and every `slangimage.diff` operation fails to
create its CUDA device.

## Common failures

| Situation | What happens | Do this |
|---|---|---|
| `pip install "slangimage[torch]"` in a fresh environment | `ERROR: PyTorch not found!`, then `metadata-generation-failed`; nothing is installed | Use one of the recipes above. |
| torch installed, then `pip install --no-build-isolation "slangimage[torch]"` without CUDA headers | `OSError: CUDA_HOME environment variable is not set` | Add the CUDA packages or wheels from a recipe above. |
| CUDA headers present but no `crt/` | `fatal error: crt/host_defines.h: No such file or directory` | Install `cuda-crt-13-0` or `cuda-toolkit[crt]==13.0.*`. |
| pip CUDA wheels without the `libcudart.so` link | `/usr/bin/ld: cannot find -lcudart` | Create the `ln -sf libcudart.so.13 ...` link. |
| No Python headers | `fatal error: Python.h: No such file or directory` | `sudo apt-get install python3-dev` |
| torch 2.14 already in the environment | pip silently replaces torch 2.14 with 2.13 (and its CUDA libraries); with torchvision 0.29 present it prints a dependency error, exits 0, and `pip check` fails afterwards | Use a fresh environment; torchvision 0.28 pairs with torch 2.13. |
| A second environment, or a torch change, without `--no-cache-dir` | pip reuses its cached bridge wheel, built against whatever torch the first build saw | Build the bridge with `--no-cache-dir` (or run `pip cache remove slangpy_torch`). After any torch change: `pip install --no-build-isolation --no-cache-dir --force-reinstall --no-deps "slangpy-torch==0.7.0"`. |
| Editable install: `pip install -e ".[dev,torch]"` | The same bridge error; adding `--no-build-isolation` fails with `No module named 'hatchling'` | Run steps 1 and 2 first, then `pip install -e ".[dev,torch]"` without the flag. |
| NVIDIA driver older than 580 (CUDA 12.x) | PyPI's torch 2.13 is the CUDA 13.0 build | Untested. PyTorch publishes CUDA 12.x builds of 2.13 on download.pytorch.org; they would need matching CUDA 12.x headers and NVRTC. |

## Checking the installation

- `python -c "import torch, slangpy_torch; print(torch.__version__, slangpy_torch.get_api_ptr())"`
  prints `2.13.0+cu130` and a non-zero integer.
- `python -c "import slangimage as si; print(si.device().info.api_name)"`
  prints `CUDA`.
- `pip check` reports no broken requirements.
- From a source checkout, `SLANGIMAGE_BACKEND=cuda pytest tests/test_diff.py tests/test_diff_core_contracts.py tests/test_nn.py -ra`
  runs the CUDA autograd tests.
