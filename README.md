# SlangImage

[![PyPI](https://img.shields.io/pypi/v/slangimage)](https://pypi.org/project/slangimage/)
[![Python](https://img.shields.io/pypi/pyversions/slangimage)](https://pypi.org/project/slangimage/)
[![checks](https://github.com/tekintatar/slangimage/actions/workflows/checks.yml/badge.svg)](https://github.com/tekintatar/slangimage/actions/workflows/checks.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/tekintatar/slangimage/blob/main/LICENSE)

**Fast GPU image processing for Python, written as plain
[Slang](https://shader-slang.org/) shaders.**

SlangImage is a library of 44 image operators, each one fused GPU kernel,
that runs the same code on Apple, NVIDIA and other Vulkan GPUs. Every
operator is a small Slang function called through
[SlangPy](https://github.com/shader-slang/slangpy), so a new one is one more
function, not a CUDA extension.

- **Image processing on any GPU.** Blur and denoise (bilateral, non-local
  means), sharpening, colour (CLAHE, white balance, LUTs, HSV), geometry,
  compositing and stylization, in fp16 or fp32. The same code runs on Metal
  (Apple Silicon), CUDA and Vulkan: prototype on a Mac, run on NVIDIA.
- **Fast where filters are heavy.** The bilateral filter is 51–147× and
  CLAHE 5–25× faster than kornia's, and a 19×19 Gaussian blur 3–9× faster
  than torchvision's.
- **Your own ops in one file.** A custom operator is one `.slang` file that
  your package registers and calls like a built-in.
- **Differentiable, too.** Nine of the operators also have fused backward
  kernels for `torch.autograd` (`slangimage.diff`): the same shader code
  serves training as well as processing.

For GPU training augmentation (torchvision's transforms with per-image
randomness, fused into a few CUDA kernels), see the sibling package
[SlangVision](https://github.com/tekintatar/slangvision).

**Status: pre-alpha, under active development.** APIs may still change.

## Quick start

```bash
pip install slangimage
python -m slangimage      # which GPU backends work here, and why not
```

```python
import slangimage as si

img = si.load("photo.jpg")                            # fp16 on the GPU
out = si.ops.surface_blur(img, radius=9, bsigma=0.05)
out = si.ops.vibrance(out, amount=0.6)
out = si.ops.clahe(out, clip_limit=2.5)
out.save("out.png")

# chaining style
out = (
    si.load("photo.jpg")
    .pipe(si.ops.kuwahara, radius=6)
    .pipe(si.ops.sharpen_luma, sharpness=0.8)
)

# numpy in / numpy out (lossless, unclipped)
import numpy as np
arr = si.from_numpy(np.random.rand(512, 512, 3).astype(np.float16))
x = si.ops.normalize(arr, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)).to_numpy()
```

The device is chosen automatically (Metal on macOS, CUDA elsewhere, Vulkan
as the fallback). Kernels compile lazily per device and precision, so the
first call of each operator is slower; services call `si.prewarm()` once at
startup. `img.to_torch()` and `si.from_torch()` convert to and from torch
tensors on any backend. More runnable scripts are in
[examples/](https://github.com/tekintatar/slangimage/tree/main/examples)
(source checkout; e.g. `python examples/01_quickstart.py`).

## Performance

Batch 1 at 1024² and 2048², on an NVIDIA L40S and an RTX PRO 4500 Blackwell
(ranges span both GPUs, fp16 and fp32 forward; training steps in fp32):

| Operation | vs kornia | vs torchvision |
|---|---|---|
| Bilateral filter, forward | 51–147× faster | no equivalent |
| Bilateral filter, training step (forward + backward) | 46–97× faster, 158–194× less GPU memory (160 MiB vs about 25–30 GiB at 2048²) | no equivalent |
| CLAHE | 5–25× faster | no equivalent |
| Hue rotation, forward | 7–15× faster | 5–28× faster |
| Gaussian blur 19×19, forward | 3–7× faster | 3–9× faster |

Tensor libraries build edge-preserving and data-dependent filters from
generic steps (unfold windows, per-pixel colour-space conversions) that
materialize many copies of the image; here each is one kernel, so the gap is
largest where the math is most complex.

Where it is not faster: tiny per-pixel operations (invert, gamma, grayscale,
autocontrast) cost about 0.05–0.7 ms per call here, mostly dispatch
overhead, and torchvision's native kernels are about 1.6–5.6× faster at that
scale; in training, PyTorch's own bilinear resize is about 2–3× faster per
step. kornia 0.9.0's CLAHE is up to about 6× faster than 0.8.3's, which
accounts for most of the CLAHE range above. Every cell, including the ones
SlangImage loses, and the raw data are linked under
[Benchmarks](https://github.com/tekintatar/slangimage#benchmarks).

## Differentiable ops (PyTorch, CUDA)

`slangimage.diff` runs nine of the operators inside `torch.autograd` with
fused forward and backward Slang kernels: `gaussian_blur`, `bilateral`,
`unsharp_mask`, `resize`, `exposure`, `sepia`, `hue_rotate`, `blend` and
`lut` (differentiable in both the image and the LUT). `slangimage.nn` wraps
them as `torch.nn.Module`s, including a learnable colour-grading LUT:

```python
import torch
import slangimage.diff as sid
import slangimage.nn as snn

img = torch.rand(512, 512, 3, device="cuda", requires_grad=True)   # one (H, W, 3) image
out = sid.bilateral(img, radius=5, sigma_range=0.15)   # edge-preserving; gradients reach img
out = sid.sepia(out, intensity=0.8)
out.mean().backward()                                  # fused backward kernels

grade = torch.nn.Sequential(snn.Bilateral(radius=5), snn.Lut()).cuda()
out = grade(img)                                       # the LUT is an nn.Parameter
```

They take one float32 `(H, W, 3)` CUDA tensor per call (fp16, bfloat16 and
`torch.autocast` inputs are computed in fp32, with gradients returned in the
input dtype) and need the compiled bridge from the
[`torch` extra](https://github.com/tekintatar/slangimage#pytorch-extra-slangimagetorch);
`sid.available()` returns `False` instead of raising when it is missing.
[examples/06_differentiable_torch.py](https://github.com/tekintatar/slangimage/blob/main/examples/06_differentiable_torch.py)
learns a colour grade as a LUT from an (original, edited) pair; the trained
LUT then works as a preset in `si.ops.lut` on any backend. Gradients are
tested against sampled fp32 finite differences with an explicit error
budget, and `resize` matches `F.interpolate(align_corners=False,
antialias=False)` to within 1e-4, forward and backward (see the
[per-op evidence](https://github.com/tekintatar/slangimage/blob/main/docs/TEST_COVERAGE.md#cuda-tensor-operators-sidiff-and-modules-sinn)).

## Operators

The texture operators (`si.ops`) support all three backends with fp16 or
fp32 storage and process one image per call. Test depth varies by operator
and precision; see the
[operator test-evidence map](https://github.com/tekintatar/slangimage/blob/main/docs/TEST_COVERAGE.md)
for references and gaps.

| Category | Operators |
|---|---|
| Blur & denoise | `gaussian_blur`, `surface_blur`, `bilateral` (full/separable, gaussian or cutoff range), `nl_means` |
| Sharpening | `unsharp_mask`, `ep_unsharp_mask`, `sharpen_luma` |
| Color | `vibrance`, `gamma`, `color_controls`, `exposure`, `sepia`, `hue_rotate`, `white_balance` (Kelvin + tint, Bradford), `false_color`, `highlight_shadow`, `clahe`, `equalize`, `autocontrast`, `solarize`, `posterize`, `grayscale`, `normalize`, `srgb_to_linear`, `linear_to_srgb`, `lut` (square 8×8-grid CLUT), `invert`, `extract_channel` |
| Compositing | `blend` (12 Photoshop/PDF modes), `clamp_multiply`, `high_pass_contrast` |
| Stylization | `kuwahara`, `pointillize`, `cmyk_halftone`, `smooth_threshold`, `smooth_threshold_bw` |
| Geometry | `resize` (Lanczos3 / bicubic), `flip`, `rot90`, `crop`, `tile` (scale/phase, periodic bilinear/nearest), `rotate` (with `expand`), `affine` (bilinear/nearest, zeros fill) |
| Generators | `random_noise` |

There is no generated API reference yet: every operator's docstring gives its
parameters, defaults and clipping behavior (`help(si.ops.clahe)`), and invalid
controls raise `ValueError` before dispatch (see the
[parameter contracts](https://github.com/tekintatar/slangimage/blob/main/docs/PARAMETER_CONTRACTS.md)).

Downstream packages can ship their own `.slang` kernels with
`ModuleCache.add_ops_dir()` and call them like the built-ins
([examples/05_custom_kernels.py](https://github.com/tekintatar/slangimage/blob/main/examples/05_custom_kernels.py)).
Higher-level effect pipelines (multi-op graphs with pooled intermediates)
live under `slangimage.fx` (`GaussianBlurEffect`, `SoftGlowEffect`,
`KuwaharaPaintEffect`), and operator implementation classes under
`slangimage.kernels`. Effects are constructed once for a device and reuse its
module cache and texture pool. Their parameters are explicit and validated;
Gaussian effects accept either `blur_amount` or the complete `radius`/`sigma`
pair:

```python
import slangimage as si

img = si.load("photo.jpg")
modules = si.ModuleCache(img.device)
glow = si.fx.SoftGlowEffect(img.device, modules)
out = glow.apply(img, radius=9, sigma=3.0, glow_strength=0.7)
```

### Backend support

| Capability | Metal | CUDA | Vulkan |
|---|:---:|:---:|:---:|
| All functional ops (`si.ops`) | ✓ | ✓ | ✓ |
| fp16 & fp32 storage | ✓ | ✓ | ✓ |
| Differentiable ops (`si.diff`, `si.nn`) | — | ✓ | — |
| Auto-selected on | macOS | Linux | fallback |

Validated on macOS 26 with an Apple M3 Max (Metal) and on Ubuntu 24.04 with an
NVIDIA L40S and an NVIDIA RTX PRO 4500 Blackwell (CUDA and Vulkan). Windows,
other GPUs and other Linux distributions have not been tested; see the
[platform notes](https://github.com/tekintatar/slangimage/blob/main/docs/PLATFORM_NOTES.md).

Differentiable ops require PyTorch and are CUDA-only (torch interop).
The normal Metal/Vulkan suites exclude CUDA autograd tests; they do not validate
gradients. (The standalone Slang autodiff experiments under `experiments/` also
ran on Metal, but they are not a supported Metal PyTorch path.)

## Installation details

Requires Python 3.10–3.13 and a Metal, CUDA, or Vulkan GPU — there is no CPU
fallback. Create the virtual environment with a supported interpreter named
explicitly: a bare `python3` can be too old (Apple's 3.9) or too new
(Homebrew's 3.14).

```bash
python3.12 -m venv .venv && . .venv/bin/activate   # or 3.10, 3.11, 3.13
pip install slangimage
python -m slangimage      # setup report: which GPU backends work, and why not
```

On Debian/Ubuntu, `sudo apt install python3-venv` first (system `pip` refuses
to install outside a virtual environment there). Minimal Linux images (Docker
base images, some cloud servers) also lack system libraries SlangPy needs:
`sudo apt install libatomic1 libx11-6` (plus `libvulkan1` for the Vulkan
backend); otherwise `import slangimage` fails with
`ImportError: libatomic.so.1` or `libX11.so.6`. Optional extras:
`slangimage[torch]` (differentiable ops — CUDA only, see
[PyTorch extra](https://github.com/tekintatar/slangimage#pytorch-extra-slangimagetorch) below), `slangimage[bench]`
(adds NVIDIA VRAM sampling to the benchmark suite; not needed on Metal) and
`slangimage[playground]` (the Gradio UI).

**Platform requirements** come from the [SlangPy](https://pypi.org/project/slangpy/)
wheels SlangImage depends on:

- **macOS:** Apple Silicon only. SlangPy 0.43 wheels need macOS 26 or later; on
  macOS 15, pip resolves SlangPy 0.42 instead. Intel Macs and macOS 14 or
  earlier cannot install SlangPy.
- **Linux:** glibc 2.34 or later (for example Ubuntu 22.04+, Debian 12+,
  RHEL 9+), x86_64 or aarch64.
- **Windows:** SlangPy publishes Windows wheels, but SlangImage has not been
  tested on Windows.

See the [backend support matrix](https://github.com/tekintatar/slangimage#backend-support) above and the
[platform notes](https://github.com/tekintatar/slangimage/blob/main/docs/PLATFORM_NOTES.md)
for tested platforms and troubleshooting.

### CUDA on Linux: install NVRTC

The NVIDIA driver alone is not enough for the CUDA backend: SlangImage compiles
CUDA kernels at run time with NVRTC, which also needs the CUDA runtime headers.
Without them, automatic selection logs `Cannot find CUDA nvrtc` and falls back
to Vulkan (which works, just not on CUDA), and an explicit `si.device("cuda")`
raises a `RuntimeError` explaining what is missing. Use an NVRTC **no newer than
the "CUDA Version" printed by `nvidia-smi`** (driver 580.x → 13.0); a newer
NVRTC fails with `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`.

System-wide (Ubuntu, from NVIDIA's CUDA apt repository — install NVIDIA's
`cuda-keyring` package first if `apt-cache policy cuda-nvrtc-dev-13-0` shows no
candidate; no environment variables needed):

```bash
sudo apt-get install -y --no-install-recommends cuda-nvrtc-dev-13-0 cuda-cudart-dev-13-0
sudo ldconfig     # refresh the shared-library cache so NVRTC is found
```

Without sudo, inside your virtual environment:

```bash
pip install "cuda-toolkit[nvrtc,cudart]==13.0.*"
CU=$(python -c 'import os, nvidia; print(os.path.join(list(nvidia.__path__)[0], "cu13"))')
ln -sf libnvrtc.so.13 "$CU/lib/libnvrtc.so"
export CUDA_PATH="$CU" LD_LIBRARY_PATH="$CU/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Both variables are required for the pip route, in every shell that runs
SlangImage: with only `CUDA_PATH` set, CUDA kernels cannot be compiled, so
automatic selection continues on Vulkan (warning if the CUDA device was created
but failed a one-pixel test kernel) and `si.device("cuda")` raises a
`RuntimeError` naming the problem. Check with
`python -c 'import slangimage as si; print(si.device().info.api_name)'`, which
prints `CUDA`. Vulkan worked without extra setup on the tested Ubuntu 24.04
host (the NVIDIA driver's Vulkan ICD plus the `libvulkan1` loader).

### PyTorch extra (`slangimage[torch]`)

The extra is for the differentiable ops (`slangimage.diff`, `slangimage.nn`).
It intentionally pins PyTorch 2.13 and SlangPy 0.42. SlangPy 0.43.0/0.43.1
bundles a compiler regression that can hang generated backward for
runtime-dependent tensor loads
([slangpy#1167](https://github.com/shader-slang/slangpy/issues/1167)); in
SlangImage this affects differentiable `gaussian_blur`, `unsharp_mask`, and
graphs containing them. Do not override the SlangPy pin when using
`slangimage.diff` or `slangimage.nn`. The texture API (`slangimage.ops`) is
unaffected and supports SlangPy 0.43.

The extra depends on SlangPy's PyTorch bridge (`slangpy-torch`), a small C++
extension published only as source and compiled against the PyTorch in your
environment. A plain `pip install "slangimage[torch]"` therefore stops with
`ERROR: PyTorch not found!`, and the extra cannot be installed on macOS. It is
supported on Linux with NVIDIA driver 580 or newer (`nvidia-smi` shows "CUDA
Version: 13.0"). Use a **fresh** virtual environment and install in three
steps (Ubuntu 24.04 with NVIDIA's CUDA apt repository, as for NVRTC above):

```bash
sudo apt-get install -y --no-install-recommends build-essential python3-dev python3-venv \
    cuda-nvrtc-dev-13-0 cuda-cudart-dev-13-0 cuda-crt-13-0
python3.12 -m venv .venv && . .venv/bin/activate
pip install "torch>=2.13,<2.14"                                          # 1. torch first
pip install --no-build-isolation --no-cache-dir "slangpy-torch==0.7.0"   # 2. build the bridge
pip install "slangimage[torch]"                                          # 3. slangimage
python -c "import slangimage.diff as sid; print(sid.available())"        # True
```

`nvcc` is not required. Installing into an environment that already has
torch 2.14 silently downgrades it to 2.13 and breaks packages pinned to 2.14
(such as torchvision 0.29). If torch changes later, rebuild the bridge by
repeating step 2 with `--force-reinstall --no-deps`. For an install without
sudo (CUDA headers from pip wheels) and troubleshooting, see
[docs/INSTALL_TORCH.md](https://github.com/tekintatar/slangimage/blob/main/docs/INSTALL_TORCH.md).

## Choosing a backend

The device is chosen automatically — Metal on macOS, CUDA elsewhere, with
Vulkan as the fallback if the preferred backend cannot start — and operators
run on the device of their input image, so user code needs no device plumbing.
To pick a backend explicitly, pass a backend name (or a device) when you
create images:

```python
import slangimage as si

img = si.load("photo.jpg", device="vulkan")  # "auto" (default), "metal", "cuda" or "vulkan"
print(img)                                   # e.g. Image(640x480, fp16, vulkan)
dev = si.device("vulkan")                    # the same device object, e.g. for effects
```

`from_numpy()` and `from_pil()` take the same `device=` argument.

Requesting a backend that is not available raises `RuntimeError` (Vulkan was
not available on the tested macOS machine; use Metal there). `si.device()`
returns a process-wide default owned by the thread that first creates it; see
[Conventions](https://github.com/tekintatar/slangimage#conventions) for multithreaded use. The `SLANGIMAGE_BACKEND`
environment variable only selects the backend for the test suite; the library
itself does not read it.

## Precision

fp16 is the default storage format: lossless for 8-bit sources and half the
texture-storage bytes of fp32. In the standard benchmark its median speedup is
about 1.5–1.6× per backend (CUDA ≈1.6×, Vulkan ≈1.5×, Metal ≈1.6× in our
runs), while individual cells range from near parity to about 2.2×; this is not
a claim that total process VRAM is halved (see
[benchmarks](https://github.com/tekintatar/slangimage/blob/main/docs/BENCHMARKS.md)).
Kernel arithmetic runs in fp32 registers regardless of storage. Every RW
texture is `[format(...)]`-annotated, so fp16 works identically on CUDA (no
[slangpy #648](https://github.com/shader-slang/slangpy/issues/648)
surface-write corruption), Metal, and Vulkan.

## Conventions

- Images are RGBA on the GPU; 3-channel input is expanded to opaque RGBA.
- Values are nominally [0, 1] but never silently clipped: `to_numpy()` is
  lossless (sharpening overshoot and `normalize` output read back exactly);
  `to_pil()`/`save()` clip for display.
- Kernel dispatch grids are row-major `(height, width)`.
- Effects run inside texture-pool frames: after warmup, repeated processing at
  one resolution is allocation-free once earlier outputs are released or
  recycled. Holding `img.texture` opts that output out of automatic recycling.
  `SLANGIMAGE_POOL_BUDGET_MB` bounds cached free-list residency (default 1536
  MiB), except that the current hot key keeps one reusable texture even if that
  texture alone exceeds the budget.
- Devices and everything derived from them are thread-affine. The process-wide
  default belongs to the thread that first creates it; another thread gets a
  `RuntimeError`. Multithreaded applications must construct a `DeviceManager`
  inside each worker and keep its Images, effects, cache access and teardown on
  that worker. `DeviceManager` takes a `si.Backend` member such as
  `si.Backend.AUTO` or its string value such as `"cuda"`; the
  [platform notes](https://github.com/tekintatar/slangimage/blob/main/docs/PLATFORM_NOTES.md#threads)
  show a complete worker.
- Prefer one long-lived device per worker. `release_device(device)` clears
  SlangImage's pool, module and operator caches but deliberately leaves the
  caller-owned device and surviving same-thread Images usable. Final teardown
  is: drop derived objects, call `release_device(device)`, then `device.close()`.
  On the validated SlangPy 0.42 stack, used native device wrappers and driver
  memory may remain resident until process exit, so repeated in-process device
  recreation is not a supported way to bound memory.

## Playground

Explore many of the operators interactively — upload a photo, drag parameter
sliders with live GPU preview, and stack ops into pipelines:

```bash
pip install "slangimage[playground]"
slangimage-playground           # http://localhost:7860
```

From a source checkout, `python playground/app.py` launches the same packaged
application; use it with an editable (`pip install -e .`) install, where the
`slangimage-playground` command is not available. It compiles the fp16 kernels
(`si.prewarm()`) before it starts serving, so the first launch takes a little
while.

## Benchmarks

SlangImage ships a standard, versioned benchmark suite, so you can run the same
measurements on your own hardware (the published reference numbers come from an
Apple M3 Max and an NVIDIA L40S):

```bash
pip install "slangimage[bench]"
slangimage-bench run --backend auto --out results.json
slangimage-bench report results.json
```

Methodology and rules: [docs/BENCHMARKING.md](https://github.com/tekintatar/slangimage/blob/main/docs/BENCHMARKING.md).
Current reference results: [docs/BENCHMARKS.md](https://github.com/tekintatar/slangimage/blob/main/docs/BENCHMARKS.md).

Head-to-head comparisons against kornia and torchvision v2 (latency,
training-step time, peak VRAM — including the cells we lose, documented
honestly) live in `benchmarks/compare_augmentation.py` and
`benchmarks/compare_kornia_diff.py`, with reference numbers in each header and
raw results for the L40S, RTX PRO 4500 Blackwell and Apple M3 Max in
[benchmarks/comparisons/](https://github.com/tekintatar/slangimage/tree/main/benchmarks/comparisons).

## Development

Requires a working GPU backend — Metal (macOS), CUDA, or Vulkan. There is no
CPU fallback for the kernels, so the test suite needs a GPU.

```bash
git clone https://github.com/tekintatar/slangimage.git && cd slangimage
python3.12 -m venv .venv && . .venv/bin/activate   # or 3.10, 3.11, 3.13
pip install -e ".[dev]"
pytest
```

The test suite runs on any of the three backends:
`SLANGIMAGE_BACKEND=vulkan pytest -ra`. This selects the shared texture fixture;
no-plumbing API/lifecycle tests also exercise automatic device selection.
The differentiable tests run only with `auto`/`cuda` selection and available
`torch` + CUDA; explicit Metal/Vulkan runs exclude all three CUDA-only files.
The summary separates selected-backend texture + host checks, intentional
AUTO-device API/lifecycle checks, and CUDA autograd results. Use
`-m 'not auto_device'` to exclude intentional AUTO checks. Prewarm tests execute
every public texture operator on the selected device at both fp16/fp32; these
are dispatch/storage checks, not numerical references. Three skipped
collection nodes (the three CUDA-only files) mean gradients were not tested. See
[test reporting](https://github.com/tekintatar/slangimage/blob/main/docs/TEST_COVERAGE.md#reporting-backend-coverage-accurately)
and [CONTRIBUTING.md](https://github.com/tekintatar/slangimage/blob/main/CONTRIBUTING.md).

## License

Copyright 2026 BeFunky, Inc. and Tekin Tatar. Licensed under
[Apache-2.0](https://github.com/tekintatar/slangimage/blob/main/LICENSE) — same license as slangpy upstream; permissive for
commercial use with an explicit patent grant (see
[NOTICE](https://github.com/tekintatar/slangimage/blob/main/NOTICE)).

SlangImage is built on Slang and slangpy but is not affiliated with the Slang
project or Khronos.
