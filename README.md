# SlangImage

Cross-platform GPU image processing library built on [Slang](https://shader-slang.org/)
via [slangpy](https://github.com/shader-slang/slangpy) — one kernel codebase
running on **Metal, CUDA, and Vulkan**, fp16-first, with **differentiable
ops for PyTorch** (`slangimage.diff`): fused forward *and* backward kernels,
including a learnable color-grading LUT.

Use it to run the same image operators (blur and denoise, sharpening, color,
geometry, stylization) from Python on macOS (Metal) or Linux (CUDA or Vulkan)
GPUs, or to put fused, differentiable image ops inside a PyTorch training loop
on CUDA. It is not a CPU library, and it processes one RGBA image at a time
(there is no batch dimension yet).

**Status: pre-alpha, under active development.** APIs may still change.

## Why SlangImage

Edge-preserving and data-dependent filters — bilateral, CLAHE, non-local
means, hue/HSV math — run here as one fused GPU kernel per operation. Tensor
libraries assemble them from generic steps (unfold/im2col windows, per-pixel
color-space conversions) that materialize many copies of the image, so the gap
is largest where the math is most complex, and it carries over to training
memory.

Measured on two NVIDIA GPU generations (L40S and RTX PRO 4500 Blackwell),
batch 1, 1024² and 2048²; forward rows cover fp16 and fp32, training-step rows
fp32. Ranges span both GPUs:

| Operation | vs kornia | vs torchvision |
|---|---|---|
| Bilateral filter, forward | 51–147× faster | no equivalent |
| Bilateral filter, training step (forward + backward) | 46–97× faster, 158–194× less GPU memory (160 MiB vs about 25–30 GiB at 2048²) | no equivalent |
| CLAHE | 5–25× faster | no equivalent |
| Hue rotation, forward | 7–15× faster | 5–28× faster |
| Gaussian blur 19×19, forward | 3–7× faster | 3–9× faster |

Where it is not faster: tiny per-pixel operations (invert, gamma, grayscale,
autocontrast) cost about 0.05–0.7 ms per call here, mostly dispatch overhead,
and torchvision's native kernels are about 1.6–5.6× faster at that scale; in
training, PyTorch's own bilinear resize is about 2–3× faster per step. Results
depend on library versions: kornia 0.9.0's CLAHE is up to about 6× faster than
0.8.3's (at 1024², fp16, measured on different GPUs), which accounts for most
of the CLAHE range above. Every cell, including the ones SlangImage loses, and
the raw data are linked under
[Benchmarks](https://github.com/tekintatar/slangimage#benchmarks).

## Installation

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

See the [backend support matrix](https://github.com/tekintatar/slangimage#backend-support) below and the
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

The CUDA differentiable extra intentionally pins PyTorch 2.13 and SlangPy 0.42.
SlangPy 0.43.0/0.43.1 bundles a compiler regression that can hang generated
backward for runtime-dependent tensor loads
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

## Quick start

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

Kernels compile lazily per device and precision, so the first call of each
operator is slower; long-running services call `si.prewarm()` once at startup
for each precision they serve. More runnable scripts live in
[examples/](https://github.com/tekintatar/slangimage/tree/main/examples)
(source checkout; run e.g. `python examples/01_quickstart.py`): 01–05 run on
any backend (add `--backend vulkan`, `cuda` or `metal` to override automatic
selection), 06 needs CUDA and the `torch` extra. Rendered images go to
`examples/out/`; without your own images in `test_images/` (or
`SLANGIMAGE_TEST_IMAGES`), a synthetic input is generated there on first run.

### Quick start: PyTorch (CUDA)

`img.to_torch()` and `si.from_torch()` copy images to and from torch tensors
on any backend (they need only `torch`). The differentiable ops in
`slangimage.diff` / `slangimage.nn` need the [`torch` extra](https://github.com/tekintatar/slangimage#pytorch-extra-slangimagetorch)
and an NVIDIA GPU, and take one image at a time as a float32 `(H, W, 3)` CUDA
tensor:

```python
import torch
import slangimage as si
import slangimage.diff as sid
import slangimage.nn as snn

photo = si.load("photo.jpg").to_torch(device="cuda").float()    # (H, W, 3) float32

# gradients flow through fused forward and backward kernels
x = photo.clone().requires_grad_()
out = sid.sepia(sid.bilateral(x, radius=5, sigma_range=0.15), intensity=0.8)
out.mean().backward()                                            # x.grad is set

# learn a color grade as a LUT from an (original, edited) pair of the same size
edited = si.load("photo_edited.jpg").to_torch(device="cuda").float()
grade = snn.Lut().cuda()                                         # identity LUT, learnable
opt = torch.optim.Adam(grade.parameters(), lr=5e-2)
for step in range(100):
    loss = (grade(photo) - edited).pow(2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
si.from_torch(grade(photo).detach().clamp(0, 1)).save("graded.png")

# the trained LUT is a preset for the texture path (any backend, no torch needed)
lut_image = si.from_torch(grade.lut.detach())
si.ops.lut(si.load("another.jpg"), lut_image).save("another_graded.png")

# torch models use NCHW batches; slangimage takes one HWC image per call
nchw = photo.permute(2, 0, 1).unsqueeze(0)                       # (1, 3, H, W)
hwc = nchw[0].permute(1, 2, 0)                                   # back to (H, W, 3)
```

`to_torch(layout="chw")` returns `(C, H, W)` directly, and `alpha=True` keeps
the alpha channel. Both conversions are lossless copies through host memory,
not zero-copy views.

### Choosing a backend

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

## Operators

The texture operators support all three backends with fp16 or fp32 storage.
Test depth varies by operator and precision; see the
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

Higher-level effect pipelines (multi-op graphs with pooled intermediates)
live under `slangimage.fx` (`GaussianBlurEffect`, `SoftGlowEffect`,
`KuwaharaPaintEffect`); operator implementation classes under
`slangimage.kernels` for advanced use. Downstream packages can ship their own
`.slang` kernels via `ModuleCache.add_ops_dir()` (see
`examples/05_custom_kernels.py`).

Effects are constructed once for a device and reuse its module cache and
texture pool. Their parameters are explicit and validated; Gaussian effects
accept either `blur_amount` or the complete `radius`/`sigma` pair:

```python
import slangimage as si

img = si.load("photo.jpg")
modules = si.ModuleCache(img.device)
glow = si.fx.SoftGlowEffect(img.device, modules)
out = glow.apply(img, radius=9, sigma=3.0, glow_strength=0.7)
```

## Differentiable ops (PyTorch)

`slangimage.diff` runs fused Slang kernels inside `torch.autograd`
(CUDA): gradients flow through the ops, so they can sit between a network
and its loss — or *be* the model:

```python
import torch
import slangimage.diff as sid

img = torch.rand(512, 512, 3, device="cuda", requires_grad=True)
out = sid.bilateral(img, radius=5, sigma_range=0.15)   # edge-preserving; gradients reach img
out = sid.sepia(out, intensity=0.8)
out.mean().backward()                                  # fused backward kernels
```

The [PyTorch quick start](https://github.com/tekintatar/slangimage#quick-start-pytorch-cuda) shows a complete
training loop that learns a color grade as a LUT.

Available: `gaussian_blur`, `bilateral` (hand-written gather backward —
46–97× faster per training step than kornia at 1–4 MP across the L40S and
RTX PRO 4500 comparison runs — 93–97× in the L40S reference run — with a
158–194× lower Torch-allocator peak),
`unsharp_mask`, `resize` (bilinear/bicubic; forward and backward match
`F.interpolate(align_corners=False, antialias=False)` to within 1e-4 in the
tests), `exposure`, `sepia`, `hue_rotate`, `blend`, `lut` (differentiable
w.r.t. both image and LUT). Gradient tests use sampled fp32 central finite
differences with an explicit error budget, or an analytic derivative for
exposure, alongside NumPy/torch references and invariants. See the
[per-op evidence](https://github.com/tekintatar/slangimage/blob/main/docs/TEST_COVERAGE.md#cuda-tensor-operators-sidiff-and-modules-sinn).
fp16, bfloat16 and `torch.autocast` inputs are handled
(upcast to fp32 compute, gradients flow back in the input dtype). See
`benchmarks/compare_kornia_diff.py` for the training-step numbers.
`sid.available()` returns `False` (instead of raising) when torch, a CUDA
device or the SlangPy torch bridge is missing.

`slangimage.nn` wraps these as `torch.nn.Module`s (kornia-style) — and
`snn.Lut()` is trainable color grading as a drop-in module:

```python
import torch
import slangimage.nn as snn

img = torch.rand(512, 512, 3, device="cuda")
grade = torch.nn.Sequential(snn.Bilateral(radius=5), snn.Lut()).cuda()
out = grade(img)                             # the LUT is an nn.Parameter
```

## GPU augmentation (PyTorch training)

`slangimage.augment` runs training augmentation on CUDA. Pipelines use
torchvision's transform names, draw separate random values for every image,
and compile into a few fused Slang kernels:

```python
import torch
import slangimage.augment as A

train = A.Compose([
    A.RandomResizedCrop(224),
    A.RandomHorizontalFlip(),
    A.ColorJitter(0.4, 0.4, 0.4, 0.1),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
images = torch.randint(0, 256, (64, 3, 480, 640), dtype=torch.uint8, device="cuda")
x = train(images)   # (64, 3, 224, 224) float16; a list of decoded images also works
```

Results match torchvision's float functionals image by image. Inputs are
read in place from any decoder (torchvision, nvImageCodec, DALI, DataLoader
workers), and you can add your own ops in a few lines of Slang. It needs
PyTorch with CUDA, but not the compiled `slangpy-torch` bridge. See
[docs/AUGMENT.md](https://github.com/tekintatar/slangimage/blob/main/docs/AUGMENT.md)
for the transforms, decoder recipes, custom ops and measured performance.

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
