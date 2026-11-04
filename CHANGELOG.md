# Changelog

All notable changes to SlangImage are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims
to adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) once it
reaches 1.0.

## [Unreleased]

### Removed

- `slangimage.augment` (GPU training augmentation), which was on `main` for
  a day and never released, moved to its own package,
  [SlangVision](https://github.com/tekintatar/slangvision).

## [0.1.1] - 2026-11-02

### Fixed

- `slangimage.diff` now runs a one-pixel test kernel when it creates its CUDA
  device. When NVRTC cannot be loaded, the first differentiable operation
  raises a `RuntimeError` with setup guidance instead of a raw
  `E52002 pass-through compiler not found` error.
- CUDA setup errors (texture and differentiable paths) say what to do when
  NVRTC is installed but not found: run `sudo ldconfig` after an apt install,
  or set both `CUDA_PATH` and `LD_LIBRARY_PATH` for the pip packages.
- `python -m slangimage` shows why the differentiable ops are unavailable.

### Documentation

- The system-wide NVRTC install steps (README, docs/INSTALL_TORCH.md) end with
  `sudo ldconfig`: right after the apt install the shared-library cache can
  miss NVRTC, so CUDA fell back to Vulkan and `slangimage.diff` reported
  unavailable.

## [0.1.0] - 2026-11-02

First public release.

### Added

- GPU image operators on Metal, CUDA and Vulkan from one Slang code base, with
  fp16 or fp32 texture storage and fp32 arithmetic: blur and denoise
  (`gaussian_blur`, `surface_blur`, `bilateral`, `nl_means`), sharpening,
  color and tone adjustments (including `white_balance`, `clahe`, `equalize`,
  `autocontrast` and square-grid `lut`), 12 Photoshop/PDF blend modes,
  stylization (`kuwahara`, `pointillize`, `cmyk_halftone`, thresholds),
  geometry (`resize`, `flip`, `rot90`, `crop`, `tile`, `rotate`, `affine`) and
  `random_noise`. The README lists every operator.
- `si.Image` handles with `si.load`, `si.from_numpy` and `si.from_pil`;
  readback through `to_numpy()` is lossless and clipping happens only on
  request or when saving. Three-channel inputs are expanded to RGBA on upload,
  EXIF orientation is honored and 16-bit images keep their full range.
- `Image.to_torch()` and `si.from_torch()`: lossless copies between images and
  torch tensors in `(H, W, C)` or `(C, H, W)` layout, with or without alpha,
  on any backend (they need `torch`, not the `torch` extra).
- Automatic backend selection (Metal on macOS, CUDA elsewhere, Vulkan as the
  fallback). A new CUDA device runs a one-pixel test kernel and falls back to
  Vulkan if it cannot compile; an explicitly requested backend that cannot
  start raises a `RuntimeError` that names the missing requirement. Backends
  can be chosen by name (`device="vulkan"`).
- A texture pool that recycles operator outputs when their `Image` is garbage
  collected, effect frames for multi-op pipelines (`slangimage.fx`:
  `GaussianBlurEffect`, `SoftGlowEffect`, `KuwaharaPaintEffect`), a memory
  budget (`SLANGIMAGE_POOL_BUDGET_MB`), `si.prewarm()` for services, and
  `release_device()` for explicit teardown. Devices are bound to the thread
  that created them.
- `python -m slangimage`: a setup report that runs a test kernel on each
  backend in its own process and explains every failure (`--json` for bug
  reports).
- Type information for the whole package (`py.typed`), checked with mypy.
- Custom `.slang` kernels from your own directories through
  `ModuleCache.add_ops_dir()`, including shadowing built-in operators.
- Differentiable operators for PyTorch on CUDA (`slangimage.diff`) and
  `torch.nn` modules (`slangimage.nn`), including a learnable color LUT, with
  fp16/bfloat16 inputs computed in fp32.
- Input validation for every public numeric control, documented in
  `docs/PARAMETER_CONTRACTS.md`.
- A versioned benchmark suite (`slangimage-bench`), comparison scripts against
  Kornia and TorchVision, a Gradio playground (`slangimage-playground`), six
  runnable examples, and reproducible wheel/sdist builds with archive checks.

### Known limitations

- Differentiable operators are CUDA-only, and the `torch` extra needs a
  three-step install (docs/INSTALL_TORCH.md). It is pinned to PyTorch 2.13 and
  SlangPy 0.42 because SlangPy 0.43's bundled compiler hangs in one generated
  backward pass (docs/UPSTREAM_ISSUES.md).
- Validated on Apple Silicon (Metal) and on NVIDIA L40S and RTX PRO 4500
  Blackwell (CUDA, Vulkan) on Ubuntu 24.04; Windows and other GPUs are untested
  (docs/PLATFORM_NOTES.md).
- Native SlangPy device memory is not fully returned until process exit after
  `release_device()`; long-running services should keep one device.
