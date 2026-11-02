# SlangImage — notes for coding agents

Cross-platform GPU image processing on SlangPy (Metal/CUDA/Vulkan),
fp16-first. Human contributors: start with [CONTRIBUTING.md](CONTRIBUTING.md);
this file adds the conventions that are expensive to rediscover. `CLAUDE.md`
only imports this file, so edit this one. Put machine- or person-specific
notes (interpreter paths, GPU hosts, identities) in your own untracked
`CLAUDE.local.md` or your tool's user-level instructions, never here.

## Environment & commands

- Setup and test commands are in [CONTRIBUTING.md](CONTRIBUTING.md): a Python
  3.10–3.13 virtual environment with `pip install -e ".[dev]"`; the suite
  needs a GPU and uses one device via `tests/conftest.py`
  (`SLANGIMAGE_BACKEND=metal|cuda|vulkan` selects it).
- **Before finishing a change, run `scripts/check.sh`** (whitespace, ruff,
  mypy over the package, tests on the available GPU; `--no-gpu` without one;
  extra arguments go to pytest, e.g. `-k bilateral`). If you could not run the
  GPU tests on a backend your change affects (CUDA autograd needs an NVIDIA
  GPU), say so instead of implying they passed.
- Setup problems: `python -m slangimage` runs a test kernel on every backend
  and explains each failure. New tests that create a device without the `ctx`
  fixture need `@pytest.mark.gpu`.
- Playground: `python playground/app.py` → localhost:7860 (Gradio).
- Benchmarks: `slangimage-bench run/report` (protocol versioned — changing
  the suite/methodology requires a PROTOCOL_VERSION bump; docs/BENCHMARKING.md).
- CUDA/Vulkan validation on a remote Linux NVIDIA host:
  `scripts/cuda_box/deploy.sh` rebuilds the host environment from zero (NVRTC
  via a pip-wheel shim; CUDA runs need
  `CUDA_PATH=~/cuda-shim LD_LIBRARY_PATH=~/cuda-shim/lib64` on that host).
  `deploy.sh` syncs with `rsync --delete`, so point it only at a disposable
  host. Never present older results as validation of new code.

## Architecture

- `src/slangimage/ops.py` — public functional API (single module ON PURPOSE:
  function names would collide with submodules and a fresh submodule import
  rebinds the package attr, clobbering the function — do not convert back
  to a package).
- `src/slangimage/kernels/*.py` — op classes (one per shader).
- `src/slangimage/shaders/ops/*.slang` — kernels; `shaders/common/` shared.
- `src/slangimage/runtime/` — device (per-backend default device cache),
  modules (ModuleCache), textures (upload/readback), texture_pool.
- `src/slangimage/image.py` — `si.Image` handle; `fx/` — effect framework;
  `bench/` — benchmark suite.

Adding an op = shaders/ops/x.slang + kernels/x.py + functional wrapper in
ops.py + a case in the `prewarm()` registry + property tests (numpy
reference or mathematical invariant, not just no-NaN). Keep ops standard and
generically named.

## Conventions & gotchas (expensive to rediscover)

- Every `RWTexture2D` param needs `[format(SLANGIMAGE_FMT)]` — CUDA does no
  format conversion on surface writes (slangpy #648). ModuleCache compiles
  fp16/fp32 variants by prepending a #define (slangpy doesn't forward
  per-module compiler defines) and hashes source into module names
  (same-name reload with different source is a slangpy hard error).
- Kernels: manual loads + clamp, NO hardware samplers; dispatch grids are
  row-major `spy.grid((h, w))`; math in fp32 registers regardless of storage.
- CUDA hot-loop rules (measured, experiments/cuda_hotloop/FINDINGS.md):
  NO float3/float2 ANYWHERE in hot kernels — loops OR epilogues; even one
  float4(x.rgb, y) swizzle-constructor per pixel is ~2.6× (slang emits
  structs NVRTC spills; use float4/scalars; filed slangpy#1059); NO
  dynamically-indexed local arrays (local memory — named accumulators +
  uniform branch; Metal prefers arrays: use __target_switch like
  kuwahara.slang); NVRTC defaults to precise software transcendentals (pow
  4.4× slower than Vulkan's default hardware approx) — device.py sets
  floating_point_mode=fast to match Vulkan/Metal driver defaults
  (downstream_args --use_fast_math is silently ignored — filed
  slangpy#1058); still hoist invariants, consolidate algebraically, integer
  classification over atan2 (fastmath.slang). Ad-hoc perf scripts:
  SLANGIMAGE_BACKEND only steers tests — create devices explicitly or you'll
  measure CUDA twice.
- Readback (`to_numpy`/`texture_to_np`) is LOSSLESS — never add clipping
  (`normalize` output and contrast overshoot depend on it); clip only in
  to_pil/save.
- 3-channel uploads auto-expand to RGBA (no 3-channel GPU formats).
- Effects run in texture-pool frames; ops allocate via `alloc_output` (never
  `device.create_texture` directly). Pool budget: SLANGIMAGE_POOL_BUDGET_MB.
  Out-of-frame op outputs auto-recycle when their Image is GC'd
  (refcount-guarded finalizer in texture_pool; slangpy's call-data cache
  holds the last dispatch's args, hence the pending-retry list). Holding
  `img.texture` blocks recycling — that's the opt-out, not a leak.
- Kernel classes (`slangimage.kernels.*`), `slangimage.fx.base` and the
  runtime helpers may be used directly by third-party packages. Treat them
  as public API: don't remove or rename them because nothing in this
  repository calls them.
- Devices are thread-affine: a device and everything derived from it stay on
  the thread that created it; workers create their own explicit device.
- Tensor-path conventions (differentiable ops): spy.grid((H,W)) gives
  pix.x=COLUMN (fast dim); Tensor<T,2> subscripts are graphics-order too
  (t[uint2(col,row)]). Differentiable slang: `import slangpy;`,
  [MaxIters(n)] on loops, and loops MUST be zero-based (`for i in 0..2r+1;
  dx = i - r`) — negative runtime loop starts segfault backward dispatch
  (slangpy#1051; also #1052: call-data cache drops autograd hooks after
  no-grad calls — diff.py uses grad/no-grad module twins; repro dirs in
  experiments/ — drop the zero-based rule once fixed upstream). Two more
  filed autodiff bugs (repros + details in experiments/): #1055 vector
  (float3) return + loop in one [Differentiable] function → silently
  WRONG gradients — use scalar-return kernels over an (H,W,C) grid; and
  #1056 two IDiffTensor params with mixed grad-ness → CUDA illegal
  memory access in backward — diff.py `_align_grads()` gives no-grad
  inputs throwaway grad aliases. Keep the `torch` extra's SlangPy 0.42 pin
  until a released wheel passes the slangpy#1167 reproducer
  (docs/UPSTREAM_ISSUES.md).
- Displayed images in gradio default to lossy webp → grid artifacts on
  smooth gradients; playground pins format="png".

## Where things are documented

- docs/ROADMAP.md — direction and what is planned next.
- docs/TEST_COVERAGE.md — per-operator evidence and how to report backend
  coverage honestly; update it when adding or removing exports or tests.
- docs/UPSTREAM_ISSUES.md — SlangPy/Slang bugs and the workarounds above.
- docs/BENCHMARKS.md (standard suite) and the head-to-head scripts
  `benchmarks/compare_augmentation.py` / `compare_kornia_diff.py` (reference
  numbers and honest notes live in each script's header).
