# SlangImage Roadmap

Direction and sequencing. Benchmark methodology lives in
[BENCHMARKING.md](BENCHMARKING.md); current operator/precision evidence and
known numerical gaps live in [TEST_COVERAGE.md](TEST_COVERAGE.md).
Implementation milestones below are dated historical records; a completed
milestone does not mean every operator, parameter or backend combination is
numerically certified.

**Status (0.1.0):** the numerical-reference, runtime-lifetime, CUDA
autograd, packaging/support-matrix and performance validation work recorded in
[TEST_COVERAGE.md](TEST_COVERAGE.md) is complete. Planned work is listed under
*Next* below.

## Positioning — the two pillars

1. **Fast fused neighborhood ops.** One Slang kernel per op on Metal/CUDA/
   Vulkan, fp16-first. Measured vs kornia on the same GPU: bilateral
   45–230× faster in the current augmentation comparison; conv-lowered ops
   (gaussian) are generally 1.4–6.8× faster in that forward-only matrix
   (`benchmarks/compare_augmentation.py`). The gap is structural — pure-tensor
   libraries pay unfold/im2col memory amplification on anything
   data-dependent per pixel.
2. **Differentiable ops (complete).** Slang has compiler-native autodiff
   (`[Differentiable]` generates backward kernels); slangpy ≥0.41 bridges it
   into `torch.autograd`. Fused backward passes attack the same structural
   gap in training pipelines — plus features tensor libraries can't match
   cheaply (learnable LUTs).

Training augmentation (torchvision-style batch pipelines on CUDA) is a
separate package, [SlangVision](https://github.com/tekintatar/slangvision);
it was developed here as `slangimage.augment` and moved before any release.

## Track D — differentiable ops ✅ COMPLETE (D0→D1→D2, 2026-07-12)

- ~~**D0 — spike.**~~ ✅ done 2026-07-12 (experiments/d0_diff_gaussian/FINDINGS.md): One op (gaussian: backward = same
  blur, trivially verifiable) as a Tensor-based `[Differentiable]` kernel
  through slangpy's torch bridge; validated with closed-form references and
  explicit fp32 finite differences. Retires the unknowns: slangpy diff-API ergonomics,
  gradient scatter/atomics behavior, dtype handling. Also doubles as the
  texture-vs-Tensor storage bake-off measurement.
- ~~**D1 — core differentiable subset**~~ ✅ complete. `si.diff` namespace,
  torch tensors in/out (CUDA), tensor-storage kernels.
  - ~~Wave 1~~ ✅ 2026-07-12: gaussian, exposure, sepia, hue_rotate.
  - ~~Wave 2~~ ✅ 2026-07-12: bilateral (gradients flow through the range
    weights), blend modes (multiply/screen/overlay, diff w.r.t. both
    layers), **LUT differentiable w.r.t. the LUT** — the headline —
    proven by `test_lut_learns_sepia` (Adam recovers a sepia grade from
    an identity LUT, loss 1.3e-07 in 120 steps). Cost: two more autodiff
    bugs found + worked around (vector-return+loop → wrong grads; mixed
    grad-ness → illegal access), see d0 FINDINGS.md wave-2 section.
  - ~~Gather backward for bilateral~~ ✅ 2026-07-12: hand-written gather
    adjoint via a plain-kernel torch.autograd.Function bridge (the D0
    manual-bridge pattern) replaced Slang's generated atomic-scatter
    backward. Training step 1024²: 261 ms → **1.00 ms** (kornia: 85 ms —
    from 3× slower to **85× faster** in that July run, with 194× less VRAM). Exact +
    deterministic (no atomics). Border semantics moved to
    truncated-window (skip out-of-bounds taps) — the symmetry w(p,q) ==
    w(q,p) is what makes the one-pass gather possible. Same pattern
    available for any future op whose generated backward scatters.
  - ~~Wave 3~~ ✅ 2026-07-12: unsharp_mask (composed from the diff
    gaussian + torch arithmetic — exact by construction) and resize
    (bilinear/bicubic, manual bridge; weights are geometry-only so the
    op is linear and the backward is two transposed 1D resamples with
    no saved activations; matches F.interpolate align_corners=False
    forward AND backward, verified against torch directly). Kuwahara
    dropped (decided 2026-07-12): a differentiable version is not planned.
    Excluded: CLAHE (histograms), pointillize/halftone (discrete).
    → D1 core subset COMPLETE: gaussian, bilateral, unsharp, resize,
    exposure/sepia/hue, blend modes, learnable LUT. D2 also completed below.
- **D2 — polish + proof.**
  - ~~si.nn Module wrappers~~ ✅ 2026-07-12: slangimage/nn.py —
    GaussianBlur/Bilateral/UnsharpMask/Resize/Exposure/Sepia/HueRotate/
    Blend + **Lut** (the headline feature: an nn.Parameter CLUT
    initialized to identity — trainable color grading in any
    nn.Sequential; state_dict roundtrip + learning loop tested).
  - ~~Mixed precision~~ ✅ 2026-07-12: fp16/bfloat16/autocast inputs
    upcast to fp32 compute (diff._f32c; amp.custom_fwd/bwd on the two
    manual bridges); gradients flow back in the input dtype. Native
    fp16 kernels deliberately not pursued (math is fp32 anyway).
  - Gradient suite exists (`test_diff.py` + `test_diff_core_contracts.py` +
    `test_nn.py`: fp32 sampled finite differences, analytic derivatives, and
    NumPy/torch references/invariants).
    It runs on a CUDA machine with `auto`/`cuda` selection and available
    torch/CUDA. Metal/Vulkan runs exclude all three files. Hosted CI has no
    GPU and does not run them; a CUDA run with skipped autograd modules is not
    gradient validation.
  - **Backward-pass benchmark vs kornia + torchvision v2** — landed and
    refreshed 2026-09-12 (benchmarks/compare_kornia_diff.py, reference numbers
    + honest notes in header): bilateral training step 93–97× faster than
    kornia at 1024²–2048² in the reference run (46–97× across the L40S and
    RTX PRO 4500 comparison runs) with a
    194× lower Torch-allocator peak (31 GB/step at 2048² — one image; torchvision lacks
    the op), hue-rotate step 5–31× vs kornia while torchvision's
    adjust_hue raises on grad input (augmentation-only). Gaussian is near
    parity and has a small Kornia loss at 2048². Honest losses: resize — native F.interpolate (which both
    wrap) is ~2–3×/step faster; ours is for pipeline completeness with
    matching the tested interpolation semantics. Historical standalone Metal
    gradient experiments are not current pytest or torch-autograd coverage;
    the supported tensor/autograd path and its normal tests are CUDA-only.

## Track C — augmentation & photo color ops ✅ COMPLETE (2026-07-12)

Positioning: the non-diff path competes in augmentation pipelines
(benchmarks/compare_augmentation.py, refreshed 2026-09-12). The specialized
paths are the wins: versus kornia, CLAHE is 11–15253×, bilateral 45–230× and
hue/saturation 6–261× across the 2026-09-12 L40S and M3 Max runs (on the
RTX PRO 4500 with kornia 0.9.0: CLAHE 5–11×, saturation 3–10×). Kornia and torchvision win
several tiny CUDA elementwise cells, and torchvision also wins a few small
Metal cells; the comparison header records those losses. Color operators kornia /
torchvision have that si.ops lacks — the parity set to build:

- ~~Parity set~~ ✅ 2026-07-12: solarize, posterize, grayscale,
  normalize (color_aug.slang) + equalize, autocontrast
  (hist_color.slang — global per-channel histogram eq reusing the
  CLAHE pattern; exact float min/max via order-preserving-uint
  atomics). Property tests on all 3 backends; in prewarm; benchmarked:
  equalize is 214–303× faster than kornia and 5–11× faster than torchvision
  on Metal, and near parity to 2.5× faster than kornia on CUDA. Autocontrast
  is 12–18× faster than torchvision on Metal but 2.6–5.3× slower on CUDA.
- ~~White balance~~ ✅ 2026-07-12: photo-grade Bradford chromatic
  adaptation — Kelvin temperature (Planckian + CIE daylight locus) and
  green/magenta tint, all color science in testable numpy
  (kernels/white_balance.py produces one 3×3 linear-RGB matrix per
  call), kernel = srgb→linear → matrix → linear→srgb. Neither kornia
  nor torchvision has an equivalent. In the playground with
  temperature/tint sliders.
- ~~Geometric transforms~~ ✅ 2026-07-12: flip / rot90 / crop as
  exact index remaps (numpy-equal in tests) + the affine engine
  (inverse-warp, bilinear/nearest, zeros fill — torchvision/kornia
  convention; forward 2×3 inverted in numpy) with rotate built on
  top (center rotation, torchvision angle convention, expand=True
  canvas growth). Track C complete.
- ~~Periodic tiling~~ ✅ implemented: `tile` with scale/phase and
  nearest/bilinear sampling; Vulkan periodic-coordinate correction and fp32
  NumPy/periodicity regressions are in place. README and prewarm include it.
  Bounded fp16/fp32 numerical and boundary references were subsequently added;
  see the evidence map for exact scope, not a claim of exhaustive tiling coverage.

## Track P — performance & correctness

- ~~Per-op dispatch floor~~ ✅ fixed 2026-07-12: profiling showed the
  ~0.35 ms CUDA floor was `device.create_texture` (~0.29 ms/call) —
  out-of-frame functional calls never returned outputs to the pool.
  Now `Image` handles auto-recycle their pooled texture on GC
  (refcount-guarded finalizer + pending retry for slangpy's transient
  call-data ref; keep `img.texture` and it's yours, never recycled).
  CUDA elementwise cells: 0.34–0.37 → **0.08 ms**; gamma/saturation/
  posterize now beat kornia; torchvision's remaining 2–4× edge is
  slangpy per-call overhead (~0.03 ms) + sync — upstream. Metal's
  floor is the command-buffer round trip (~0.3 ms), which MPS pays
  identically — nothing to fix there at single-op granularity.
- ~~CUDA vs Vulkan effect-chain gap~~ ✅ closed 2026-07-12 — and
  inverted. Root causes (experiments/cuda_hotloop/FINDINGS.md): float3
  arithmetic in hot loops is ~3× on the CUDA path (use float4/scalars);
  dynamically-indexed local arrays go to local memory (named registers
  + uniform branch on CUDA/Vulkan, arrays on Metal — kuwahara selects
  via __target_switch); NVRTC transcendentals are precise software
  routines and fast-math flags don't reach it through slangpy
  (upstream candidate) — hoist, consolidate algebraically, avoid
  atan2. Results at 12MP: kuwahara CUDA 21.1→5.9 ms, Vulkan
  12.7→6.8 ms; surface_blur CUDA 4.6→2.4 ms. CUDA is now the fastest
  backend on every suite op but resize (July 2026) — no more torch-interop
  tension.
- ~~float3 sweep of remaining neighborhood kernels~~ ✅ 2026-07-12:
  bilateral 4.4→2.0 ms, nl_means 31→10.4 ms (3×), sharpen_rgb
  3.7→1.08 ms (3.4×), resize-lanczos + gaussians cleaned — every
  neighborhood op now at Vulkan parity on CUDA. Two MORE rules found (FINDINGS addendum): float3 poisons codegen
  ANYWHERE in a hot kernel, not just loops — one float4(x.rgb, a)
  swizzle-constructor per pixel cost sharpen_rgb 2.6×; and [unroll]
  on runtime trip counts + per-tap fdiv are ~20% each. Knock-on:
  bilateral vs kornia in the augmentation benchmark went from 22× to
  61–131×.
- Deferred benchmark protocol work: pooled "server-loop" scenario and
  differentiable forward/backward cases in the versioned suite. D1 is already
  complete and standalone differentiable comparisons exist; changing the suite
  methodology still requires a protocol bump, not an automatic version change.
- ✅ Same-input cross-backend capture/compare harness
  (`scripts/cross_backend_numeric.py`) covers all 44 texture exports at
  fp16/fp32 with per-case policies. It uses temporary archives rather than a
  permanent machine golden; same-backend PSNR remains a separate
  storage-precision check.

## Next

- Generated API-reference site (operator docstrings already document
  parameters, defaults and clipping).
- **Hugging Face Space** of the playground (CUDA via the pip NVRTC shim), so
  the operators can be tried without installing.
- GPU validation stays explicit rather than CI-driven: hosted CI runs only
  the GPU-free checks (lint, types, `-m "not gpu"` tests, archive build), so
  fresh-install checks and each GPU group still run on real hardware before a
  release, with CUDA autograd reported separately.

## Later / ideas

- Zero-copy torch interop on CUDA: `Image.to_torch()` / `si.from_torch()`
  copy through host memory today; sharing the device with
  `spy.create_torch_device()` plus a texture-to-buffer copy kernel would keep
  the data on the GPU. Batch API (B×H×W×C).
- Video: temporal ops, stream processing on the pooled path.
- Playground: "export pipeline as Python" button; before/after slider.
- Shader disk cache to cut cold-start compilation (CUDA first call, compile +
  dispatch, was 143–887 ms per op × precision in the current reference run,
  the largest including one-time session setup; see [BENCHMARKS.md](BENCHMARKS.md)).
- bfloat16 texture storage (bf16 tensor inputs are already supported and
  tested; there is no bf16 texture format).

## Sequencing

The core C, D and P implementation milestones landed in July; later tiling and
release-validation work is also implemented. This is not a claim of exhaustive
verification: the remaining numerical and cross-backend gaps are listed at the
end of [TEST_COVERAGE.md](TEST_COVERAGE.md). Deferred benchmark work is
separate from the completed optimization milestones. *Next* lists planned
work; the "Later / ideas" backlog is unscheduled.
