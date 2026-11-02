# Upstream SlangPy / Slang issue ledger

Last verified **2026-09-27**. Tracker states were checked on GitHub, and every
reproducer was re-run against the published `slangpy==0.42.0` and
`slangpy==0.43.1` wheels on an NVIDIA L40S (driver 580.126.16, NVRTC 13.0,
Ubuntu 24.04, Python 3.12) with the CUDA and Vulkan backends. Performance
reproducers were only checked to run; their timings were not re-measured.

This is the release-status ledger for the six reports filed by the SlangImage
maintainer on 2026-07-12 and for
[#1167](https://github.com/shader-slang/slangpy/issues/1167) (filed
2026-09-17; see its section below). The reports and standalone reproducers live under
[`experiments/`](../experiments/README.md). A closed tracker issue does **not**
mean the fix is available to SlangImage: SlangPy wheels bundle a prebuilt Slang
compiler, so a compiler fix reaches users only in a SlangPy release that
bundles a Slang version containing it. Each workaround remains until a
published wheel is verified on the matching reproducer.

## Release boundary

The newest PyPI release on 2026-09-27 is
[`slangpy==0.43.1`](https://pypi.org/project/slangpy/0.43.1/), published
2026-07-16. SlangPy 0.43.0 and 0.43.1 bundle Slang 2026.12 (tagged
2026-06-25); 0.42.0 bundles Slang 2026.5.2. **None of the six fixes below is in
a released SlangPy wheel.** Keep all corresponding workarounds.

Not yet released: SlangPy `main` pins Slang 2026.17.1
([`external/CMakeLists.txt`](https://github.com/shader-slang/slangpy/blob/main/external/CMakeLists.txt)),
which contains the Slang fixes for #1051, #1055 and #1059, and it contains
SlangPy #1095 for #1058. Nothing from `main` was tested here.

| SlangPy issue | Status upstream (2026-09-27) | Released? Result of the 2026-09-27 re-run | SlangImage workaround (in place) |
| --- | --- | --- | --- |
| [#1051](https://github.com/shader-slang/slangpy/issues/1051) — runtime loop start crashes backward | Closed 2026-08-03. Fixed by Slang [#12299](https://github.com/shader-slang/slang/pull/12299), merged 2026-08-03 (Slang issue [#12070](https://github.com/shader-slang/slang/issues/12070)). First Slang release with the fix: 2026.16. | **Not released.** 0.42.0: SIGSEGV in `bwds()` on CUDA and Vulkan. 0.43.1: still broken; the backward now fails to compile instead (CUDA: NVRTC `identifier ... is undefined`; Vulkan: Slang internal error). The zero-based control hangs on 0.43.1 (#1167). | Zero-based differentiable loops with an offset index (`src/slangimage/shaders/diff/diff_kernels.slang`). On 0.43.x that form hits #1167, so the Torch extra stays on 0.42.0. See `experiments/d0_diff_gaussian/`. |
| [#1052](https://github.com/shader-slang/slangpy/issues/1052) — no-grad call poisons the autograd cache | Open. SlangPy [#1054](https://github.com/shader-slang/slangpy/pull/1054) is still a draft PR. | **Not released.** Reproduces on 0.42.0 and 0.43.1. | Separate grad and no-grad module instances in `src/slangimage/diff.py`. See `experiments/cache_poisoning/`. |
| [#1055](https://github.com/shader-slang/slangpy/issues/1055) — looped vector return has wrong gradients | Closed 2026-07-30. Fixed by Slang [#12095](https://github.com/shader-slang/slang/pull/12095), merged 2026-07-30 (Slang issue [#12071](https://github.com/shader-slang/slang/issues/12071)). First Slang release with the fix: 2026.16. | **Not released.** 0.42.0: wrong gradients reproduced. 0.43.1: the reproducer hangs in backward (#1167) before gradients can be compared. | Scalar-return kernels over an `(H, W, C)` grid for Gaussian blur; bilateral uses a hand-written plain-kernel backward. See `experiments/vector_return_grads/`. |
| [#1056](https://github.com/shader-slang/slangpy/issues/1056) — mixed grad-ness causes CUDA illegal access | Open. SlangPy [#1057](https://github.com/shader-slang/slangpy/pull/1057) is still a draft PR. | **Not released.** Aborts with `CUDA_ERROR_ILLEGAL_ADDRESS` on 0.42.0 and 0.43.1. | Throwaway gradient aliases for no-grad inputs via `_align_grads()` in `src/slangimage/diff.py`. See `experiments/mixed_gradness_crash/`. |
| [#1058](https://github.com/shader-slang/slangpy/issues/1058) — CUDA drops `downstream_args` | Open. SlangPy [#1095](https://github.com/shader-slang/slangpy/pull/1095), which forwards `downstream_args` to NVRTC, merged 2026-08-07. | **Not released**; #1095 is not in 0.43.1. On 0.42.0 and 0.43.1 an invalid NVRTC option passed through `downstream_args` compiles without error, and `--use_fast_math` gives results bit-identical to the default, so the arguments are still dropped. | `floating_point_mode=fast` on devices created by `src/slangimage/runtime/device.py`; do not rely on `downstream_args=["--use_fast_math"]`. See `experiments/nvrtc_fastmath/`. |
| [#1059](https://github.com/shader-slang/slangpy/issues/1059) — CUDA re-evaluates folded vector swizzles | Closed 2026-07-14. Fixed by Slang [#12078](https://github.com/shader-slang/slang/pull/12078), merged 2026-07-14 (Slang issue [#12073](https://github.com/shader-slang/slang/issues/12073)). First Slang release with the fix: 2026.14. | **Not released.** Slang 2026.12, bundled by 0.43.x, predates the fix. Performance only: the reproducer runs on both wheels but was not re-timed. | Avoid folded `float3` / multi-component-swizzle expressions in CUDA hot paths; use `float4` or named scalars. See `experiments/float3_cuda_penalty/`. |

## [#1167](https://github.com/shader-slang/slangpy/issues/1167): Slang 2026.12 dynamic-control-flow backward hang

Status on 2026-09-27: **open**. Upstream triage classified it as a Slang
compiler regression and escalated it to
[shader-slang/slang#13169](https://github.com/shader-slang/slang/issues/13169)
(open). The same triage reports that the hang persists with Slang 2026.17.1,
the compiler that SlangPy `main` pins, so a SlangPy release built on it may
still need the 0.42.0 pin. That report was not verified here.

The 2026-09-17 investigation explains the SlangPy 0.43.0/0.43.1 autograd
failure without treating the entire release as broken. A direct `.bwds()`
reproducer hangs only when runtime-dependent control flow encloses a
differentiable tensor load. Fixed-count loops and dynamic scalar-only loops
complete, as do texture operations and seven of SlangImage's nine tensor paths.

The controlled version matrix isolates the regression to the bundled compiler:

| SlangPy | Slang | Runtime-loop `.bwds()` |
|---|---|---|
| 0.42.0 | 2026.5.2 | pass (re-run 2026-09-27 on CUDA and Vulkan) |
| 0.43.0 | 2026.12 | **hang** |
| 0.43.1 | 2026.12 | **hang** (re-run 2026-09-27 on CUDA and Vulkan) |
| 0.43.0 rebuilt against 2026.5.2 | 2026.5.2 | pass |

The hang is in shader compilation, not on the GPU. On 2026-09-27, three native
stack samples (py-spy, at about 15, 25 and 35 seconds) of the stuck 0.43.1 CUDA
process all showed the main thread inside `libslang-compiler.so.0.2026.12`
(`linkAndOptimizeIR` → `specializeModule`) while it compiled the backward entry
point, at about 100% CPU with the GPU idle.

The reproducer does not import PyTorch, so neither PyTorch 2.13 nor
`slangpy-torch` is causal. SlangPy 0.43.1 is a packaging release: its
[changelog entry](https://github.com/shader-slang/slangpy/blob/v0.43.1/docs/changelog.rst)
lists only wheel-build and classifier changes, and the
[`v0.43.0...v0.43.1` diff](https://github.com/shader-slang/slangpy/compare/v0.43.0...v0.43.1)
touches only build, packaging, documentation and version files. (The GitHub
release page for v0.43.1 has no notes.) The upstream source pins confirm
[Slang 2026.5.2 in v0.42.0](https://github.com/shader-slang/slangpy/blob/v0.42.0/external/CMakeLists.txt)
and Slang 2026.12 in [v0.43.0](https://github.com/shader-slang/slangpy/blob/v0.43.0/external/CMakeLists.txt)
and [v0.43.1](https://github.com/shader-slang/slangpy/blob/v0.43.1/external/CMakeLists.txt).

The same hang changes what two older reproducers show on 0.43.x: the zero-based
`control` variant in `experiments/d0_diff_gaussian/` and the
`experiments/vector_return_grads/` reproducer both hang in backward. Stack
samples of both showed the same compiler frames as above. Use SlangPy 0.42.0
to observe #1051 and #1055 as reported.

This affects generated backward for SlangImage Gaussian blur and unsharp mask,
their `nn` wrappers and graphs containing them. Exposure, sepia, hue rotation,
blend and LUT generated backward passed; bilateral and resize manual backward
passed. A 2026-09-27 re-check ran one small forward and backward per
`slangimage.diff` op with the Python torch bridge. It matched: on 0.43.1 those
seven ops completed while Gaussian blur and unsharp mask hung in backward; on
0.42.0 all nine completed. Keep the Torch extra on SlangPy 0.42.0. Base installs may retain the
wider 0.42–0.43 range because texture upload/dispatch/readback passed on 0.43.1.

Full evidence, the build fingerprint and the bounded reproducer are in
[`experiments/runtime_loop_autodiff_hang/`](../experiments/runtime_loop_autodiff_hang/).
No matching prior upstream issue was found; the report was filed on 2026-09-17.

## Latest-wheel verification

### 2026-09-27

Fresh virtual environments contained exactly `slangpy==0.42.0` or
`slangpy==0.43.1` with NumPy 2.5.3. The PyTorch reproducers added PyTorch
2.13.0 and used SlangPy's Python torch bridge (`SLANGPY_ALLOW_TORCH_FALLBACK=1`).
On 0.42.0 they were repeated with the compiled `slangpy-torch` 0.7.0 bridge,
with identical results. Hardware: NVIDIA L40S, CUDA and Vulkan. Every run was
bounded by `timeout 120`; exit 124 means the bound was reached.

| Reproducer | SlangPy 0.42.0 | SlangPy 0.43.1 |
|---|---|---|
| #1051 `repro_bwds_segfault.py`, device type set to CUDA or Vulkan | SIGSEGV in `bwds()` (exit 139); `control` passes | backward compile error (exit 1); `control` hangs (exit 124, #1167) |
| #1051 `spike.py`, stage 2 (PyTorch) | SIGSEGV in backward (exit 139) | NVRTC compile error (exit 1) |
| #1052 `repro_cache_poisoning.py` | `backward FAILED` (exit 1); `control` passes | same as 0.42.0 |
| #1055 `repro_vector_return_grads.py` | vector-return error 5.14e+00 `WRONG`; scalar-return `OK` | hangs in the first backward (exit 124, #1167) |
| #1056 `repro_mixed_gradness.py` | `first` and `second` abort (exit 134); `both` passes | same as 0.42.0 |
| #1058 `repro_fastmath.py` | runs in all four modes (not timed) | same as 0.42.0 |
| #1058 invalid-option check (below) | invalid option accepted | invalid option accepted |
| #1059 `repro_float3_cuda.py` | runs on CUDA and Vulkan (not timed) | same as 0.42.0 |
| #1167 `repro_runtime_loop_autodiff_hang.py` | `runtime` and `fixed` pass (gradient sum 320) on CUDA and Vulkan | `runtime` hangs (exit 124), `fixed` passes, on CUDA and Vulkan |

A timing-free check for #1058: create a CUDA device with
`compiler_options={"downstream_args": ["--definitely-not-an-nvrtc-option"]}`
and call any function. A wheel that forwards the arguments makes NVRTC reject
the option; 0.42.0 and 0.43.1 compile and run it. In the same check,
`downstream_args=["--use_fast_math"]` returned results bit-identical to the
default, while `floating_point_mode=fast` changed them.

### 2026-09-07 (timings; not re-measured)

These ran against a fresh virtual environment containing exactly
`slangpy==0.43.1` on the L40S:

- #1059: `f4_all` took 0.95 ms; `f3_loop` took 2.76 ms (**2.90x**). This is
  consistent with the compiler fix being absent from the bundled Slang 2026.12.
- #1058: CUDA default took 1.47 ms; CUDA with
  `downstream_args=["--use_fast_math"]` took 1.46 ms; explicit
  `floating_point_mode=fast` took 0.34 ms. Thus downstream arguments were
  ignored by that wheel.

When a newer SlangPy wheel is published, run the named reproducers on a CUDA
machine before changing code or removing a workaround. The timing reproducers
need an otherwise idle GPU. Record the candidate version, command, and result in
this file.
