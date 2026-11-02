# Slang 2026.12 runtime-loop autodiff hang

Investigated **2026-09-17** on an NVIDIA L40S (driver 580.126.16),
Python 3.12.3 and CUDA 13.0. The standalone reproducer is
[`repro_runtime_loop_autodiff_hang.py`](repro_runtime_loop_autodiff_hang.py).

## Finding

SlangPy 0.43.0 and 0.43.1 do not generally break CUDA or autograd. Their
bundled **Slang 2026.12** compiler hangs while generating the reverse-mode
derivative of a differentiable tensor load inside runtime-dependent control
flow. The forward call completes. The direct `Function.bwds()` call then
remains inside the native backward path until the process is terminated; the
2026-09-27 stack samples below place it in shader compilation, before any GPU
dispatch.

This is a compiler regression relative to Slang 2026.5.2, not a PyTorch or
`slangpy-torch` failure:

- the minimal failure uses SlangPy's direct tensor and `.bwds()` APIs without
  importing PyTorch;
- the Python autograd hook is byte-identical in SlangPy 0.42.0 and 0.43.1;
- a controlled SlangPy 0.43.0 build linked against Slang 2026.5.2 completes
  the same runtime-count backward and returns the correct gradient.

## Controlled matrix

Every hanging case used `timeout -k 2s 12s`; exit 124 means the call did not
return before the bound.

| SlangPy | Bundled Slang | Case | Result |
|---|---|---|---|
| 0.42.0 | 2026.5.2 | runtime-count loop | pass, gradient sum 320 |
| 0.43.0 | 2026.12 | runtime-count loop | **hang**, exit 124 |
| 0.43.1 | 2026.12 | runtime-count loop | **hang**, exit 124 |
| 0.43.1 | 2026.12 | fixed-count loop control | pass, gradient sum 320 |
| 0.43.0 hybrid | 2026.5.2 | runtime-count loop | pass, gradient sum 320 |

The 0.42 runtime case, 0.43.1 runtime/fixed cases and hybrid runtime case had
the same result on CUDA and Vulkan. The stock 0.43.0 row was checked on CUDA.

The hybrid was built from the official SlangPy `v0.43.0` tag (commit
`e1927ed90ef5e856bc4835ad5117d81840ce5c77`) with
`CMAKE_ARGS=-DSGL_SLANG_VERSION=2026.5.2`.
Crashpad was omitted from that experimental build because it is disabled by
the build configuration and unrelated to shader compilation or dispatch.

The installed native library names independently confirm the compiler matrix:

```text
slangpy 0.42.0          libslang-compiler.so.0.2026.5.2
slangpy 0.43.0          libslang-compiler.so.0.2026.12
slangpy 0.43.1          libslang-compiler.so.0.2026.12
slangpy 0.43.0 hybrid   libslang-compiler.so.0.2026.5.2
```

## Minimal boundary

The following 0.43.1 controls passed:

- no loop around the differentiable tensor load;
- a statically fixed loop count, including a dynamically indexed load;
- a runtime loop containing only scalar/non-differentiable work, followed by a
  differentiable tensor load outside the loop.

The following 0.43.1 cases hung:

- a runtime-count loop containing a differentiable tensor load;
- a runtime-count loop repeatedly loading the same tensor element;
- a fixed maximum loop with a runtime `if` guard around the load;
- a fixed maximum loop with a runtime `break` before the load.

The problem is therefore dynamic control flow around the differentiable load,
not merely dynamic indexing or a negative loop start. It is distinct from the
older zero-based-loop workaround tracked as
[SlangPy #1051](https://github.com/shader-slang/slangpy/issues/1051) /
[Slang #12070](https://github.com/shader-slang/slang/issues/12070).

## SlangImage impact

The generated-backward paths for `slangimage.diff.gaussian_blur` and
`unsharp_mask` are affected because their differentiable Gaussian passes use
runtime-count loops. The matching `slangimage.nn` wrappers and composed graphs
containing those operations inherit the hang. A full autograd selection reaches
`test_gaussian_backward_closed_form` and stops there.

Direct checks on SlangPy 0.43.1 showed that generated backward still completes
for exposure, sepia, hue rotation, blend and LUT. The manual backward paths for
bilateral and resize also complete. Base texture upload, dispatch and readback
remain healthy. This is why SlangPy 0.43.1 remains valid for the non-Torch base
install while the `torch` extra remains pinned to the validated SlangPy 0.42.0
stack.

As a repository-level confirmation, the unchanged Gaussian/unsharp selection
from `test_diff.py`, `test_nn.py` and `test_diff_core_contracts.py` passed all
**22 tests** under the SlangPy 0.43.0/Slang 2026.5.2 hybrid. The same 0.43 code
therefore works for the real SlangImage paths when only the compiler is changed.

No production workaround was added. Replacing Gaussian's generated derivative
with a manual backward would duplicate a compiler responsibility and is not
justified while the package can select the known-good dependency. Before
relaxing the pin, rerun this reproducer against a released wheel and then run
the complete CUDA tensor suite.

No matching prior upstream report was found during this investigation. The
report was filed as
[SlangPy #1167](https://github.com/shader-slang/slangpy/issues/1167).
Upstream escalated it to
[shader-slang/slang#13169](https://github.com/shader-slang/slang/issues/13169);
both were open on 2026-09-27.

## Re-check on 2026-09-27

Published wheels on an NVIDIA L40S (driver 580.126.16), every run bounded by
`timeout 120`:

| SlangPy | `runtime` case | `fixed` case |
|---|---|---|
| 0.42.0 | pass, gradient sum 320 (CUDA and Vulkan) | pass (CUDA and Vulkan) |
| 0.43.1 | **hang**, exit 124 (CUDA and Vulkan) | pass (CUDA and Vulkan) |

Three `py-spy dump --native` samples of the stuck 0.43.1 CUDA process, at about
15, 25 and 35 seconds, all showed the main thread inside
`libslang-compiler.so.0.2026.12` (`linkAndOptimizeIR` → `specializeModule`),
called from `ComponentType::getEntryPointCode` while slang-rhi compiled the
backward pipeline. The process used about 100% of one CPU core and the GPU was
idle. The same frames appeared when sampling the hangs of the zero-based
`control` variant in `../d0_diff_gaussian/` and of
`../vector_return_grads/repro_vector_return_grads.py` on 0.43.1.
