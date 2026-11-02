# Experiments

Standalone investigations kept for provenance. They are not part of the
installed package or the test suite. Most directories hold a reproducer for a
bug reported upstream to SlangPy or Slang. Each `UPSTREAM_ISSUE.md` is a copy of
the report as it was filed; its code blocks call the script `repro.py`, which is
the script in the same directory. Upstream status, and the SlangImage
workaround each bug requires, are tracked in
[`docs/UPSTREAM_ISSUES.md`](../docs/UPSTREAM_ISSUES.md).

## Prerequisites

- A GPU backend: CUDA or Vulkan on Linux, Metal on macOS. CUDA also needs
  NVRTC at run time; see "CUDA on Linux: install NVRTC" in the
  [README](../README.md).
- `pip install numpy "slangpy==0.42.0"`. The reports were filed against SlangPy
  0.42.0; on 0.43.x two reproducers show a different symptom (see below).
- The PyTorch reproducers (`cache_poisoning`, `mixed_gradness_crash`,
  `vector_return_grads`, and `spike.py` / `manual_bridge.py` in
  `d0_diff_gaussian`) also need an NVIDIA GPU, PyTorch 2.13 and SlangPy's torch
  bridge (`slangpy[torch]`, which compiles `slangpy-torch` against the installed
  PyTorch; see the README's PyTorch extra section). Without the compiled bridge,
  set `SLANGPY_ALLOW_TORCH_FALLBACK=1` to use SlangPy's slower Python bridge;
  the bugs reproduce with either bridge.
- `cuda_hotloop/micro_run.py` also needs SlangImage installed.

Run each script from its own directory, one case per process: a crash or a
CUDA error leaves the process unusable. Run hang reproducers under `timeout`.
When redirecting output to a file, use `python -u`, because a crashing process
loses buffered output.

## Index

| Directory | Upstream issue | What it shows | Command | Bug present | Bug fixed |
|---|---|---|---|---|---|
| `cache_poisoning/` | SlangPy [#1052](https://github.com/shader-slang/slangpy/issues/1052) | A no-grad call silently drops the autograd hook from later grad-requiring calls to the same function. PyTorch. | `python repro_cache_poisoning.py`; control: `python repro_cache_poisoning.py control` | `backward FAILED: ...`, exit 1 | `backward OK, x.grad = [2.0, 4.0, 6.0]` |
| `mixed_gradness_crash/` | SlangPy [#1056](https://github.com/shader-slang/slangpy/issues/1056) | Backward aborts when only one of two tensor inputs requires grad. PyTorch. | `python repro_mixed_gradness.py first` (or `second`); control: `both` | `CUDA error: an illegal memory access was encountered`, then abort (exit 134) | `backward OK, gradients exact` |
| `vector_return_grads/` | SlangPy [#1055](https://github.com/shader-slang/slangpy/issues/1055), Slang [#12071](https://github.com/shader-slang/slang/issues/12071) | A `[Differentiable]` function with a loop and a `float3` return gets silently wrong gradients. PyTorch. | `python repro_vector_return_grads.py` | last line ends in `WRONG` | both final lines end in `OK` |
| `d0_diff_gaussian/` | SlangPy [#1051](https://github.com/shader-slang/slangpy/issues/1051), Slang [#12070](https://github.com/shader-slang/slang/issues/12070) | Generated backward crashes when a differentiable loop starts at a runtime value. The directory also holds the first differentiable-op spike (`spike.py`, `manual_bridge.py`, PyTorch). | `python repro_bwds_segfault.py`; control: `python repro_bwds_segfault.py control`. A second argument picks the backend (`repro vulkan`); the default is Metal on macOS and CUDA elsewhere. | `calling bwds...`, then SIGSEGV (exit 139) | `bwds ok, grad sum: 320.0` |
| `runtime_loop_autodiff_hang/` | SlangPy [#1167](https://github.com/shader-slang/slangpy/issues/1167), Slang [#13169](https://github.com/shader-slang/slang/issues/13169) | SlangPy 0.43.x never finishes compiling the backward of a runtime-count loop around a tensor load. No PyTorch; needs `slangpy==0.43.1` to see the bug. | `timeout 15s python repro_runtime_loop_autodiff_hang.py runtime cuda` (or `vulkan`); control: `fixed` | no output after `calling bwds()`; `timeout` exits 124 | `bwds ok: grad_sum=320.0` |
| `nvrtc_fastmath/` | SlangPy [#1058](https://github.com/shader-slang/slangpy/issues/1058) | CUDA compiles transcendentals precisely by default, and `downstream_args` never reach NVRTC. Timing only. | `python repro_fastmath.py cuda default`, then `cuda fast`, `cuda nvrtcfast`, `vulkan default` | `nvrtcfast` about as slow as `default` | `nvrtcfast` close to `fast` |
| `float3_cuda_penalty/` | SlangPy [#1059](https://github.com/shader-slang/slangpy/issues/1059), Slang [#12073](https://github.com/shader-slang/slang/issues/12073) | In CUDA output, a `.rgb`-style swizzle re-evaluates its base expression once per component, so loops built on it run about 3x slower. Timing only. | `python repro_float3_cuda.py cuda`; reference: `python repro_float3_cuda.py vulkan` | CUDA `f3_loop` about 3x `f4_all` | CUDA `f3_loop` close to `f4_all` |
| `cuda_hotloop/` | background for #1058 and #1059 | The CUDA-versus-Vulkan investigation behind the hot-loop rules in its `FINDINGS.md`. Timing only; no pass/fail. | `python micro_run.py cuda` (or `vulkan`) | — | — |

Timing reproducers print milliseconds and exit 0 either way. Their numbers mean
something only on an otherwise idle GPU; compare the variants within one run.
`docs/UPSTREAM_ISSUES.md` describes a timing-free check for #1058.

On SlangPy 0.43.x (checked 2026-09-27), `repro_bwds_segfault.py` fails with a
backward compile error (exit 1) instead of SIGSEGV and its `control` variant
hangs, and `repro_vector_return_grads.py` hangs in backward. Both hangs are
#1167. Use SlangPy 0.42.0 for those two reproducers.
