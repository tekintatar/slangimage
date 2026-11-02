# D0 findings — differentiable ops spike (2026-07-12, CUDA L40S)

> **Status 2026-09-27:** this is a dated log of the July spike ("D0" in
> [`docs/ROADMAP.md`](../../docs/ROADMAP.md)). Of the bugs below, slangpy#1051
> and #1055 are closed upstream, but their Slang compiler fixes are not in any
> released SlangPy wheel; #1056 is open. Upstream triage refined the #1051
> trigger to any runtime (non-constant) loop start, not only a negative one.
> Current status: [`docs/UPSTREAM_ISSUES.md`](../../docs/UPSTREAM_ISSUES.md).

**Verdict: differentiable slangimage is viable today.** Working proof in
`manual_bridge.py`: gaussian through `torch.autograd` with gradients
verified against a NumPy reference (2.98e-07), closed-form transpose
(exact), finite differences incl. edges, and `torch.autograd.gradcheck`.
fwd+bwd at 1024²/r9: **0.59 ms**.

## The D1 architecture (proven here)

`torch.autograd.Function` bridging two NORMAL slangpy dispatches:
forward kernel + explicit backward kernel (hand-written gather — no
atomics, no scatter). Uses the same production call path as `si.ops`.
slangpy's `create_torch_device` + torch-tensor marshalling works well
for this (values verified exact both directions).

## Upstream bug discovered — root cause isolated

Backward dispatch **segfaults when the [Differentiable] function contains
a loop with a negative runtime start value** (`for (int dx = -radius;
...)`), the canonical convolution form. Reproduced on Metal, CUDA and
Vulkan; pure-Tensor and torch paths alike; forward always correct.
(Earlier "generator argument" theory was a confound — generators are
fine.) Filed as slangpy#1051; triaged **valid, P1** — root cause confirmed in
the Slang compiler's reverse-mode autodiff (loop replay uses a synthetic
zero-based counter, mishandles negative start offsets in checkpoint/
replay indexing); escalated to the Slang compiler repo. Our zero-based
workaround matches the internal replay form exactly. Re-verified
reproducing on all 3 backends incl. the cu13 toolchain.
Full bisection matrix + report: `UPSTREAM_ISSUE.md`;
self-contained repro: `repro_bwds_segfault.py`.

**Workaround (verified)**: zero-based loops with an offset
(`for i in 0..2r+1; dx = i - r`) make BOTH paths work — .bwds() and full
torch autograd produce float32-exact gradients. So the automatic
Slang-generated backward is usable in D1 after all; hand-written gather
backwards remain the perf optimization, not a necessity.

## Conventions learned (cost hours; now pinned)

- `spy.grid((H, W))` yields `pix.x = column` (fastest dim), `pix.y = row`.
- `Tensor<T,2>` subscripts are ALSO graphics-order: `t[uint2(x=col, y=row)]`
  — consistent with grid, opposite of numpy dim-order intuition.
- Differentiable slang needs: `import slangpy;` (ITensor/IDiffTensor),
  `[MaxIters(n)]` on loops, `no_diff` on non-diff params (not on calls).
- `slangpy-torch` native extension requires a full CUDA toolkit at build
  time (`CUDA_HOME`; apt nvidia-cuda-toolkit works) — the pip NVRTC shim
  covers runtime only. The extension was then built on the test machine.

## Files

- `diff_gaussian.slang` — [Differentiable] forward variants + hand-written
  backward kernel (`gaussian_h_bwd`)
- `manual_bridge.py` — the working D1-pattern proof (all checks PASS)
- `spike.py` — original staged spike; stage 2 backward is the upstream
  segfault repro

## Wave-2 additions (2026-07-12): two more autodiff bugs, both worked around

- **Bug #3 — vector return + loop → silently WRONG gradients.** Identical
  kernel body: scalar return gives exact gradients (1.4e-07 vs finite
  differences); `float3` return gives garbage (errors ~1.4, sign flips).
  Trigger requires BOTH a vector return type and a loop in the function
  (LUT: float3 return, no loop → correct; gaussian: loop, scalar → correct).
  Workaround: scalar-return kernels over an (H,W,C) grid — bilateral reads
  all channels per tap for the joint weight but accumulates/returns only
  its own channel.
- **Bug #4 — mixed grad-ness across tensor params → CUDA illegal memory
  access.** A [Differentiable] function with two IDiffTensor params crashes
  in backward when only one input requires grad (generated backward
  scatters into the absent grad buffer). Both-grad works. Workaround in
  diff.py `_align_grads`: no-grad inputs get a throwaway detached
  `requires_grad_(True)` alias; torch discards the unused leaf grads.

Both FILED 2026-07-12: bug #3 = slangpy#1055, bug #4 = slangpy#1056.
experiments/vector_return_grads/ and experiments/mixed_gradness_crash/
hold the verified minimal repros + issue text. Note: #1056 is NOT fixed
by PR #1054 (the #1052 cache-keying fix) — it crashes on the first call
in a fresh process; see the issue's "not a duplicate" section.
