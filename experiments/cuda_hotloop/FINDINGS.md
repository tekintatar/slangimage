# CUDA hot-loop rules (CUDA-vs-Vulkan gap investigation, 2026-07-12)

Same L40S, same Slang kernels: kuwahara ran 1.9x and surface_blur 3.3x
slower on the CUDA backend than on Vulkan. Bisected with the synthetic
kernels in microbench.slang (`python micro_run.py cuda|vulkan`, needs
a machine with both CUDA and Vulkan; 3456^2 fp16). Findings, in order of
discovery:

1. **float3 arithmetic in a hot loop is ~3x on CUDA** — the whole
   surface_blur gap. Identical loops: float4 accumulator 0.96 ms /
   scalar 0.95 ms on BOTH backends; any float3 in the loop -> 2.75 ms
   on CUDA, unchanged on Vulkan. slang emits float3 through a struct
   the NVRTC path keeps in local memory; float4 stays vectorized.
   RULE: float4 or scalars in hot loops, never float3 (or float2).

2. **Dynamically-indexed local arrays go to local memory** on CUDA and
   Vulkan — kuwahara's per-sector accumulator arrays. Named registers
   + a divergence-free branch (the index is loop-uniform): CUDA
   18.4 -> 4.6 ms, Vulkan 8.1 -> 3.9 ms. Metal is the opposite: the
   register form tanks occupancy (62 -> 84 ms), so kuwahara.slang
   selects per target with __target_switch (case metal: arrays).

3. **NVRTC compiles ALL transcendentals as precise software routines**
   (pow measured 4.3x slower than Vulkan's hardware approx; atan2 is a
   huge routine; even exp2f is precise — CUDA has no __exp2f, and the
   __expf intrinsic made no measurable difference once float3 was
   fixed). Neither Slang floating_point_mode=fast nor nvrtc
   --use_fast_math via downstream_args changed the emitted code
   through slangpy 0.42 — UPSTREAM ISSUE CANDIDATE. Mitigations that
   do work: hoist loop-invariant subexpressions, consolidate
   algebraically (surface_blur: 3 exps + 5 divs + 2 sqrts per tap
   reduce to ONE exp — the prefactors cancel in the normalized ratio),
   and replace atan2-based classification with integer comparisons
   (kuwahara_sector: exact match, verified exhaustively to r=64).

Outcome (3456^2 fp16, kuwahara_r6 / surface_blur_r9):
  CUDA   21.1 / 4.6 ms  ->  5.9 / 2.4 ms   (3.6x / 1.9x)
  Vulkan 12.7 / 3.4 ms  ->  6.8 / 3.4 ms   (1.9x / ~1x)
  Metal  61.4 / 18.7 ms -> 66.0 / 17.4 ms  (-7% / +7% — kuwahara keeps
  its array form; the small loss comes with the shared restructuring)
CUDA is now the fastest backend for every op in the suite except
resize; the old "Vulkan 2x faster on chains" advice is obsolete.

Measurement trap that cost an hour: SLANGIMAGE_BACKEND only steers the
TEST suite (conftest) — the library ignores it. Ad-hoc scripts must
create the device explicitly or they silently measure CUDA twice.

## Sweep addendum (same day): two more rules

The float3 sweep of the remaining neighborhood kernels (bilateral,
nl_means, sharpen_luma, resize-lanczos, gaussians)
surfaced two more NVRTC-path rules, bisected via sharpen_rgb (stuck at
3.7 ms while an "identical" clone ran 1.1):

4. **float3 ANYWHERE in a hot kernel poisons codegen — not just in
   loops.** A single once-per-pixel `dst = float4(x.rgb, s.a)`
   swizzle-constructor in the epilogue kept sharpen_rgb at 2.9 ms;
   replacing it with lane assignment (`x.a = s.a; dst = x;`) dropped it
   to 1.1. The materialized float3 temporary appears to force the whole
   downstream dataflow through local memory. RULE: kernels touching
   big loops use float4/scalars everywhere, epilogues included.

5. **[unroll] on runtime trip counts** and **per-tap fdiv** are worth
   ~20-25% each on NVRTC (hoist reciprocals; drop the attribute unless
   the bound is compile-time).

Sweep results (3456^2 fp16, CUDA before -> after / Vulkan after):
  bilateral_r5    4.42 -> 1.97 / 1.76     nl_means_s3  31.2 -> 10.4 / 10.5
  sharpen_rgb_r4  3.66 -> 1.08 / 1.06     bilateral_1d 0.51 -> 0.34 / 0.35
  gaussians: already clean
Vulkan gained on several too (the fixes are portable). pointillize was
already at parity (small fixed 3x3 loop) and was left alone.

## Correction (same day, experiments/nvrtc_fastmath)

Rule 3 was half wrong — confounded by the float3 bottleneck in the
kernels used to test it. On a clean pow-chain kernel,
`floating_point_mode = fast` DOES reach NVRTC: pow 1.49 -> 0.34 ms
(4.4x, faster than Vulkan's default 0.75). What IS broken upstream:
`downstream_args = ["--use_fast_math"]` is silently ignored, and the
cross-backend DEFAULTS are inconsistent (Vulkan/Metal drivers compile
fast approximations by default; NVRTC compiles precise) — FILED as
slangpy#1058; the float3 penalty (rules 1 and 4) as slangpy#1059. slangimage now
sets floating_point_mode=fast on its devices (runtime/device.py) —
with it, kuwahara CUDA lands at 3.85 ms vs Vulkan 3.77: every gap in
this investigation is fully closed (kuwahara started at 21.1).

## Upstream status (2026-09-27)

- slangpy#1059 (rules 1 and 4) was closed after the Slang fix
  [slang#12078](https://github.com/shader-slang/slang/pull/12078) merged on
  2026-07-14. Upstream's analysis
  ([slang#12073](https://github.com/shader-slang/slang/issues/12073)) differs
  from the struct explanation in rule 1: Slang's C++/CUDA output expanded a
  multi-component swizzle such as `.rgb` into one expression per component,
  re-evaluating its base (for example a texture fetch) each time. SlangPy
  0.43.x bundles Slang 2026.12, which predates the fix, so the rules still apply.
- slangpy#1058 is open. SlangPy
  [#1095](https://github.com/shader-slang/slangpy/pull/1095), merged 2026-08-07
  but not yet released, forwards `downstream_args` to NVRTC.

## Running micro_run.py

`micro_run.py` needs SlangImage installed; it loads `microbench.slang` from
its own directory:

```bash
python micro_run.py cuda     # or: vulkan
```

The timings are meaningful only on an otherwise idle GPU.
