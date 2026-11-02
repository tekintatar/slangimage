# SlangImage Benchmarks

Protocol v2 — see docs/BENCHMARKING.md for methodology and how to reproduce. Generated 2026-09-13T02:14:04+00:00.

These are protocol v2 measurements, reported under protocol v3: the methodology
is identical and v3 removed one case, whose rows are deleted here (each case is
timed independently, so the other cells are unaffected). Measured with
slangimage 0.1.0.dev20260821.

Re-checked 2026-09-27 with the 0.1.0 release candidate on a second L40S host:
GPU-bound CUDA cells (≥4 MP) were within 3% and the CUDA VRAM table reproduced
exactly; Vulkan and 1 MP cells ran 15–21% slower on that host with the
same library version too (host variation; see BENCHMARKING.md). Since 0.1.0,
creating a CUDA device compiles a one-pixel test kernel, so the CUDA vibrance
fp16 first-call cell below no longer includes one-time session setup.

Steady-state = median of synced iterations after warmup (shader/module caches hot). First-call = module load + slang/backend compile + first dispatch, size-independent. Bare op calls allocate their output per call.

- **metal**: Apple M3 Max — slangimage 0.1.0.dev20260821, slangpy 0.42.0, python 3.12.12, macOS-26.5-arm64-arm-64bit, 15 iters, 2026-09-13T02:13:57+00:00
- **cuda**: NVIDIA L40S, driver 580.126.16 — slangimage 0.1.0.dev20260821, slangpy 0.42.0, python 3.12.3, Linux-6.17.0-14-generic-x86_64-with-glibc2.39, 15 iters, 2026-09-13T02:07:21+00:00
- **vulkan**: NVIDIA L40S — slangimage 0.1.0.dev20260821, slangpy 0.42.0, python 3.12.3, Linux-6.17.0-14-generic-x86_64-with-glibc2.39, 15 iters, 2026-09-13T02:07:35+00:00

## Steady-state (ms, median)

### 1024×1024 (~1.0 MP)

| op | metal fp16 | metal fp32 | cuda fp16 | cuda fp32 | vulkan fp16 | vulkan fp32 |
|---|---|---|---|---|---|---|
| vibrance | 0.89 | 1.16 | 0.30 | 0.47 | 0.67 | 0.90 |
| gaussian_r9 | 1.44 | 1.82 | 0.34 | 0.55 | 0.87 | 1.29 |
| unsharp_mask_r8 | 1.48 | 2.10 | 0.64 | 1.00 | 1.39 | 2.00 |
| surface_blur_r9 | 1.81 | 3.12 | 0.38 | 0.59 | 0.69 | 0.89 |
| kuwahara_r6 | 5.89 | 6.06 | 0.63 | 0.86 | 0.83 | 1.00 |
| resize_half_lanczos3 | 0.99 | 1.30 | 0.39 | 0.51 | 1.05 | 1.22 |

### 2048×2048 (~4.2 MP)

| op | metal fp16 | metal fp32 | cuda fp16 | cuda fp32 | vulkan fp16 | vulkan fp32 |
|---|---|---|---|---|---|---|
| vibrance | 1.82 | 2.44 | 0.86 | 1.45 | 1.22 | 1.95 |
| gaussian_r9 | 2.44 | 4.67 | 1.01 | 1.71 | 2.00 | 3.28 |
| unsharp_mask_r8 | 3.05 | 6.22 | 1.83 | 3.53 | 3.08 | 5.40 |
| surface_blur_r9 | 5.31 | 11.13 | 1.14 | 1.79 | 1.24 | 1.92 |
| kuwahara_r6 | 23.02 | 23.45 | 2.14 | 2.86 | 2.16 | 2.83 |
| resize_half_lanczos3 | 1.40 | 2.53 | 0.81 | 1.46 | 1.55 | 2.29 |

### 3456×3456 (~11.9 MP)

| op | metal fp16 | metal fp32 | cuda fp16 | cuda fp32 | vulkan fp16 | vulkan fp32 |
|---|---|---|---|---|---|---|
| vibrance | 3.82 | 5.88 | 1.91 | 3.13 | 2.54 | 5.47 |
| gaussian_r9 | 7.54 | 14.37 | 2.05 | 4.19 | 4.00 | 7.45 |
| unsharp_mask_r8 | 10.17 | 19.39 | 4.54 | 7.31 | 7.17 | 13.18 |
| surface_blur_r9 | 17.04 | 35.05 | 2.31 | 3.94 | 2.78 | 5.46 |
| kuwahara_r6 | 65.70 | 67.34 | 5.11 | 5.87 | 5.37 | 7.25 |
| resize_half_lanczos3 | 3.72 | 7.95 | 2.02 | 3.70 | 2.92 | 5.01 |

## First call: compile + dispatch (ms)

Note: the first kernel compiled in the process additionally absorbs one-time session init — the first suite entry overstates that op's compile cost.

| op | metal fp16 | metal fp32 | cuda fp16 | cuda fp32 | vulkan fp16 | vulkan fp32 |
|---|---|---|---|---|---|---|
| vibrance | 362 | 132 | 887 | 165 | 793 | 75 |
| gaussian_r9 | 234 | 221 | 289 | 270 | 103 | 80 |
| unsharp_mask_r8 | 118 | 114 | 155 | 143 | 54 | 46 |
| surface_blur_r9 | 126 | 119 | 162 | 148 | 70 | 49 |
| kuwahara_r6 | 161 | 149 | 206 | 171 | 116 | 90 |
| resize_half_lanczos3 | 240 | 228 | 301 | 281 | 128 | 90 |

## CUDA peak VRAM (MB, absolute device usage during steady loop)

| op | 1024px fp16 | 1024px fp32 | 2048px fp16 | 2048px fp32 | 3456px fp16 | 3456px fp32 |
|---|---|---|---|---|---|---|
| vibrance | 1129 | 1536 | 1204 | 1754 | 1406 | 2282 |
| gaussian_r9 | 1137 | 1553 | 1238 | 1821 | 1599 | 2475 |
| unsharp_mask_r8 | 1129 | 1536 | 1204 | 1754 | 1599 | 2475 |
| surface_blur_r9 | 1129 | 1536 | 1238 | 1754 | 1406 | 2282 |
| kuwahara_r6 | 1129 | 1536 | 1204 | 1687 | 1502 | 2089 |
| resize_half_lanczos3 | 1123 | 1523 | 1171 | 1620 | 1360 | 1997 |

Baseline device usage before first sampled loop: ~1129 MB (CUDA context + caches).
