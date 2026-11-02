# Benchmarking SlangImage

SlangImage ships a standard benchmark suite so results are reproducible and
comparable across machines, backends, and versions. Published numbers live in
[BENCHMARKS.md](BENCHMARKS.md).

## Quick start

```bash
pip install "slangimage[bench]"        # [bench] adds NVML for CUDA VRAM tracking

slangimage-bench run --backend cuda --out results_cuda.json      # one run per backend:
slangimage-bench run --backend vulkan --out results_vulkan.json  # metal on macOS
slangimage-bench report results_*.json -o BENCHMARKS.md
```

One `run` per backend; `report` merges one result file for each backend into one
table. Strict reporting rejects duplicate backends and differences in protocol,
SlangImage/SlangPy versions, iteration count, or suite cells before merging.
`--allow-incompatible` is an explicit escape hatch for exploratory reports.
`--iters` and `--sizes` exist for quick local iteration, but published results
must use the defaults.

## Protocol

Every result JSON carries `meta.protocol` (currently **v3**). The protocol
pins the op suite, parameters, image sizes, and timing methodology. **Numbers
are only comparable within the same protocol version.** Changing ops, params,
sizes, or methodology requires bumping `PROTOCOL_VERSION` in
`slangimage/bench/__init__.py`.

Protocol history:
- **v3** (2026-11): one case removed from the suite; the remaining cases,
  parameters, sizes and methodology are unchanged. The published tables are
  the v2 measurements with that case's rows deleted (no re-run; each case is
  timed independently, so the other cells are unaffected).
- **v2** (2026-07): one effect-chain case replaced; all other cases
  unchanged, so per-op rows remain comparable with v1.
- **v1** (2026-07): initial suite.

### What is measured (protocol v3)

- **Input**: deterministic synthetic RGBA image (fixed RNG seed), sizes
  1024² (~1 MP), 2048² (~4.2 MP), 3456² (~11.9 MP), at fp16 and fp32
  texture precision. Upload is outside all timings.
- **First call** (per op × precision): wall time of the first invocation =
  module load + Slang/backend compile (NVRTC on CUDA) + first dispatch.
  Size-independent; measured once at the smallest size. The very first
  kernel in the process also absorbs one-time session init — except on CUDA
  since 0.1.0, where device creation (untimed) compiles and runs a one-pixel
  test kernel and absorbs it instead. The published CUDA first-call row
  predates that change.
- **Steady state**: after an additional untimed warmup at each size, the
  median of N=15 iterations, each synced with `device.wait_for_idle()`.
  Sync-per-iteration measures honest per-op wall time (no pipelining).
  Whatever allocation behavior the library ships is intentionally included.
  The standard suite instantiates low-level op classes, whose raw output
  textures are not automatically recycled; this measures one output allocation
  per call. Public functional calls returning `Image` handles have a
  different service lifecycle: dropped outputs can auto-recycle.
- **Peak VRAM** (CUDA only): NVML device-memory sampler (5 ms interval)
  during the steady loop; absolute usage plus the pre-loop baseline.
  Note this is device-wide, so run on an otherwise idle GPU.
  With the texture pool active, per-op peaks converge to roughly
  *CUDA context + pool free-list residency + active working set* — they
  measure cumulative pool fill, not one op's footprint. The pool budget
  is the lever: `SLANGIMAGE_POOL_BUDGET_MB` (default 1536) bounds
  free-list bytes, with a one-current-hot-texture floor so an oversized texture
  can still be reused. Thus cached residency is bounded by the larger of the
  configured budget and one texture, separate from live working data and driver/
  compiler caches.

### Op suite (protocol v3)

| case | params |
|---|---|
| vibrance | amount 0.6 |
| gaussian_r9 | radius 9, sigma 3.0 |
| unsharp_mask_r8 | amount 1.5, radius 8, sigma 3.0 |
| surface_blur_r9 | radius 9, bsigma 0.1 |
| kuwahara_r6 | radius 6 |
| resize_half_lanczos3 | 50% downscale |

## Rules for published results

1. Defaults only (15 iters, standard sizes) — flags are for local iteration.
2. Idle GPU: no other compute/graphics workload during the run.
3. Report the JSON, not just the table — `meta` captures adapter, driver,
   OS, python, slangpy/slangimage versions, timestamp.
4. Don't compare across protocol versions or mixed library versions.
5. AC power / no thermal throttling for laptops (run twice, keep the second).

### Comparing across machines

Published tables come from one host per backend. A 2026-09-27 re-run on a
second L40S host (same GPU, driver, kernel and Python) reproduced the
GPU-bound CUDA cells (≥4 MP) within 3% and the CUDA VRAM table exactly, while
Vulkan cells, ≤1 MP cells, first-call compile times and sub-millisecond
comparison cells differed by 15–45% — with the same library version as
well, so the difference is the host, not the code. Compare numbers within one
host. Some 3456² CUDA cells are bimodal across processes (per-call output
allocation), so compare medians over at least three processes rather than one
run. The CUDA VRAM column is NVML device memory in use, which includes about
0.6 GB that the driver reserves even with no process running (L40S, ECC on).

The standard table is not an end-to-end request benchmark: upload/readback,
encoding, network work and unsynchronized pipelining are excluded. The
standalone comparison scripts likewise report GPU-resident operator costs and
document where border, color-model or filtering semantics are only analogous.

## Regenerating the reference results

- Reference results live in `benchmarks/results_{metal,cuda,vulkan}.json`;
  regenerate the published table from a source checkout with
  `slangimage-bench report benchmarks/results_*.json -o docs/BENCHMARKS.md`.
- The Metal reference run used an Apple M3 Max; the CUDA/Vulkan reference runs
  used an NVIDIA L40S Linux host prepared with the helper scripts in
  `scripts/cuda_box/` (they assume Ubuntu 24.04 with an NVIDIA driver installed
  and SSH access to the host):
  `SLANGIMAGE_CUDA_HOST=<ssh-host> ./scripts/cuda_box/deploy.sh`, then
  `SLANGIMAGE_CUDA_HOST=<ssh-host> ./scripts/cuda_box/run_bench.sh`.
  `run_bench.sh` copies the result JSONs back and regenerates the table with
  the local `python3` on your `PATH` (set `PYTHON` to use another interpreter
  with SlangImage installed).
