# [Perf] CUDA target defaults to precise transcendentals — 4x slower than Vulkan's default on the same GPU; downstream_args --use_fast_math silently ignored

> FILED: https://github.com/shader-slang/slangpy/issues/1058 (2026-07-12).
> Repro: repro_fastmath.py (verified 2026-07-12, L40S).
>
> **Status 2026-09-27:** open. SlangPy
> [#1095](https://github.com/shader-slang/slangpy/pull/1095), merged 2026-08-07
> but not yet released, forwards `downstream_args` to NVRTC; SlangPy 0.42.0 and
> 0.43.1 still drop them. The title's "4x" compares CUDA default with CUDA
> `floating_point_mode = fast`; CUDA default is about 2x slower than the Vulkan
> default. The timings below need an otherwise idle GPU. Current status:
> [`docs/UPSTREAM_ISSUES.md`](../../docs/UPSTREAM_ISSUES.md).

## Summary

The same Slang source compiles to very different transcendental
implementations per target, with no warning:

- **Vulkan / Metal**: drivers compile `exp` / `pow` to fast hardware
  approximations **by default**.
- **CUDA (NVRTC)**: `exp` / `pow` (and `atan2`, even `exp2`) compile to
  precise software routines by default — a pow-chain kernel runs **4.4x
  slower** than with fast math, and 2x slower than the Vulkan *default*
  on the same physical GPU.

`SlangCompilerOptions.floating_point_mode = fast` fixes it (and is how
we resolved a persistent CUDA-vs-Vulkan benchmark gap), so this is
partly a documentation/defaults report:

1. The cross-backend default inconsistency is surprising: identical
   Slang code is "fast math" on two backends and "precise math" on the
   third. A note in the docs (or aligned defaults) would save users a
   long investigation — kernels dense in transcendentals mysteriously
   lose to Vulkan on the same GPU.
2. `downstream_args = ["--use_fast_math"]` is accepted without error
   but has **no effect** on the generated CUDA code or timing — either
   it should reach NVRTC or it should be rejected.

## Environment

- slangpy **0.42.0** (NVRTC 13.0 via CUDA 13 toolchain)
- Ubuntu 24.04, NVIDIA L40S, driver 580.126.16
- Python 3.12

## Reproduction

`python repro_fastmath.py <backend> <mode>` — a 64-iteration dependent
`pow` chain per pixel at 3456^2.

Observed output:

```text
cuda/default   (NVIDIA L40S): pow x64 chain at 3456^2 ->  1.49 ms
cuda/fast      (NVIDIA L40S): pow x64 chain at 3456^2 ->  0.34 ms
cuda/nvrtcfast (NVIDIA L40S): pow x64 chain at 3456^2 ->  1.50 ms   <- --use_fast_math ignored
vulkan/default (NVIDIA L40S): pow x64 chain at 3456^2 ->  0.75 ms
```

<details>
<summary>Full repro script (repro_fastmath.py)</summary>

```python
# Minimal repro: transcendentals on the CUDA backend default to precise
# software routines — 2x slower than Vulkan's DEFAULT (hardware approx)
# on the same GPU, 4.4x slower than CUDA with floating_point_mode=fast
# (which works). downstream_args = ["--use_fast_math"] is accepted but
# silently changes nothing.
#
#   python repro_fastmath.py cuda default
#   python repro_fastmath.py cuda fast        # floating_point_mode
#   python repro_fastmath.py cuda nvrtcfast   # downstream --use_fast_math
#   python repro_fastmath.py vulkan default   # reference: hardware approx
#
# slangpy 0.42.0.

import sys
import time

import numpy as np
import slangpy as spy

SHADER = """
void pow_loop(uint2 tid, Texture2D<float4> src,
              [format("rgba16f")] RWTexture2D<float4> dst, int iters)
{
    uint w, h; src.GetDimensions(w, h);
    if (tid.x >= w || tid.y >= h) return;
    float acc = src[tid].x + 0.5;
    for (int i = 0; i < iters; ++i)
        acc = pow(abs(acc) + 0.5, 1.7) * 0.31 + src[tid].y * 1e-3;
    dst[tid] = float4(acc, 0.0, 0.0, 1.0);
}
"""

S, N, ITERS = 3456, 20, 64


def main() -> None:
    backend = sys.argv[1] if len(sys.argv) > 1 else "cuda"
    mode = sys.argv[2] if len(sys.argv) > 2 else "default"

    opts: dict = {"include_paths": [spy.SHADER_PATH]}
    if mode == "fast":
        opts["floating_point_mode"] = spy.SlangFloatingPointMode.fast
    elif mode == "nvrtcfast":
        opts["downstream_args"] = ["--use_fast_math"]
    device = spy.Device(type=getattr(spy.DeviceType, backend),
                        compiler_options=opts)
    module = spy.Module.load_from_source(device, "fastmath_repro", SHADER)

    rng = np.random.default_rng(3)
    usage = spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access
    src = device.create_texture(width=S, height=S, format=spy.Format.rgba16_float,
                                usage=usage,
                                data=rng.random((S, S, 4)).astype(np.float16))
    dst = device.create_texture(width=S, height=S, format=spy.Format.rgba16_float,
                                usage=usage)
    grid = spy.grid((S, S))

    module.pow_loop(grid, src, dst, ITERS)
    device.wait_for_idle()
    t0 = time.perf_counter()
    for _ in range(N):
        module.pow_loop(grid, src, dst, ITERS)
    device.wait_for_idle()
    ms = (time.perf_counter() - t0) * 1000 / N
    print(f"{backend}/{mode} ({device.info.adapter_name}): "
          f"pow x{ITERS} chain at {S}^2 -> {ms:6.2f} ms")


if __name__ == "__main__":
    main()
```

</details>

- CUDA default is 2x slower than the Vulkan default and 4.4x slower
  than CUDA with `floating_point_mode = fast`.
- `downstream_args = ["--use_fast_math"]` produces timing identical to
  the default (1.50 vs 1.49 ms).

## Expected

Either consistent cross-target defaults for transcendental precision,
or a prominent note on the CUDA target's behavior; and
`downstream_args` that don't reach the downstream compiler should
error rather than silently no-op.

## Workaround (for anyone else hitting this)

Set `floating_point_mode = spy.SlangFloatingPointMode.fast` in the
device's compiler options for the CUDA backend — this matches what the
Vulkan and Metal drivers already do to the same source, so precision
behavior becomes consistent across backends rather than looser.
