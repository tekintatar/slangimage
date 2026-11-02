# [Perf] float3 arithmetic in loops makes CUDA kernels ~3x slower than identical float4 code (CUDA target only)

> FILED: https://github.com/shader-slang/slangpy/issues/1059 (2026-07-12).
> Repro: repro_float3_cuda.py (verified 2026-07-12, L40S). Likely a Slang
> CUDA-emission issue rather than slangpy's — filed there as the
> integration surface; may transfer to shader-slang/slang.
>
> **Status 2026-09-27:** closed upstream, fixed in the Slang compiler by
> [shader-slang/slang#12078](https://github.com/shader-slang/slang/pull/12078)
> (Slang issue [#12073](https://github.com/shader-slang/slang/issues/12073)).
> No released SlangPy wheel contains the fix: SlangPy 0.43.x bundles Slang
> 2026.12, which predates it. Upstream found a different root cause from the
> struct hypothesis below: in C++/CUDA output a multi-component swizzle such as
> `.rgb` re-evaluates its base expression once per component, so a folded
> texture fetch runs three times. The timings below need an otherwise idle GPU.
> Current status: [`docs/UPSTREAM_ISSUES.md`](../../docs/UPSTREAM_ISSUES.md).

## Summary

A compute kernel whose hot loop accumulates in `float3` runs **~2.9x
slower on the CUDA backend** than the mathematically identical kernel
written with a `float4` accumulator (or plain scalars). On Vulkan, all
variants run at the same speed — the penalty is specific to the CUDA
emission path. The generated CUDA C++ routes vector-of-3 values through
a struct that NVRTC does not keep in registers, while float4 stays
vectorized.

float3 is the natural type for image-processing math (`tex[q].rgb` is
everywhere in shader code), so this silently costs most naive ports ~3x
on CUDA. We found it because the same Slang kernels benchmarked 2-3x
slower on CUDA than on Vulkan on the same physical GPU (L40S) — after
rewriting our neighborhood kernels to float4/scalars, CUDA reached
parity or better on every kernel.

## Environment

- slangpy **0.42.0** (NVRTC 13.0 via CUDA 13 toolchain)
- Ubuntu 24.04, NVIDIA L40S, driver 580.126.16
- Python 3.12

## Reproduction

`python repro_float3_cuda.py cuda` / `... vulkan` — one 81-tap
neighborhood loop over a 3456^2 rgba16f texture, three variants with
identical math (weight 0.5 per tap, normalized at the end).

Observed output:

```text
cuda (NVIDIA L40S), 3456^2, 81 taps, median-ish of 20:
  f4_all  (float4 everywhere)        0.96 ms
  f3_loop (float3 in the loop)       2.76 ms  (2.87x)
  f3_epi  (float4 loop, float3 only in the final write)  0.95 ms

vulkan (NVIDIA L40S), 3456^2, 81 taps, median-ish of 20:
  f4_all  (float4 everywhere)        1.09 ms
  f3_loop (float3 in the loop)       0.95 ms  (0.87x — even slightly
                                     faster than float4 on Vulkan)
  f3_epi                             0.95 ms
```

<details>
<summary>Full repro script (repro_float3_cuda.py)</summary>

```python
# Minimal repro: float3 arithmetic makes CUDA kernels ~3x slower than the
# IDENTICAL math written with float4 — on the CUDA backend only (Vulkan
# runs all variants at the same speed).
#
#   python repro_float3_cuda.py cuda
#   python repro_float3_cuda.py vulkan   # control: all variants equal
#
# Same 81-tap neighborhood loop, three ways:
#   f4_all   — float4 accumulator and reads              (fast everywhere)
#   f3_loop  — float3 accumulator + .rgb reads in loop   (~2.9x on CUDA)
#   f3_epi   — float4 loop, float4(acc.rgb/z, 1.0) write (parity HERE —
#              but the same epilogue pattern cost 2.6x in a larger
#              kernel in this library with a helper call; the boundary is not
#              fully characterized, see issue text)
#
# slangpy 0.42.0. Works on any GPU with both backends (tested: L40S).

import sys
import time

import numpy as np
import slangpy as spy

SHADER = """
void f4_all(uint2 tid, Texture2D<float4> src,
            [format("rgba16f")] RWTexture2D<float4> dst, int radius)
{
    uint w, h; src.GetDimensions(w, h);
    if (tid.x >= w || tid.y >= h) return;
    float4 acc = float4(0.0);
    float z = 0.0;
    for (int j = -radius; j <= radius; ++j)
        for (int i = -radius; i <= radius; ++i)
        {
            uint2 q = uint2(clamp(int(tid.x) + i, 0, int(w) - 1),
                            clamp(int(tid.y) + j, 0, int(h) - 1));
            float4 s = src[q];
            z += 0.5;
            acc += 0.5 * s;
        }
    dst[tid] = acc / max(z, 1e-5);
}

void f3_loop(uint2 tid, Texture2D<float4> src,
             [format("rgba16f")] RWTexture2D<float4> dst, int radius)
{
    uint w, h; src.GetDimensions(w, h);
    if (tid.x >= w || tid.y >= h) return;
    float3 acc = float3(0.0);
    float z = 0.0;
    for (int j = -radius; j <= radius; ++j)
        for (int i = -radius; i <= radius; ++i)
        {
            uint2 q = uint2(clamp(int(tid.x) + i, 0, int(w) - 1),
                            clamp(int(tid.y) + j, 0, int(h) - 1));
            float3 s = src[q].rgb;
            z += 0.5;
            acc += 0.5 * s;
        }
    dst[tid] = float4(acc / max(z, 1e-5), 1.0);
}

void f3_epi(uint2 tid, Texture2D<float4> src,
            [format("rgba16f")] RWTexture2D<float4> dst, int radius)
{
    uint w, h; src.GetDimensions(w, h);
    if (tid.x >= w || tid.y >= h) return;
    float4 acc = float4(0.0);
    float z = 0.0;
    for (int j = -radius; j <= radius; ++j)
        for (int i = -radius; i <= radius; ++i)
        {
            uint2 q = uint2(clamp(int(tid.x) + i, 0, int(w) - 1),
                            clamp(int(tid.y) + j, 0, int(h) - 1));
            float4 s = src[q];
            z += 0.5;
            acc += 0.5 * s;
        }
    // identical loop to f4_all; only this line differs. In THIS minimal
    // form it stays fast — in a larger kernel in this library the same
    // pattern cost 2.6x (see issue text)
    dst[tid] = float4(acc.rgb / max(z, 1e-5), 1.0);
}
"""

S, N, RADIUS = 3456, 20, 4


def main() -> None:
    backend = sys.argv[1] if len(sys.argv) > 1 else "cuda"
    device = spy.Device(type=getattr(spy.DeviceType, backend),
                        compiler_options={"include_paths": [spy.SHADER_PATH]})
    module = spy.Module.load_from_source(device, "float3_repro", SHADER)

    rng = np.random.default_rng(3)
    usage = spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access
    src = device.create_texture(width=S, height=S, format=spy.Format.rgba16_float,
                                usage=usage,
                                data=rng.random((S, S, 4)).astype(np.float16))
    dst = device.create_texture(width=S, height=S, format=spy.Format.rgba16_float,
                                usage=usage)
    grid = spy.grid((S, S))

    def timed(fn):
        fn(grid, src, dst, RADIUS)
        device.wait_for_idle()
        t0 = time.perf_counter()
        for _ in range(N):
            fn(grid, src, dst, RADIUS)
        device.wait_for_idle()
        return (time.perf_counter() - t0) * 1000 / N

    f4 = timed(module.f4_all)
    f3l = timed(module.f3_loop)
    f3e = timed(module.f3_epi)
    print(f"{backend} ({device.info.adapter_name}), {S}^2, "
          f"{(2 * RADIUS + 1) ** 2} taps, median-ish of {N}:")
    print(f"  f4_all  (float4 everywhere)      {f4:6.2f} ms")
    print(f"  f3_loop (float3 in the loop)     {f3l:6.2f} ms  ({f3l / f4:.2f}x)")
    print(f"  f3_epi  (float3 ONLY in epilogue){f3e:6.2f} ms  ({f3e / f4:.2f}x)")


if __name__ == "__main__":
    main()
```

</details>

The pure-scalar formulation (three named float accumulators) also runs
at f4_all speed on both backends — the penalty tracks the float3 type,
not the amount of arithmetic.

## Notes from a larger codebase

- The minimal epilogue-only variant (`f3_epi`) does NOT reproduce the
  penalty here, but in a larger kernel in this library (bicubic accumulation
  in a helper function + `saturate`), a single once-per-thread
  `dst[tid] = float4(x.rgb, s.a)` swizzle-constructor kept the whole
  kernel at 2.6x until replaced by a lane assignment
  (`x.a = s.a; dst[tid] = x;`). The trigger boundary seems to involve
  more surrounding structure than this minimal form captures.
- float2 loop arithmetic appears to behave like float3 (not isolated
  as rigorously).
- Fixing this pattern across an image-processing library (bilateral,
  non-local means, kuwahara, unsharp, resize) took CUDA from 2-3x
  slower than Vulkan to parity-or-faster on every kernel, same GPU.

## Expected

float3 loop math within ~10-20% of float4, as on the Vulkan/Metal
paths — or at least a documented performance note for the CUDA target.

## Workaround (for anyone else hitting this)

In hot kernels on the CUDA target: accumulate in float4 (let the alpha
lane ride along) or in named scalars; read `float4 s = tex[q];` and use
`s.r/s.g/s.b` components rather than `.rgb` swizzles; write results via
lane assignment instead of `float4(v.rgb, a)` constructors.
