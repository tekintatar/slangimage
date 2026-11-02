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
          f"{(2 * RADIUS + 1) ** 2} taps, mean of {N}:")
    print(f"  f4_all  (float4 everywhere)      {f4:6.2f} ms")
    print(f"  f3_loop (float3 in the loop)     {f3l:6.2f} ms  ({f3l / f4:.2f}x)")
    print(f"  f3_epi  (float3 ONLY in epilogue){f3e:6.2f} ms  ({f3e / f4:.2f}x)")


if __name__ == "__main__":
    main()
