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
