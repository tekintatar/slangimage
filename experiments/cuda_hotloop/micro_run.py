import pathlib
import sys
import time

import numpy as np
import slangpy as spy

import slangimage
from slangimage.runtime.modules import ModuleCache
from slangimage.runtime.textures import np_to_texture
from slangimage.runtime.texture_pool import alloc_output

backend = sys.argv[1]
shaders_root = pathlib.Path(slangimage.__file__).resolve().parent / "shaders"
opts = {
    "include_paths": [str(spy.SHADER_PATH), str(shaders_root),
                      str(shaders_root / "common"), str(shaders_root / "ops")],
    "debug_info": spy.SlangDebugInfoLevel.none,
    "optimization": spy.SlangOptimizationLevel.default,
}
dev = spy.Device(type=getattr(spy.DeviceType, backend), compiler_options=opts)
mc = ModuleCache(dev, extra_ops_dirs=[str(pathlib.Path(__file__).resolve().parent)])

S, N = 3456, 20
rng = np.random.default_rng(3)
arr = rng.random((S, S, 4)).astype(np.float16)
arr[..., 3] = 1.0
src = np_to_texture(dev, arr, usage="read", precision="fp16")
dst = alloc_output(dev, S, S, src.format)
grid = spy.grid((S, S))
mod = mc.load_variant("microbench", "fp16")


def timed(fn, n=N):
    fn()
    dev.wait_for_idle()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    dev.wait_for_idle()
    return (time.perf_counter() - t0) * 1000 / n


print(f"{backend}: "
      f"reads49 {timed(lambda: mod.reads49(grid, src, dst)):6.2f} ms | "
      f"exp64 {timed(lambda: mod.exp_loop(grid, src, dst, 64)):6.2f} | "
      f"pow64 {timed(lambda: mod.pow_loop(grid, src, dst, 64)):6.2f} | "
      f"fma512 {timed(lambda: mod.fma_loop(grid, src, dst, 512)):6.2f}")
print(f"{backend}: "
      f"reads_rt(r3,49taps) {timed(lambda: mod.reads_rt(grid, src, dst, 3)):6.2f} ms | "
      f"reads_rt(r4,81) {timed(lambda: mod.reads_rt(grid, src, dst, 4)):6.2f} | "
      f"surfb_sim(r4,81) {timed(lambda: mod.surfb_sim(grid, src, dst, 4)):6.2f}")
print(f"{backend}: surfb_hw(r4,81) {timed(lambda: mod.surfb_hw(grid, src, dst, 4)):6.2f} ms")
print(f"{backend}: surfb_noexp {timed(lambda: mod.surfb_noexp(grid, src, dst, 4)):6.2f} ms | "
      f"surfb_constf {timed(lambda: mod.surfb_constf(grid, src, dst, 4)):6.2f}")
print(f"{backend}: surfb_constf4 {timed(lambda: mod.surfb_constf4(grid, src, dst, 4)):6.2f} ms")
print(f"{backend}: surfb_f4acc {timed(lambda: mod.surfb_f4acc(grid, src, dst, 4)):6.2f} ms | "
      f"surfb_scalar3 {timed(lambda: mod.surfb_scalar3(grid, src, dst, 4)):6.2f}")
