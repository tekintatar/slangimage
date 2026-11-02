# examples/03_pool_and_pipelines.py
# Texture-pool demo: effects run inside pool frames automatically, so a
# repeated-processing loop stops allocating once the pool is warm.

from _common import backend_arg, demo_image, out_path

import slangimage as si
from slangimage.runtime.texture_pool import get_pool
from slangimage.runtime.textures import (
    load_rgba_texture_from_image,
    save_texture_to_image,
)
from slangimage.fx.soft_glow import SoftGlowEffect


def main():
    manager = si.DeviceManager(backend_arg())  # --backend, default "auto"
    device = manager.device
    modules = si.ModuleCache(device)
    pool = get_pool(device)

    src = load_rgba_texture_from_image(device, demo_image(), precision="fp16")
    effect = SoftGlowEffect(device, modules)

    out = None
    for i in range(5):
        if out is not None:
            pool.recycle(out)  # done with the previous result
        out = effect.apply(src, glow_strength=0.7)
        print(f"run {i}: pool stats = {pool.stats()}")

    dst = out_path("pool_demo_out.png")
    save_texture_to_image(out, str(dst))
    print(f"saved {dst}")
    print("note: 'created' stays flat after run 0 — that is the pool working")


if __name__ == "__main__":
    main()
