# examples/05_custom_kernels.py
# Ship your own .slang kernels on top of slangimage: register a directory
# with ModuleCache.add_ops_dir and dispatch like any built-in op. This is
# how downstream packages add custom operators without forking the library.

import tempfile
from pathlib import Path

import numpy as np
import slangpy as spy
from _common import pick_device

import slangimage as si
from slangimage.runtime.modules import get_module_cache
from slangimage.runtime.texture_pool import alloc_output

MY_KERNEL = """
// posterize.slang — quantize each channel to n levels
void posterize(
    uint2               coord,
    Texture2D<float4>   src,
    [format(SLANGIMAGE_FMT)] RWTexture2D<float4> dst,
    float               levels
)
{
    uint w, h;
    src.GetDimensions(w, h);
    if (coord.x >= w || coord.y >= h)
        return;
    float4 c = src[coord];
    float n = max(levels - 1.0, 1.0);
    dst[coord] = float4(round(c.rgb * n) / n, c.a);
}
"""


def posterize(image: si.Image, levels: float = 4.0) -> si.Image:
    device = image.device
    cache = get_module_cache(device)  # the same cache si.ops uses
    dst = alloc_output(device, image.width, image.height, image.format)
    cache.get_op("posterize").posterize(
        spy.grid((image.height, image.width)), image.texture, dst, float(levels)
    )
    # wrap_pooled (not Image(dst)) so the output returns to the pool when the
    # caller drops it — the same lifecycle the built-in ops use.
    return si.wrap_pooled(dst)


def main():
    # write the kernel to a directory and register it (a real package would
    # ship a shaders/ dir and register it once at import time)
    kernel_dir = Path(tempfile.mkdtemp(prefix="my_kernels_"))
    (kernel_dir / "posterize.slang").write_text(MY_KERNEL)
    device = pick_device()  # --backend to override automatic selection
    get_module_cache(device).add_ops_dir(kernel_dir)

    rng = np.random.default_rng(5)
    img = si.from_numpy(rng.random((256, 256, 4)).astype(np.float16), device=device)

    out = posterize(img, levels=5)
    arr = out.to_numpy()
    print("unique values per channel ~", len(np.unique(arr[..., 0].round(3))))
    print("custom kernel ran on:", str(img.device.info.type).split(".")[-1])


if __name__ == "__main__":
    main()
