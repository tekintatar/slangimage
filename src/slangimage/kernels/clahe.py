# kernels/clahe.py

from __future__ import annotations

import numpy as np
import slangpy as spy

from .base import ImageOperator
from ..runtime import ModuleCache
from ..runtime.texture_pool import alloc_output

_BINS = 256
_BUF_USAGE = spy.BufferUsage.unordered_access | spy.BufferUsage.shader_resource


class ClaheOp(ImageOperator):
    """
    Contrast-Limited Adaptive Histogram Equalization on luminance.

    apply(src, clip_limit=3.0, tiles=(8, 8), strength=1.0)

    clip_limit is relative, with a minimum effective bin cap of one count;
    excess is redistributed as fractional counts. tiles is the CLAHE grid.
    Strength is clamped to [0,1] and mixes original/mapped luminance. RGB ratio
    scaling preserves channel ratios before the final [0,1] clamp; even zero
    strength clips HDR RGB. Alpha is preserved. Not an OpenCV/Lab parity claim.
    """

    def __init__(self, device: spy.Device, modules: ModuleCache):
        super().__init__(device)
        self.module = modules.get_op("clahe")

    def apply(
        self,
        src_tex: spy.Texture,
        clip_limit: float = 3.0,
        tiles: tuple[int, int] = (8, 8),
        strength: float = 1.0,
    ) -> spy.Texture:
        tiles_x, tiles_y = int(tiles[0]), int(tiles[1])
        w, h = src_tex.width, src_tex.height
        n = tiles_x * tiles_y * _BINS

        hist = self.device.create_buffer(
            element_count=n, struct_size=4, usage=_BUF_USAGE,
            data=np.zeros(n, dtype=np.uint32),
        )
        mapping = self.device.create_buffer(
            element_count=n, struct_size=4, usage=_BUF_USAGE,
            data=np.zeros(n, dtype=np.float32),
        )

        grid = spy.grid((h, w))
        self.module.clahe_hist(grid, src_tex, hist, tiles_x, tiles_y)
        self.module.clahe_cdf(
            spy.grid((tiles_y, tiles_x)), hist, mapping,
            tiles_x, tiles_y, float(clip_limit),
        )

        dst = alloc_output(self.device, w, h, src_tex.format)
        self.module.clahe_apply(
            grid, src_tex, dst, mapping, tiles_x, tiles_y, float(strength)
        )
        return dst
