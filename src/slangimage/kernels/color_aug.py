# kernels/color_aug.py
# Augmentation color-op parity set, pointwise half: solarize, posterize,
# grayscale, normalize. Grouped wrapper like PointOps — one method per
# kernel in color_aug.slang.

from __future__ import annotations

import slangpy as spy

from ..runtime import ModuleCache
from ..runtime.texture_pool import alloc_output


class ColorAugOps:
    def __init__(self, device: spy.Device, modules: ModuleCache):
        self.device = device
        self.module = modules.get_op("color_aug")

    def _out(self, like: spy.Texture) -> tuple:
        w, h = like.width, like.height
        return spy.grid((h, w)), alloc_output(self.device, w, h, like.format)

    def solarize(self, src_tex: spy.Texture, threshold: float = 0.5) -> spy.Texture:
        grid, dst = self._out(src_tex)
        self.module.solarize(grid, src_tex, dst, float(threshold))
        return dst

    def posterize(self, src_tex: spy.Texture, bits: int = 4) -> spy.Texture:
        if not 1 <= int(bits) <= 8:
            raise ValueError(f"bits must be in [1, 8], got {bits}")
        grid, dst = self._out(src_tex)
        self.module.posterize(grid, src_tex, dst, int(bits))
        return dst

    def grayscale(self, src_tex: spy.Texture) -> spy.Texture:
        grid, dst = self._out(src_tex)
        self.module.grayscale(grid, src_tex, dst)
        return dst

    def normalize(self, src_tex: spy.Texture, mean, std) -> spy.Texture:
        mean, std = tuple(map(float, mean)), tuple(map(float, std))
        if len(mean) != 3 or len(std) != 3:
            raise ValueError("mean and std must each have 3 components")
        if any(s == 0 for s in std):
            raise ValueError("std components must be non-zero")
        grid, dst = self._out(src_tex)
        self.module.normalize(grid, src_tex, dst, spy.float3(*mean), spy.float3(*std))
        return dst
