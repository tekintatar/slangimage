# kernels/bilateral.py

from __future__ import annotations

import slangpy as spy

from .base import ImageOperator
from ..runtime import ModuleCache
from ..runtime.texture_pool import alloc_output, get_pool

_RANGE_TYPES = {"gaussian": 0, "cutoff": 1}


class BilateralOp(ImageOperator):
    """
    Classic bilateral filter.

    apply(src, radius, sigma_spatial, sigma_range, range_type)
        full 2D bilateral. sigma_spatial <= 0 selects a box spatial kernel.
        range_type "gaussian" (default) or "cutoff" (hard color threshold,
        the fast/epsilon-filter formulation).

    apply_separable(...) runs a horizontal then vertical 1D pass — the usual
    fast approximation.
    """

    def __init__(self, device: spy.Device, modules: ModuleCache):
        super().__init__(device)
        self.module = modules.get_op("bilateral")

    def apply(
        self,
        src_tex: spy.Texture,
        radius: int = 5,
        sigma_spatial: float = 0.0,
        sigma_range: float = 0.15,
        range_type: str = "gaussian",
    ) -> spy.Texture:
        rt = _RANGE_TYPES[range_type]
        w, h = src_tex.width, src_tex.height
        grid = spy.grid((h, w))
        dst = alloc_output(self.device, w, h, src_tex.format)
        self.module.bilateral(
            grid, src_tex, dst,
            int(radius), float(sigma_spatial), float(sigma_range), int(rt),
        )
        return dst

    def apply_separable(
        self,
        src_tex: spy.Texture,
        radius: int = 5,
        sigma_spatial: float = 0.0,
        sigma_range: float = 0.15,
        range_type: str = "gaussian",
    ) -> spy.Texture:
        rt = _RANGE_TYPES[range_type]
        w, h = src_tex.width, src_tex.height
        grid = spy.grid((h, w))

        tmp = alloc_output(self.device, w, h, src_tex.format)
        dst = alloc_output(self.device, w, h, src_tex.format)

        self.module.bilateral_1d(
            grid, src_tex, tmp,
            int(radius), 1, 0, float(sigma_spatial), float(sigma_range), int(rt),
        )
        self.module.bilateral_1d(
            grid, tmp, dst,
            int(radius), 0, 1, float(sigma_spatial), float(sigma_range), int(rt),
        )
        get_pool(self.device).recycle(tmp)
        return dst
