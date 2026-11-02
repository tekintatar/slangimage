# kernels/gamma_adjust.py

from __future__ import annotations

import slangpy as spy

from .base import ImageOperator
from ..runtime import ModuleCache
from ..runtime.texture_pool import alloc_output


class GammaAdjustOp(ImageOperator):
    """
    Gamma adjustment of RGB clipped to [0, 1]; source alpha is unchanged.

    power:
      < 1.0  brightens (0.7 is a gentle lift)
      = 1.0  no change for RGB already in [0, 1]
      > 1.0  darkens
    Finite power is clamped to [0.01, 10] by the shader.
    """

    def __init__(
        self,
        device: spy.Device,
        modules: ModuleCache,
        op_name: str = "gamma_adjust",
    ):
        super().__init__(device)
        self.module = modules.get_op(op_name)

    def apply(
        self,
        src_tex: spy.Texture,
        power: float = 1.0,
    ) -> spy.Texture:
        w, h = src_tex.width, src_tex.height
        fmt = src_tex.format

        dst_tex = alloc_output(self.device, w, h, fmt)

        grid = spy.grid((h, w))
        self.module.gamma_adjust(
            grid,
            src_tex,
            dst_tex,
            float(power),
        )
        return dst_tex
