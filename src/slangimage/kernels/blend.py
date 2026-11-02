# kernels/blend.py

from __future__ import annotations

import slangpy as spy

from .base import ImageOperator
from ..runtime import ModuleCache
from ..runtime.texture_pool import alloc_output


class BlendSoftGlowOp(ImageOperator):
    """
    Blend operator that applies a soft glow:
    out = lerp(src, screen(src, blurred), strength)
    """

    def __init__(
        self,
        device: spy.Device,
        modules: ModuleCache,
        op_name: str = "blend",
    ):
        super().__init__(device)
        self.module = modules.get_op(op_name)

    def apply(
        self,
        src_tex: spy.Texture,
        blur_tex: spy.Texture,
        strength: float,
    ) -> spy.Texture:

        w, h = src_tex.width, src_tex.height
        fmt = src_tex.format

        grid = spy.grid((h, w))

        dst_tex = alloc_output(self.device, w, h, fmt)

        self.module.soft_glow(
            grid,
            src_tex,
            blur_tex,
            dst_tex,
            float(strength),
        )

        return dst_tex
