# kernels/vibrance.py

from __future__ import annotations

import slangpy as spy

from .base import ImageOperator
from ..runtime import ModuleCache
from ..runtime.texture_pool import alloc_output


class VibranceOp(ImageOperator):
    """
    Vibrance operator.

    Works in the texture's working color space (SRGB-ish float domain),
    matching shaders/ops/vibrance.slang:

        void vibrance(
            uint2               coord,
            Texture2D<float4>   src,
            RWTexture2D<float4> dst,
            float               amount
        )

    Parameters
    ----------
    amount : float
        Main vibrance strength; the shader clamps to [-1, 1].
        Skin protection is a fixed RGB-based mask, not a public parameter.
    """

    def __init__(
        self,
        device: spy.Device,
        modules: ModuleCache,
        op_name: str = "vibrance",
    ) -> None:
        super().__init__(device)
        self.module = modules.get_op(op_name)

    def apply(
        self,
        src_tex: spy.Texture,
        amount: float = 0.5,
    ) -> spy.Texture:
        """
        Apply vibrance to a texture.

        src_tex:       input texture (fp16 or fp32, SRGB working space)
        amount:        vibrance strength (0..1 is typical)
        Skin protection is built in; source alpha is preserved.
        """
        w, h = src_tex.width, src_tex.height
        fmt = src_tex.format

        dst_tex = alloc_output(self.device, w, h, fmt)

        # Soft clamp user parameters to sane ranges
        a = float(amount)
        a = max(-1.0, min(a, 3.0))     # allow slight negative / strong positive

        grid = spy.grid((h, w))
        self.module.vibrance(
            grid,
            src_tex,
            dst_tex,
            a,
        )
        return dst_tex
