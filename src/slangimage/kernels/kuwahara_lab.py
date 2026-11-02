# kernels/kuwahara_lab.py

from __future__ import annotations

import slangpy as spy

from .base import ImageOperator
from ..runtime import ModuleCache
from ..runtime.texture_pool import alloc_output


class KuwaharaLabBlendOp(ImageOperator):
    """
    Blend Kuwahara grayscale luminance into original image in Lab space.
    Input:
      - src_tex: original RGBA float
      - kuw_tex: Kuwahara grayscale RGBA float (value in RGB)
    Output:
      - RGBA float, colors preserved, structure from Kuwahara in L
    """

    def __init__(
        self,
        device: spy.Device,
        modules: ModuleCache,
        op_name: str = "kuwahara_lab",
    ):
        super().__init__(device)
        self.module = modules.get_op(op_name)

    def apply(
        self,
        src_tex: spy.Texture,
        kuw_tex: spy.Texture,
        strength: float = 1.0,
    ) -> spy.Texture:
        
        w, h = src_tex.width, src_tex.height
        fmt = src_tex.format
        grid = spy.grid((h, w))

        dst_tex = alloc_output(self.device, w, h, fmt)

        self.module.kuwahara_lab_blend(
            grid,
            src_tex,
            kuw_tex,
            dst_tex,
            float(strength),
        )

        return dst_tex