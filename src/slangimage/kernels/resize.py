# kernels/resize.py

from __future__ import annotations

import slangpy as spy

from .base import ImageOperator
from ..runtime import ModuleCache
from ..runtime.texture_pool import alloc_output


class ResizeOp(ImageOperator):
    """
    GPU resize operator (shaders/ops/resize.slang).

    Modes:
      - 'lanczos3' (default): separable, anti-aliased; the footprint widens
        with the reduction ratio for clean downscaling.
      - 'bicubic': Catmull-Rom, fixed 4x4 footprint.
      - 'bilinear': fixed 2x2 footprint.
    """

    def __init__(
        self,
        device: spy.Device,
        modules: ModuleCache,
        op_name: str = "resize",
    ):
        super().__init__(device)
        self.module = modules.get_op(op_name)

    def apply(
        self,
        src_tex: spy.Texture,
        out_w: int,
        out_h: int,
        mode: str = "lanczos3",  # "auto" | "bilinear" | "lanczos3" | bicubic
    ) -> spy.Texture:
        """
        Core resize API required by ImageOperator.

        Parameters
        ----------
        src_tex : spy.Texture
            Input texture (RGBA float).
        out_w : int
            Output width in pixels.
        out_h : int
            Output height in pixels.

        Returns
        -------
        spy.Texture
            Resized texture.
        """
        fmt = src_tex.format

        dst_tex = alloc_output(self.device, out_w, out_h, fmt)
        grid = spy.grid((out_h, out_w))

        if mode == "lanczos3":
            # Separable Lanczos3: horizontal pass to an intermediate (out_w x
            # src_h), then vertical pass to the output. Mathematically the 2D
            # footprint, but the intermediate rounds to the input precision.
            # O(ratio) taps/pixel with full occupancy; the tmp texture returns
            # to the pool with the op's frame.
            tmp = alloc_output(self.device, out_w, src_tex.height, fmt)
            self.module.resize_lanczos3_h(spy.grid((src_tex.height, out_w)), src_tex, tmp)
            self.module.resize_lanczos3_v(grid, tmp, dst_tex)
        elif mode == "bicubic":
            self.module.resize_bicubic(grid, src_tex, dst_tex)
        elif mode == "bilinear":
            self.module.resize_bilinear(grid, src_tex, dst_tex)
        else:
            self.module.resize_bicubic(grid, src_tex, dst_tex)
        return dst_tex
