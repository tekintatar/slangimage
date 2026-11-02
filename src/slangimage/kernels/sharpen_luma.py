# kernels/sharpen_luma.py

from __future__ import annotations

import slangpy as spy

from .base import ImageOperator
from ..runtime import ModuleCache
from ..runtime.texture_pool import alloc_output


class SharpenLumaOp(ImageOperator):
    """
    Cubic-filter sharpening of RGB or HSL lightness (preserving hue/saturation).
    Shader: shaders/ops/sharpen_luma.slang
    Entry:  sharpen_luma
    """

    def __init__(
        self,
        device: spy.Device,
        modules: ModuleCache,
        op_name: str = "sharpen_luma",
    ):
        super().__init__(device)
        self.module = modules.get_op(op_name)

    def apply(
        self,
        src_tex: spy.Texture,
        radius: float,
        sharpness: float,
        sharpen_rgb: bool =  True
    ) -> spy.Texture:
        """
        Parameters
        ----------
        src_tex : spy.Texture
            Input sRGB RGBA texture in [0, 1].
        radius : float
            Blur radius in pixels used to estimate Y_smooth.
            Bigger radius = larger scale sharpening.
        sharpness : float
            Strength of sharpening. 1.0 is a good starting point.

        Returns
        -------
        spy.Texture
            Sharpened RGBA texture.
        """
        w, h = src_tex.width, src_tex.height
        fmt = src_tex.format
        grid = spy.grid((h, w))

        dst_tex = alloc_output(self.device, w, h, fmt)

        if sharpen_rgb:
            self.module.sharpen_rgb(
                grid,
                src_tex,
                dst_tex,
                float(radius),
                float(sharpness),
            )
        else:
            self.module.sharpen_luma(
                grid,
                src_tex,
                dst_tex,
                float(radius),
                float(sharpness),
            )


        return dst_tex

    def apply_default(
        self,
        src_tex: spy.Texture,
        sharpness: float = 1.0,
        megapixels: float | None = None,
    ) -> spy.Texture:
        """
        Convenience wrapper to pick a radius based on image size.

        Ensures:
        - radius >= 2.0
        - radius <= 4.0
        """

        w = src_tex.width
        h = src_tex.height

        if megapixels is None:
            megapixels = (w * h) / 1e6

        # base radius before clamping
        # grows slowly with sqrt of MP
        base_radius = 1.5
        scaled = base_radius * (megapixels ** 0.5)

        # enforce final bounds: [2, 4]
        radius = max(2.0, min(scaled, 4.0))
        return self.apply(src_tex, radius=radius, sharpness=sharpness)
