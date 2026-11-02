# kernels/unsharp_mask.py

from __future__ import annotations
import slangpy as spy

from .base import ImageOperator
from ..runtime import ModuleCache
from ..runtime.texture_pool import alloc_output
from .gaussian import GaussianBlurOp
import math


class UnsharpMaskOp(ImageOperator):
    """
    Unsharp mask in stored RGB (default) or linear-light YCbCr luma.

    RGB: clip(src.rgb + amount * (src.rgb - blur.rgb), 0, 1).
    Luma: sharpen/clamp Y, retain source Cb/Cr through the rounded color
    transform, then encode/clip RGB. Both modes preserve source alpha.
    """

    def __init__(
        self,
        device: spy.Device,
        modules: ModuleCache,
        op_name: str = "unsharp_mask",
    ):
        super().__init__(device)
        self.module = modules.get_op(op_name)

        # Internal Gaussian op to make unsharp mask self-contained
        self.gauss = GaussianBlurOp(device, modules)

    def apply(
        self,
        src_tex: spy.Texture,
        radius: float = 2.0,
        sigma: float = 1.5,
        amount: float = 1.0,
        colorSpace: str = "rgb"
    ) -> spy.Texture:
        """
        Fully self-contained USM:

            blur = Gaussian(src)
            out = unsharp(src, blur, amount)

        Parameters
        ----------
        radius : float
            Gaussian blur radius.
        sigma : float
            Gaussian sigma.
        amount : float
            Sharpen strength. 1.0 is typical. At 0.0 RGB is only clipped;
            luma still passes through the approximate color-space roundtrip.
        colorSpace: str
            rgb or luma (linear-light YCbCr)

        Returns
        -------
        spy.Texture
            The sharpened output texture.
        """

        # 1) Gaussian blur
        blur_tex = self.gauss.apply(
            src_tex,
            radius=int(radius),
            sigma=float(sigma)
        )

        # 2) Allocate output
        w, h = src_tex.width, src_tex.height
        fmt = src_tex.format
        grid = spy.grid((h, w))

        dst_tex = alloc_output(self.device, w, h, fmt)

        if colorSpace == "rgb":
            self.module.unsharp_mask(
                grid,
                src_tex,
                blur_tex,
                dst_tex,
                float(amount),
            )
        else:
            self.module.unsharp_mask_luma(
                grid,
                src_tex,
                blur_tex,
                dst_tex,
                float(amount),
            )

        return dst_tex

    def apply_default(
        self,
        src_tex: spy.Texture,
        amount: float = 1.0,
        megapixels: float | None = None,
        colorSpace: str = "rgb"
    ) -> spy.Texture:
        """
        Convenience wrapper that chooses radius and sigma from image size.

        - For small images, uses a tight blur.
        - For large images, increases blur footprint a bit.

        Parameters
        ----------
        amount : float
            Sharpen strength. 0.0 to 2.0 is a good working range.
        megapixels : float | None
            Override for image size if you want. If None it is computed
            from src_tex dimensions.
        """
        w, h = src_tex.width, src_tex.height

        if megapixels is None:
            megapixels = (w * h) / 1.0e6

        # Scale with sqrt of megapixels, clamped to a reasonable range
        #   1 MP  -> scale ~ 1
        #   4 MP  -> scale ~ 2
        #   16 MP -> scale ~ 4 (clamped)
        scale = max(1.0, min(math.sqrt(megapixels), 4.0))

        # Base sigma for 1 MP around 1.0, grows with scale
        base_sigma = 1.0
        sigma = base_sigma * scale
        sigma = max(0.8, min(sigma, 3.0))  # keep it sane

        # Radius about 3 * sigma, clamped
        radius = int(max(1, min(round(3.0 * sigma), 21)))

        return self.apply(
            src_tex,
            radius=radius,
            sigma=sigma,
            amount=amount,
            colorSpace=colorSpace
        )
