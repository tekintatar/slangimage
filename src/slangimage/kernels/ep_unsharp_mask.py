# kernels/ep_unsharp_mask.py

from __future__ import annotations
import slangpy as spy

from .base import ImageOperator
from ..runtime import ModuleCache
from ..runtime.texture_pool import alloc_output
from .surface_blur import SurfaceBlurOp


class EdgePreservingUnsharpMaskOp(ImageOperator):
    """
    Edge-preserving unsharp mask using SurfaceBlurOp.

        1) Y_smooth = linear-light Y of SurfaceBlur(RGB)
        2) detail   = Y - Y_smooth
        3) Y_sharp  = Y + amount * detail
    """

    def __init__(self, device: spy.Device, modules: ModuleCache, op_name="ep_unsharp_mask"):
        super().__init__(device)
        self.module = modules.get_op(op_name)
        self.surface = SurfaceBlurOp(device, modules)   # internal

    def apply(
        self,
        src_tex: spy.Texture,
        radius: float = 3.0,
        bsigma: float = 0.12,
        amount: float = 1.0,
    ) -> spy.Texture:
        # 1) surface-blurred base, driven by radius and bsigma
        blur_tex = self.surface.apply(src_tex, int(round(radius)), float(bsigma))

        w, h = src_tex.width, src_tex.height
        fmt = src_tex.format
        grid = spy.grid((h, w))

        dst = alloc_output(self.device, w, h, fmt)

        grid = spy.grid((h, w))

        self.module.ep_unsharp_mask(
            grid,
            src_tex,
            blur_tex,
            dst,
            float(amount),
        )

        return dst


    def apply_default(self, src_tex: spy.Texture, amount: float = 1.0):
        """
        A simple preset mapping where amount controls ONLY sharpening strength.
        The blur parameters are tuned for stability across resolutions.
        """

        # Keep blur soft and edge-aware
        radius = 3.0
        bsigma = 0.12

        return self.apply(
            src_tex,
            radius=radius,
            bsigma=bsigma,
            amount=amount,
        )
