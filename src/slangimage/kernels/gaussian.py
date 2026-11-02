# kernels/gaussian.py

from __future__ import annotations

import math
import slangpy as spy

from .base import ImageOperator
from ..runtime import ModuleCache
from ..runtime.texture_pool import alloc_output, get_pool
from .. import _validation as validate


def validate_gaussian_params(radius: int, sigma: float) -> tuple[int, float]:
    """Normalize and validate explicit Gaussian kernel parameters."""
    return validate.radius("radius", radius), validate.sigma("sigma", sigma)


def validate_blur_amount(amount: float) -> float:
    """Normalize a user-facing blur amount before size-aware conversion."""
    amount = validate.finite_float("amount", amount)
    if not math.isfinite(amount) or amount < 0:
        raise ValueError("amount must be a finite value >= 0")
    return amount


def blur_params_from_amount(
    amount: float,
    width: int,
    height: int,
    ref_mp: float = 1.0,
) -> tuple[int, float]:
    """Convert a user-facing blur amount to size-aware radius and sigma."""
    amount = max(0.1, validate_blur_amount(amount))
    mp = (width * height) / 1e6
    scale = math.sqrt(mp / ref_mp)
    sigma = amount * scale
    if sigma > validate.MAX_RADIUS / 3.0:
        raise ValueError(f"amount produces a radius exceeding {validate.MAX_RADIUS}")
    radius = max(1, int(math.ceil(3.0 * sigma)))
    return validate_gaussian_params(radius, sigma)


class GaussianBlurOp(ImageOperator):
    """
    Separable Gaussian blur operator with:
      - normal allocation path (apply)
      - preallocated reuse (apply_reuse)
    """

    def __init__(self, device: spy.Device, modules: ModuleCache):
        super().__init__(device)
        self.module = modules.get_op("gaussian_s")

    # -----------------------------------------------------------
    # Convert "amount" → (radius, sigma)
    # -----------------------------------------------------------


    def blur_params_from_amount(self,
                                amount: float,
                                width: int,
                                height: int,
                                ref_mp: float = 1.0) -> tuple[int, float]:
        """
        Compute Gaussian blur (radius, sigma) from a blur 'amount', scaled by megapixels.

        amount:
            blur strength; sigma = amount * sqrt(megapixels / ref_mp), and values
            below 0.1 are raised to 0.1
        width, height:
            image size in pixels
        ref_mp:
            megapixel baseline (keep at 1.0 unless you want a different reference)

        Returns:
            (radius, sigma)
        """
        return blur_params_from_amount(amount, width, height, ref_mp)

    # -----------------------------------------------------------
    # Original API (allocates intermediate textures every call)
    # -----------------------------------------------------------
    def apply(self, src_tex: spy.Texture, radius: int, sigma: float) -> spy.Texture:
        radius, sigma = validate_gaussian_params(radius, sigma)
        w, h = src_tex.width, src_tex.height
        fmt = src_tex.format
        grid = spy.grid((h, w))

        tmp_tex = alloc_output(self.device, w, h, fmt)

        dst_tex = alloc_output(self.device, w, h, fmt)

        self.module.gaussian_blur_h(grid, src_tex, tmp_tex, radius, sigma)
        self.module.gaussian_blur_v(grid, tmp_tex, dst_tex, radius, sigma)

        # scratch texture goes straight back to the pool (queue order makes
        # reuse safe); no device sync here — readback syncs when needed
        get_pool(self.device).recycle(tmp_tex)
        return dst_tex

    def apply_with_amount(self, src_tex: spy.Texture, amount: float) -> spy.Texture:
        w = src_tex.width
        h = src_tex.height
        radius, sigma = blur_params_from_amount(amount, w, h)
        return self.apply(src_tex, radius, sigma)

    # -----------------------------------------------------------
    # Reuse externally preallocated textures (zero allocations)
    # -----------------------------------------------------------
    def apply_reuse(
        self,
        src_tex: spy.Texture,
        tmp_tex: spy.Texture,
        dst_tex: spy.Texture,
        radius: int,
        sigma: float,
        wait_for_idle: bool = False,
    ) -> spy.Texture:

        radius, sigma = validate_gaussian_params(radius, sigma)
        w, h = src_tex.width, src_tex.height
        grid = spy.grid((h, w))

        assert tmp_tex.width == w and tmp_tex.height == h
        assert dst_tex.width == w and dst_tex.height == h

        self.module.gaussian_blur_h(grid, src_tex, tmp_tex, radius, sigma)
        self.module.gaussian_blur_v(grid, tmp_tex, dst_tex, radius, sigma)

        if wait_for_idle:
            self.device.wait_for_idle()

        return dst_tex
