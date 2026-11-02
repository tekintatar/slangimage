# fx/gaussian_blur.py

from __future__ import annotations

import slangpy as spy

from ..runtime import ModuleCache
from ..kernels.gaussian import GaussianBlurOp
from ._params import apply_gaussian
from .base import Effect, EffectInputT


class GaussianBlurEffect(Effect):
    """
    Simple Gaussian blur effect.

    Core logic is texture based via _apply_impl.
    """

    def __init__(self, device: spy.Device, modules: ModuleCache):
        super().__init__(device, modules)
        self.op = GaussianBlurOp(device, modules)

    def apply(  # type: ignore[override]  # subclasses intentionally narrow effect params
        self,
        src_tex: EffectInputT,
        *,
        blur_amount: float | None = None,
        radius: int | None = None,
        sigma: float | None = None,
    ) -> EffectInputT:
        """Blur an image using either ``blur_amount`` or an explicit pair."""
        return super().apply(
            src_tex,
            blur_amount=blur_amount,
            radius=radius,
            sigma=sigma,
        )

    def _apply_impl(  # type: ignore[override]  # each effect owns its parameter schema
        self,
        src_tex: spy.Texture,
        *,
        blur_amount: float | None = None,
        radius: int | None = None,
        sigma: float | None = None,
    ) -> spy.Texture:
        return apply_gaussian(
            self.op,
            src_tex,
            blur_amount=blur_amount,
            default_amount=5.0,
            radius=radius,
            sigma=sigma,
        )
