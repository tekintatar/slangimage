# fx/soft_glow.py

from __future__ import annotations

import slangpy as spy

from ..runtime import ModuleCache
from ..kernels.gaussian import GaussianBlurOp
from ..kernels.blend import BlendSoftGlowOp
from ._params import apply_gaussian, finite_float
from .base import Effect, EffectInputT


class SoftGlowEffect(Effect):
    """
    Soft glow effect:
      1) Gaussian blur
      2) Screen blend original and blurred
      3) Mix based on glow_strength
    """

    def __init__(self, device: spy.Device, modules: ModuleCache):
        super().__init__(device, modules)
        self.gauss = GaussianBlurOp(device, modules)
        self.blend = BlendSoftGlowOp(device, modules)

    def apply(  # type: ignore[override]  # subclasses intentionally narrow effect params
        self,
        src_tex: EffectInputT,
        *,
        blur_amount: float | None = None,
        glow_strength: float = 0.7,
        radius: int | None = None,
        sigma: float | None = None,
    ) -> EffectInputT:
        """Apply a Gaussian glow using an amount or explicit radius/sigma."""
        return super().apply(
            src_tex,
            blur_amount=blur_amount,
            glow_strength=glow_strength,
            radius=radius,
            sigma=sigma,
        )

    def _apply_impl(  # type: ignore[override]  # each effect owns its parameter schema
        self,
        src_tex: spy.Texture,
        *,
        blur_amount: float | None = None,
        glow_strength: float = 0.7,
        radius: int | None = None,
        sigma: float | None = None,
    ) -> spy.Texture:
        blur_tex = apply_gaussian(
            self.gauss,
            src_tex,
            blur_amount=blur_amount,
            default_amount=6.0,
            radius=radius,
            sigma=sigma,
        )

        # 2) blend soft glow
        dst_tex = self.blend.apply(
            src_tex,
            blur_tex,
            strength=finite_float("glow_strength", glow_strength),
        )
        return dst_tex
