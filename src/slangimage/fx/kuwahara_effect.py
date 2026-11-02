# fx/kuwahara_effect.py

from __future__ import annotations

import slangpy as spy

from ..runtime import ModuleCache
from ..kernels.kuwahara import KuwaharaOp
from ..kernels.kuwahara_lab import KuwaharaLabBlendOp
from ._params import finite_float, positive_float
from .base import Effect, EffectInputT


class KuwaharaPaintEffect(Effect):
    """
    Painterly generalized-Kuwahara stylization applied to luminance: the
    Kuwahara result's luminance (Lab L) is blended into the image's L while the
    original color (a*/b*) is kept, so flat regions stay put and only luminance
    is stylized.
    """

    def __init__(self, device: spy.Device, modules: ModuleCache):
        super().__init__(device, modules)
        self.kuwahara = KuwaharaOp(device, modules)
        self.lab_blend = KuwaharaLabBlendOp(device, modules)

    def apply(  # type: ignore[override]  # subclasses intentionally narrow effect params
        self,
        src_tex: EffectInputT,
        *,
        radius: int = 6,
        sigma_spatial: float = 12.0,
        var_power: float = 7.0,
        strength: float = 1.0,
    ) -> EffectInputT:
        """Apply luminance-preserving Kuwahara stylization."""
        return super().apply(
            src_tex,
            radius=radius,
            sigma_spatial=sigma_spatial,
            var_power=var_power,
            strength=strength,
        )

    def _apply_impl(  # type: ignore[override]  # each effect owns its parameter schema
        self,
        src_tex: spy.Texture,
        *,
        radius: int = 6,
        sigma_spatial: float = 12.0,
        var_power: float = 7.0,
        strength: float = 1.0,
    ) -> spy.Texture:
        radius = int(radius)
        sigma_spatial = positive_float("sigma_spatial", sigma_spatial)
        var_power = positive_float("var_power", var_power)
        strength = finite_float("strength", strength)
        if radius < 1:
            raise ValueError("radius must be >= 1")

        kuw_tex = self.kuwahara.apply(
            src_tex,
            radius=radius,
            sigma_spatial=sigma_spatial,
            var_power=var_power,
        )

        dst_tex = self.lab_blend.apply(
            src_tex,
            kuw_tex,
            strength=strength,
        )
        return dst_tex
