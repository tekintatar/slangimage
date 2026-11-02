# kernels/surface_blur.py

from __future__ import annotations

from dataclasses import dataclass

import slangpy as spy

from ..runtime.texture_pool import alloc_output


@dataclass
class SurfaceBlurParams:
    radius: int
    bsigma: float          # range sigma


def params_from_amount(amount: float, megapixels: float | None = None) -> SurfaceBlurParams:
    """
    Adaptive SurfaceBlur parameters based on:
        - a blur strength 'amount' (values below 0.1 are raised to 0.1)
        - image megapixels (optional)

    Rules:
        * radius grows with image size (sqrt-ish of MP)
        * radius capped at 15
        * bsigma grows gently with amount
        * spatial_sigma tied to radius
    """
    amount = float(max(0.1, amount))

    # Base radius from amount
    base_radius = 1.0 + amount * 2.0  # ~3..7 for typical amounts

    # Scale with megapixels if provided
    if megapixels is not None:
        mp_scale = megapixels**0.2
        base_radius *= mp_scale

    # Final radius with cap
    radius = int(round(min(base_radius, 15.0)))

    # Color sigma follows amount slightly
    bsigma = 0.10 + 0.10 * amount

    return SurfaceBlurParams(
        radius=radius,
        bsigma=bsigma,
    )


class SurfaceBlurOp:
    """
    Low level texture operator for surface blur.

    Works directly on Texture2D<float4> / Texture2D<half4> underneath
    and is reusable inside effects.
    """

    def __init__(self, device: spy.Device, modules) -> None:
        """
        device  - slangpy Device
        modules - ModuleCache from runtime.modules
        """
        self.device = device
        # op name maps to shaders/ops/surface_blur.slang via ModuleCache
        self.module = modules.get_op("surface_blur")


    def apply(
        self,
        src_tex: spy.Texture,
        radius: int,
        bsigma: float,
    ) -> spy.Texture:
        """
        Run surface blur on a texture.

        src_tex       - input texture (can be fp32 or fp16)
        radius        - legacy sample count; half-width = floor((radius-1)/2)
        bsigma        - color sigma (range)
        Spatial sigma equals the half-width; output keeps src_tex.format.
        RGB is filtered with edge replication; source alpha is preserved.
        """

        w, h = src_tex.width, src_tex.height
        fmt = src_tex.format
        grid = spy.grid((h, w))


        dst_tex = alloc_output(self.device, w, h, fmt)
        self.module.surface_blur(
            grid,
            src_tex,
            dst_tex,
            int(radius),
            float(bsigma)
        )

        return dst_tex

    def apply_with_amount(
        self,
        src_tex: spy.Texture,
        amount: float,
    ) -> spy.Texture:
        """
        Convenience wrapper that takes a single 'amount' slider.

        Output keeps src_tex.format.
        """
        mp = (src_tex.width * src_tex.height) / 1e6
        p = params_from_amount(amount, megapixels=mp)
        return self.apply(
            src_tex,
            radius=p.radius,
            bsigma=p.bsigma
        )
