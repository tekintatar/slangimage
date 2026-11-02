# kernels/color_adjust.py
# Standard color utilities: exposure, sepia, hue rotate, false color,
# highlight/shadow. Parameter conventions follow the CI built-ins.

from __future__ import annotations

import slangpy as spy

from ..runtime import ModuleCache
from ..runtime.texture_pool import alloc_output


class ColorAdjustOps:
    """Grouped wrapper for color_adjust.slang (one method per kernel)."""

    def __init__(self, device: spy.Device, modules: ModuleCache):
        self.device = device
        self.module = modules.get_op("color_adjust")

    def _out(self, like: spy.Texture) -> tuple:
        w, h = like.width, like.height
        return spy.grid((h, w)), alloc_output(self.device, w, h, like.format)

    def exposure(self, src_tex: spy.Texture, ev: float = 0.0) -> spy.Texture:
        """CIExposureAdjust: rgb * 2^ev."""
        grid, dst = self._out(src_tex)
        self.module.exposure_adjust(grid, src_tex, dst, float(ev))
        return dst

    def sepia(self, src_tex: spy.Texture, intensity: float = 1.0) -> spy.Texture:
        """CISepiaTone: classic sepia matrix, lerped by intensity."""
        grid, dst = self._out(src_tex)
        self.module.sepia_tone(grid, src_tex, dst, float(intensity))
        return dst

    def hue(self, src_tex: spy.Texture, angle: float = 0.0) -> spy.Texture:
        """Rotate encoded-RGB YIQ chroma in radians, with positive I-to-Q rotation.

        No sRGB decoding; rounded matrices are not exact inverses. Final RGB
        clips to [0, 1], alpha is unchanged. Y is untouched internally, but
        reconstructed/clipped output luma is not exactly preserved. Angle is
        passed as fp32 without modulo reduction; Core Image parity is unverified.
        """
        grid, dst = self._out(src_tex)
        self.module.hue_adjust(grid, src_tex, dst, float(angle))
        return dst

    def false_color(
        self,
        src_tex: spy.Texture,
        color0: tuple[float, float, float] = (0.3, 0.0, 0.0),
        color1: tuple[float, float, float] = (1.0, 0.9, 0.8),
    ) -> spy.Texture:
        """Duotone between fp32 RGB endpoints using clipped encoded-RGB luma.

        Uses weights (0.299, 0.587, 0.114), with no sRGB decoding or input-channel
        clipping. Endpoints/final RGB are unclipped; alpha is retained exactly.
        Exact Core Image parity is not established.
        """
        grid, dst = self._out(src_tex)
        self.module.false_color(
            grid, src_tex, dst, spy.float3(*color0), spy.float3(*color1)
        )
        return dst

    def highlight_shadow(
        self,
        src_tex: spy.Texture,
        shadow_amount: float = 0.0,
        highlight_amount: float = 1.0,
    ) -> spy.Texture:
        """Global pointwise power-law highlight/shadow curve.

        shadow_amount: usual range -1..1, not clamped; positive lifts shadows.
        highlight_amount: clamped to 0..1, 1 neutral, lower compresses brights.
        Uses encoded-RGB luma, without linearization. Clips luma/final RGB,
        preserves alpha, and scales RGB for every positive luma. Zero/clipped-
        negative luma bypasses division. Exact Core Image parity is not established.
        """
        grid, dst = self._out(src_tex)
        self.module.highlight_shadow(
            grid, src_tex, dst, float(shadow_amount), float(highlight_amount)
        )
        return dst
