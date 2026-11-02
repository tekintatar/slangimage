# kernels/white_balance.py
# Photographic white balance: Bradford chromatic adaptation from an
# assumed illuminant (Kelvin temperature on the blackbody/daylight locus,
# plus a green/magenta tint offset) to the 6500K daylight neutral.
# All color science happens here in numpy — the kernel just applies the
# resulting 3x3 matrix in linear RGB.

from __future__ import annotations

import numpy as np
import slangpy as spy

from .base import ImageOperator
from ..runtime import ModuleCache
from ..runtime.texture_pool import alloc_output

# Bradford cone-response matrix and sRGB(D65) <-> XYZ (IEC 61966-2-1)
_BRADFORD = np.array([
    [0.8951, 0.2664, -0.1614],
    [-0.7502, 1.7135, 0.0367],
    [0.0389, -0.0685, 1.0296],
])
_RGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])
_XYZ_TO_RGB = np.linalg.inv(_RGB_TO_XYZ)

# tint slider scale: +-150 spans a strong but usable green<->magenta range
_TINT_TO_DY = 3.0e-4

NEUTRAL_TEMPERATURE = 6500.0


def illuminant_xy(temperature: float) -> tuple[float, float]:
    """Chromaticity of the illuminant at a correlated color temperature.

    Uses the CIE daylight locus for T >= 4000K and the Krystek Planckian
    approximation below (2000..4000K) — the standard photo-app model.
    """
    t = float(np.clip(temperature, 2000.0, 25000.0))
    if t >= 4000.0:
        it = 1e3 / t
        if t <= 7000.0:
            x = 0.244063 + 0.09911 * it + 2.9678 * it**2 - 4.6070 * it**3
        else:
            x = 0.237040 + 0.24748 * it + 1.9018 * it**2 - 2.0064 * it**3
        y = -3.000 * x * x + 2.870 * x - 0.275
    else:
        it = 1e3 / t
        x = 0.179910 + 0.8776956 * it - 0.2343589 * it**2 - 0.2661239 * it**3
        if t <= 2222.0:
            y = -0.20219683 + 2.18555832 * x - 1.34811020 * x * x - 1.1063814 * x**3
        else:
            y = -0.16748867 + 2.09137015 * x - 1.37418593 * x * x - 0.9549476 * x**3
    return x, y


def _white_xyz(x: float, y: float) -> np.ndarray:
    return np.array([x / y, 1.0, (1.0 - x - y) / y])


def white_balance_matrix(temperature: float, tint: float) -> np.ndarray:
    """3x3 linear-RGB matrix adapting the assumed illuminant to neutral.

    temperature: assumed scene illuminant in Kelvin (higher -> the light
    was bluer -> the correction warms the image). tint > 0 assumes a
    greener illuminant -> the correction pushes magenta (Lightroom
    convention). (6500, 0) is exactly identity.
    """
    sx, sy = illuminant_xy(temperature)
    sy += float(np.clip(tint, -150.0, 150.0)) * _TINT_TO_DY
    dx, dy = illuminant_xy(NEUTRAL_TEMPERATURE)

    src_lms = _BRADFORD @ _white_xyz(sx, sy)
    dst_lms = _BRADFORD @ _white_xyz(dx, dy)
    cat = np.linalg.inv(_BRADFORD) @ np.diag(dst_lms / src_lms) @ _BRADFORD
    return _XYZ_TO_RGB @ cat @ _RGB_TO_XYZ


class WhiteBalanceOp(ImageOperator):
    """apply(src, temperature=6500, tint=0) — photographic white balance."""

    def __init__(self, device: spy.Device, modules: ModuleCache):
        super().__init__(device)
        self.module = modules.get_op("white_balance")

    def apply(
        self,
        src_tex: spy.Texture,
        temperature: float = NEUTRAL_TEMPERATURE,
        tint: float = 0.0,
    ) -> spy.Texture:
        m = white_balance_matrix(float(temperature), float(tint))
        w, h = src_tex.width, src_tex.height
        dst = alloc_output(self.device, w, h, src_tex.format)
        self.module.white_balance(
            spy.grid((h, w)), src_tex, dst,
            spy.float3(*m[0]), spy.float3(*m[1]), spy.float3(*m[2]),
        )
        return dst
