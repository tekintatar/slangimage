# kernels/color_controls.py

from __future__ import annotations

import slangpy as spy

from .base import ImageOperator
from ..runtime import ModuleCache
from ..runtime.texture_pool import alloc_output


class ColorControlsOp(ImageOperator):
    """
    Combined color controls with linear-light YCbCr-style intermediates.

    Hue, saturation and brightness are signed controls, clamped to [-1, 1]:

      hue:
        -1.0 → -180 degrees
         0.0 → no hue shift
         1.0 → +180 degrees

      saturation:
        -1.0 → monochrome
         0.0 → original saturation
         1.0 → much stronger saturation

      brightness:
        -1.0 → subtract 0.5 from linear luma
         0.0 → original brightness
         1.0 → add 0.5 to linear luma

    Contrast uses the legacy FACTOR convention, unlike si.ops.color_controls:
    1 is neutral, factors between 1e-4 and 1 flatten, and factors above 1
    strengthen encoded RGB around 0.5. The shader clamps factors to [0, 5];
    values <=1e-4 bypass contrast entirely, including the default 0. This
    low-level bypass is discontinuous; use the signed public API for controls.

    Luma is clipped to [0, 1], chroma magnitude to 0.6. Final RGB is unclipped;
    alpha is retained. Rounded conversion matrices are not exact inverses.
    """

    def __init__(
        self,
        device: spy.Device,
        modules: ModuleCache,
        op_name: str = "color_controls",
    ):
        super().__init__(device)
        self.module = modules.get_op(op_name)

    def apply(
        self,
        src_tex: spy.Texture,
        hue: float = 0.0,
        saturation: float = 0.0,
        contrast: float = 0.0,
        brightness: float = 0.0,
    ) -> spy.Texture:
        # Preserve legacy factor-style contrast, including its shader bypass.
        hue = max(-1.0, min(1.0, float(hue)))
        saturation = max(-1.0, min(1.0, float(saturation)))
        contrast = max(-1.0, min(5.0, float(contrast)))
        brightness = max(-1.0, min(1.0, float(brightness)))

        w, h = src_tex.width, src_tex.height
        fmt = src_tex.format

        dst_tex = alloc_output(self.device, w, h, fmt)

        grid = spy.grid((h, w))
        self.module.color_controls(
            grid,
            src_tex,
            dst_tex,
            float(hue),
            float(saturation),
            float(contrast),
            float(brightness),
        )
        return dst_tex
