# kernels/smooth_threshold.py

from __future__ import annotations

import math

import slangpy as spy

from .base import ImageOperator
from ..runtime import ModuleCache
from ..runtime.texture_pool import alloc_output
from .. import _validation as validate


def validate_threshold_range(
    black_threshold: float,
    white_threshold: float,
) -> tuple[float, float]:
    black_threshold = validate.finite_float("black_threshold", black_threshold)
    white_threshold = validate.finite_float("white_threshold", white_threshold)
    if not math.isfinite(black_threshold) or not math.isfinite(white_threshold):
        raise ValueError("black_threshold and white_threshold must be finite")
    if white_threshold <= black_threshold:
        raise ValueError("white_threshold must be greater than black_threshold")
    if validate.fp32(white_threshold) <= validate.fp32(black_threshold):
        raise ValueError("white_threshold must remain greater than black_threshold in fp32")
    gap = validate.fp32(validate.fp32(white_threshold) - validate.fp32(black_threshold))
    if not math.isfinite(gap) or gap < 2.0**-126:
        raise ValueError("white_threshold - black_threshold must be finite and normal in fp32")
    return black_threshold, white_threshold


class SmoothThresholdOp(ImageOperator):
    """
    Luminance thresholding.

    apply(src, threshold, smooth_amount)      — smoothstep around threshold
    apply_bw(src, black_threshold, white_threshold) — linear ramp

    Two-point threshold controls must have a positive, finite, normal fp32 gap.
    """

    def __init__(self, device: spy.Device, modules: ModuleCache):
        super().__init__(device)
        self.module = modules.get_op("smooth_threshold")

    def apply(
        self,
        src_tex: spy.Texture,
        threshold: float = 0.5,
        smooth_amount: float = 0.28,
    ) -> spy.Texture:
        w, h = src_tex.width, src_tex.height
        grid = spy.grid((h, w))
        dst_tex = alloc_output(self.device, w, h, src_tex.format)
        self.module.smooth_threshold(
            grid, src_tex, dst_tex, float(threshold), float(smooth_amount)
        )
        return dst_tex

    def apply_bw(
        self,
        src_tex: spy.Texture,
        black_threshold: float,
        white_threshold: float,
    ) -> spy.Texture:
        black_threshold, white_threshold = validate_threshold_range(
            black_threshold, white_threshold
        )
        w, h = src_tex.width, src_tex.height
        grid = spy.grid((h, w))
        dst_tex = alloc_output(self.device, w, h, src_tex.format)
        self.module.smooth_threshold_bw(
            grid, src_tex, dst_tex, black_threshold, white_threshold
        )
        return dst_tex
