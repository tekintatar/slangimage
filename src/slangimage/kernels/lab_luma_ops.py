# kernels/lab_luma_ops.py
# Advanced building blocks for "process luminance, preserve color" pipelines:
# split an image's CIELAB L* channel out, run any luma-domain op on it, then
# recombine it with the original's a*/b* chrominance. Not exposed as a single
# si.ops.* verb (there's no one-shot op — it's a two-step extract/blend
# primitive); use the op classes directly.

from __future__ import annotations

import slangpy as spy
from ..runtime.texture_pool import alloc_output
from ..runtime import ModuleCache
from .base import ImageOperator


class LabLumaExtractOp(ImageOperator):
    """Extract CIELAB luminance (L*) as a grayscale texture for luma-domain
    processing; recombine with LabLumaBlendOp."""

    def __init__(self, device: spy.Device, modules: ModuleCache, op_name: str = "lab_luma_extract"):
        super().__init__(device)
        self.device = device
        self.module = modules.get_op(op_name)

    def apply(self, src_tex: spy.Texture) -> spy.Texture:
        w, h = src_tex.width, src_tex.height
        fmt = src_tex.format
        grid = spy.grid((h, w))

        dst = alloc_output(self.device, w, h, fmt)
        
        self.module.lab_luma_extract(grid, src_tex, dst)
        return dst


class LabLumaBlendOp(ImageOperator):
    """Recombine a (processed) L* luminance texture with the original RGB's
    a*/b* chrominance — the inverse pairing of LabLumaExtractOp."""

    def __init__(self, device: spy.Device, modules: ModuleCache, op_name: str = "lab_luma_blend"):
        super().__init__(device)
        self.module = modules.get_op(op_name)

    def apply(self, luma_tex: spy.Texture, src_rgb_tex: spy.Texture) -> spy.Texture:
        w, h = src_rgb_tex.width, src_rgb_tex.height
        fmt  = src_rgb_tex.format
        grid = spy.grid((h, w))

        dst_tex = alloc_output(self.device, w, h, fmt)

        self.module.lab_luma_blend(grid, luma_tex, src_rgb_tex, dst_tex)
        return dst_tex