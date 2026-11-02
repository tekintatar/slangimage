# kernels/stylize.py
# Standard stylization filters: pointillize, CMYK halftone.

from __future__ import annotations

import slangpy as spy

from ..runtime import ModuleCache
from ..runtime.texture_pool import alloc_output


class PointillizeOp:
    """Jittered-cell pointillist stylization (CIPointillize equivalent)."""

    def __init__(self, device: spy.Device, modules: ModuleCache):
        self.device = device
        self.module = modules.get_op("pointillize")

    def apply(
        self,
        src_tex: spy.Texture,
        radius: float = 20.0,
        jitter: float = 1.0,
        seed: float = 0.0,
    ) -> spy.Texture:
        w, h = src_tex.width, src_tex.height
        grid = spy.grid((h, w))
        dst = alloc_output(self.device, w, h, src_tex.format)
        self.module.pointillize(
            grid, src_tex, dst, float(radius), float(jitter), float(seed)
        )
        return dst


class CmykHalftoneOp:
    """Rotated-screen CMYK halftone (CICMYKHalftone equivalent)."""

    def __init__(self, device: spy.Device, modules: ModuleCache):
        self.device = device
        self.module = modules.get_op("cmyk_halftone")

    def apply(
        self,
        src_tex: spy.Texture,
        width: float = 6.0,
        angle_deg: float = 0.0,
        sharpness: float = 0.08,
    ) -> spy.Texture:
        w, h = src_tex.width, src_tex.height
        grid = spy.grid((h, w))
        dst = alloc_output(self.device, w, h, src_tex.format)
        self.module.cmyk_halftone(
            grid, src_tex, dst, float(width), float(angle_deg), float(sharpness)
        )
        return dst
