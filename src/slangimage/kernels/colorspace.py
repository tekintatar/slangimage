# kernels/colorspace.py
# sRGB <-> linear conversion ops.

from __future__ import annotations

import slangpy as spy
from ..runtime.texture_pool import alloc_output
from ..runtime import ModuleCache


class SRGBToLinearOp:
    def __init__(self, device: spy.Device, modules: ModuleCache):
        self.device = device
        self.module = modules.get_op("srgb_to_linear")

    def apply(self, src_tex: spy.Texture) -> spy.Texture:
        w, h = src_tex.width, src_tex.height
        fmt = src_tex.format

        dst = alloc_output(self.device, w, h, fmt)

        grid = spy.grid((h, w))
        self.module.srgb_to_linear_pass(grid, src_tex, dst)
        return dst


class LinearToSRGBOp:
    def __init__(self, device: spy.Device, modules: ModuleCache):
        self.device = device
        self.module = modules.get_op("linear_to_srgb")

    def apply(self, src_tex: spy.Texture) -> spy.Texture:
        w, h = src_tex.width, src_tex.height
        fmt = src_tex.format

        dst = alloc_output(self.device, w, h, fmt)

        grid = spy.grid((h, w))
        self.module.linear_to_srgb_pass(grid, src_tex, dst)
        return dst