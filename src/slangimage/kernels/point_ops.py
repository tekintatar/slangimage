# kernels/point_ops.py
# Trivial per-pixel ops: invert, extract_channel, clamp_multiply,
# high_pass_contrast, random_noise. One module, small wrapper classes.

from __future__ import annotations

import slangpy as spy

from ..runtime import ModuleCache
from ..runtime.texture_pool import alloc_output

_CHANNELS = {"r": 0, "g": 1, "b": 2, 0: 0, 1: 1, 2: 2}


class PointOps:
    """Grouped wrapper for the point_ops.slang kernels. Not an ImageOperator
    subclass — it exposes one method per kernel instead of a single apply()."""

    def __init__(self, device: spy.Device, modules: ModuleCache):
        self.device = device
        self.module = modules.get_op("point_ops")

    def _out(self, like: spy.Texture) -> tuple:
        w, h = like.width, like.height
        return spy.grid((h, w)), alloc_output(self.device, w, h, like.format)

    def invert(self, src_tex: spy.Texture) -> spy.Texture:
        grid, dst = self._out(src_tex)
        self.module.invert(grid, src_tex, dst)
        return dst

    def extract_channel(self, src_tex: spy.Texture, channel="b") -> spy.Texture:
        ch = _CHANNELS.get(channel)
        if ch is None:
            raise ValueError(f"channel must be r/g/b, got {channel!r}")
        grid, dst = self._out(src_tex)
        self.module.extract_channel(grid, src_tex, dst, int(ch))
        return dst

    def clamp_multiply(self, src_tex: spy.Texture, src2_tex: spy.Texture) -> spy.Texture:
        grid, dst = self._out(src_tex)
        self.module.clamp_multiply(grid, src_tex, src2_tex, dst)
        return dst

    def high_pass_contrast(
        self, src_tex: spy.Texture, blur_tex: spy.Texture, strength: float = 1.0
    ) -> spy.Texture:
        grid, dst = self._out(src_tex)
        self.module.high_pass_contrast(grid, src_tex, blur_tex, dst, float(strength))
        return dst

    def random_noise(
        self,
        width: int,
        height: int,
        seed: float = 0.0,
        fmt: spy.Format = spy.Format.rgba32_float,
    ) -> spy.Texture:
        grid = spy.grid((height, width))
        dst = alloc_output(self.device, width, height, fmt)
        self.module.random_noise(grid, dst, float(seed))
        return dst
