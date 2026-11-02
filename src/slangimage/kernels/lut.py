# kernels/lut.py

from __future__ import annotations

import numpy as np
import slangpy as spy

from .base import ImageOperator
from ..runtime import ModuleCache
from ..runtime.texture_pool import alloc_output
from .. import _validation as validate


def validate_lut_size(width: int, height: int) -> int:
    """Validate the shared square 8x8-grid CLUT dimension contract."""
    width, height = validate.integer("width", width, 1), validate.integer("height", height, 1)
    if width != height or width < 16 or width % 8 != 0:
        raise ValueError(
            "LUT must be square, divisible by 8, and at least 16x16; "
            f"got {width}x{height}"
        )
    return width


def identity_lut_np(size: int = 512) -> np.ndarray:
    """Generate the identity 8x8-grid CLUT (float32 RGBA, HxWx4).

    Blue selects one of 64 slices (row-major in an 8x8 grid); red varies
    horizontally and green vertically within each size/8 by size/8 slice.
    The default 512 atlas has 64 samples along all three color axes.

    `size` must be a multiple of 8 and at least 16 (each 8x8-grid cell needs to
    be >= 2px, or the r/g ramp divides by zero and the borders come out zero).
    """
    cells = 8
    size = validate_lut_size(size, size)
    cell = size // cells  # 64 for the default 512
    lut = np.zeros((size, size, 4), dtype=np.float32)
    for slice_idx in range(cells * cells):
        cy, cx = divmod(slice_idx, cells)
        rg = (np.arange(cell, dtype=np.float32) + 0.0) / (cell - 1)
        r = np.tile(rg[None, :], (cell, 1))
        g = np.tile(rg[:, None], (1, cell))
        b = slice_idx / (cells * cells - 1)
        y0, x0 = cy * cell, cx * cell
        lut[y0 : y0 + cell, x0 : x0 + cell, 0] = r
        lut[y0 : y0 + cell, x0 : x0 + cell, 1] = g
        lut[y0 : y0 + cell, x0 : x0 + cell, 2] = b
        lut[y0 : y0 + cell, x0 : x0 + cell, 3] = 1.0
    return lut


class LutOp(ImageOperator):
    """Square 8x8-grid color lookup table (CLUT), size >=16 and divisible by 8."""

    def __init__(self, device: spy.Device, modules: ModuleCache):
        super().__init__(device)
        self.module = modules.get_op("lut")

    def apply(
        self,
        src_tex: spy.Texture,
        lut_tex: spy.Texture,
        intensity: float = 1.0,
    ) -> spy.Texture:
        validate_lut_size(lut_tex.width, lut_tex.height)
        w, h = src_tex.width, src_tex.height
        grid = spy.grid((h, w))
        dst_tex = alloc_output(self.device, w, h, src_tex.format)
        self.module.lut_apply(grid, src_tex, lut_tex, dst_tex, float(intensity))
        return dst_tex
