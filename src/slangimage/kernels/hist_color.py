# kernels/hist_color.py
# Augmentation color-op parity set, histogram half: global per-channel
# equalize (3-pass, same shape as ClaheOp) and autocontrast (exact
# per-channel min/max via order-preserving atomic reduction).

from __future__ import annotations

import numpy as np
import slangpy as spy

from ..runtime import ModuleCache
from ..runtime.texture_pool import alloc_output

_BINS = 256
_BUF_USAGE = spy.BufferUsage.unordered_access | spy.BufferUsage.shader_resource


class HistColorOps:
    def __init__(self, device: spy.Device, modules: ModuleCache):
        self.device = device
        self.module = modules.get_op("hist_color")

    def equalize(self, src_tex: spy.Texture) -> spy.Texture:
        w, h = src_tex.width, src_tex.height
        n = 3 * _BINS
        hist = self.device.create_buffer(
            element_count=n, struct_size=4, usage=_BUF_USAGE,
            data=np.zeros(n, dtype=np.uint32),
        )
        mapping = self.device.create_buffer(
            element_count=n, struct_size=4, usage=_BUF_USAGE,
            data=np.zeros(n, dtype=np.float32),
        )

        grid = spy.grid((h, w))
        self.module.equalize_hist(grid, src_tex, hist)
        self.module.equalize_cdf(spy.grid((1, 3)), hist, mapping)

        dst = alloc_output(self.device, w, h, src_tex.format)
        self.module.equalize_apply(grid, src_tex, dst, mapping)
        return dst

    def autocontrast(self, src_tex: spy.Texture) -> spy.Texture:
        w, h = src_tex.width, src_tex.height
        # [min r,g,b | max r,g,b] as order-preserving keys; seed with the
        # identities (all-ones key floors min, zero key floors max)
        mm = self.device.create_buffer(
            element_count=6, struct_size=4, usage=_BUF_USAGE,
            data=np.array([0xFFFFFFFF] * 3 + [0] * 3, dtype=np.uint32),
        )

        grid = spy.grid((h, w))
        self.module.autocontrast_minmax(grid, src_tex, mm)

        dst = alloc_output(self.device, w, h, src_tex.format)
        self.module.autocontrast_apply(grid, src_tex, dst, mm)
        return dst
