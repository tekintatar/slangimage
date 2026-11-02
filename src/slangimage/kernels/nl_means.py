# kernels/nl_means.py

from __future__ import annotations

import slangpy as spy

from .base import ImageOperator
from ..runtime import ModuleCache
from ..runtime.texture_pool import alloc_output


class NLMeansOp(ImageOperator):
    """
    Non-local means denoise (Buades et al.).

    apply(src, search_radius=5, patch_radius=1, h=0.1, sigma=0.0)

    h is the filtering strength (higher = smoother); sigma is an optional
    noise-std estimate subtracted from patch distances. Cost grows with
    (2*search+1)^2 * (2*patch+1)^2 — this is the quality denoiser, not a
    realtime blur.
    """

    def __init__(self, device: spy.Device, modules: ModuleCache):
        super().__init__(device)
        self.module = modules.get_op("nl_means")

    def apply(
        self,
        src_tex: spy.Texture,
        search_radius: int = 5,
        patch_radius: int = 1,
        h: float = 0.1,
        sigma: float = 0.0,
    ) -> spy.Texture:
        w, ht = src_tex.width, src_tex.height
        grid = spy.grid((ht, w))
        dst = alloc_output(self.device, w, ht, src_tex.format)
        self.module.nl_means(
            grid, src_tex, dst,
            int(search_radius), int(patch_radius), float(h), float(sigma),
        )
        return dst
