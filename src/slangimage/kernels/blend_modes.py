# kernels/blend_modes.py

from __future__ import annotations

import slangpy as spy

from .base import ImageOperator
from ..runtime import ModuleCache
from ..runtime.texture_pool import alloc_output

# ids must match apply_blend_mode() in shaders/ops/blend_modes.slang
BLEND_MODES = {
    "multiply": 0,
    "screen": 1,
    "overlay": 2,
    "darken": 3,
    "lighten": 4,
    "color_dodge": 5,
    "color_burn": 6,
    "hard_light": 7,
    "soft_light": 8,
    "linear_light": 9,
    "vivid_light": 10,
    "color": 11,
}


class BlendModesOp(ImageOperator):
    """
    Photoshop/PDF blend modes (Core Image CI*BlendMode equivalents).

    apply(base, blend, mode="multiply", opacity=1.0) -> texture
    base is the background; blend is the foreground layer.
    """

    def __init__(self, device: spy.Device, modules: ModuleCache):
        super().__init__(device)
        self.module = modules.get_op("blend_modes")

    def apply(
        self,
        base_tex: spy.Texture,
        blend_tex: spy.Texture,
        mode: str = "multiply",
        opacity: float = 1.0,
    ) -> spy.Texture:
        mode_id = BLEND_MODES.get(mode)
        if mode_id is None:
            raise ValueError(f"Unknown blend mode '{mode}'. One of: {sorted(BLEND_MODES)}")

        w, h = base_tex.width, base_tex.height
        grid = spy.grid((h, w))
        dst_tex = alloc_output(self.device, w, h, base_tex.format)

        self.module.blend_mode(
            grid,
            base_tex,
            blend_tex,
            dst_tex,
            int(mode_id),
            float(opacity),
        )
        return dst_tex
