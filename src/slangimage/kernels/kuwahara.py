# kernels/kuwahara.py

from __future__ import annotations

import slangpy as spy

from .base import ImageOperator
from ..runtime import ModuleCache
from ..runtime.texture_pool import alloc_output


class KuwaharaOp(ImageOperator):
    """
    Generalized Kuwahara filter (8 sectors).

    Works on RGBA float textures. Blends RGB sector means using inverse
    variance of RGB vector magnitude; output retains color and original alpha.
    """

    def __init__(
        self,
        device: spy.Device,
        modules: ModuleCache,
        op_name: str = "kuwahara",
    ):
        super().__init__(device)
        self.module = modules.get_op(op_name)

    def apply(
        self,
        src_tex: spy.Texture,
        radius: int = 7,
        sigma_spatial: float = 12.0,
        var_power: float = 7.0,
    ) -> spy.Texture:
        """
        Apply Kuwahara filter.

        Parameters
        ----------
        src_tex : spy.Texture
            Input texture (RGBA float).
        radius : int
            Square-neighborhood radius.
        sigma_spatial : float, optional
            Spatial weight is exp(-(dx*dx+dy*dy)/(sigma_spatial*radius)).
        var_power : float, optional
            Positive exponent used on inverse sector variance.

        Returns
        -------
        spy.Texture
            Output texture (smoothed RGB, original alpha).
        """
        w, h = src_tex.width, src_tex.height
        fmt = src_tex.format
        grid = spy.grid((h, w))

        dst_tex = alloc_output(self.device, w, h, fmt)

        self.module.generalized_kuwahara(
            grid,
            src_tex,
            dst_tex,
            int(radius),
            float(sigma_spatial),
            float(var_power),
        )

        return dst_tex
