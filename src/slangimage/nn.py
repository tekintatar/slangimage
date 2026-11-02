# slangimage/nn.py — torch.nn.Module wrappers over slangimage.diff
# (kornia-style), so the fused Slang ops drop into nn.Sequential and
# training pipelines. CUDA only, like slangimage.diff.
#
#     import torch
#     import slangimage.nn as snn
#
#     grade = torch.nn.Sequential(
#         snn.Bilateral(radius=5, sigma_range=0.15),
#         snn.Lut(),                       # learnable color grading
#     ).cuda()
#     loss = (grade(img) - target).pow(2).mean()
#     loss.backward()                      # fused Slang backward kernels
#
# Scalar settings (radius, sigma, ev, …) are fixed hyperparameters — the
# kernels take them as non-differentiable uniforms. The learnable state
# is the Lut module's 512x512x3 CLUT parameter (differentiable w.r.t.
# both image and LUT). Inputs are HWC tensors on the CUDA device; fp16 /
# autocast inputs are upcast to fp32 by the ops (see diff._f32c).

from __future__ import annotations

import numpy as np
import torch
from torch import nn as _tnn

from . import diff as _diff
from .ops import identity_lut_np
from .kernels.lut import validate_lut_size


class _ImageModule(_tnn.Module):
    """Narrow torch's generic ``Module.__call__`` for HWC image modules."""

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return super().__call__(image)


class _TwoImageModule(_tnn.Module):
    """Typed two-image counterpart used by blend modules."""

    def __call__(self, base: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        return super().__call__(base, blend)


class GaussianBlur(_ImageModule):
    """Separable gaussian blur; (H, W) or (H, W, C)."""

    def __init__(self, radius: int = 9, sigma: float = 3.0):
        super().__init__()
        self.radius, self.sigma = int(radius), float(sigma)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return _diff.gaussian_blur(image, radius=self.radius, sigma=self.sigma)

    def extra_repr(self) -> str:
        return f"radius={self.radius}, sigma={self.sigma}"


class Bilateral(_ImageModule):
    """Edge-preserving bilateral filter; gradients flow through the range
    weights (the edge-preservation itself is trainable end-to-end)."""

    def __init__(self, radius: int = 5, sigma_spatial: float = 2.5,
                 sigma_range: float = 0.15):
        super().__init__()
        self.radius = int(radius)
        self.sigma_spatial = float(sigma_spatial)
        self.sigma_range = float(sigma_range)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return _diff.bilateral(image, radius=self.radius,
                               sigma_spatial=self.sigma_spatial,
                               sigma_range=self.sigma_range)

    def extra_repr(self) -> str:
        return (f"radius={self.radius}, sigma_spatial={self.sigma_spatial}, "
                f"sigma_range={self.sigma_range}")


class UnsharpMask(_ImageModule):
    """Unsharp mask, src + amount * (src - gaussian(src)), unclamped;
    (H, W) or (H, W, C). Wraps slangimage.diff.unsharp_mask."""

    def __init__(self, radius: int = 2, sigma: float = 1.5, amount: float = 1.0):
        super().__init__()
        self.radius, self.sigma, self.amount = int(radius), float(sigma), float(amount)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return _diff.unsharp_mask(image, radius=self.radius, sigma=self.sigma,
                                  amount=self.amount)

    def extra_repr(self) -> str:
        return f"radius={self.radius}, sigma={self.sigma}, amount={self.amount}"


class Resize(_ImageModule):
    """Bilinear/bicubic resize matching F.interpolate(align_corners=False)."""

    def __init__(self, width: int | None = None, height: int | None = None,
                 scale: float | None = None, mode: str = "bilinear"):
        super().__init__()
        self.width, self.height, self.scale, self.mode = width, height, scale, mode

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return _diff.resize(image, width=self.width, height=self.height,
                            scale=self.scale, mode=self.mode)

    def extra_repr(self) -> str:
        return f"width={self.width}, height={self.height}, scale={self.scale}, mode={self.mode}"


class Exposure(_ImageModule):
    """Exposure, rgb * 2**ev, unclamped; (H, W, 3). Wraps
    slangimage.diff.exposure."""

    def __init__(self, ev: float = 0.0):
        super().__init__()
        self.ev = float(ev)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return _diff.exposure(image, ev=self.ev)

    def extra_repr(self) -> str:
        return f"ev={self.ev}"


class Sepia(_ImageModule):
    """Classic sepia matrix, blended by intensity; (H, W, 3). Wraps
    slangimage.diff.sepia."""

    def __init__(self, intensity: float = 1.0):
        super().__init__()
        self.intensity = float(intensity)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return _diff.sepia(image, intensity=self.intensity)

    def extra_repr(self) -> str:
        return f"intensity={self.intensity}"


class HueRotate(_ImageModule):
    """YIQ chroma rotation by ``angle`` radians; (H, W, 3). Wraps
    slangimage.diff.hue_rotate."""

    def __init__(self, angle: float = 0.0):
        super().__init__()
        self.angle = float(angle)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return _diff.hue_rotate(image, angle=self.angle)

    def extra_repr(self) -> str:
        return f"angle={self.angle}"


class Blend(_TwoImageModule):
    """Two-input module: forward(base, blend) with a fixed mode/opacity."""

    def __init__(self, mode: str = "multiply", opacity: float = 1.0):
        super().__init__()
        self.mode, self.opacity = mode, float(opacity)

    def forward(self, base: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        return _diff.blend(base, blend, mode=self.mode, opacity=self.opacity)

    def extra_repr(self) -> str:
        return f"mode={self.mode!r}, opacity={self.opacity}"


class Lut(_ImageModule):
    """Learnable color grading as a 512x512 (8x8-grid) CLUT.

    With learnable=True (default) the LUT is an nn.Parameter initialized
    to identity — optimize it like any other weight and the module learns
    a color grade; the trained tensor doubles as a shareable preset
    (layout matches slangimage.ops.lut / identity_lut_np). Pass lut= to
    start from an existing grade, or learnable=False to apply a fixed one.
    """

    def __init__(self, lut: object | None = None, learnable: bool = True):
        super().__init__()
        if lut is None:
            lut = torch.tensor(identity_lut_np()[..., :3].copy())
        lut = torch.as_tensor(np.asarray(lut) if not isinstance(lut, torch.Tensor) else lut,
                              dtype=torch.float32)
        if lut.ndim != 3 or lut.shape[2] != 3:
            raise ValueError(f"expected an (N, N, 3) LUT, got {tuple(lut.shape)}")
        validate_lut_size(lut.shape[1], lut.shape[0])
        if learnable:
            self.lut = _tnn.Parameter(lut)
        else:
            self.register_buffer("lut", lut)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return _diff.lut(image, self.lut)

    def extra_repr(self) -> str:
        learnable = isinstance(self.lut, _tnn.Parameter)
        return f"size={self.lut.shape[0]}, learnable={learnable}"
