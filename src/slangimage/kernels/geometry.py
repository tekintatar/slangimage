# kernels/geometry.py
# Geometric transforms: flip / rot90 / crop (exact index remaps), wrapped
# tiling, and the affine engine (inverse-warp, bilinear or nearest, zeros
# outside).
# Matrix conventions and inversion live here in numpy; rotate matrices
# are built by ops.rotate.

from __future__ import annotations

import numpy as np
import slangpy as spy

from ..runtime import ModuleCache
from ..runtime.texture_pool import alloc_output
from .. import _validation as validate

_MODES = {"bilinear": 0, "nearest": 1}


def invert_affine(matrix) -> np.ndarray:
    """Invert a forward 2x3 affine (source->output, pixel-center coords)."""
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (2, 3):
        raise ValueError(f"expected a 2x3 affine matrix, got {m.shape}")
    if not np.isfinite(m).all():
        raise ValueError("affine matrix must contain finite values")
    aug = np.vstack([m, [0.0, 0.0, 1.0]])
    try:
        inv = np.linalg.inv(aug)[:2]
    except np.linalg.LinAlgError as error:
        raise ValueError("affine matrix must be nonsingular") from error
    if not np.isfinite(inv).all() or np.any(abs(inv) > validate.FLOAT32_MAX):
        raise ValueError("affine inverse must contain finite fp32 values")
    return inv


def rotation_matrix(angle_deg: float, cx: float, cy: float) -> np.ndarray:
    """Forward 2x3 for a rotation about (cx, cy). Positive angle rotates
    counter-clockwise on screen (torchvision convention; y points down,
    so the sine signs are flipped vs the textbook matrix). Exact multiples
    of 90 degrees use exact coefficients to avoid inflating ceil-rounded
    canvas bounds. Nearby and other arbitrary angles are not snapped."""
    if angle_deg % 90.0 == 0.0:
        quarter = int((angle_deg % 360.0) / 90.0)
        c, s = ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))[quarter]
    else:
        t = np.deg2rad(angle_deg)
        c, s = np.cos(t), np.sin(t)
    # x' = c*(x-cx) + s*(y-cy) + cx ; y' = -s*(x-cx) + c*(y-cy) + cy
    return np.array([
        [c, s, cx - c * cx - s * cy],
        [-s, c, cy + s * cx - c * cy],
    ])


class GeometryOps:
    """Grouped wrapper for geometry.slang (one method per kernel)."""

    def __init__(self, device: spy.Device, modules: ModuleCache):
        self.device = device
        self.module = modules.get_op("geometry")

    def flip(self, src_tex: spy.Texture, horizontal: bool = False,
             vertical: bool = False) -> spy.Texture:
        w, h = src_tex.width, src_tex.height
        dst = alloc_output(self.device, w, h, src_tex.format)
        self.module.flip(spy.grid((h, w)), src_tex, dst,
                         int(bool(horizontal)), int(bool(vertical)))
        return dst

    def rot90(self, src_tex: spy.Texture, k: int = 1) -> spy.Texture:
        k = int(k) % 4
        w, h = src_tex.width, src_tex.height
        if k == 0:
            dst = alloc_output(self.device, w, h, src_tex.format)
            self.module.flip(spy.grid((h, w)), src_tex, dst, 0, 0)  # copy
            return dst
        dw, dh = (h, w) if k % 2 == 1 else (w, h)
        dst = alloc_output(self.device, dw, dh, src_tex.format)
        self.module.rot90(spy.grid((dh, dw)), src_tex, dst, k)
        return dst

    def crop(self, src_tex: spy.Texture, x: int, y: int,
             width: int, height: int) -> spy.Texture:
        x, y, width, height = int(x), int(y), int(width), int(height)
        if width <= 0 or height <= 0:
            raise ValueError("crop width/height must be positive")
        if x < 0 or y < 0 or x + width > src_tex.width or y + height > src_tex.height:
            raise ValueError(
                f"crop rect ({x},{y},{width},{height}) outside "
                f"{src_tex.width}x{src_tex.height} source")
        dst = alloc_output(self.device, width, height, src_tex.format)
        self.module.crop(spy.grid((height, width)), src_tex, dst, x, y)
        return dst

    def tile(
        self,
        src_tex: spy.Texture,
        width: int,
        height: int,
        scale_x: float = 1.0,
        scale_y: float = 1.0,
        offset_x: float = 0.0,
        offset_y: float = 0.0,
        mode: str = "bilinear",
    ) -> spy.Texture:
        """Repeat ``src_tex`` over a new extent with wrapped sampling.

        Scale controls the displayed tile size (values above one enlarge it).
        Offsets are source-pixel phase added after inverse scaling.
        """
        mode_int = _MODES.get(mode)
        if mode_int is None:
            raise ValueError(f"mode must be one of {sorted(_MODES)}")

        width, height = validate.integer("width", width), validate.integer("height", height)
        if width <= 0 or height <= 0:
            raise ValueError("tile width/height must be positive")

        scale_x, scale_y = float(scale_x), float(scale_y)
        offset_x, offset_y = float(offset_x), float(offset_y)
        if not np.isfinite(scale_x) or not np.isfinite(scale_y) or scale_x <= 0 or scale_y <= 0:
            raise ValueError("scale_x and scale_y must be finite and > 0")
        if not np.isfinite(offset_x) or not np.isfinite(offset_y):
            raise ValueError("offset_x and offset_y must be finite")

        # Tile phase is periodic. Keeping coordinates non-negative avoids a
        # Vulkan codegen issue with floor followed by a signed integer cast.
        offset_x %= src_tex.width
        offset_y %= src_tex.height

        width, height = validate.dimensions(width, height, src_tex.device)
        # float-to-int casts and the +1 bilinear tap must remain in int32.
        # Reserve rounding headroom at the upper boundary (fp32 ULP is 128).
        for name, extent, size, scale, offset in (
            ("scale_x", width, src_tex.width, scale_x, offset_x),
            ("scale_y", height, src_tex.height, scale_y, offset_y),
        ):
            scale = validate.fp32(validate.finite_float(name, scale))
            if scale <= 0 or extent / scale + offset + size >= 2**31 - 1024:
                raise ValueError(f"{name} produces tile coordinates outside signed shader indexing")

        dst = alloc_output(self.device, width, height, src_tex.format)
        self.module.tile_repeat(
            spy.grid((height, width)),
            src_tex,
            dst,
            scale_x,
            scale_y,
            offset_x,
            offset_y,
            mode_int,
        )
        return dst

    def affine(self, src_tex: spy.Texture, matrix,
               width: int | None = None, height: int | None = None,
               mode: str = "bilinear") -> spy.Texture:
        mode_int = _MODES.get(mode)
        if mode_int is None:
            raise ValueError(f"mode must be one of {sorted(_MODES)}")
        inv = invert_affine(matrix)
        w = src_tex.width if width is None else width
        h = src_tex.height if height is None else height
        w, h = validate.dimensions(w, h, src_tex.device)
        # Bound every intermediate multiply/add, not just corner sums that
        # could conceal overflow by cancellation. Does not reject ordinary
        # out-of-image coordinates, which still produce transparent fill.
        coordinate_bound = abs(inv) @ np.array([w + 0.5, h + 0.5, 1.0])
        if np.any(coordinate_bound >= 2**31 - 1024):
            raise ValueError("affine inverse produces coordinates outside signed shader indexing")
        dst = alloc_output(self.device, w, h, src_tex.format)
        self.module.affine(
            spy.grid((h, w)), src_tex, dst,
            spy.float3(*inv[0]), spy.float3(*inv[1]), mode_int,
        )
        return dst
