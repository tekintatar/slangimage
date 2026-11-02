from __future__ import annotations

from typing import Optional
import numpy as np
from PIL import Image
import slangpy as spy
from pathlib import Path


# -------------------------------------------------------------------
# Usage flags
# -------------------------------------------------------------------
TEX_USAGE_READ = spy.TextureUsage.shader_resource
TEX_USAGE_RW   = spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access


# -------------------------------------------------------------------
# Helpers: usage + format inference
# -------------------------------------------------------------------
def _parse_usage(u) -> spy.TextureUsage:
    if isinstance(u, spy.TextureUsage):
        return u

    if isinstance(u, str):
        key = u.lower().strip()
        if key in ("read", "r"):
            return TEX_USAGE_READ
        if key in ("rw", "readwrite", "read_write", "write", "uav"):
            return TEX_USAGE_RW
        raise ValueError(f"Unknown usage string '{u}'")

    raise TypeError(f"usage must be spy.TextureUsage or str, got {type(u).__name__}")


def _infer_format(channels: int, precision: str) -> spy.Format:
    """
    Infer the Texture2D format from channel count and precision.
    Only 1- and 4-channel formats exist on GPUs; np_to_texture expands
    3-channel input to RGBA before this is called.
    """
    precision = precision.lower()
    if precision == "fp32":
        if channels == 4:
            return spy.Format.rgba32_float
        if channels == 1:
            return spy.Format.r32_float
    elif precision == "fp16":
        if channels == 4:
            return spy.Format.rgba16_float
        if channels == 1:
            return spy.Format.r16_float

    raise ValueError(f"Unsupported channels={channels} precision={precision}")


# -------------------------------------------------------------------
# Core: upload NumPy → Texture2D
# -------------------------------------------------------------------
def np_to_texture(
    device: spy.Device,
    np_img: np.ndarray,
    usage: spy.TextureUsage | str = "read",
    precision: str = "fp32",
) -> spy.Texture:
    """
    Upload a HxWxC NumPy array (float32 or float16, [0..1]) as a Texture2D.

    precision:
        "fp32" → convert to float32 and use rgba32/rgb32 formats
        "fp16" → convert to float16 and use rgba16/rgb16 formats
    """
    from .device import _claim_device_thread

    _claim_device_thread(device)
    if np_img.ndim != 3:
        raise ValueError(f"Expected HxWxC array, got {np_img.shape}")

    h, w, c = np_img.shape
    if h < 1 or w < 1:
        raise ValueError(
            f"image height and width must both be >= 1, got {h}x{w}"
        )

    # slang/create_texture requires C-contiguous data; normalize views
    # (transposed / negative-stride / sliced) up front instead of failing deep
    # in the backend with "Data is not contiguous".
    np_img = np.ascontiguousarray(np_img)

    # GPUs have no 3-channel texture formats — expand RGB to opaque RGBA
    if c == 3:
        alpha = np.ones((h, w, 1), dtype=np_img.dtype)
        np_img = np.concatenate([np_img, alpha], axis=-1)
        c = 4
    if c == 1:
        # every op module is compiled for rgba storage; a real r16/r32 texture
        # silently mismatches (wrong readback shape, save() crashes, corrupt
        # output on some backends). Reject with an actionable message.
        raise ValueError(
            "1-channel (grayscale) input is not supported; broadcast to 3 or 4 "
            "channels first, e.g. np.repeat(arr, 3, axis=-1)."
        )
    if c != 4:
        raise ValueError(f"Expected 3 or 4 channels, got {c}")

    if precision == "fp32":
        data = np_img.astype(np.float32, copy=False)
    elif precision == "fp16":
        data = np_img.astype(np.float16)
    else:
        raise ValueError("precision must be 'fp32' or 'fp16'")

    # SlangPy's ndarray binding requires writable data even for upload. Copy
    # read-only inputs rather than changing the caller's writeability flag.
    if not data.flags.writeable:
        data = data.copy(order="C")

    # NumPy ignores singleton-axis strides when deciding C-contiguity, whereas
    # SlangPy requires canonical byte strides even for those axes. Flattening
    # the already-contiguous data then restoring its shape normalizes zero,
    # negative and oversized singleton strides without copying pixel storage
    # or mutating the caller's array. Keep ordinary uploads on the same path.
    itemsize = data.dtype.itemsize
    if data.strides != (w * c * itemsize, c * itemsize, itemsize):
        data = data.reshape(-1).reshape(h, w, c)

    fmt = _infer_format(c, precision)
    usage_flag = _parse_usage(usage)

    tex = device.create_texture(
        width=w,
        height=h,
        format=fmt,
        usage=usage_flag,
        data=data,
    )
    return tex


# -------------------------------------------------------------------
# Readback: Texture2D → NumPy float32
# -------------------------------------------------------------------
def texture_to_np(tex: spy.Texture, clip: bool = False) -> np.ndarray:
    """
    Read a texture into float32 — lossless by default.

    Values are returned exactly as stored (normalized tensors, HDR data and
    contrast overshoot legitimately leave [0, 1]). Pass clip=True, or use
    texture_to_pil, when you want display-ready color data.
    """
    from .device import _claim_device_thread

    _claim_device_thread(tex.device)
    arr = tex.to_numpy().astype(np.float32, copy=False)
    if clip:
        arr = np.clip(arr, 0.0, 1.0)
    return arr


def texture_to_pil(tex: spy.Texture, mode: Optional[str] = None) -> Image.Image:
    arr = texture_to_np(tex, clip=True)
    arr_u8 = (arr * 255.0).round().astype(np.uint8)

    if mode is None:
        c = arr_u8.shape[2]
        mode = {1: "L", 3: "RGB", 4: "RGBA"}.get(c)
        if mode is None:
            raise ValueError(f"Cannot infer PIL mode for shape {arr_u8.shape}")

    return Image.fromarray(arr_u8, mode=mode)


# -------------------------------------------------------------------
# PIL → NumPy helpers
# -------------------------------------------------------------------
def pil_to_np_rgba32(pil_img: Image.Image) -> np.ndarray:
    """
    Convert PIL image to RGBA, normalized to float32 in [0..1].

    Honors EXIF orientation (a rotated camera JPEG comes out upright) and scales
    16-bit ("I;16") sources from their full range instead of clipping through
    8-bit.
    """
    from PIL import ImageOps

    pil_img = ImageOps.exif_transpose(pil_img)  # apply camera orientation tag
    if pil_img.mode in ("I;16", "I;16B", "I;16L"):
        gray = np.asarray(pil_img).astype(np.float32) / 65535.0
        return np.dstack([gray, gray, gray, np.ones_like(gray)]).astype(np.float32)
    arr = np.asarray(pil_img.convert("RGBA")).astype(np.float32)
    return arr / 255.0


def load_rgba_texture_from_image(
    device: spy.Device,
    img_path: str,
    usage: spy.TextureUsage | str = "read",
    precision: str = "fp16",
) -> spy.Texture:
    """
    Convenience helper:
      - loads an image from disk
      - converts to RGBA float [0,1]
      - casts to the requested precision ("fp16" or "fp32")
      - uploads to GPU as a Texture2D

    precision:
        "fp16" (default): half-float storage, the library default
        "fp32": full float
    """
    img = Image.open(img_path)
    np_img = pil_to_np_rgba32(img)
    np.clip(np_img, 0.0, 1.0, out=np_img)

    # Apply requested precision
    if precision == "fp16":
        np_img = np_img.astype("float16")
    elif precision == "fp32":
        np_img = np_img.astype("float32")
    else:
        raise ValueError(f"Unsupported precision '{precision}'. Use 'fp16' or 'fp32'.")

    tex = np_to_texture(device, np_img, usage=usage, precision=precision)
    return tex


def save_texture_to_image(
    tex: spy.Texture,
    out_path: str,
    mode: str | None = "RGBA",
    quality: int = 100,
) -> str:
    img = texture_to_pil(tex, mode=None)

    if mode is not None and img.mode != mode:
        img = img.convert(mode)

    suffix = Path(out_path).suffix.lower()

    if suffix in (".jpg", ".jpeg"):
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")   # JPEG has no alpha channel
        img.save(
            out_path,
            format="JPEG",
            quality=quality,
            subsampling=0,   # VERY important: disable chroma subsampling
            optimize=True,
        )
    else:
        img.save(out_path)

    return out_path
