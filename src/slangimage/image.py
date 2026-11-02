# slangimage/image.py
# The public image handle and constructors. An Image wraps a GPU texture and
# hides the storage layer; ops resolve the device (and kernel caches) from it,
# so user code never plumbs (device, modules) around.

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from PIL import Image as PILImage

import slangpy as spy

from .runtime.device import get_default_device, Backend
from .runtime.textures import (
    np_to_texture,
    pil_to_np_rgba32,
    texture_to_np,
    texture_to_pil,
    save_texture_to_image,
)

_SUPPORTED_IMAGE_FORMATS = {spy.Format.rgba16_float, spy.Format.rgba32_float}


def _validate_image_texture(texture: spy.Texture) -> spy.Texture:
    """Validate the storage contract shared by Image, ops, and effects."""
    if not isinstance(texture, spy.Texture):
        raise TypeError(
            "Image expects a slangpy Texture, "
            f"got {type(texture).__name__}."
        )
    from .runtime.device import _claim_device_thread

    _claim_device_thread(texture.device)
    if texture.format not in _SUPPORTED_IMAGE_FORMATS:
        raise ValueError(
            f"unsupported image texture format {texture.format}; slangimage "
            "requires rgba16_float or rgba32_float storage. Convert scalar or "
            "multi-plane data to RGBA before wrapping it."
        )
    if not (
        int(texture.desc.usage) & int(spy.TextureUsage.shader_resource)
    ):
        raise ValueError(
            "image textures must include TextureUsage.shader_resource so "
            "slangimage kernels can read them."
        )
    return texture


class Image:
    """A GPU image: a thin handle over a slangpy texture.

    Create with slangimage.load / from_numpy / from_pil, or wrap an existing
    spy.Texture. Ops in slangimage.ops take and return Image instances.
    """

    # __weakref__ lets the texture pool auto-recycle op outputs when their
    # Image handle is garbage collected (runtime/texture_pool.auto_recycle)
    __slots__ = ("texture", "__weakref__")

    def __init__(self, texture: spy.Texture):
        self.texture = _validate_image_texture(texture)

    # ------------------------------------------------------------ properties
    @property
    def device(self) -> spy.Device:
        _validate_image_texture(self.texture)
        return self.texture.device

    @property
    def width(self) -> int:
        _validate_image_texture(self.texture)
        return self.texture.width

    @property
    def height(self) -> int:
        _validate_image_texture(self.texture)
        return self.texture.height

    @property
    def format(self) -> spy.Format:
        _validate_image_texture(self.texture)
        return self.texture.format

    @property
    def precision(self) -> str:
        _validate_image_texture(self.texture)
        return "fp16" if self.texture.format == spy.Format.rgba16_float else "fp32"

    # ------------------------------------------------------------ conversion
    def to_numpy(self, clip: bool = False) -> np.ndarray:
        """Read back as float32 HxWxC. Lossless unless clip=True."""
        return texture_to_np(self.texture, clip=clip)

    def to_pil(self) -> PILImage.Image:
        return texture_to_pil(self.texture)

    def save(self, path: str | Path, quality: int = 95) -> str:
        """Write to disk (JPEG quality applies to .jpg/.jpeg only)."""
        return save_texture_to_image(self.texture, str(path), quality=quality)

    # -------------------------------------------------------------- chaining
    def pipe(self, fn: Callable[..., "Image"], *args: Any, **kwargs: Any) -> "Image":
        """out = img.pipe(si.ops.gaussian_blur, sigma=2.0)"""
        return fn(self, *args, **kwargs)

    def __repr__(self) -> str:
        return f"Image({self.width}x{self.height}, {self.precision}, {str(self.device.info.type).split('.')[-1]})"


# ------------------------------------------------------------- pooling
def wrap_pooled(texture: spy.Texture) -> Image:
    """Wrap an op/effect output texture in an Image that returns the texture to
    the device's pool when the Image is garbage-collected — this is what keeps
    repeated processing allocation-free. Holding ``img.texture`` opts out (the
    texture is then yours and never recycled). Use this for any custom op or
    effect that produces a texture the caller owns."""
    from .runtime.texture_pool import get_pool

    img = Image(texture)
    get_pool(texture.device).auto_recycle(img, texture)
    return img


# ------------------------------------------------------------- constructors
def _resolve_device(device: spy.Device | Backend | str | None) -> spy.Device:
    """``None`` is the automatic default; a backend name or member is that
    backend's process-wide default (as :func:`device` returns)."""
    if device is None:
        return get_default_device()
    if isinstance(device, str):  # includes Backend members (a str enum)
        return get_default_device(device)
    return device


def from_numpy(
    arr: np.ndarray,
    precision: Optional[str] = None,
    device: spy.Device | Backend | str | None = None,
) -> Image:
    """Upload an HxWxC float array in [0,1] (C = 3 or 4; 3 is expanded to
    opaque RGBA). precision defaults to fp16 for float16 input, fp32 otherwise.
    Integer arrays are rejected — normalize to float [0,1] first (e.g. / 255).
    ``device`` is a device (e.g. ``si.device("vulkan")``) or a backend name
    (``"auto"``, ``"metal"``, ``"cuda"``, ``"vulkan"``); ``None`` means auto."""
    if not np.issubdtype(arr.dtype, np.floating):
        raise ValueError(
            f"expected a float array in [0, 1], got dtype {arr.dtype}; "
            "normalize first, e.g. arr.astype(np.float32) / 255."
        )
    if precision is None:
        precision = "fp16" if arr.dtype == np.float16 else "fp32"
    device = _resolve_device(device)
    return Image(np_to_texture(device, arr, usage="read", precision=precision))


def from_pil(
    img: PILImage.Image,
    precision: str = "fp16",
    device: spy.Device | Backend | str | None = None,
) -> Image:
    """Upload a PIL image. fp16 by default — lossless for 8-bit sources.
    ``device`` is a device or a backend name, as for :func:`from_numpy`."""
    arr = pil_to_np_rgba32(img)
    device = _resolve_device(device)
    return Image(np_to_texture(device, arr.astype(np.float16 if precision == "fp16" else np.float32), usage="read", precision=precision))


def load(
    path: str | Path,
    precision: str = "fp16",
    device: spy.Device | Backend | str | None = None,
) -> Image:
    """Load an image file onto the GPU. fp16 by default — lossless for 8-bit
    sources, with half the texture storage and lower memory traffic than fp32.
    ``device`` is a device or a backend name, as for :func:`from_numpy`."""
    return from_pil(PILImage.open(str(path)), precision=precision, device=device)


def device(backend: Backend | str = Backend.AUTO) -> spy.Device:
    """The calling thread's process-wide default for a backend.

    The first caller owns the cached device; another thread must construct its
    own DeviceManager inside that worker rather than sharing this default.
    AUTO selects Metal on macOS and CUDA on Linux when available.
    """
    return get_default_device(backend)
