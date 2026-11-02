# slangimage/image.py
# The public image handle and constructors. An Image wraps a GPU texture and
# hides the storage layer; ops resolve the device (and kernel caches) from it,
# so user code never plumbs (device, modules) around.

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

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

if TYPE_CHECKING:
    import torch

_SUPPORTED_IMAGE_FORMATS = {spy.Format.rgba16_float, spy.Format.rgba32_float}
_TORCH_LAYOUTS = ("hwc", "chw")


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

    def to_torch(
        self, device: Any = None, layout: str = "hwc", alpha: bool = False
    ) -> "torch.Tensor":
        """Copy into a new torch tensor. Lossless, but not zero-copy: the data
        goes through host memory, like :meth:`to_numpy`.

        dtype follows storage: float16 for fp16 images, float32 for fp32.
        ``layout="hwc"`` gives (H, W, C), the layout ``slangimage.diff`` takes;
        ``"chw"`` gives (C, H, W) for torch models (``unsqueeze(0)`` makes a
        batch of one). C is 3 unless ``alpha=True``. ``device`` is any torch
        device (e.g. ``"cuda"``); ``None`` keeps the tensor on the CPU. Needs
        PyTorch, but not the ``torch`` extra.
        """
        torch = _import_torch("Image.to_torch")
        _check_layout(layout)
        arr = self.to_numpy()
        if not alpha:
            arr = arr[..., :3]
        if layout == "chw":
            arr = arr.transpose(2, 0, 1)
        dtype = torch.float16 if self.precision == "fp16" else torch.float32
        tensor = torch.from_numpy(np.ascontiguousarray(arr)).to(dtype)
        return tensor if device is None else tensor.to(device)

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


def from_torch(
    tensor: "torch.Tensor",
    layout: str = "hwc",
    precision: Optional[str] = None,
    device: spy.Device | Backend | str | None = None,
) -> Image:
    """Copy one image from a float torch tensor in [0, 1] (any torch device).

    The tensor is (H, W, C) for ``layout="hwc"`` or (C, H, W) for ``"chw"``,
    with C = 3 or 4 (3 is expanded to opaque RGBA). Batches are not accepted:
    convert each image, e.g. ``[si.from_torch(t, layout="chw") for t in x]``.
    precision defaults to fp16 for float16 tensors, fp32 otherwise (bfloat16
    and float64 are converted). ``device`` selects the SlangImage device as for
    :func:`from_numpy`; the data is copied through host memory. Gradients are
    not tracked.
    """
    torch = _import_torch("from_torch")
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"expected a torch.Tensor, got {type(tensor).__name__}")
    _check_layout(layout)
    if tensor.dim() != 3:
        raise ValueError(
            f"expected one image as a 3-D tensor, got shape {tuple(tensor.shape)}; "
            "for a batch, convert each image separately"
        )
    if not tensor.is_floating_point():
        raise ValueError(
            f"expected a float tensor in [0, 1], got dtype {tensor.dtype}; "
            "normalize first, e.g. tensor.float() / 255"
        )
    channels = tensor.shape[0] if layout == "chw" else tensor.shape[-1]
    if channels not in (3, 4):
        looks_chw = layout == "hwc" and tensor.shape[0] in (3, 4)
        hint = ' (pass layout="chw" for C, H, W tensors)' if looks_chw else ""
        raise ValueError(
            f"expected 3 or 4 channels, got shape {tuple(tensor.shape)} "
            f"with layout {layout!r}{hint}"
        )
    if precision is None:
        precision = "fp16" if tensor.dtype == torch.float16 else "fp32"
    data = tensor.detach()
    if layout == "chw":
        data = data.permute(1, 2, 0)
    if data.dtype not in (torch.float16, torch.float32):
        data = data.float()
    return from_numpy(data.contiguous().cpu().numpy(), precision=precision, device=device)


def _import_torch(caller: str) -> Any:
    try:
        import torch
    except ImportError:
        raise ImportError(
            f"{caller} needs PyTorch (pip install torch); the slangimage "
            "'torch' extra is only required for slangimage.diff."
        ) from None
    return torch


def _check_layout(layout: str) -> None:
    if layout not in _TORCH_LAYOUTS:
        raise ValueError(f"layout must be 'hwc' or 'chw', got {layout!r}")


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
