# augment/_runtime.py — the CUDA device, the table of input images and the
# compiled-module cache. Kernels run on PyTorch's current CUDA stream.

from __future__ import annotations

import contextlib
import functools
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import slangpy as spy

if TYPE_CHECKING:
    import torch

_LIBRARY = Path(__file__).resolve().parent.parent / "shaders" / "augment" / "augment.slang"

_DEVICES: dict[int, spy.Device] = {}
_MODULES: dict[tuple[int, str], spy.Module] = {}


@functools.lru_cache(maxsize=1)
def library_source() -> str:
    return _LIBRARY.read_text()


def device() -> spy.Device:
    """SlangPy's CUDA device sharing PyTorch's context on the current GPU."""
    import torch

    from ..runtime import device as runtime

    if not torch.cuda.is_available():
        raise RuntimeError("slangimage.augment needs a CUDA GPU and a CUDA build of PyTorch")
    index = torch.cuda.current_device()
    if index in _DEVICES:
        return _DEVICES[index]
    try:
        dev = spy.create_torch_device(type=spy.DeviceType.cuda)
    except RuntimeError as e:
        raise RuntimeError(f"slangimage.augment could not create SlangPy's CUDA device "
                           f"({runtime._summarize(e)}). {runtime._NVRTC_HINT}.") from e
    try:
        runtime._verify_runtime_compiler(dev)
    except Exception as e:
        runtime._close_quietly(dev)
        raise RuntimeError("slangimage.augment created SlangPy's CUDA device but could not "
                           f"compile a test kernel on it ({runtime._summarize(e)}). "
                           f"{runtime._NVRTC_HINT}.") from e
    _DEVICES[index] = dev
    return dev


@contextlib.contextmanager
def torch_bridge():
    """Allow SlangPy's Python torch bridge for augmentation calls only.

    It costs well under 0.1 ms per batch here, so the compiled slangpy-torch
    extension is optional for this path. The setting is process-wide in
    SlangPy, so it is restored afterwards: slangimage.diff keeps reporting a
    missing compiled bridge."""
    if not spy.is_torch_bridge_using_fallback():
        yield
        return
    from slangpy.core import calldata

    previous = bool(getattr(calldata, "_ALLOW_TORCH_FALLBACK", False))
    spy.set_allow_torch_fallback(True)
    try:
        yield
    finally:
        spy.set_allow_torch_fallback(previous)


def load_module(source: str) -> spy.Module:
    """Compile a generated pipeline once per GPU (cached by source)."""
    import torch

    dev = device()
    digest = hashlib.sha1(source.encode()).hexdigest()[:16]
    key = (torch.cuda.current_device(), digest)
    if key not in _MODULES:
        try:
            _MODULES[key] = spy.Module.load_from_source(dev, f"slangimage_augment_{digest}", source)
        except Exception as e:
            raise RuntimeError("slangimage.augment could not compile this pipeline; "
                               "Compose.source shows the generated Slang (a ColorOp or "
                               f"NeighbourhoodOp you defined is the usual cause): {e}") from e
    return _MODULES[key]


@dataclass
class ImageTable:
    """Device addresses and sizes of the input images, read in place."""

    ptrs: "torch.Tensor | None"     # (B,) int64 device addresses
    meta: "torch.Tensor | None"     # (B, 2) int32 rows: (H, W)
    count: int
    size: tuple[int, int] | None    # (H, W) when every image has it
    device: "torch.device"
    keep: list                      # the (contiguous) inputs, alive during the call


def _checked(t: Any, dev: "torch.device", layout: str, batched: bool) -> "torch.Tensor":
    import torch

    want = 4 if batched else 3
    shape = "(B, 3, H, W)" if batched else "(3, H, W)"
    if layout == "hwc":
        shape = "(B, H, W, 3)" if batched else "(H, W, 3)"
    if not isinstance(t, torch.Tensor) or t.ndim != want:
        raise ValueError(f"expected uint8 CUDA images of shape {shape}, got "
                         f"{getattr(t, 'shape', type(t).__name__)}")
    if t.dtype != torch.uint8:
        raise ValueError(f"images must be uint8 (decoder output), got {t.dtype}")
    if t.device != dev:
        raise ValueError(f"images must be on the current CUDA device {dev}, got {t.device}")
    channels = t.shape[-1] if layout == "hwc" else t.shape[-3]
    if channels != 3:
        raise ValueError(f"images must have 3 (RGB) channels in layout {layout!r}, "
                         f"got shape {tuple(t.shape)}")
    if 0 in (t.shape[-2:] if layout == "chw" else t.shape[-3:-1]):
        raise ValueError(f"images must not be empty, got shape {tuple(t.shape)}")
    return t.contiguous()


def image_table(images: Any, layout: str) -> ImageTable:
    import torch

    if layout not in ("chw", "hwc"):
        raise ValueError(f"layout must be 'chw' or 'hwc', got {layout!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("slangimage.augment needs a CUDA GPU and a CUDA build of PyTorch")
    dev = torch.device("cuda", torch.cuda.current_device())
    if isinstance(images, torch.Tensor):
        x = _checked(images, dev, layout, batched=True)
        B = x.shape[0]
        H, W = (x.shape[2], x.shape[3]) if layout == "chw" else (x.shape[1], x.shape[2])
        if B == 0:
            return ImageTable(None, None, 0, (H, W), dev, [])
        ptrs = x.data_ptr() + torch.arange(B, dtype=torch.int64, device=dev) * (3 * H * W)
        meta = torch.tensor([H, W], dtype=torch.int32, device=dev).repeat(B, 1)
        return ImageTable(ptrs, meta, B, (H, W), dev, [x])
    imgs = [_checked(t, dev, layout, batched=False) for t in images]
    if not imgs:
        return ImageTable(None, None, 0, None, dev, [])
    dims = [(t.shape[1], t.shape[2]) if layout == "chw" else (t.shape[0], t.shape[1])
            for t in imgs]
    ptrs = torch.tensor([t.data_ptr() for t in imgs], dtype=torch.int64).pin_memory()
    meta = torch.tensor(dims, dtype=torch.int32).pin_memory()
    size = dims[0] if all(d == dims[0] for d in dims) else None
    return ImageTable(ptrs.to(dev, non_blocking=True), meta.to(dev, non_blocking=True),
                      len(imgs), size, dev, imgs)
