# fx/base.py

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Literal, Optional, TypeVar, cast, overload

import time

from PIL import Image as PILImage
import slangpy as spy

from ..image import Image as SlangImage
from ..runtime.textures import (
    pil_to_np_rgba32,
    np_to_texture,
    texture_to_pil,
)
from ..runtime.texture_pool import get_pool


EffectInputT = TypeVar("EffectInputT", SlangImage, spy.Texture)


class Effect(ABC):
    """
    High level effect base.

    Core API is texture based. PIL helpers are convenience entry points.

    Subclasses implement:
        _apply_impl(self, src_tex, **params) -> spy.Texture

    Public entry points:
        apply(src_tex, **params)        # texture in, texture out
        apply_pil(img, **params)        # PIL in, PIL out by default
    """

    def __init__(
        self,
        device: spy.Device,
        modules: Any,
        debug_timing: bool = False,
        timing_logger: Optional[Callable[[str, float], None]] = None,
    ):
        """
        device:
            SlangPy device

        modules:
            ModuleCache or similar (kept opaque on purpose)

        debug_timing:
            If True, apply() will print or log time in milliseconds.

        timing_logger:
            Optional callback: fn(effect_name: str, ms: float)
            If None, timing is printed to stdout when debug_timing is True.
        """
        from ..runtime.device import _claim_device_thread

        _claim_device_thread(device)
        self.device = device
        self.modules = modules
        modules_device = getattr(modules, "device", None)
        if modules_device is not None and modules_device is not device:
            raise ValueError(
                "effect device and module-cache device must be the same "
                "device instance; construct both from one DeviceManager."
            )
        self.debug_timing = bool(debug_timing)
        self._timing_logger = timing_logger

    @abstractmethod
    def _apply_impl(self, src_tex: spy.Texture, **params: Any) -> spy.Texture:
        """
        Core implementation that subclasses must provide.

        Contract:
            src_tex: input texture
            returns: new texture with the effect applied

        Do not add timing here. The base class handles that
        in apply() when debug_timing is enabled.
        """
        raise NotImplementedError

    def apply(self, src_tex: EffectInputT, **params: Any) -> EffectInputT:
        """
        Main entry point. Accepts an Image or a raw texture and returns the
        same kind.

        Runs _apply_impl inside a texture-pool frame: intermediates return
        to the pool when the frame ends, the result texture escapes.
        Optional debug timing wraps the whole frame.
        """
        from ..image import _validate_image_texture, wrap_pooled

        wrapped = isinstance(src_tex, SlangImage)
        input_tex = (
            cast(SlangImage, src_tex).texture
            if wrapped
            else cast(spy.Texture, src_tex)
        )
        input_tex = _validate_image_texture(input_tex)
        if input_tex.device is not self.device:
            raise ValueError(
                "effect input and effect must live on the same device. They "
                "don't — note that AUTO and an explicit backend, or two "
                "DeviceManager instances, are distinct devices; construct the "
                "effect and image from one device."
            )

        pool = get_pool(self.device)

        if not self.debug_timing:
            with pool.frame() as fr:
                out_tex = self._apply_impl(input_tex, **params)
                fr.escape(out_tex)
            return cast(EffectInputT, wrap_pooled(out_tex) if wrapped else out_tex)

        t0 = time.perf_counter()
        with pool.frame() as fr:
            out_tex = self._apply_impl(input_tex, **params)
            fr.escape(out_tex)
        t1 = time.perf_counter()

        ms = (t1 - t0) * 1000.0
        name = self.__class__.__name__

        if self._timing_logger is not None:
            self._timing_logger(name, ms)
        else:
            print(f"[Effect timing] {name}: {ms:.3f} ms")

        return cast(EffectInputT, wrap_pooled(out_tex) if wrapped else out_tex)

    @overload
    def apply_pil(
        self,
        img: PILImage.Image,
        return_texture: Literal[False] = False,
        precision: str = "fp16",
        **params: Any,
    ) -> PILImage.Image: ...

    @overload
    def apply_pil(
        self,
        img: PILImage.Image,
        return_texture: Literal[True],
        precision: str = "fp16",
        **params: Any,
    ) -> spy.Texture: ...

    def apply_pil(
        self,
        img: PILImage.Image,
        return_texture: bool = False,
        precision: str = "fp16",
        **params: Any,
    ) -> PILImage.Image | spy.Texture:
        """
        Convenience wrapper for PIL images.

        Steps:
          1) PIL -> NumPy -> Texture (fp16 by default: lossless for 8-bit
             sources with half the fp32 texture storage; pass precision="fp32"
             to override)
          2) apply(...) with optional debug timing
          3) Texture -> PIL (unless return_texture=True)
        """
        np_img = pil_to_np_rgba32(img)
        src_tex = np_to_texture(self.device, np_img, usage="read", precision=precision)

        dst_tex = self.apply(src_tex, **params)

        if return_texture:
            return dst_tex

        out_img = texture_to_pil(dst_tex, mode="RGBA")
        # the output texture never leaves this function — hand it back to
        # the pool so repeated apply_pil calls are allocation-free
        get_pool(self.device).recycle(dst_tex)
        return out_img
