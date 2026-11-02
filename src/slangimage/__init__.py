"""SlangImage — cross-platform GPU image processing built on Slang/slangpy.

Runs the same Slang kernels on Metal, CUDA, and Vulkan. Convention:

    import slangimage as si

    img = si.load("photo.jpg")                     # fp16 on the GPU
    out = si.ops.surface_blur(img, radius=9, bsigma=0.05)
    out = si.ops.vibrance(out, amount=0.6)
    out.save("out.png")
"""

from __future__ import annotations

__version__ = "0.1.0"

from .image import Image, load, from_numpy, from_pil, device, wrap_pooled

from .runtime import (
    Backend,
    DeviceManager,
    ModuleCache,
    get_default_device,
    release_device,
    load_rgba_texture_from_image,
    save_texture_to_image,
)

from . import ops
from . import fx
from .ops import prewarm

__all__ = [
    "__version__",
    # public API
    "Image",
    "load",
    "from_numpy",
    "from_pil",
    "device",
    "wrap_pooled",
    "prewarm",
    "ops",
    "fx",
    # lower-level building blocks
    "Backend",
    "DeviceManager",
    "ModuleCache",
    "get_default_device",
    "release_device",
    "load_rgba_texture_from_image",
    "save_texture_to_image",
]
