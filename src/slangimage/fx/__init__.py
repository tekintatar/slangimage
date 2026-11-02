# slangimage.fx — higher-level effect pipelines (multi-op graphs with pooled
# intermediates). Each effect subclasses Effect and runs inside a texture-pool
# frame; construct with (device, modules).

from .base import Effect
from .gaussian_blur import GaussianBlurEffect
from .kuwahara_effect import KuwaharaPaintEffect
from .soft_glow import SoftGlowEffect

__all__ = [
    "Effect",
    "GaussianBlurEffect",
    "KuwaharaPaintEffect",
    "SoftGlowEffect",
]
