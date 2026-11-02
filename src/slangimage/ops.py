# slangimage.ops — the functional operator API.
#
#     import slangimage as si
#     img = si.load("photo.jpg")                  # fp16 by default
#     out = si.ops.surface_blur(img, radius=9, bsigma=0.05)
#     out = si.ops.vibrance(out, amount=0.6)
#     out.save("out.png")
#
# Every function takes and returns an Image (raw spy.Texture also accepted,
# returning a raw texture). The device and kernel caches are resolved from
# the input — no (device, modules) plumbing. Kernels compile lazily on first
# use per device/precision; call slangimage.prewarm() in servers.

from __future__ import annotations

import weakref
from collections.abc import Sequence
from typing import Any, Optional, TypeVar, cast

import slangpy as spy

from . import _validation as _v
from .image import Image, _validate_image_texture, wrap_pooled
from .runtime.device import get_default_device
from .runtime.modules import get_module_cache
from .runtime.texture_pool import get_pool

# op classes (also usable directly for advanced/stateful use)
from .kernels.gaussian import (
    GaussianBlurOp,
    validate_blur_amount,
    validate_gaussian_params,
)
from .kernels.surface_blur import SurfaceBlurOp
from .kernels.unsharp_mask import UnsharpMaskOp
from .kernels.ep_unsharp_mask import EdgePreservingUnsharpMaskOp
from .kernels.sharpen_luma import SharpenLumaOp
from .kernels.vibrance import VibranceOp
from .kernels.gamma_adjust import GammaAdjustOp
from .kernels.color_controls import ColorControlsOp
from .kernels.colorspace import SRGBToLinearOp, LinearToSRGBOp
from .kernels.resize import ResizeOp
from .kernels.kuwahara import KuwaharaOp
from .kernels.bilateral import BilateralOp
from .kernels.clahe import ClaheOp
from .kernels.nl_means import NLMeansOp
from .kernels.stylize import PointillizeOp, CmykHalftoneOp
from .kernels.blend_modes import BlendModesOp, BLEND_MODES  # noqa: F401 (re-export)
from .kernels.point_ops import PointOps
from .kernels.lut import (  # noqa: F401 (identity_lut_np re-export)
    LutOp,
    identity_lut_np,
    validate_lut_size,
)
from .kernels.smooth_threshold import SmoothThresholdOp, validate_threshold_range
from .kernels.color_adjust import ColorAdjustOps
from .kernels.color_aug import ColorAugOps
from .kernels.hist_color import HistColorOps
from .kernels.white_balance import WhiteBalanceOp
from .kernels.geometry import GeometryOps, rotation_matrix

# Instances own their device/module cache, so release_device() is the lifecycle
# boundary; weak keys prevent stale id reuse rather than making this self-clearing.
_instances: "weakref.WeakKeyDictionary[spy.Device, dict[type, Any]]" = weakref.WeakKeyDictionary()
_ImageT = TypeVar("_ImageT", Image, spy.Texture)
_ImageLike = Image | spy.Texture


def _unwrap(x: _ImageLike) -> tuple[spy.Texture, bool]:
    if isinstance(x, Image):
        return _validate_image_texture(x.texture), True
    return _validate_image_texture(x), False


def _op_for_device(device: spy.Device, cls: type):
    from .runtime.device import _claim_device_thread

    _claim_device_thread(device)
    per_device = _instances.get(device)
    if per_device is None:
        per_device = {}
        _instances[device] = per_device
    inst = per_device.get(cls)
    if inst is None:
        inst = cls(device, get_module_cache(device))
        per_device[cls] = inst
    return inst


def _op(tex: spy.Texture, cls: type):
    return _op_for_device(tex.device, cls)


def _wrap_out(out: spy.Texture) -> Image:
    """Wrap an op output; when the Image is dropped the pool reclaims the
    texture (kills the per-call device.create_texture dispatch floor —
    ~0.3 ms on CUDA). Raw-texture callers manage lifetime themselves."""
    return wrap_pooled(out)


def _run1(cls: type, method: str, image: _ImageT, *args: Any, **kwargs: Any) -> _ImageT:
    tex, wrapped = _unwrap(image)
    # Run inside a pool frame so any scratch the op allocates (e.g. the blur in
    # unsharp) returns to the free list; only the escaped output survives.
    with get_pool(tex.device).frame() as fr:
        out = getattr(_op(tex, cls), method)(tex, *args, **kwargs)
        fr.escape(out)
    return cast(_ImageT, _wrap_out(out) if wrapped else out)


def _run2(
    cls: type,
    method: str,
    image: _ImageT,
    image2: _ImageLike,
    *args: Any,
    same_size_for: str | None = None,
    **kwargs: Any,
) -> _ImageT:
    tex, wrapped = _unwrap(image)
    tex2, _ = _unwrap(image2)
    if tex.device is not tex2.device:
        raise ValueError(
            "both inputs must live on the same device. They don't — note that "
            "AUTO and an explicit backend, or two DeviceManager instances, are "
            "distinct devices; resolve both images from one device."
        )
    if same_size_for is not None and (
        tex.width != tex2.width or tex.height != tex2.height
    ):
        raise ValueError(
            f"{same_size_for} requires inputs with matching width and height; "
            f"got {tex.width}x{tex.height} and {tex2.width}x{tex2.height}. "
            "Resize the secondary input explicitly before calling the op."
        )
    with get_pool(tex.device).frame() as fr:
        out = getattr(_op(tex, cls), method)(tex, tex2, *args, **kwargs)
        fr.escape(out)
    return cast(_ImageT, _wrap_out(out) if wrapped else out)


# ---------------------------------------------------------------- blurs
def gaussian_blur(image: _ImageT, radius: Optional[int] = None,
                  sigma: Optional[float] = None,
                  amount: Optional[float] = None) -> _ImageT:
    """Separable Gaussian blur of stored RGBA values, including alpha.

    Uses normalized weights and edge replication, without linear-light
    conversion or clipping. The horizontal intermediate uses the input storage
    precision. Give (radius, sigma), or an amount with sigma = max(0.1, amount)
    * sqrt(width * height / 1e6) and radius = max(1, ceil(3 * sigma)).
    """
    if amount is not None:
        if radius is not None or sigma is not None:
            raise ValueError("choose amount or radius/sigma, not both")
        amount = validate_blur_amount(amount)
        return _run1(GaussianBlurOp, "apply_with_amount", image, amount)
    if radius is None or sigma is None:
        raise ValueError("give either amount=… or both radius=… and sigma=…")
    radius, sigma = validate_gaussian_params(radius, sigma)
    return _run1(GaussianBlurOp, "apply", image, radius, sigma)


def surface_blur(image: _ImageT, radius: int = 9, bsigma: float = 0.1) -> _ImageT:
    """RGB bilateral surface blur; source alpha is unchanged.

    The legacy ``radius`` parameter is a sample-count control: the pixel
    half-width and spatial sigma are floor((radius-1)/2). Thus 3 and 4 both
    use a 3x3 footprint, and 1/2 return the source pixel.
    ``bsigma`` is the range sigma on stored RGB, without linear-light
    conversion or clipping. Borders replicate edge pixels.
    """
    radius, bsigma = _v.radius("radius", radius), _v.sigma("bsigma", bsigma)
    return _run1(SurfaceBlurOp, "apply", image, radius, bsigma)


def bilateral(image: _ImageT, radius: int = 5, sigma_spatial: float = 0.0,
              sigma_range: float = 0.15, range_type: str = "gaussian",
              separable: bool = False) -> _ImageT:
    """RGB bilateral filtering with a square, edge-replicated neighborhood.

    sigma_spatial<=0 selects box spatial weights; otherwise weights are
    Gaussian in pixel distance. Range weights use Euclidean RGB distance:
    Gaussian with sigma_range, or an inclusive hard cutoff at sigma_range.
    Zero range sigma is allowed only for cutoff (equal RGB samples only).

    RGB is not clipped or linearized; alpha is preserved and does not affect
    weights. separable=True is an approximation: horizontal then vertical
    1D passes, with the intermediate stored at the input precision. Vertical
    range weights are recomputed from that intermediate, not the original.
    """
    radius = _v.radius("radius", radius)
    sigma_spatial = _v.finite_float("sigma_spatial", sigma_spatial)
    if sigma_spatial > 0:
        sigma_spatial = _v.sigma("sigma_spatial", sigma_spatial)
    if range_type not in ("gaussian", "cutoff"):
        raise ValueError("range_type must be 'gaussian' or 'cutoff'")
    sigma_range = _v.finite_float("sigma_range", sigma_range)
    if not (range_type == "cutoff" and sigma_range == 0):
        sigma_range = _v.sigma("sigma_range", sigma_range)
    separable = _v.boolean("separable", separable)
    method = "apply_separable" if separable else "apply"
    return _run1(BilateralOp, method, image, radius, sigma_spatial, sigma_range, range_type)


def nl_means(image: _ImageT, search_radius: int = 5, patch_radius: int = 1,
             h: float = 0.1, sigma: float = 0.0) -> _ImageT:
    """Patch-based non-local means filtering of unclipped stored RGB.

    Patch distance is squared difference averaged over all patch pixels and
    three RGB channels. Neighbor weights are exp(-max(distance-2*sigma**2, 0)
    / max(h**2, 1e-8)); h=0 therefore still permits smoothing. The center uses
    max(largest neighbor weight, 1e-6), not a fixed weight of 1.

    Patch samples and candidate RGB use edge replication. Candidate patch
    offsets are added before clamping, so replicated candidate centers can
    still have different patches. Search offsets retain their multiplicity.
    Alpha is preserved and ignored by weights; no linear-light conversion.
    search_radius=0 gives identity subject to arithmetic/storage rounding.
    Cost grows with the product of search-window and patch areas.
    """
    search_radius = _v.radius("search_radius", search_radius, minimum=0)
    patch_radius = _v.radius("patch_radius", patch_radius, minimum=0)
    h, sigma = _v.finite_float("h", h), _v.finite_float("sigma", sigma)
    if not 0 <= h <= _v.MAX_SIGMA or not 0 <= sigma <= _v.MAX_SIGMA:
        raise ValueError(f"h and sigma must be in [0, {_v.MAX_SIGMA:g}]")
    return _run1(NLMeansOp, "apply", image, search_radius, patch_radius, h, sigma)


# ------------------------------------------------------------- sharpening
def unsharp_mask(image: _ImageT, radius: float = 2.0, sigma: float = 1.5,
                 amount: float = 1.0, color_space: str = "rgb") -> _ImageT:
    """Texture unsharp-mask sharpening, with source alpha unchanged.

    'rgb' returns clip(rgb + amount * (rgb - gaussian_rgb), 0, 1), operating
    on stored color values. 'luma' decodes source and blurred RGB to linear
    light, sharpens/clamps Y in the library's rounded BT.601-style YCbCr
    transform, retains source Cb/Cr, and encodes/clips RGB on return. That
    rounded color transform is not an exact identity even at amount=0.
    The Gaussian uses edge replication and input-precision intermediates.
    Unlike si.diff.unsharp_mask, the texture result is clipped to [0, 1].
    Larger radius/sigma widen the halo.
    """
    if color_space not in ("rgb", "luma"):
        raise ValueError("color_space must be 'rgb' or 'luma'")
    # The float-radius API historically truncates; keep that convention.
    radius = _v.finite_float("radius", radius)
    radius, sigma = validate_gaussian_params(int(radius), sigma)
    amount = _v.finite_float("amount", amount)
    return _run1(UnsharpMaskOp, "apply", image, radius, sigma, amount, color_space)


def ep_unsharp_mask(image: _ImageT, radius: float = 3.0, bsigma: float = 0.12,
                    amount: float = 1.0) -> _ImageT:
    """Sharpen linear-light Y against a surface-blurred RGB base.

    ``radius`` is rounded (ties to even) and passed as surface_blur's sample
    count, not a pixel half-width. The default 3 uses a 3x3 footprint;
    explicit radii rounding to 1/2 give no base smoothing.
    The intermediate uses input precision.
    The rounded BT.601-style YCbCr transform preserves source Cb/Cr; Y is
    clipped to [0,1], but returned RGB is NOT clipped. Alpha is unchanged.
    Strong range edges reduce smoothing, without a general no-halo guarantee.
    """
    radius = _v.finite_float("radius", radius)
    _v.radius("radius (rounded)", round(radius))
    bsigma, amount = _v.sigma("bsigma", bsigma), _v.finite_float("amount", amount)
    return _run1(EdgePreservingUnsharpMaskOp, "apply", image, radius, bsigma, amount)


def sharpen_luma(image: _ImageT, radius: float = 4.0, sharpness: float = 1.0,
                 sharpen_rgb: bool = True) -> _ImageT:
    """Cubic-filter sharpening of RGB (default) or HSL lightness.

    ``sharpen_rgb=False`` retains HSL hue/saturation, not YCbCr chroma.
    RGB inputs and outputs are clipped to [0,1]; alpha is unchanged.
    Radius is clamped to [0.5,64] and scales cubic weights within a footprint
    of half-width ceil(clamp(radius,1,4)); borders replicate edge pixels.
    Sharpness is clamped to [-2,4]. Radius <=1 gives an identity filter.
    """
    radius, sharpness = _v.finite_float("radius", radius), _v.finite_float("sharpness", sharpness)
    sharpen_rgb = _v.boolean("sharpen_rgb", sharpen_rgb)
    return _run1(SharpenLumaOp, "apply", image, radius, sharpness, sharpen_rgb)


# ------------------------------------------------------------------ color
def vibrance(image: _ImageT, amount: float = 0.5) -> _ImageT:
    """HSL saturation adjustment with a fixed, RGB-based skin-protection mask.

    Amount is effectively clamped to [-1,1]; negative values desaturate.
    Operates on clipped stored RGB, not linear light. Output RGB is in [0,1]
    and alpha is unchanged. Gray pixels remain gray.
    """
    return _run1(VibranceOp, "apply", image, _v.finite_float("amount", amount))


def gamma(image: _ImageT, power: float = 1.0) -> _ImageT:
    """Per-channel gamma: clip(rgb, 0, 1) ** clip(power, 0.01, 10).

    Alpha is unchanged. Finite power < 1 brightens midtones and > 1 darkens;
    power=1 is identity only for RGB already in [0, 1]. No linear-light
    conversion is performed.
    """
    return _run1(GammaAdjustOp, "apply", image, _v.finite_float("power", power))


def color_controls(image: _ImageT, hue: float = 0.0, saturation: float = 0.0,
                   contrast: float = 0.0, brightness: float = 0.0) -> _ImageT:
    """Adjust hue, saturation, brightness and contrast; preserve alpha exactly.

    Decode sRGB, then use rounded BT.601-style coefficients on linear RGB.
    Brightness adds half its value to luma, which is clipped to [0, 1]. Hue
    rotates the chroma plane by hue*pi, saturation scales chroma by 1+saturation,
    and chroma magnitude is capped at 0.6. Hue/saturation/brightness clamp to
    [-1, 1]. After conversion back to sRGB, contrast scales RGB around 0.5
    by clamp(1+contrast, 0.001, 5): signed 0 is neutral, negative flattens,
    positive strengthens. Contrast beyond the usual [-1, 1] remains accepted.

    Final RGB is NOT clipped; negative/HDR output is possible. Neutral controls
    are only approximately identity because the rounded conversion matrices
    are not exact inverses; luma/chroma limits can also change input colors.
    The low-level ColorControlsOp instead takes factor-style contrast.
    """
    # The kernel takes factor-style contrast (1 = neutral, >1 stronger, <1
    # flatter) and is non-monotonic on a raw signed value; map the public signed
    # contrast onto it so positive always adds contrast. (The op class keeps the
    # factor convention for advanced/downstream callers.)
    hue = _v.finite_float("hue", hue)
    saturation = _v.finite_float("saturation", saturation)
    brightness = _v.finite_float("brightness", brightness)
    contrast_factor = max(1e-3, 1.0 + _v.finite_float("contrast", contrast))
    return _run1(ColorControlsOp, "apply", image, hue, saturation, contrast_factor, brightness)


def srgb_to_linear(image: _ImageT) -> _ImageT:
    """Decode RGB using the piecewise sRGB transfer function; preserve alpha.

    RGB <= 0.04045 uses x/12.92; higher values use ((x+.055)/1.055)**2.4.
    Neither input nor output is clipped: all negative values use the linear
    branch, not a sign-mirrored power law. Values above 1 use the power branch.
    Computation and branch cutoffs are fp32; output keeps input precision.
    Approximately inverse to linear_to_srgb, subject to rounded joins/storage.
    """
    return _run1(SRGBToLinearOp, "apply", image)


def linear_to_srgb(image: _ImageT) -> _ImageT:
    """Encode linear-light RGB using the sRGB transfer function; preserve alpha.

    RGB <= 0.0031308 uses 12.92*x; higher values use 1.055*x**(1/2.4)-.055.
    Neither input nor output is clipped: all negative values use the linear
    branch, not a sign-mirrored power law. Values above 1 use the power branch.
    Computation and branch cutoffs are fp32; output keeps input precision.
    Approximately inverse to srgb_to_linear, subject to rounded joins/storage.
    """
    return _run1(LinearToSRGBOp, "apply", image)


def exposure(image: _ImageT, ev: float = 0.0) -> _ImageT:
    """Multiply encoded RGB by ``2**ev``, clip RGB to [0, 1], and preserve
    alpha. ``ev`` is passed as fp32 and limited to [-126, 127]."""
    ev = _v.finite_float("ev", ev)
    if not -126 <= ev <= 127:
        raise ValueError("ev must be in [-126, 127] so exp2(ev) is normal and finite in fp32")
    return _run1(ColorAdjustOps, "exposure", image, ev)


def sepia(image: _ImageT, intensity: float = 1.0) -> _ImageT:
    """Sepia tone via a fixed color matrix, lerped from the original by
    intensity (0 = original, 1 = full sepia). Matrix RGB and intensity are
    each clipped to [0, 1] before interpolation; the original RGB endpoint is
    not clipped, so partial-intensity output may remain outside [0, 1]. Alpha
    is preserved."""
    return _run1(ColorAdjustOps, "sepia", image, _v.finite_float("intensity", intensity))


def hue_rotate(image: _ImageT, angle: float = 0.0) -> _ImageT:
    """Rotate encoded-RGB chroma in YIQ by angle radians; preserve alpha.

    Uses rounded YIQ conversion/reconstruction coefficients without sRGB
    decoding. Positive angles rotate I toward Q. Final RGB is clipped to [0, 1].
    Y is unchanged inside the transform, but output luma is only approximately
    preserved before clipping and can change substantially when channels clip.
    Zero/full-turn rotations are approximately, not exactly, identity.

    Finite fp32-range angles are accepted, including negative/multiple turns;
    they are rounded to fp32 and passed to shader sin/cos without host modulo
    reduction. Very large angles lose phase precision. Exact Core Image parity
    is not established.
    """
    return _run1(ColorAdjustOps, "hue", image, _v.finite_float("angle", angle))


def false_color(image: _ImageT, color0: Sequence[float] = (0.3, 0.0, 0.0),
                color1: Sequence[float] = (1.0, 0.9, 0.8)) -> _ImageT:
    """Map encoded-RGB luma to a duotone; preserve alpha exactly.

    Interpolate color0 to color1 using clamp(dot(rgb, (0.299, 0.587, 0.114)),
    0, 1), without sRGB decoding or per-channel input clipping. Endpoint RGB
    triples use fp32 uniforms even for fp16 textures. Endpoints and output RGB
    are not clipped: custom negative/HDR colors are retained subject to storage
    precision. Output keeps the input image's device and precision.
    """
    return _run1(ColorAdjustOps, "false_color", image, _v.vector("color0", color0), _v.vector("color1", color1))


def highlight_shadow(image: _ImageT, shadow_amount: float = 0.0,
                     highlight_amount: float = 1.0) -> _ImageT:
    """Pointwise highlight/shadow curve on encoded-RGB luma; preserve alpha.

    Uses weights (0.299, 0.587, 0.114), without sRGB decoding. Positive shadows
    lift darks; negative shadows apply power-law deepening. Shadow values are
    finite but not clamped (usual range [-1, 1]). Highlights clamp to [0, 1],
    with 1 neutral and lower values compressing brights. Both deltas use the
    original luma, not sequentially adjusted luma.

    Luma and final RGB are clipped to [0, 1]; defaults (0, 1) are approximately
    identity only for normalized RGB. RGB is scaled by adjusted/original luma,
    for positive luma; zero/clipped-negative luma bypasses division. Exact black
    stays black, and positive near-black pixels follow the curve. Alpha is unchanged.
    """
    shadow_amount = _v.finite_float("shadow_amount", shadow_amount)
    highlight_amount = _v.finite_float("highlight_amount", highlight_amount)
    return _run1(ColorAdjustOps, "highlight_shadow", image, shadow_amount, highlight_amount)


def solarize(image: _ImageT, threshold: float = 0.5) -> _ImageT:
    """Independently replace each encoded-RGB channel at or above the fp32
    threshold with ``1-value``; preserve lower channels and alpha. Input and
    output RGB are not otherwise clipped."""
    return _run1(ColorAugOps, "solarize", image, _v.finite_float("threshold", threshold))


def posterize(image: _ImageT, bits: int = 4) -> _ImageT:
    """Quantize each channel to the top `bits` bits of its 8-bit value
    (PIL/torchvision semantics). RGB is first clipped to [0, 1], rounded on
    the fp32 0..255 scale, and bit-masked; alpha is preserved."""
    return _run1(ColorAugOps, "posterize", image, _v.integer("bits", bits, 1, 8))


def grayscale(image: _ImageT) -> _ImageT:
    """ITU-R 601 luma grayscale — same weights as kornia/torchvision
    rgb_to_grayscale. RGB is not clipped and alpha is preserved.
    (color_controls(saturation=-1) is the linear-light YCbCr take.)"""
    return _run1(ColorAugOps, "grayscale", image)


def normalize(image: _ImageT, mean: Sequence[float], std: Sequence[float]) -> _ImageT:
    """(x - mean) / std per channel, e.g. ImageNet constants. The output is
    unclamped by design (training-tensor convention) — readback is lossless.
    Mean/std are fp32 triples; std components may be negative but must be
    nonzero normal values. Alpha is preserved."""
    mean, std = _v.vector("mean", mean), _v.vector("std", std)
    if any(abs(s) < 2.0**-126 for s in std):
        raise ValueError("std components must be non-zero normal fp32 values")
    return _run1(ColorAugOps, "normalize", image, mean, std)


def white_balance(image: _ImageT, temperature: float = 6500.0,
                  tint: float = 0.0) -> _ImageT:
    """Photographic white balance using Bradford adaptation in linear RGB.

    Encoded input RGB is clipped to [0,1], decoded to linear sRGB, adapted from
    the assumed illuminant to 6500 K, clipped again, and encoded to sRGB. The
    temperature is clamped to 2000..25000 K; higher values warm the correction.
    Tint is clamped to -150..150, with positive values pushing magenta and
    negative values green. Alpha is preserved. (6500, 0) is identity for RGB
    already in [0,1], apart from transfer/matrix rounding.
    """
    temperature, tint = _v.finite_float("temperature", temperature), _v.finite_float("tint", tint)
    return _run1(WhiteBalanceOp, "apply", image, temperature, tint)


def equalize(image: _ImageT) -> _ImageT:
    """Global, independent RGB equalization using 256 nearest-value bins.

    Clamp RGB to [0,1], bin via floor(value*255 + .5), then map by the normalized
    CDF after subtracting its first occupied-bin count. A single-bin channel
    maps to bin/255, not the exact original value. All pixels contribute,
    including transparent ones; alpha is preserved without clipping. The
    resulting RGB lies in [0,1]. For local contrast use clahe.
    """
    return _run1(HistColorOps, "equalize", image)


def autocontrast(image: _ImageT) -> _ImageT:
    """Stretch each RGB channel by (x - min) / (max - min), without input clipping.

    Channels with an fp32 range <= 1e-12 pass through unchanged, including
    constant negative/HDR values. All pixels determine the extrema, including
    transparent ones; alpha is preserved without clipping.
    """
    return _run1(HistColorOps, "autocontrast", image)


def clahe(image: _ImageT, clip_limit: float = 3.0, tiles: tuple[int, int] = (8, 8),
          strength: float = 1.0) -> _ImageT:
    """Local contrast equalization of .299R + .587G + .114B luminance.

    Clamp luminance to [0,1] and assign 256 nearest-value bins. Each tile clips
    counts at max(clip_limit * pixel_count / 256, 1), redistributes excess
    uniformly as fractional counts, and normalizes the CDF without subtracting
    its first occupied-bin count. This is not an OpenCV/Lab parity contract.

    Strength is clamped to [0,1] and mixes mapped/original luminance. RGB is
    scaled by their ratio only above luminance 1e-4, then always clamped to
    [0,1]; strength=0 therefore still clips HDR RGB. Alpha is unchanged and
    all pixels contribute regardless of alpha. Histogram and interpolation
    share ceil-sized tile footprints; edge lookup extends the last occupied
    tile mapping, never an empty trailing tile.
    """
    tiles = tuple(_v.integer(f"tiles[{i}]", v, 1) for i, v in enumerate(_v.vector("tiles", tiles, 2)))
    if tiles[0] * tiles[1] > _v.INT32_MAX // 256:
        raise ValueError("tiles product exceeds signed 256-bin histogram indexing capacity")
    clip_limit = _v.finite_float("clip_limit", clip_limit)
    if clip_limit <= 0:
        raise ValueError("clip_limit must be > 0")
    strength = _v.finite_float("strength", strength)
    return _run1(ClaheOp, "apply", image, clip_limit, tiles, strength)


# ------------------------------------------------------------- geometry
def resize(image: _ImageT, width: Optional[int] = None,
           height: Optional[int] = None, scale: Optional[float] = None,
           mode: str = "lanczos3") -> _ImageT:
    """Resize using pixel-center sampling in texture value space.

    Give scale=, or width/height; a single dimension preserves aspect ratio.
    Derived dimensions round ties to even, with a minimum of one pixel.
    Lanczos3 widens its support for downscaling, drops out-of-bounds taps and
    renormalizes. Its horizontal intermediate uses the input storage precision.
    Bicubic uses fixed 4x4 Catmull-Rom taps with repeated edge pixels; it is not
    an antialiased downscaler. Both filter RGBA independently without color
    conversion, premultiplication or clipping (including alpha overshoot).
    """
    if mode not in ("lanczos3", "bicubic"):
        raise ValueError("mode must be 'lanczos3' or 'bicubic'")
    if width is not None:
        width = _v.integer("width", width, 1)
    if height is not None:
        height = _v.integer("height", height, 1)
    if scale is not None:
        scale = _v.finite_float("scale", scale)
        if scale <= 0:
            raise ValueError("scale must be > 0")
        if width is not None or height is not None:
            raise ValueError("choose scale or width/height, not both")
    tex, _ = _unwrap(image)
    if scale is not None:
        width, height = round(tex.width * scale), round(tex.height * scale)
    elif width is not None and height is None:
        height = round(tex.height * width / tex.width)
    elif height is not None and width is None:
        width = round(tex.width * height / tex.height)
    if width is None or height is None:
        raise ValueError("give scale=, or width= and/or height=")
    # aspect-preserving rounding can hit 0 for tiny inputs — never go degenerate
    width, height = max(1, int(width)), max(1, int(height))
    width, height = _v.dimensions(width, height, tex.device)
    return _run1(ResizeOp, "apply", image, width, height, mode)


def flip(image: _ImageT, horizontal: bool = False, vertical: bool = False) -> _ImageT:
    """Mirror the image along either or both axes."""
    horizontal, vertical = _v.boolean("horizontal", horizontal), _v.boolean("vertical", vertical)
    return _run1(GeometryOps, "flip", image, horizontal, vertical)


def rot90(image: _ImageT, k: int = 1) -> _ImageT:
    """Rotate by k counter-clockwise quarter turns (numpy.rot90 semantics);
    output dimensions swap for odd k."""
    return _run1(GeometryOps, "rot90", image, _v.integer("k", k, -_v.INT32_MAX) % 4)


def crop(image: _ImageT, x: int, y: int, width: int, height: int) -> _ImageT:
    """Extract the (x, y, width, height) rectangle (top-left origin)."""
    x, y = _v.integer("x", x), _v.integer("y", y)
    width, height = _v.integer("width", width, 1), _v.integer("height", height, 1)
    return _run1(GeometryOps, "crop", image, x, y, width, height)


def tile(
    image: _ImageT,
    width: int,
    height: int,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    mode: str = "bilinear",
) -> _ImageT:
    """Repeat an image over an explicit output extent.

    ``scale_x`` and ``scale_y`` control the displayed tile size (values above
    one enlarge it). ``offset_x`` and ``offset_y`` are wrapped source-pixel
    phase offsets. Sampling is ``bilinear`` or ``nearest``.
    """
    width, height = _v.integer("width", width, 1), _v.integer("height", height, 1)
    scale_x, scale_y = _v.finite_float("scale_x", scale_x), _v.finite_float("scale_y", scale_y)
    if scale_x <= 0 or scale_y <= 0:
        raise ValueError("scale_x and scale_y must be > 0")
    offset_x, offset_y = _v.finite_float("offset_x", offset_x), _v.finite_float("offset_y", offset_y)
    return _run1(
        GeometryOps,
        "tile",
        image,
        width,
        height,
        scale_x,
        scale_y,
        offset_x,
        offset_y,
        mode,
    )


def affine(image: _ImageT, matrix: Any, width: int = 0, height: int = 0,
           mode: str = "bilinear") -> _ImageT:
    """General affine warp. matrix is the FORWARD 2x3 (source -> output,
    pixel-center coords); sampling is bilinear or nearest, with zeros
    (transparent black) outside the source — torchvision/kornia fill
    convention. width/height set the output canvas (default: input size)."""
    if width < 0 or height < 0:
        raise ValueError("width and height must be >= 0 (0 uses the input size)")
    width, height = _v.integer("width", width), _v.integer("height", height)
    return _run1(GeometryOps, "affine", image, matrix,
                 width or None, height or None, mode)


def rotate(image: _ImageT, angle: float, mode: str = "bilinear",
           expand: bool = False) -> _ImageT:
    """Rotate by angle degrees about the center, positive counterclockwise.

    Sample pixel centers with bilinear or nearest interpolation; outside taps
    are transparent black. RGBA is sampled independently without color
    conversion, premultiplication or clipping. expand=True fits the rotated
    image bounds, anchors their minimum to the output origin and ceil-rounds
    the canvas dimensions; otherwise the input canvas clips rotated corners.
    Exact multiples of 90 degrees use exact rotation coefficients, so expanded
    quarter turns swap width/height without adding rounding-induced pixels.
    """
    import numpy as _np

    angle = _v.finite_float("angle", angle)
    expand = _v.boolean("expand", expand)
    if mode not in ("bilinear", "nearest"):
        raise ValueError("mode must be 'bilinear' or 'nearest'")
    tex, _ = _unwrap(image)
    w, h = tex.width, tex.height
    m = rotation_matrix(float(angle), w / 2.0, h / 2.0)
    out_w, out_h = 0, 0
    if expand:
        corners = _np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], dtype=float).T
        moved = _np.vstack([m, [0, 0, 1]]) @ corners
        out_w = int(_np.ceil(moved[0].max() - moved[0].min()))
        out_h = int(_np.ceil(moved[1].max() - moved[1].min()))
        m[0, 2] -= moved[0].min()
        m[1, 2] -= moved[1].min()
    return affine(image, m, width=out_w, height=out_h, mode=mode)


# -------------------------------------------------------------- stylize
def kuwahara(image: _ImageT, radius: int = 7, sigma_spatial: float = 12.0,
             var_power: float = 7.0) -> _ImageT:
    """Blend eight angular-sector RGB means using inverse-variance weights.

    The square neighborhood uses edge replication and spatial weights
    exp(-(dx*dx+dy*dy)/(sigma_spatial*radius)). Sector variance is of RGB vector
    magnitude, not weighted luminance or per-channel variance. Variance is
    floored at 1e-5; sectors with spatial weight sum <=1e-5 are skipped.
    Sector means are blended, not selected by an argmin; the center belongs
    to sector 4. RGB is neither clipped nor linearized, and alpha is preserved
    without affecting weights. This is not a grayscale or four-quadrant filter.

    Sector weights are scaled relative to the smallest variance before
    normalization to avoid overflowing raw inverse-variance powers.
    """
    radius = _v.radius("radius", radius)
    sigma_spatial, var_power = _v.sigma("sigma_spatial", sigma_spatial), _v.finite_float("var_power", var_power)
    if var_power <= 0:
        raise ValueError("var_power must be > 0")
    return _run1(KuwaharaOp, "apply", image, radius, sigma_spatial, var_power)


def pointillize(image: _ImageT, radius: float = 20.0, jitter: float = 1.0,
                seed: float = 0.0) -> _ImageT:
    """Repaint RGB from the nearest center of a jittered square-cell grid.

    ``radius`` is the cell size and is clamped to a minimum of 2 pixels;
    ``jitter`` is clamped to [0,1]. ``seed`` deterministically varies the fp32
    cell hash on a given backend. The chosen center is truncated to an integer
    source coordinate and clamped at image edges. RGB is copied without color
    conversion or clipping, while alpha remains that of the output pixel rather
    than the sampled center.
    """
    radius = _v.finite_float("radius", radius)
    jitter, seed = _v.finite_float("jitter", jitter), _v.finite_float("seed", seed)
    if radius > _v.INT32_MAX / 4:
        raise ValueError("radius exceeds pointillize cell-coordinate capacity")
    return _run1(PointillizeOp, "apply", image, radius, jitter, seed)


def cmyk_halftone(image: _ImageT, width: float = 6.0, angle_deg: float = 0.0,
                  sharpness: float = 0.08) -> _ImageT:
    """Normalized-CMYK dot screens recombined multiplicatively on white.

    Input RGB is clipped to [0,1], separated into normalized C/M/Y plus K, and
    screened at base angles 15/75/0/45 degrees respectively. ``angle_deg`` adds
    a common rotation. ``width`` is the cell size in pixels (minimum 1), and
    ``sharpness`` is the smooth dot-edge width in cell units (minimum .001).
    Output RGB lies in [0,1] and alpha is preserved.
    """
    width = _v.finite_float("width", width)
    angle_deg = _v.finite_float("angle_deg", angle_deg)
    sharpness = _v.finite_float("sharpness", sharpness)
    return _run1(CmykHalftoneOp, "apply", image, width, angle_deg, sharpness)


def smooth_threshold(image: _ImageT, threshold: float = 0.5,
                     smooth_amount: float = 0.28) -> _ImageT:
    """Soft threshold on encoded-RGB luma weights (.2125, .7154, .0721).
    Applies smoothstep from ``max(threshold-smooth_amount, 0)`` to
    ``min(threshold+smooth_amount, 1)``, emitting grayscale RGB in [0, 1] and
    preserving alpha. Controls/edges use fp32."""
    threshold = _v.finite_float("threshold", threshold)
    smooth_amount = _v.finite_float("smooth_amount", smooth_amount)
    # smoothstep requires strictly ordered edges AFTER fp32 rounding.
    t, s = _v.fp32(threshold), _v.fp32(smooth_amount)
    lo = max(_v.fp32(t - s), 0.0)
    hi = min(_v.fp32(t + s), 1.0)
    if smooth_amount <= 0 or lo >= hi:
        raise ValueError("smooth_amount must be > 0 and produce ordered fp32 threshold edges")
    return _run1(SmoothThresholdOp, "apply", image, threshold, smooth_amount)


def smooth_threshold_bw(image: _ImageT, black_threshold: float,
                        white_threshold: float) -> _ImageT:
    """Two-point encoded-RGB luma ramp using weights (.299, .587, .114):
    black at/below ``black_threshold``, white at/above ``white_threshold``,
    and linear grayscale between them. Alpha is preserved. Controls and their
    positive gap must remain finite and normal after fp32 rounding."""
    black_threshold, white_threshold = validate_threshold_range(
        black_threshold, white_threshold
    )
    return _run1(SmoothThresholdOp, "apply_bw", image, black_threshold, white_threshold)


def invert(image: _ImageT) -> _ImageT:
    """Invert encoded RGB as ``1-value`` without clipping; preserve alpha."""
    return _run1(PointOps, "invert", image)


def extract_channel(image: _ImageT, channel: str = "b") -> _ImageT:
    """Broadcast a single channel to grayscale RGB. `channel` is 'r', 'g',
    'b', or its integer index 0/1/2. Values are not clipped; alpha is
    preserved."""
    if channel not in ("r", "g", "b", 0, 1, 2):
        raise ValueError("channel must be r/g/b or 0/1/2")
    return _run1(PointOps, "extract_channel", image, channel)


# ----------------------------------------------------------- two-input
def blend(base: _ImageT, blend_image: _ImageLike, mode: str = "multiply",
          opacity: float = 1.0) -> _ImageT:
    """Blend straight (unpremultiplied) RGBA with source-over compositing.

    Base is the background; blend_image is the foreground. Opacity is clamped
    to [0,1] and scales foreground alpha, not its RGB. Output alpha is the
    source-over union; RGB is unpremultiplied (zero when output alpha <=1e-6).
    The base sets output extent and storage precision. A smaller foreground is
    extended by replicating its final row/column; mixed fp16/fp32 inputs are
    accepted. Blend formulas operate directly on stored RGB without a common
    input/output clip, although individual modes have their own bounds/guards.
    For RGB in [0,1], color mode transfers base luminosity (weights .30/.59/.11)
    to the foreground, reducing chroma to fit the gamut; not an HSL conversion.
    See slangimage.ops.BLEND_MODES for the available color formulas.
    """
    if mode not in BLEND_MODES:
        raise ValueError(f"Unknown blend mode {mode!r}")
    return _run2(BlendModesOp, "apply", base, blend_image, mode, _v.finite_float("opacity", opacity))


def clamp_multiply(image: _ImageT, image2: _ImageLike) -> _ImageT:
    """Clip each input RGB to [0, 1], multiply per channel, and preserve the
    first image's alpha. The first image sets the output extent; a smaller
    second image is extended by replicating its last row/column."""
    return _run2(PointOps, "clamp_multiply", image, image2)


def high_pass_contrast(image: _ImageT, blur_image: _ImageLike,
                       strength: float = 1.0) -> _ImageT:
    """Unclipped local-contrast RGB as
    ``0.5 + strength*(image-blur_image)/2``; preserve the first image's alpha.
    Inputs must have matching extents and strength is an fp32 scalar."""
    strength = _v.finite_float("strength", strength)
    return _run2(
        PointOps,
        "high_pass_contrast",
        image,
        blur_image,
        strength,
        same_size_for="high_pass_contrast",
    )


def lut(image: _ImageT, lut_image: _ImageLike, intensity: float = 1.0) -> _ImageT:
    """Apply a square 8x8-grid color LUT, size >=16 and divisible by 8.

    The 64 blue slices are row-major tiles; red varies across each tile and
    green down it (top-left image origin). Interpolate linearly in all three
    axes; each tile has size/8 red and green samples, independent of blue.
    Input RGBA and intensity are clamped to [0,1] before lookup/mixing, so
    intensity=0 returns the clamped input, not an HDR passthrough. LUT alpha
    is ignored; output alpha is clamped input alpha. Mapped RGB is not clipped.
    Inputs may have different storage precisions; output uses image precision.
    """
    intensity = _v.finite_float("intensity", intensity)
    lut_tex, _ = _unwrap(lut_image)
    validate_lut_size(lut_tex.width, lut_tex.height)
    return _run2(LutOp, "apply", image, lut_image, intensity)


# ----------------------------------------------------------- generators
def random_noise(width: int, height: int, seed: float = 0.0,
                 precision: str = "fp16", device: Optional[spy.Device] = None) -> Image:
    """Generate opaque grayscale fp32 sin/fmod hash noise stored at ``precision``.

    Values occupy approximately [.05,.95], RGB channels are identical and alpha
    is 1. Results repeat for a given fp32 seed on a given backend, but are not
    identical across backends because transcendental/hash evaluation diverges;
    do not rely on cross-platform bit reproducibility.
    """
    width, height = _v.integer("width", width, 1), _v.integer("height", height, 1)
    seed = _v.finite_float("seed", seed)
    if precision not in ("fp16", "fp32"):
        raise ValueError("precision must be 'fp16' or 'fp32'")
    device = device or get_default_device()
    width, height = _v.dimensions(width, height, device)
    fmt = spy.Format.rgba16_float if precision == "fp16" else spy.Format.rgba32_float
    inst = _op_for_device(device, PointOps)
    return _wrap_out(inst.random_noise(width, height, seed, fmt))


# --------------------------------------------------------------- prewarm
def prewarm(device: Optional[spy.Device] = None, precision: str = "fp16",
            size: int = 64) -> int:
    """Warm representative kernels for every texture operator on a tiny image.

    Uses one device/precision; call separately for each precision you need.
    Returns the number of registry cases, not unique operators or shader
    variants. This is a startup dispatch check, not numerical validation.
    """
    import numpy as np

    if precision not in ("fp16", "fp32"):
        raise ValueError("precision must be 'fp16' or 'fp32'")
    size = _v.finite_float("size", size)
    if size < 20:
        raise ValueError("size must be >= 20 so every prewarm case has valid geometry")
    size = _v.integer("size", size, 20)
    device = device or get_default_device()
    _v.dimensions(size, size, device)
    rng = np.random.default_rng(0)
    arr = rng.random((size, size, 4)).astype(np.float16 if precision == "fp16" else np.float32)
    arr[..., 3] = 1.0
    from .image import from_numpy

    img = from_numpy(arr, precision=precision, device=device)
    lut_img = from_numpy(identity_lut_np().astype(arr.dtype), precision=precision, device=device)

    cases = [
        lambda: gaussian_blur(img, radius=5, sigma=2.0),
        lambda: surface_blur(img, radius=5),
        lambda: bilateral(img, radius=3),
        lambda: bilateral(img, radius=3, separable=True),
        lambda: nl_means(img, search_radius=2),
        lambda: unsharp_mask(img),
        lambda: ep_unsharp_mask(img),
        lambda: sharpen_luma(img),
        lambda: vibrance(img),
        lambda: gamma(img, 1.1),
        lambda: color_controls(img),
        lambda: srgb_to_linear(img),
        lambda: linear_to_srgb(img),
        lambda: exposure(img, 0.1),
        lambda: sepia(img, 0.5),
        lambda: hue_rotate(img, 0.1),
        lambda: false_color(img),
        lambda: highlight_shadow(img, 0.2, 0.9),
        lambda: clahe(img, tiles=(2, 2)),
        lambda: solarize(img),
        lambda: posterize(img),
        lambda: grayscale(img),
        lambda: normalize(img, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        lambda: equalize(img),
        lambda: autocontrast(img),
        lambda: white_balance(img, 4500.0, 20.0),
        lambda: flip(img, horizontal=True),
        lambda: rot90(img),
        lambda: crop(img, 4, 4, 16, 16),
        lambda: tile(img, size + 3, size + 5, scale_x=1.25, scale_y=0.75, offset_x=0.5),
        lambda: rotate(img, 30.0),
        lambda: resize(img, scale=0.5),
        lambda: kuwahara(img, radius=3),
        lambda: pointillize(img, radius=6.0),
        lambda: cmyk_halftone(img),
        lambda: smooth_threshold(img),
        lambda: smooth_threshold_bw(img, 0.2, 0.8),
        lambda: invert(img),
        lambda: extract_channel(img),
        lambda: blend(img, img, "overlay"),
        lambda: clamp_multiply(img, img),
        lambda: high_pass_contrast(img, img),
        lambda: lut(img, lut_img),
        lambda: random_noise(size, size, precision=precision, device=device),
    ]
    for case in cases:
        case()
    device.wait_for_idle()
    return len(cases)


# Deliberate public surface (keeps `from slangimage.ops import *` and doc tools
# from picking up imported helpers, op classes, and cache internals).
__all__ = [
    # blurs & denoise
    "gaussian_blur", "surface_blur", "bilateral", "nl_means",
    # sharpening
    "unsharp_mask", "ep_unsharp_mask", "sharpen_luma",
    # color
    "vibrance", "gamma", "color_controls", "srgb_to_linear", "linear_to_srgb",
    "exposure", "sepia", "hue_rotate", "false_color", "highlight_shadow",
    "solarize", "posterize", "grayscale", "normalize", "white_balance",
    "equalize", "autocontrast", "clahe",
    # geometry
    "resize", "flip", "rot90", "crop", "tile", "affine", "rotate",
    # stylize
    "kuwahara", "pointillize", "cmyk_halftone", "smooth_threshold",
    "smooth_threshold_bw", "invert", "extract_channel",
    # two-input
    "blend", "clamp_multiply", "high_pass_contrast", "lut",
    # generators
    "random_noise",
    # lifecycle + re-exports
    "prewarm", "BLEND_MODES", "identity_lut_np",
]
