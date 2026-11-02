"""Independent texture references for surface blur, edge-preserving unsharp,
vibrance and luma sharpen.

Float64 NumPy sums and standard-library HLS are the oracles, never production
color/filter helpers or another GPU implementation.
Shared fixtures/storage bounds come from test_core_numeric.
"""

import colorsys

import numpy as np
import pytest

import slangimage as si
import test_core_numeric as core
from test_core_numeric import (
    COMPUTE_ATOL,
    LUMA_WEIGHTS,
    check_output,
    convolution,
    decode_srgb,
    encode_srgb,
    storage_error,
)

# Explicit fixture aliases avoid collecting tests from the shared module.
precision = core.precision
sample = core.sample


def output_error(reference, precision, arithmetic=COMPUTE_ATOL):
    return arithmetic + storage_error(reference, precision, arithmetic)


def shifted(image, dx, dy):
    h, w = image.shape[:2]
    return image[np.clip(np.arange(h) + dy, 0, h - 1)[:, None],
                 np.clip(np.arange(w) + dx, 0, w - 1)[None, :]]


def surface_reference(image, samples, sigma):
    """Range Gaussian over RGB, spatial sigma = half the sample footprint."""
    half = (samples - 1) // 2
    if half == 0:
        # Explicit one-pixel identity, with no undefined spatial weights.
        return image[..., :3].copy()
    center = image[..., :3]
    numerator, denominator = np.zeros_like(center), np.zeros(center.shape[:2])
    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            neighbor = shifted(center, dx, dy)
            distance = np.sum((neighbor - center) ** 2, axis=-1)
            weight = np.exp(-0.5 * (distance / sigma**2 + (dx**2 + dy**2) / half**2))
            numerator += neighbor * weight[..., None]
            denominator += weight
    return numerator / denominator[..., None]


def ep_reference(image, blurred, blur_error, amount):
    linear = decode_srgb(image[..., :3])
    y = linear @ LUMA_WEIGHTS
    cb, cr = (linear[..., 2] - y) * 0.564, (linear[..., 0] - y) * 0.713
    chroma = np.stack((1.403 * cr, -0.714 * cr - 0.344 * cb, 1.770 * cb), axis=-1)

    def sharpen(base):
        return np.clip(y + amount * (y - decode_srgb(base) @ LUMA_WEIGHTS), 0, 1)

    reference = encode_srgb(sharpen(blurred)[..., None] + chroma)
    ends = np.stack((sharpen(blurred - blur_error), sharpen(blurred + blur_error)))
    arithmetic = (1 + 2 * abs(amount)) * 2e-6
    low = np.clip(ends.min(axis=0) - arithmetic, 0, 1)
    high = np.clip(ends.max(axis=0) + arithmetic, 0, 1)
    lower = encode_srgb(low[..., None] + chroma - 2e-6)
    upper = encode_srgb(high[..., None] + chroma + 2e-6)
    return reference, np.maximum(reference - lower, upper - reference) + COMPUTE_ATOL


def hls(rgb):
    return np.apply_along_axis(lambda c: colorsys.rgb_to_hls(*c), -1, rgb)


def from_hls(value):
    return np.apply_along_axis(lambda c: colorsys.hls_to_rgb(*c), -1, value)


def vibrance_reference(image, amount):
    rgb = np.clip(image[..., :3], 0, 1)
    value = hls(rgb)
    amount = float(np.clip(np.float32(amount), -1, 1))
    if amount >= 0:
        value[..., 2] = np.minimum(1, value[..., 2] / max(1 - amount, 1e-15))
    else:
        value[..., 2] *= 1 + amount
    adjusted = from_hls(value)
    adjusted = np.where((np.ptp(rgb, axis=-1) <= 1e-6)[..., None], rgb, adjusted)
    # max_rgb's actual convention is max(R,G,B), not weighted luminance.
    maximum = rgb.max(axis=-1)
    red_saturation = (maximum - rgb[..., 0]) / (maximum + 1e-6)
    skin = np.minimum(rgb[..., 0] - rgb[..., 1], 2 * rgb[..., 1] - rgb[..., 2])
    skin *= 4 * (1 - red_saturation) / (maximum + 1e-6)
    mask = 0.85 - 0.7 * np.clip(skin, 0, 1)
    t = np.clip((mask - 0.5) / 0.5, 0, 1)
    mix = mask * (1 - 0.5 * t**2 * (3 - 2 * t))
    return np.clip(rgb + mix[..., None] * (adjusted - rgb), 0, 1)


def sharpen_reference(image, radius, sharpness, rgb_mode):
    rgb = np.clip(image[..., :3], 0, 1)
    radius, sharpness = np.clip(radius, 0.5, 64), np.clip(sharpness, -2, 4)
    n = int(np.ceil(np.clip(radius, 1, 4)))
    x = np.abs(np.arange(-n, n + 1) / radius)
    # Factored cubic pieces, independently evaluated and separably convolved.
    weights = np.where(x <= 1, (1 - x) * (1 + x - x*x),
                       np.where(x < 2, (2 - x)**2 * (1 - x), 0))
    weights /= weights.sum()
    values = rgb if rgb_mode else hls(rgb)[..., 1]
    blurred = convolution(convolution(values, weights, 0), weights, 1)
    result = np.clip(values + sharpness * (values - blurred), 0, 1)
    if not rgb_mode:
        value = hls(rgb)
        value[..., 1] = result
        result = from_hls(value)
    return result


@pytest.mark.parametrize("radius,bsigma", [(1, 0.1), (2, 0.1), (3, 0.05), (4, 0.3), (9, 1.0), (15, 0.3)])
def test_surface_matches_numpy(sample, precision, radius, bsigma):
    image, source = sample
    reference = surface_reference(image, radius, float(np.float32(bsigma)))
    check_output(si.ops.surface_blur(source, radius, bsigma), source, reference,
                 output_error(reference, precision), preserve_alpha=True)


@pytest.mark.parametrize("radius,bsigma,amount", [(3, 0.3, 0), (5, 0.3, -0.5), (5, 0.1, 1), (9, 0.5, 2)])
def test_ep_unsharp_matches_numpy(sample, precision, radius, bsigma, amount):
    image, source = sample
    blur = surface_reference(image, radius, float(np.float32(bsigma)))
    reference, error = ep_reference(image, blur, output_error(blur, precision), amount)
    error += storage_error(reference, precision, error)
    check_output(si.ops.ep_unsharp_mask(source, radius, bsigma, amount), source,
                 reference, error, preserve_alpha=True)


@pytest.mark.parametrize("amount", [-2, -0.5, 0, 0.25, 0.75, 1, 3])
def test_vibrance_matches_hls(sample, precision, amount):
    image, source = sample
    reference = vibrance_reference(image, amount)
    check_output(si.ops.vibrance(source, amount), source, reference,
                 output_error(reference, precision), preserve_alpha=True)


@pytest.mark.parametrize("rgb_mode", [False, True])
@pytest.mark.parametrize("radius,sharpness", [(-1, 1), (1, 1), (1.5, -0.5), (2.5, 0), (4, 1), (8, 2), (80, 5), (2, -3)])
def test_sharpen_matches_hls_and_cubic(sample, precision, rgb_mode, radius, sharpness):
    image, source = sample
    reference = sharpen_reference(image, radius, sharpness, rgb_mode)
    # <=81 tap float32 sum, strength up to4, HLS conversion up to2x slope.
    arithmetic = (1 + abs(np.clip(sharpness, -2, 4))) * 5e-6 * (1 if rgb_mode else 2)
    check_output(si.ops.sharpen_luma(source, radius, sharpness, rgb_mode), source,
                 reference, output_error(reference, precision, arithmetic), preserve_alpha=True)


def test_ep_unsharp_default_adds_detail(ctx, precision):
    # A low-contrast patch that the default range sigma should smooth. Comparing
    # amount=0 avoids mistaking the approximate YCbCr roundtrip for sharpening.
    image = np.random.default_rng(71).uniform(0.3, 0.7, (7, 9, 4)).astype(np.float32)
    source = si.from_numpy(image, precision=precision, device=ctx[0])
    baseline = si.ops.ep_unsharp_mask(source, amount=0).to_numpy().astype(np.float64)
    sharpened = si.ops.ep_unsharp_mask(source).to_numpy().astype(np.float64)
    assert np.max(abs(sharpened[..., :3] - baseline[..., :3])) > 0.001
    stored = source.to_numpy().astype(np.float64)
    blurred = surface_reference(stored, 3, float(np.float32(0.12)))
    reference, error = ep_reference(stored, blurred, output_error(blurred, precision), 1)
    error += storage_error(reference, precision, error)
    check_output(si.ops.ep_unsharp_mask(source), source, reference, error, preserve_alpha=True)


def test_ep_default_entry_points_agree(sample, precision, ctx):
    from slangimage.kernels.ep_unsharp_mask import EdgePreservingUnsharpMaskOp
    from slangimage.runtime.textures import texture_to_np

    _, source = sample
    operator = EdgePreservingUnsharpMaskOp(*ctx)
    expected = si.ops.ep_unsharp_mask(source, radius=3).to_numpy()
    np.testing.assert_array_equal(si.ops.ep_unsharp_mask(source).to_numpy(), expected)
    np.testing.assert_array_equal(texture_to_np(operator.apply(source.texture)), expected)
    np.testing.assert_array_equal(texture_to_np(operator.apply_default(source.texture)), expected)


@pytest.mark.parametrize("radius", [1, 2])
def test_surface_one_pixel_footprint_is_exact_identity(sample, radius):
    _, source = sample
    np.testing.assert_array_equal(si.ops.surface_blur(source, radius=radius).to_numpy(),
                                  source.to_numpy())


@pytest.mark.parametrize("amount", [-1, 0, 0.5, 1])
def test_vibrance_color_boundaries(ctx, precision, amount):
    rgb = np.array([(0, 0, 0), (1, 1, 1), (0.5, 0.5, 0.5),
                    (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (0, 1, 1), (1, 0, 1),
                    (0.8, 0.5, 0.3), (0.3, 0.5, 0.8), (0.6, 0.4, 0.5),
                    (0.001, 0.002, 0.003), (0.99, 0.98, 0.97), (0.5000004, 0.5, 0.5)])
    image = np.concatenate((rgb, np.linspace(0, 1, len(rgb))[:, None]), axis=-1)[None]
    source = si.from_numpy(image.astype(np.float32), precision=precision, device=ctx[0])
    reference = vibrance_reference(source.to_numpy().astype(np.float64), amount)
    check_output(si.ops.vibrance(source, amount), source, reference,
                 output_error(reference, precision), preserve_alpha=True)


def test_reference_closed_forms():
    constant = np.full((3, 5, 4), 0.375)
    np.testing.assert_allclose(surface_reference(constant, 9, 0.1), constant[..., :3], atol=1e-15)
    colors = np.array([[[0, 0, 0, 0], [1, 1, 1, 1], [0.4, 0.4, 0.4, 0.3]]])
    np.testing.assert_allclose(vibrance_reference(colors, 1), colors[..., :3], atol=1e-15)
