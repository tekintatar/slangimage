"""Bounded numeric references for color augmentation and threshold operators.

All image values are first quantized to fp16/fp32 texture storage. References
model fp32 controls and declared clipping/branch behavior without calling
production math helpers.
"""

import numpy as np
import pytest

import slangimage as si
from slangimage.kernels.color_aug import ColorAugOps
from slangimage.kernels.smooth_threshold import SmoothThresholdOp
from slangimage.runtime.textures import texture_to_np
from test_core_numeric import storage_error


EPS = np.finfo(np.float32).eps
GRAY = np.array([.299, .587, .114])
SMOOTH_LUMA = np.array([.2125, .7154, .0721])


def make_image(pattern, precision, shape=(7, 11)):
    if pattern == "palette":
        colors = [[0, 0, 0], [1, 1, 1], [1, 0, 0], [0, 1, 0],
                  [0, 0, 1], [.25, .5, .75], [.5, .5, .5]]
    elif pattern == "hdr":
        colors = [[-.25, -.1, 0], [1.25, 1.5, 1], [-.25, .5, 1.25],
                  [1.5, .1, -.1], [.1, 1.25, -.25]]
    else:
        colors = np.random.default_rng(1202).uniform(-.25, 1.5, (77, 3))
    dtype = np.float16 if precision == "fp16" else np.float32
    stored = np.ones((*shape, 4), dtype=dtype)
    stored[..., :3] = np.resize(colors, (*shape, 3))
    stored[..., 3] = np.resize([-1, 0, .125, .5, 1, 2], shape)
    return stored


def output(ctx, stored, precision, operation, expected, compute):
    source = si.from_numpy(stored, precision=precision, device=ctx[0])
    result = operation(source)
    assert result.device is source.device and result.precision == precision
    actual = result.to_numpy().astype(np.float64)
    assert actual.shape == stored.shape and np.isfinite(actual).all()
    np.testing.assert_array_equal(actual[..., 3], stored[..., 3])
    error = compute+storage_error(expected, precision, compute)
    np.testing.assert_array_less(
        np.abs(actual[..., :3]-expected), np.nextafter(error, np.inf)
    )
    return actual


def solarize_reference(stored, threshold):
    rgb = stored[..., :3].astype(np.float64)
    threshold = float(np.float32(threshold))
    expected = np.where(rgb >= threshold, 1-rgb, rgb)
    return expected, 2*EPS*(1+np.abs(expected))


def posterize_reference(stored, bits):
    rgb = stored[..., :3].astype(np.float32)
    step = 1 << (8-bits)
    # The classification is defined on an 8-bit scale computed in fp32.
    # Model that discrete decision explicitly rather than hiding a neighboring
    # level inside a numeric output tolerance.
    scaled = np.float32(np.clip(rgb, 0, 1)*np.float32(255)+np.float32(.5))
    integer = np.floor(scaled).astype(np.float64)
    expected = (integer-np.mod(integer, step))/255
    return expected, np.full_like(expected, 3*EPS)


def grayscale_reference(stored):
    rgb = stored[..., :3].astype(np.float64)
    expected = rgb @ GRAY
    compute = 4*EPS*np.einsum("...c,c->...", np.abs(rgb), GRAY)[..., None]
    return np.repeat(expected[..., None], 3, axis=-1), compute


def normalize_reference(stored, mean, std):
    rgb = stored[..., :3].astype(np.float64)
    mean = np.asarray(mean, dtype=np.float32).astype(np.float64)
    std = np.asarray(std, dtype=np.float32).astype(np.float64)
    expected = (rgb-mean)/std
    compute = 5*EPS*(1+np.abs(expected)+np.abs(rgb/std))
    return expected, compute


def smooth_edges(threshold, amount):
    threshold, amount = np.float32(threshold), np.float32(amount)
    lo = max(float(np.float32(threshold-amount)), 0.)
    hi = min(float(np.float32(threshold+amount)), 1.)
    return lo, hi


def smooth_curve(x, lo, hi):
    t = np.clip((x-lo)/(hi-lo), 0, 1)
    return t*t*(3-2*t)


def smooth_reference(stored, threshold, amount):
    rgb = stored[..., :3].astype(np.float64)
    lo, hi = smooth_edges(threshold, amount)
    luma = rgb @ SMOOTH_LUMA
    expected = smooth_curve(luma, lo, hi)
    luma_error = 4*EPS*np.einsum("...c,c->...", np.abs(rgb), SMOOTH_LUMA)
    low, high = smooth_curve(luma-luma_error, lo, hi), smooth_curve(luma+luma_error, lo, hi)
    compute = np.maximum(expected-low, high-expected)+3e-6
    return np.repeat(expected[..., None], 3, axis=-1), compute[..., None]


def bw_curve(x, black, white):
    result = (x-black)/(white-black)
    return np.where(x < black, 0, np.where(x > white, 1, result))


def smooth_bw_reference(stored, black, white):
    rgb = stored[..., :3].astype(np.float64)
    black, white = float(np.float32(black)), float(np.float32(white))
    luma = rgb @ GRAY
    expected = bw_curve(luma, black, white)
    luma_error = 4*EPS*np.einsum("...c,c->...", np.abs(rgb), GRAY)
    low = bw_curve(luma-luma_error, black, white)
    high = bw_curve(luma+luma_error, black, white)
    compute = np.maximum(expected-low, high-expected)+2e-6
    return np.repeat(expected[..., None], 3, axis=-1), compute[..., None]


def test_augmentation_reference_hand_values():
    stored = np.array([[[.25, .5, .75, 1.]]])
    solarized, _ = solarize_reference(stored, .5)
    np.testing.assert_array_equal(solarized, [[[.25, .5, .25]]])
    gray, _ = grayscale_reference(stored)
    np.testing.assert_allclose(gray, [[[.45375, .45375, .45375]]], rtol=0, atol=1e-15)


def test_threshold_reference_hand_values():
    stored = np.array([[[.5, .5, .5, 1.]]])
    smooth, _ = smooth_reference(stored, .5, .25)
    ramp, _ = smooth_bw_reference(stored, .25, .75)
    np.testing.assert_allclose(smooth, [[[.5, .5, .5]]], rtol=0, atol=1e-15)
    np.testing.assert_allclose(ramp, [[[.5, .5, .5]]], rtol=0, atol=1e-15)


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("pattern", ["palette", "hdr", "random"])
@pytest.mark.parametrize("threshold", [-.5, .25, .5, 1, 1.5])
def test_solarize_independent_reference(ctx, precision, pattern, threshold):
    stored = make_image(pattern, precision)
    expected, compute = solarize_reference(stored, threshold)
    output(ctx, stored, precision, lambda image: si.ops.solarize(image, threshold), expected, compute)


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("pattern", ["palette", "hdr", "random"])
@pytest.mark.parametrize("bits", [1, 2, 4, 7, 8])
def test_posterize_independent_reference(ctx, precision, pattern, bits):
    stored = make_image(pattern, precision)
    expected, compute = posterize_reference(stored, bits)
    actual = output(ctx, stored, precision, lambda image: si.ops.posterize(image, bits), expected, compute)
    assert actual[..., :3].min() >= 0 and actual[..., :3].max() <= 1


@pytest.mark.parametrize("bits", [1, 4, 8])
def test_posterize_fp32_rounding_boundary_neighbors(ctx, bits):
    boundary = np.float32((64-.5)/255)
    values = [np.nextafter(boundary, np.float32(-np.inf)), boundary,
              np.nextafter(boundary, np.float32(np.inf))]
    stored = np.ones((1, 3, 4), dtype=np.float32)
    stored[0, :, :3] = np.asarray(values)[:, None]
    expected, compute = posterize_reference(stored, bits)
    output(ctx, stored, "fp32", lambda image: si.ops.posterize(image, bits), expected, compute)


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("pattern", ["palette", "hdr", "random"])
def test_grayscale_independent_reference(ctx, precision, pattern):
    stored = make_image(pattern, precision)
    expected, compute = grayscale_reference(stored)
    actual = output(ctx, stored, precision, si.ops.grayscale, expected, compute)
    np.testing.assert_array_equal(actual[..., 0], actual[..., 1])
    np.testing.assert_array_equal(actual[..., 1], actual[..., 2])


NORMALIZATIONS = [((0, 0, 0), (1, 1, 1)),
                  ((.485, .456, .406), (.229, .224, .225)),
                  ((-.25, .5, 1.25), (-.5, .125, 2))]


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("pattern", ["palette", "hdr", "random"])
@pytest.mark.parametrize("mean,std", NORMALIZATIONS)
def test_normalize_independent_reference(ctx, precision, pattern, mean, std):
    stored = make_image(pattern, precision)
    expected, compute = normalize_reference(stored, mean, std)
    output(ctx, stored, precision, lambda image: si.ops.normalize(image, mean, std), expected, compute)


SMOOTH_CONTROLS = [(.5, .1), (.5, .5), (-.1, .2), (1.1, .2)]


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("pattern", ["palette", "hdr", "random"])
@pytest.mark.parametrize("threshold,amount", SMOOTH_CONTROLS)
def test_smooth_threshold_independent_reference(ctx, precision, pattern, threshold, amount):
    stored = make_image(pattern, precision)
    expected, compute = smooth_reference(stored, threshold, amount)
    actual = output(
        ctx, stored, precision,
        lambda image: si.ops.smooth_threshold(image, threshold, amount), expected, compute
    )
    assert actual[..., :3].min() >= 0 and actual[..., :3].max() <= 1


BW_CONTROLS = [(-.2, .2), (.25, .75), (.8, 1.2)]


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("pattern", ["palette", "hdr", "random"])
@pytest.mark.parametrize("black,white", BW_CONTROLS)
def test_smooth_threshold_bw_independent_reference(ctx, precision, pattern, black, white):
    stored = make_image(pattern, precision)
    expected, compute = smooth_bw_reference(stored, black, white)
    actual = output(
        ctx, stored, precision,
        lambda image: si.ops.smooth_threshold_bw(image, black, white), expected, compute
    )
    assert actual[..., :3].min() >= 0 and actual[..., :3].max() <= 1


def test_smooth_threshold_bw_exact_white_endpoint_for_narrow_fp32_range(ctx):
    black = np.float32(.5)
    white = np.nextafter(black, np.float32(1))
    stored = np.ones((1, 1, 4), dtype=np.float32)*white
    source = si.from_numpy(stored, precision="fp32", device=ctx[0])
    actual = si.ops.smooth_threshold_bw(source, float(black), float(white)).to_numpy()
    np.testing.assert_array_equal(actual[..., :3], 1)


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("shape", [(1, 1), (1, 11), (9, 1)])
def test_augmentation_threshold_tiny_alpha_repeated_and_low_level(ctx, precision, shape):
    stored = make_image("hdr", precision, shape)
    source = si.from_numpy(stored, precision=precision, device=ctx[0])
    aug, threshold = ColorAugOps(*ctx), SmoothThresholdOp(*ctx)
    calls = [
        (si.ops.solarize(source, .3), aug.solarize(source.texture, .3)),
        (si.ops.posterize(source, 3), aug.posterize(source.texture, 3)),
        (si.ops.grayscale(source), aug.grayscale(source.texture)),
        (si.ops.normalize(source, (0, .5, 1), (1, .5, .25)),
         aug.normalize(source.texture, (0, .5, 1), (1, .5, .25))),
        (si.ops.smooth_threshold(source, .4, .2), threshold.apply(source.texture, .4, .2)),
        (si.ops.smooth_threshold_bw(source, .2, .8), threshold.apply_bw(source.texture, .2, .8)),
    ]
    for public, low in calls:
        public_array = public.to_numpy()
        np.testing.assert_array_equal(public_array, texture_to_np(low))
        np.testing.assert_array_equal(public_array[..., 3], stored[..., 3])
