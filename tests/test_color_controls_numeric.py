"""Bounded independent reference for the combined color-control pipeline.

The library uses rounded BT.601-style coefficients on LINEAR RGB, a radial
chroma cap, then contrast on encoded RGB. This is not standard encoded YCbCr
or an exact RGB roundtrip. No production conversion helper is an oracle.
"""

import cmath
import math

import numpy as np
import pytest

import slangimage as si
from slangimage.kernels.color_controls import ColorControlsOp
from slangimage.runtime.textures import texture_to_np
from test_core_numeric import decode_srgb, encode_srgb, storage_error


LUMA = np.array([0.299, 0.587, 0.114])
TO_YCC = np.stack((LUMA, 0.564*(np.array([0, 0, 1])-LUMA),
                   0.713*(np.array([1, 0, 0])-LUMA)))
FROM_YCC = np.array([[1, 0, 1.403], [1, -0.344, -0.714], [1, 1.770, 0]])


def contrast_factor(contrast, public):
    if public:
        contrast = max(0.001, 1+float(contrast))
    return float(np.float32(np.clip(contrast, 0, 5)))


def reference(stored, *, hue=0, saturation=0, contrast=0, brightness=0, public=True):
    linear = decode_srgb(stored[..., :3].astype(np.float64))
    ycc = linear @ TO_YCC.T
    hue, saturation, brightness = [float(np.float32(np.clip(v, -1, 1)))
                                   for v in (hue, saturation, brightness)]
    ycc[..., 0] = np.clip(ycc[..., 0] + brightness/2, 0, 1)
    # Complex multiplication independently represents the two-dimensional
    # rotation; radial projection is continuous at the cap.
    chroma = (ycc[..., 1]+1j*ycc[..., 2])*cmath.exp(1j*math.pi*hue)*(1+saturation)
    magnitude = np.abs(chroma)
    chroma *= 0.6/np.maximum(0.6, magnitude)
    ycc[..., 1], ycc[..., 2] = chroma.real, chroma.imag
    encoded = encode_srgb(ycc @ FROM_YCC.T)
    factor = contrast_factor(contrast, public)
    active = factor > float(np.float32(1e-4))
    expected = (encoded-0.5)*factor+0.5 if active else encoded
    # For bounded input [-.25,1.5]: decode/matrix/trig/radial-projection fp32
    # allowance in linear RGB, propagated through the monotone sRGB encoder.
    # Rounded matrix coefficients are part of the reference, not error budget.
    linear_error = 2e-6*(1+np.max(np.abs(linear), axis=-1, keepdims=True))
    output_linear = ycc @ FROM_YCC.T
    encoded_error = np.maximum(encode_srgb(output_linear+linear_error)-encoded,
                               encoded-encode_srgb(output_linear-linear_error))
    compute_error = encoded_error*(factor if active else 1) + 2e-6*(1+np.abs(expected))
    return expected, compute_error


def make_image(pattern, precision, shape=(11, 17)):
    if pattern == "palette":
        values = [[0, 0, 0], [1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                  [1, 1, 0], [1, 0, 1], [0, 1, 1], [.125, .25, .375]]
    elif pattern == "gray":
        levels = np.array([0, .001, .04045, .1, .25, .5, .75, 1])
        values = np.repeat(levels[:, None], 3, axis=-1)
    elif pattern == "hdr":
        values = [[-.25, -.25, -.25], [1.5, 1.5, 1.5], [-.25, .1, 1.5],
                  [1.25, -.1, .5], [.25, 1.5, -.25]]
    elif pattern == "cap":
        magenta = np.array([1., 0, 1])
        radius = np.linalg.norm((TO_YCC @ magenta)[1:])
        levels = .6/radius*np.array([.99, .9995, 1, 1.0005, 1.01])
        values = encode_srgb(levels[:, None]*magenta)
    else:
        values = np.random.default_rng(719).uniform(0, 1, (math.prod(shape), 3))
    dtype = np.float16 if precision == "fp16" else np.float32
    stored = np.ones((*shape, 4), dtype=dtype)
    stored[..., :3] = np.resize(values, (*shape, 3))
    stored[..., 3] = np.resize([-1, 0, .125, .5, 1, 2], math.prod(shape)).reshape(shape)
    return stored


def check(ctx, stored, precision, *, public=True, **params):
    source = si.from_numpy(stored, device=ctx[0], precision=precision)
    if public:
        result = si.ops.color_controls(source, **params)
        assert result.device is source.device and result.precision == precision
        actual = result.to_numpy().astype(np.float64)
    else:
        texture = ColorControlsOp(*ctx).apply(source.texture, **params)
        assert texture.format == source.texture.format
        actual = texture_to_np(texture).astype(np.float64)
    assert actual.shape == stored.shape and np.isfinite(actual).all()
    np.testing.assert_array_equal(actual[..., 3], stored[..., 3])
    expected, compute_error = reference(stored, public=public, **params)
    assert compute_error.max() < .001  # Reject an uninformative reference bound.
    error = compute_error + storage_error(expected, precision, compute_error)
    np.testing.assert_array_less(np.abs(actual[..., :3]-expected), np.nextafter(error, np.inf))
    return actual


def test_color_controls_reference_gray_hand_calculation():
    stored = np.array([[[.5, .5, .5, 1]]])
    expected, _ = reference(stored, brightness=.25, contrast=.5)
    linear = ((.5+.055)/1.055)**2.4 + .125
    encoded = 1.055*linear**(1/2.4)-.055
    np.testing.assert_allclose(expected, [[[.5+1.5*(encoded-.5)]*3]], atol=1e-15, rtol=0)


def test_color_controls_reference_desaturated_primary():
    stored = np.array([[[1., 0, 0, 1]]])
    expected, _ = reference(stored, saturation=-1)
    gray = 1.055*.299**(1/2.4)-.055
    np.testing.assert_allclose(expected, [[[gray]*3]], atol=1e-15, rtol=0)


CONTROLS = [{}, {"brightness": -.6}, {"brightness": .6},
            {"saturation": -1}, {"saturation": .7}, {"contrast": -.8},
            {"contrast": .8}, {"hue": -.5}, {"hue": .5},
            {"hue": .23, "saturation": .4, "contrast": .6, "brightness": -.2}]


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("pattern", ["palette", "gray", "random", "hdr", "cap"])
@pytest.mark.parametrize("params", CONTROLS)
def test_color_controls_reference(ctx, precision, pattern, params):
    check(ctx, make_image(pattern, precision), precision, **params)


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("control,low,high", [("hue", -1, 1), ("saturation", -1, 1),
                                             ("brightness", -1, 1), ("contrast", -1, 4)])
def test_color_controls_control_clamps(ctx, precision, control, low, high):
    stored = make_image("palette", precision)
    lower = check(ctx, stored, precision, **{control: low})
    upper = check(ctx, stored, precision, **{control: high})
    np.testing.assert_array_equal(lower, check(ctx, stored, precision, **{control: low-2}))
    np.testing.assert_array_equal(upper, check(ctx, stored, precision, **{control: high+2}))


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("shape", [(1, 1), (1, 13), (17, 1)])
def test_color_controls_tiny_alpha_and_repeated_calls(ctx, precision, shape):
    stored = make_image("hdr", precision, shape)
    params = CONTROLS[-1]
    first = check(ctx, stored, precision, **params)
    check(ctx, stored, precision, hue=-1, contrast=-1)
    np.testing.assert_array_equal(first, check(ctx, stored, precision, **params))
    stored[..., 3] = 0
    transparent = check(ctx, stored, precision, **params)
    np.testing.assert_array_equal(first[..., :3], transparent[..., :3])


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
def test_color_controls_brightness_luma_clamp_and_unclipped_rgb(ctx, precision):
    stored = make_image("gray", precision)
    black = check(ctx, stored, precision, brightness=-1)
    white = check(ctx, stored, precision, brightness=1)
    np.testing.assert_array_equal(black[0, 0, :3], 0)
    np.testing.assert_allclose(white[0, 7, :3], 1, atol=6e-4 if precision == "fp16" else 2e-6, rtol=0)
    contrasted = check(ctx, stored, precision, contrast=1)
    assert contrasted[..., :3].min() < 0 and contrasted[..., :3].max() > 1
    gray = check(ctx, make_image("palette", precision), precision, saturation=-1)
    np.testing.assert_array_equal(gray[..., 0], gray[..., 1])
    np.testing.assert_array_equal(gray[..., 0], gray[..., 2])


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("factor", [-1., 0., float(np.nextafter(np.float32(1e-4), np.float32(0))),
                                     float(np.float32(1e-4)),
                                     float(np.nextafter(np.float32(1e-4), np.float32(1))),
                                     .001, .5, 1., 2., 5.])
def test_color_controls_low_level_factor_compatibility(ctx, precision, factor):
    # Existing bypass branch is intentionally characterized, not prescribed as
    # a good numeric-control design. Public mapping never reaches that branch.
    stored = make_image("palette", precision)
    actual = check(ctx, stored, precision, public=False, contrast=factor)
    if .001 <= factor <= 5:
        public = check(ctx, stored, precision, contrast=factor-1)
        np.testing.assert_array_equal(actual, public)


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
def test_color_controls_public_contrast_floor_neighbors(ctx, precision):
    stored = make_image("gray", precision)
    floor = check(ctx, stored, precision, contrast=-1)
    np.testing.assert_array_equal(floor, check(ctx, stored, precision, contrast=-.999))
    above = check(ctx, stored, precision, contrast=-.998)
    assert np.ptp(above[..., :3]) > np.ptp(floor[..., :3])
