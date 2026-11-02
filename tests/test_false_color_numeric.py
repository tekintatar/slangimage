"""Independent barycentric duotone reference on quantized images/endpoints.

The library maps encoded-RGB luma, without sRGB decoding. Only the interpolation
coordinate is clipped; custom endpoint colors and output RGB remain unclipped.
"""

import numpy as np
import pytest

import slangimage as si
from slangimage.kernels.color_adjust import ColorAdjustOps
from slangimage.runtime.textures import texture_to_np
from test_core_numeric import storage_error


LUMA = np.array([.299, .587, .114])
DEFAULTS = ((.3, 0, 0), (1, .9, .8))
PAIRS = [DEFAULTS, ((0, 0, 0), (1, 1, 1)), ((1, 1, 1), (0, 0, 0)),
         ((1, 0, .5), (0, .75, .125)), ((.25, .5, .75), (.25, .5, .75)),
         ((-.5, 1.5, .25), (2, -.25, 1.25))]
EPS = np.finfo(np.float32).eps


def reference(stored, pair):
    rgb = stored[..., :3].astype(np.float64)
    endpoints = np.array(pair, dtype=np.float32).astype(np.float64)
    raw = np.einsum('...c,c->...', rgb, LUMA)
    t = np.clip(raw, 0, 1)
    expected = np.stack((1-t, t), axis=-1) @ endpoints
    # Three-term fp32 dot, rounded coefficients, clipping (1-Lipschitz), then
    # lerp arithmetic; its factor error is amplified by endpoint separation.
    luma_error = 4*EPS*np.einsum('...c,c->...', np.abs(rgb), LUMA)
    coordinate_error = np.maximum(t-np.clip(raw-luma_error, 0, 1),
                                  np.clip(raw+luma_error, 0, 1)-t)
    compute = coordinate_error[..., None]*np.abs(endpoints[1]-endpoints[0])
    compute += 4*EPS*(np.abs(endpoints[0])+np.abs(endpoints[1])+np.abs(expected))
    return expected, compute


def make_image(pattern, precision, shape=(11, 17)):
    if pattern == "palette":
        colors = [[0, 0, 0], [1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                  [1, 1, 0], [1, 0, 1], [0, 1, 1]]
    elif pattern == "ramp":
        colors = np.repeat(np.linspace(0, 1, 33)[:, None], 3, axis=1)
    elif pattern == "hdr":
        colors = [[-.25, -.1, -.2], [1.5, 1.5, 1.5], [-.25, .5, 1.25],
                  [1.5, .1, -.1], [.1, 1.25, -.25]]
    else:
        colors = np.random.default_rng(925).uniform(-.25, 1.5, (187, 3))
    dtype = np.float16 if precision == "fp16" else np.float32
    stored = np.ones((*shape, 4), dtype=dtype)
    stored[..., :3] = np.resize(colors, (*shape, 3))
    stored[..., 3] = np.resize([-1, 0, .125, .5, 1, 2], shape)
    return stored


def check(ctx, stored, precision, pair=DEFAULTS, *, defaults=False):
    source = si.from_numpy(stored, device=ctx[0], precision=precision)
    result = si.ops.false_color(source) if defaults else si.ops.false_color(source, *pair)
    assert result.device is source.device and result.precision == precision
    actual = result.to_numpy().astype(np.float64)
    assert actual.shape == stored.shape and np.isfinite(actual).all()
    np.testing.assert_array_equal(actual[..., 3], stored[..., 3])
    expected, compute = reference(stored, pair)
    assert compute.max() < 1e-5
    error = compute+storage_error(expected, precision, compute)
    np.testing.assert_array_less(np.abs(actual[..., :3]-expected), np.nextafter(error, np.inf))
    return actual


def test_false_color_reference_primary_luma_hand_values():
    stored = np.ones((1, 3, 4))
    stored[0, :, :3] = np.eye(3)
    actual, _ = reference(stored, ((0, 0, 0), (1, 1, 1)))
    np.testing.assert_allclose(actual[0], np.repeat(LUMA[:, None], 3, axis=1), rtol=0, atol=1e-16)


def test_false_color_reference_quarter_gray_hand_value():
    stored = np.full((1, 1, 4), .25)
    actual, _ = reference(stored, ((-.5, 1.5, .25), (2, -.25, 1.25)))
    np.testing.assert_allclose(actual, [[[.125, 1.0625, .5]]], rtol=0, atol=1e-16)


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("pattern", ["palette", "ramp", "hdr", "random"])
@pytest.mark.parametrize("pair", PAIRS)
def test_false_color_independent_reference(ctx, precision, pattern, pair):
    check(ctx, make_image(pattern, precision), precision, pair)


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("shape", [(1, 1), (1, 17), (19, 1)])
def test_false_color_tiny_defaults_alpha_repeated_and_low_level(ctx, precision, shape):
    stored = make_image("hdr", precision, shape)
    first = check(ctx, stored, precision, defaults=True)
    check(ctx, stored, precision, PAIRS[-1])
    np.testing.assert_array_equal(first, check(ctx, stored, precision))
    source = si.from_numpy(stored, device=ctx[0], precision=precision)
    low = ColorAdjustOps(*ctx).false_color(source.texture)
    np.testing.assert_array_equal(first, texture_to_np(low))
    stored[..., 3] = 0
    transparent = check(ctx, stored, precision)
    np.testing.assert_array_equal(first[..., :3], transparent[..., :3])


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
def test_false_color_constant_map_and_unclipped_endpoints(ctx, precision):
    stored = make_image("random", precision)
    constant = check(ctx, stored, precision, ((-.5, .25, 1.5),)*2)
    np.testing.assert_array_equal(constant[..., :3], np.broadcast_to([-.5, .25, 1.5], constant[..., :3].shape))
    mapped = check(ctx, make_image("ramp", precision), precision, PAIRS[-1])
    assert mapped[..., :3].min() < 0 and mapped[..., :3].max() > 1


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("boundary", [0., 1.])
def test_false_color_luma_clipping_neighbors(ctx, precision, boundary):
    dtype = np.float16 if precision == "fp16" else np.float32
    # Non-gray cancellation fixture crossing the clipping plane; avoids fp32
    # subnormals while testing that INPUT channels are not clipped separately.
    green = (boundary-.299*.5)/.587
    stored = np.ones((1, 3, 4), dtype=dtype)
    for index, delta in enumerate([-.002, 0, .002]):
        stored[0, index, :3] = [.5, green+delta, 0]
    actual = check(ctx, stored, precision, ((0, 0, 0), (1, 1, 1)))
    if boundary == 0:
        assert actual[0, 0, 0] == 0 and actual[0, 2, 0] > .0005
    else:
        assert actual[0, 0, 0] < .9995 and actual[0, 2, 0] == 1


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
def test_false_color_fp32_endpoint_controls(ctx, precision):
    # Endpoint uniforms use fp32 even when texture storage is fp16. Constants
    # straddle an fp16 rounding midpoint; references must not quantize them twice.
    middle = .5+2**-12
    pair = ((middle-2**-20, .123456789, -.23456789),
            (middle+2**-20, 1.23456789, .987654321))
    check(ctx, make_image("ramp", precision), precision, pair)
