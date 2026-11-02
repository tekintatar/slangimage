"""Bounded independent references for exposure and sepia texture operators.

Inputs are quantized to their texture precision before float64 reference math.
The references model the documented fp32 controls, clipping order, and alpha
retention without using the differentiable kernels or production helpers.
"""

import numpy as np
import pytest

import slangimage as si
from slangimage.kernels.color_adjust import ColorAdjustOps
from slangimage.runtime.textures import texture_to_np
from test_core_numeric import storage_error


EPS = np.finfo(np.float32).eps
SEPIA = np.array([
    [.393, .769, .189],
    [.349, .686, .168],
    [.272, .534, .131],
])


def make_image(pattern, precision, shape=(9, 13)):
    if pattern == "palette":
        colors = [[0, 0, 0], [1, 1, 1], [1, 0, 0], [0, 1, 0],
                  [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]]
    elif pattern == "ramp":
        x = np.linspace(0, 1, 29)
        colors = np.stack((x, x[::-1], np.mod(3*x, 1)), axis=-1)
    elif pattern == "hdr":
        colors = [[-.25, -.1, 0], [1.25, 1.5, 1], [-.25, .5, 1.25],
                  [1.5, .1, -.1], [.1, 1.25, -.25]]
    else:
        colors = np.random.default_rng(1201).uniform(-.25, 1.5, (117, 3))
    dtype = np.float16 if precision == "fp16" else np.float32
    stored = np.ones((*shape, 4), dtype=dtype)
    stored[..., :3] = np.resize(colors, (*shape, 3))
    stored[..., 3] = np.resize([-1, 0, .125, .5, 1, 2], shape)
    return stored


def assert_result(result, source, stored, expected, compute):
    assert isinstance(result, si.Image)
    assert result.device is source.device and result.precision == source.precision
    actual = result.to_numpy().astype(np.float64)
    assert actual.shape == stored.shape and np.isfinite(actual).all()
    np.testing.assert_array_equal(actual[..., 3], stored[..., 3])
    error = compute + storage_error(expected, source.precision, compute)
    np.testing.assert_array_less(
        np.abs(actual[..., :3]-expected), np.nextafter(error, np.inf)
    )
    return actual


def exposure_reference(stored, ev):
    factor = 2.0**float(np.float32(ev))
    scaled = stored[..., :3].astype(np.float64)*factor
    expected = np.clip(scaled, 0, 1)
    # fp32 exp2/multiply followed by clipping. The bounded corpus keeps
    # |ev| <= 4 and RGB in [-.25, 1.5].
    compute = 4e-6*(1+np.abs(scaled))
    assert compute.max() < 1.1e-4
    return expected, compute


def sepia_reference(stored, intensity):
    rgb = stored[..., :3].astype(np.float64)
    amount = float(np.clip(np.float32(intensity), 0, 1))
    sepia = np.clip(rgb @ SEPIA.T, 0, 1)
    expected = rgb+amount*(sepia-rgb)
    magnitude = np.max(np.abs(rgb), axis=-1, keepdims=True)
    compute = 4e-6*(1+magnitude+np.abs(expected))
    assert compute.max() < 2.5e-5
    return expected, compute


def check_exposure(ctx, stored, precision, ev):
    source = si.from_numpy(stored, precision=precision, device=ctx[0])
    expected, compute = exposure_reference(stored, ev)
    return assert_result(si.ops.exposure(source, ev), source, stored, expected, compute)


def check_sepia(ctx, stored, precision, intensity):
    source = si.from_numpy(stored, precision=precision, device=ctx[0])
    expected, compute = sepia_reference(stored, intensity)
    return assert_result(si.ops.sepia(source, intensity), source, stored, expected, compute)


def test_exposure_reference_hand_values():
    stored = np.array([[[.25, .75, -.25, .4]]])
    expected, _ = exposure_reference(stored, 1)
    np.testing.assert_array_equal(expected, [[[.5, 1, 0]]])


def test_sepia_reference_red_hand_value_and_clipping_order():
    stored = np.array([[[1., 0, 0, 1], [2., 0, 0, 1]]])
    expected, _ = sepia_reference(stored, 1)
    np.testing.assert_allclose(expected[0, 0], [.393, .349, .272], rtol=0, atol=1e-15)
    np.testing.assert_allclose(expected[0, 1], [.786, .698, .544], rtol=0, atol=1e-15)


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("pattern", ["palette", "ramp", "hdr", "random"])
@pytest.mark.parametrize("ev", [-4, -1, 0, .5, 1, 4])
def test_exposure_independent_reference(ctx, precision, pattern, ev):
    check_exposure(ctx, make_image(pattern, precision), precision, ev)


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("pattern", ["palette", "ramp", "hdr", "random"])
@pytest.mark.parametrize("intensity", [-1, 0, .25, 1, 2])
def test_sepia_independent_reference(ctx, precision, pattern, intensity):
    check_sepia(ctx, make_image(pattern, precision), precision, intensity)


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("shape", [(1, 1), (1, 13), (11, 1)])
def test_simple_color_tiny_alpha_repeated_defaults_and_low_level(ctx, precision, shape):
    stored = make_image("hdr", precision, shape)
    exposure = check_exposure(ctx, stored, precision, 0)
    sepia = check_sepia(ctx, stored, precision, 1)
    np.testing.assert_array_equal(exposure, check_exposure(ctx, stored, precision, 0))
    np.testing.assert_array_equal(sepia, check_sepia(ctx, stored, precision, 1))

    source = si.from_numpy(stored, precision=precision, device=ctx[0])
    low = ColorAdjustOps(*ctx)
    np.testing.assert_array_equal(exposure, texture_to_np(low.exposure(source.texture, 0)))
    np.testing.assert_array_equal(sepia, texture_to_np(low.sepia(source.texture, 1)))


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
def test_simple_color_alpha_does_not_affect_rgb(ctx, precision):
    stored = make_image("random", precision)
    first_exposure = check_exposure(ctx, stored, precision, .5)
    first_sepia = check_sepia(ctx, stored, precision, .25)
    stored[..., 3] = np.resize([-.5, 0, 2], stored.shape[:2])
    second_exposure = check_exposure(ctx, stored, precision, .5)
    second_sepia = check_sepia(ctx, stored, precision, .25)
    np.testing.assert_array_equal(first_exposure[..., :3], second_exposure[..., :3])
    np.testing.assert_array_equal(first_sepia[..., :3], second_sepia[..., :3])

