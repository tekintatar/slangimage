"""Bounded independent YIQ hue references, with input/control quantization.

The rounded forward/reconstruction matrices define existing library behavior;
they are not exact inverses or a claim of Core Image parity. No production
color helper is used. Complex multiplication independently rotates chroma.
"""

import cmath
import math

import numpy as np
import pytest

import slangimage as si
from slangimage.kernels.color_adjust import ColorAdjustOps
from slangimage.runtime.textures import texture_to_np
from test_core_numeric import storage_error


FORWARD = np.array([[.299, .587, .114], [.596, -.274, -.322], [.211, -.523, .312]])
BACK = np.array([[1, .956, .621], [1, -.272, -.647], [1, -1.106, 1.703]])


def reference(rgb, angle):
    yiq = np.asarray(rgb, dtype=np.float64) @ FORWARD.T
    rotated = (yiq[..., 1]+1j*yiq[..., 2])*cmath.exp(1j*float(np.float32(angle)))
    yiq[..., 1], yiq[..., 2] = rotated.real, rotated.imag
    return np.clip(yiq @ BACK.T, 0, 1)


def make_image(pattern, precision, shape=(11, 17)):
    if pattern == "palette":
        colors = [[0, 0, 0], [1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                  [1, 1, 0], [1, 0, 1], [0, 1, 1]]
    elif pattern == "gray":
        colors = np.repeat(np.array([0, .001, .04045, .1, .25, .5, .9, 1])[:, None], 3, axis=1)
    elif pattern == "hdr":
        colors = [[-.25, -.1, -.2], [1.5, 1.5, 1.5], [-.25, .5, 1.25],
                  [1.5, .1, -.1], [.1, 1.25, -.25]]
    else:
        colors = np.random.default_rng(903).uniform(0, 1, (187, 3))
    dtype = np.float16 if precision == "fp16" else np.float32
    stored = np.ones((*shape, 4), dtype=dtype)
    stored[..., :3] = np.resize(colors, (*shape, 3))
    stored[..., 3] = np.resize([-1, 0, .125, .5, 1, 2], shape)
    return stored


def check(ctx, stored, precision, angle):
    source = si.from_numpy(stored, device=ctx[0], precision=precision)
    result = si.ops.hue_rotate(source, angle)
    assert result.device is source.device and result.precision == precision
    actual = result.to_numpy().astype(np.float64)
    assert actual.shape == stored.shape and np.isfinite(actual).all()
    np.testing.assert_array_equal(actual[..., 3], stored[..., 3])
    assert actual[..., :3].min() >= 0 and actual[..., :3].max() <= 1
    expected = reference(stored[..., :3], angle)
    # Bounded input [-.25,1.5], |angle| <= 8*pi+.37: two small matrices,
    # fp32 trig and rotation, then one storage operation. Quantization is
    # handled before reference math, not folded into this compute allowance.
    compute = 2e-6*(1+np.max(np.abs(stored[..., :3].astype(np.float64)), axis=-1, keepdims=True))
    error = compute+storage_error(expected, precision, compute)
    np.testing.assert_array_less(np.abs(actual[..., :3]-expected), np.nextafter(error, np.inf))
    return actual


def test_hue_reference_zero_angle_red_hand_value():
    np.testing.assert_allclose(reference(np.array([1., 0, 0]), 0),
                               [.999807, .000371, 0], atol=2e-16, rtol=0)


def test_hue_reference_gray_has_zero_chroma():
    for angle in [-2.1, 0, .7, 2*math.pi]:
        np.testing.assert_allclose(reference(np.array([.5, .5, .5]), angle),
                                   [.5, .5, .5], atol=2e-16, rtol=0)


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("pattern", ["palette", "gray", "hdr", "random"])
@pytest.mark.parametrize("angle", [0, .37, -.73, math.pi/2, -math.pi/2, math.pi,
                                   -math.pi, 2*math.pi, -2*math.pi, 8*math.pi+.37])
def test_hue_independent_reference(ctx, precision, pattern, angle):
    check(ctx, make_image(pattern, precision), precision, angle)


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("center", [0., math.pi/2, math.pi, 2*math.pi, -2*math.pi])
def test_hue_angle_fp32_neighbors(ctx, precision, center):
    # Control precision is fp32 even for fp16 image storage. Avoid subnormal
    # controls around zero: host policy/fast-math subnormals are separate scope.
    center = np.float32(center)
    angles = [-1e-6, 0, 1e-6] if center == 0 else [
        np.nextafter(center, np.float32(-np.inf)), center,
        np.nextafter(center, np.float32(np.inf))]
    stored = make_image("palette", precision)
    for angle in angles:
        check(ctx, stored, precision, float(angle))


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("shape", [(1, 1), (1, 17), (19, 1)])
def test_hue_tiny_alpha_repeated_and_low_level(ctx, precision, shape):
    stored = make_image("hdr", precision, shape)
    first = check(ctx, stored, precision, .73)
    check(ctx, stored, precision, -1.1)
    np.testing.assert_array_equal(first, check(ctx, stored, precision, .73))
    source = si.from_numpy(stored, device=ctx[0], precision=precision)
    low = ColorAdjustOps(*ctx).hue(source.texture, .73)
    np.testing.assert_array_equal(first, texture_to_np(low))
    stored[..., 3] = 0
    transparent = check(ctx, stored, precision, .73)
    np.testing.assert_array_equal(first[..., :3], transparent[..., :3])


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("boundary", [0., 1.])
def test_hue_clipping_boundary_neighbors(ctx, precision, boundary):
    # Construct a gray-offset red ray crossing the blue output clipping plane.
    # Derive its coefficient from a scalar red Y/I/Q rotation, not the GPU.
    angle = float(np.float32(.73))
    chroma = complex(.596, .211)*cmath.exp(1j*angle)
    blue = .299-1.106*chroma.real+1.703*chroma.imag
    offset = boundary-.25*blue
    dtype = np.float16 if precision == "fp16" else np.float32
    stored = np.ones((1, 3, 4), dtype=dtype)
    # Offset span exceeds fp16 quantization yet stays close to the plane.
    for index, delta in enumerate([-.002, 0, .002]):
        stored[0, index, :3] = [offset+.25+delta, offset+delta, offset+delta]
    actual = check(ctx, stored, precision, angle)
    if boundary == 0:
        assert actual[0, 0, 2] == 0 and actual[0, 2, 2] > .001
    else:
        assert actual[0, 0, 2] < .999 and actual[0, 2, 2] == 1


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
def test_hue_full_turn_quantization_and_clipping_changes_luma(ctx, precision):
    stored = make_image("palette", precision)
    zero = check(ctx, stored, precision, 0)
    turn = check(ctx, stored, precision, 2*math.pi)
    # A float32 full turn is not exactly 2*pi; allow compute + both stores.
    error = 1e-6+storage_error(zero[..., :3], precision)+storage_error(turn[..., :3], precision)
    np.testing.assert_array_less(np.abs(zero[..., :3]-turn[..., :3]), np.nextafter(error, np.inf))
    rotated = check(ctx, stored, precision, math.pi/2)
    old_luma = stored[..., :3].astype(np.float64) @ FORWARD[0]
    new_luma = rotated[..., :3] @ FORWARD[0]
    assert np.max(np.abs(new_luma-old_luma)) > .05
