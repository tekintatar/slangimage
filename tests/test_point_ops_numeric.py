"""Independent bounded references for the four image-input point utilities."""

import numpy as np
import pytest

import slangimage as si
from slangimage.kernels.point_ops import PointOps
from slangimage.runtime.textures import texture_to_np
from test_core_numeric import storage_error


EPS = np.finfo(np.float32).eps


def make_image(pattern, precision, shape=(8, 12), seed=1203):
    if pattern == "palette":
        colors = [[0, 0, 0], [1, 1, 1], [1, 0, 0], [0, 1, 0],
                  [0, 0, 1], [.25, .5, .75]]
    elif pattern == "hdr":
        colors = [[-.25, -.1, 0], [1.25, 1.5, 1], [-.25, .5, 1.25],
                  [1.5, .1, -.1], [.1, 1.25, -.25]]
    else:
        colors = np.random.default_rng(seed).uniform(-.25, 1.5, (np.prod(shape), 3))
    dtype = np.float16 if precision == "fp16" else np.float32
    stored = np.ones((*shape, 4), dtype=dtype)
    stored[..., :3] = np.resize(colors, (*shape, 3))
    stored[..., 3] = np.resize([-1, 0, .125, .5, 1, 2], shape)
    return stored


def output(result, source, stored, expected, compute):
    assert result.device is source.device and result.precision == source.precision
    actual = result.to_numpy().astype(np.float64)
    assert actual.shape == stored.shape and np.isfinite(actual).all()
    np.testing.assert_array_equal(actual[..., 3], stored[..., 3])
    error = compute+storage_error(expected, source.precision, compute)
    np.testing.assert_array_less(
        np.abs(actual[..., :3]-expected), np.nextafter(error, np.inf)
    )
    return actual


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("pattern", ["palette", "hdr", "random"])
def test_invert_independent_reference(ctx, precision, pattern):
    stored = make_image(pattern, precision)
    source = si.from_numpy(stored, precision=precision, device=ctx[0])
    expected = 1-stored[..., :3].astype(np.float64)
    compute = 2*EPS*(1+np.abs(expected))
    output(si.ops.invert(source), source, stored, expected, compute)


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("pattern", ["palette", "hdr", "random"])
@pytest.mark.parametrize("channel", ["r", "g", "b", 0, 1, 2])
def test_extract_channel_independent_reference(ctx, precision, pattern, channel):
    stored = make_image(pattern, precision)
    source = si.from_numpy(stored, precision=precision, device=ctx[0])
    index = {"r": 0, "g": 1, "b": 2, 0: 0, 1: 1, 2: 2}[channel]
    expected = np.repeat(stored[..., index:index+1].astype(np.float64), 3, axis=-1)
    actual = output(si.ops.extract_channel(source, channel), source, stored, expected, 0.)
    np.testing.assert_array_equal(actual[..., 0], actual[..., 1])
    np.testing.assert_array_equal(actual[..., 1], actual[..., 2])


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("pattern", ["palette", "hdr", "random"])
def test_clamp_multiply_independent_reference(ctx, precision, pattern):
    first = make_image(pattern, precision)
    second = make_image("random", precision, seed=1213)
    source = si.from_numpy(first, precision=precision, device=ctx[0])
    other = si.from_numpy(second, precision=precision, device=ctx[0])
    expected = np.clip(first[..., :3].astype(np.float64), 0, 1)
    expected *= np.clip(second[..., :3].astype(np.float64), 0, 1)
    compute = 4*EPS*(1+np.abs(expected))
    actual = output(si.ops.clamp_multiply(source, other), source, first, expected, compute)
    assert actual[..., :3].min() >= 0 and actual[..., :3].max() <= 1


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
def test_clamp_multiply_replicates_smaller_second_image_border(ctx, precision):
    first = make_image("random", precision, shape=(5, 7))
    second = make_image("hdr", precision, shape=(2, 3))
    source = si.from_numpy(first, precision=precision, device=ctx[0])
    other = si.from_numpy(second, precision=precision, device=ctx[0])
    rows = np.minimum(np.arange(5), 1)
    cols = np.minimum(np.arange(7), 2)
    extended = second[rows[:, None], cols[None, :], :3]
    expected = np.clip(first[..., :3].astype(np.float64), 0, 1)
    expected *= np.clip(extended.astype(np.float64), 0, 1)
    compute = 4*EPS*(1+np.abs(expected))
    output(si.ops.clamp_multiply(source, other), source, first, expected, compute)


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("pattern", ["palette", "hdr", "random"])
@pytest.mark.parametrize("strength", [-2, 0, .5, 1, 3])
def test_high_pass_contrast_independent_reference(ctx, precision, pattern, strength):
    first = make_image(pattern, precision)
    second = make_image("random", precision, seed=1223)
    source = si.from_numpy(first, precision=precision, device=ctx[0])
    other = si.from_numpy(second, precision=precision, device=ctx[0])
    amount = float(np.float32(strength))
    a, b = first[..., :3].astype(np.float64), second[..., :3].astype(np.float64)
    expected = amount*(a*.5-b*.5)+.5
    compute = 5*EPS*(1+abs(amount)*(np.abs(a)+np.abs(b))+np.abs(expected))
    output(si.ops.high_pass_contrast(source, other, strength), source, first, expected, compute)


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("shape", [(1, 1), (1, 9), (7, 1)])
def test_point_ops_tiny_alpha_independence_repeated_and_low_level(ctx, precision, shape):
    first = make_image("hdr", precision, shape)
    second = make_image("random", precision, shape, seed=1233)
    source = si.from_numpy(first, precision=precision, device=ctx[0])
    other = si.from_numpy(second, precision=precision, device=ctx[0])
    low = PointOps(*ctx)
    calls = [
        (si.ops.invert(source), low.invert(source.texture)),
        (si.ops.extract_channel(source, "g"), low.extract_channel(source.texture, "g")),
        (si.ops.clamp_multiply(source, other), low.clamp_multiply(source.texture, other.texture)),
        (si.ops.high_pass_contrast(source, other, 1.5),
         low.high_pass_contrast(source.texture, other.texture, 1.5)),
    ]
    for public, direct in calls:
        public_array = public.to_numpy()
        np.testing.assert_array_equal(public_array, texture_to_np(direct))
        np.testing.assert_array_equal(public_array[..., 3], first[..., 3])

    repeated = si.ops.high_pass_contrast(source, other, 1.5).to_numpy()
    np.testing.assert_array_equal(repeated, calls[-1][0].to_numpy())
    second_alpha_changed = second.copy()
    second_alpha_changed[..., 3] *= -2
    changed = si.from_numpy(second_alpha_changed, precision=precision, device=ctx[0])
    np.testing.assert_array_equal(
        si.ops.clamp_multiply(source, changed).to_numpy(), calls[2][0].to_numpy()
    )

