"""Independent bounded highlight/shadow references and near-black continuity checks.

This library's pointwise curve uses encoded RGB luma, not linear-light luma.
Decimal evaluates an algebraically simplified scalar curve; no production
shader or conversion helper is used as the reference.
"""

from decimal import Decimal, localcontext
from functools import lru_cache

import numpy as np
import pytest

import slangimage as si
from slangimage.kernels.color_adjust import ColorAdjustOps
from slangimage.runtime.textures import texture_to_np
from test_core_numeric import storage_error


WEIGHTS = np.array([.299, .587, .114])
EPS = np.finfo(np.float32).eps


@lru_cache(maxsize=8192)
def tone(luma, shadow, highlight):
    with localcontext() as context:
        context.prec = 50
        lum, s, h = [Decimal.from_float(float(v)) for v in (luma, shadow, highlight)]
        if s >= 0:
            t = lum**(1/(1+s))
            sd = max(t*(1-Decimal('.76')*t)-lum, Decimal(0))
        else:
            sd = lum**(1-s)-lum
        q = 1-lum
        u = q**(1/(2-h))
        hd = min(q-u*(1-Decimal('.8')*u), Decimal(0))
        return float(min(max(lum+sd+hd, Decimal(0)), Decimal(1)))


def curve_interval(lo, hi, shadow, highlight):
    """Conservative fp32 power/coefficient and luma uncertainty propagation."""
    def power_bounds(a, b, exponent):
        # Bases in [0,1]; larger exponent decreases the result. Covers fp32
        # exponent division and a bounded fast-pow arithmetic allowance.
        low = np.maximum(0, a**(exponent*(1+2*EPS))*(1-8*EPS))
        high = b**(exponent*(1-2*EPS))*(1+8*EPS)
        return low, high

    if shadow >= 0:
        a, b = power_bounds(lo, hi, 1/(1+shadow))
        aa, bb = power_bounds(lo, hi, 2/(1+shadow))
        sl = np.clip(a-.76*(1+EPS)*bb-hi, 0, 1)
        sh = np.clip(b-.76*(1-EPS)*aa-lo, 0, 1)
    else:
        a, b = power_bounds(lo, hi, 1-shadow)
        sl, sh = a-hi, b-lo
    ql, qh = 1-hi, 1-lo
    a, b = power_bounds(ql, qh, 1/(2-highlight))
    aa, bb = power_bounds(ql, qh, 2/(2-highlight))
    # Absolute allowance includes subtraction from 1 in the shader's original
    # expression, which the algebraically simplified reference avoids.
    hl = np.clip(ql-b+.8*(1-EPS)*aa-2*EPS, -1, 0)
    hh = np.clip(qh-a+.8*(1+EPS)*bb+2*EPS, -1, 0)
    return np.clip(lo+sl+hl-2*EPS, 0, 1), np.clip(hi+sh+hh+2*EPS, 0, 1)


def reference(stored, shadow=0, highlight=1):
    rgb = stored[..., :3].astype(np.float64)
    s, h = float(np.float32(shadow)), float(np.float32(np.clip(highlight, 0, 1)))
    raw = rgb @ WEIGHTS
    lum = np.clip(raw, 0, 1)
    le = 4*EPS*(np.abs(rgb) @ WEIGHTS)
    lo, hi = np.clip(raw-le, 0, 1), np.clip(raw+le, 0, 1)
    active = lum > 0
    # Do not hide ambiguous fp32 dot-product branch selection in a tolerance.
    assert np.all((hi <= 0) | (lo > 0))
    mapped = np.array([tone(float(value), s, h) for value in lum.flat]).reshape(lum.shape)
    ratio = np.ones_like(lum)
    ratio[active] = mapped[active]/lum[active]
    expected = np.clip(rgb*ratio[..., None], 0, 1)
    lower, upper = curve_interval(lo, hi, s, h)
    rl, rh = np.ones_like(lum), np.ones_like(lum)
    rl[active] = lower[active]/hi[active]
    rh[active] = upper[active]/lo[active]
    ends = np.stack((rgb*rl[..., None], rgb*rh[..., None]))
    low = np.clip(ends.min(axis=0), 0, 1)
    high = np.clip(ends.max(axis=0), 0, 1)
    error = np.maximum(expected-low, high-expected)+2*EPS*(1+np.abs(expected))
    return expected, error


def make_image(pattern, precision, shape=(9, 13)):
    if pattern == "palette":
        values = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0],
                  [1, 0, 1], [0, 1, 1], [.025, .05, .1], [.7, .9, 1]]
    elif pattern == "gray":
        levels = [0, .00005, .000099, .000101, .001, .01, .1, .25, .5, .75, .9, .99, 1]
        values = np.repeat(np.array(levels)[:, None], 3, axis=-1)
    elif pattern == "hdr":
        values = [[-.25, -.1, -.2], [1.5, 1.5, 1.5], [-.25, .5, 1.25],
                  [1.5, .1, -.1], [.1, 1.25, -.25]]
    else:
        values = np.random.default_rng(811).uniform(.005, .995, (117, 3))
    dtype = np.float16 if precision == "fp16" else np.float32
    stored = np.ones((*shape, 4), dtype=dtype)
    stored[..., :3] = np.resize(values, (*shape, 3))
    stored[..., 3] = np.resize([-1, 0, .125, .5, 1, 2], shape)
    return stored


def check(ctx, stored, precision, shadow=0, highlight=1):
    source = si.from_numpy(stored, device=ctx[0], precision=precision)
    result = si.ops.highlight_shadow(source, shadow_amount=shadow, highlight_amount=highlight)
    assert result.device is source.device and result.precision == precision
    actual = result.to_numpy().astype(np.float64)
    assert actual.shape == stored.shape and np.isfinite(actual).all()
    np.testing.assert_array_equal(actual[..., 3], stored[..., 3])
    expected, compute_error = reference(stored, shadow, highlight)
    assert compute_error.max() < .001
    error = compute_error+storage_error(expected, precision, compute_error)
    np.testing.assert_array_less(np.abs(actual[..., :3]-expected), np.nextafter(error, np.inf))
    return actual


def test_highlight_shadow_reference_hand_values():
    # At l=.25 and s=1, sqrt(l)-.76*l=.31. At l=.75,h=0:
    # 1-sqrt(1-l)+.8*(1-l)=.7. Negative s=-1 squares l.
    assert tone(.25, 1, 1) == .31
    assert tone(.75, 0, 0) == .7
    assert tone(.25, -1, 1) == .0625


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("pattern", ["palette", "gray", "hdr", "random"])
@pytest.mark.parametrize("shadow,highlight", [(0, 1), (-1, 1), (-.5, 1), (.5, 1), (1, 1),
                                              (0, 0), (0, .5), (.7, .3), (-.7, .3)])
def test_highlight_shadow_reference(ctx, precision, pattern, shadow, highlight):
    check(ctx, make_image(pattern, precision), precision, shadow, highlight)


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("shape", [(1, 1), (1, 17), (19, 1)])
def test_highlight_shadow_tiny_alpha_repeat_and_low_level(ctx, precision, shape):
    stored = make_image("palette", precision, shape)
    first = check(ctx, stored, precision, .8, .2)
    check(ctx, stored, precision, -1, 0)
    np.testing.assert_array_equal(first, check(ctx, stored, precision, .8, .2))
    source = si.from_numpy(stored, device=ctx[0], precision=precision)
    low_level = ColorAdjustOps(*ctx).highlight_shadow(source.texture, .8, .2)
    np.testing.assert_array_equal(first, texture_to_np(low_level))
    stored[..., 3] = 0
    transparent = check(ctx, stored, precision, .8, .2)
    np.testing.assert_array_equal(first[..., :3], transparent[..., :3])


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
def test_highlight_shadow_highlight_clamps_and_shadow_extension(ctx, precision):
    stored = make_image("gray", precision)
    np.testing.assert_array_equal(check(ctx, stored, precision, .5, 0),
                                  check(ctx, stored, precision, .5, -2))
    np.testing.assert_array_equal(check(ctx, stored, precision, .5, 1),
                                  check(ctx, stored, precision, .5, 3))
    # Existing public contract accepts finite shadow values, not a [-1,1] clamp.
    for shadow in [-2, 2]:
        actual = check(ctx, stored, precision, shadow)
        endpoint = check(ctx, stored, precision, np.sign(shadow))
        assert np.max(np.abs(actual-endpoint)) > .01


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
def test_highlight_shadow_defaults_clip_rgb_not_alpha(ctx, precision):
    stored = make_image("hdr", precision)
    actual = check(ctx, stored, precision)
    expected = np.clip(stored[..., :3].astype(np.float64), 0, 1)
    np.testing.assert_allclose(actual[..., :3], expected, rtol=0,
                               atol=2e-3 if precision == "fp16" else 2e-6)


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("shadow", [.5, 1.])
def test_highlight_shadow_near_black_has_no_hard_jump(ctx, precision, shadow):
    dtype = np.float16 if precision == "fp16" else np.float32
    stored = np.ones((1, 2, 4), dtype=dtype)
    stored[0, :, :3] = np.array([.000099, .000101])[:, None]
    source = si.from_numpy(stored, device=ctx[0], precision=precision)
    actual = si.ops.highlight_shadow(source, shadow_amount=shadow).to_numpy()
    # The continuous positive-luma formula changes by <.00015 over this gap.
    # The old bypass instead introduced .002--.01 jumps at the cutoff.
    assert np.max(np.abs(actual[0, 1, :3]-actual[0, 0, :3])) < .0005


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("shadow", [-1., -.5, 0., .5, 1., 2.])
def test_highlight_shadow_dark_curve_to_black(ctx, precision, shadow):
    dtype = np.float16 if precision == "fp16" else np.float32
    levels = np.array([0, 1e-6, 1e-5, 5e-5, 9.9e-5, 1e-4, 1.01e-4, 2e-4], dtype=dtype)
    # Gray and chromatic rays, all nonnegative and approaching exact black.
    rays = np.array([[1, 1, 1], [1, .25, .5], [0, 0, 1]])
    stored = np.ones((3, len(levels), 4), dtype=dtype)
    stored[..., :3] = rays[:, None, :]*levels[None, :, None]
    stored[..., 3] = np.linspace(-1, 2, len(levels))
    actual = check(ctx, stored, precision, shadow, .3)
    np.testing.assert_array_equal(actual[:, 0, :3], 0)
    # Use a tighter direct curve budget here; the general interval includes
    # conservative subtraction errors that are too loose for very dark colors.
    expected, _ = reference(stored, shadow, .3)
    error = 2e-7 + 5e-5*np.abs(expected)
    error += storage_error(expected, precision, error)
    np.testing.assert_array_less(np.abs(actual[..., :3]-expected), np.nextafter(error, np.inf))


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("which", ["shadow", "highlight"])
def test_highlight_shadow_curve_activation_neighbors(ctx, precision, which):
    # Algebraic roots at maximum strength, independent of production branches:
    # sqrt(l)=1/1.76 and sqrt(1-l)=1/1.8 respectively.
    root = (1/1.76)**2 if which == "shadow" else 1-(1/1.8)**2
    dtype = np.float16 if precision == "fp16" else np.float32
    center = dtype(root)
    levels = [np.nextafter(center, dtype(0)), center, np.nextafter(center, dtype(1))]
    stored = np.ones((1, 3, 4), dtype=dtype)
    stored[0, :, :3] = np.array(levels)[:, None]
    check(ctx, stored, precision, 1 if which == "shadow" else 0,
          1 if which == "shadow" else 0)


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
def test_highlight_shadow_shadow_zero_neighbors(ctx, precision):
    stored = make_image("gray", precision)
    for shadow in [-.0001, 0, .0001]:
        check(ctx, stored, precision, shadow, .4)
