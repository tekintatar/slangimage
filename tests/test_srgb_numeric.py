"""Independent scalar Decimal references for the two public transfer operators.

Normalized-range equations: https://registry.color.org/rgb-registry/files/sRGB.pdf
Negative linear extension and unclipped HDR are library behavior, not a claim
of sign-preserving extended-sRGB compliance. No production math helper is used.
"""

from decimal import Decimal, localcontext
from functools import lru_cache

import numpy as np
import pytest

import slangimage as si
from test_core_numeric import storage_error


DIRECTIONS = ["srgb_to_linear", "linear_to_srgb"]
CUTOFFS = {"srgb_to_linear": np.float32(0.04045), "linear_to_srgb": np.float32(0.0031308)}


@lru_cache(maxsize=16384)
def scalar_reference(value, direction):
    # Compare against the actual shader-literal cutoff, then evaluate each
    # branch with decimal constants at 50 digits on the exact stored input.
    with localcontext() as context:
        context.prec = 50
        x = Decimal.from_float(float(value))
        if direction == "srgb_to_linear":
            if value <= CUTOFFS[direction]:
                return float(x/Decimal("12.92"))
            return float(((x+Decimal("0.055"))/Decimal("1.055"))**Decimal("2.4"))
        if value <= CUTOFFS[direction]:
            return float(x*Decimal("12.92"))
        return float(Decimal("1.055")*x**(Decimal(1)/Decimal("2.4"))-Decimal("0.055"))


def reference(values, direction):
    unique, inverse = np.unique(values, return_inverse=True)
    mapped = np.array([scalar_reference(float(value), direction) for value in unique])
    return mapped[inverse].reshape(values.shape)


@pytest.mark.parametrize("direction,midpoint,negative", [
    ("srgb_to_linear", 0.21404114048223255, -0.03869969040247678),
    ("linear_to_srgb", 0.7353569830524495, -6.46),
])
def test_srgb_reference_known_values(direction, midpoint, negative):
    actual = reference(np.array([0, 0.5, 1, -0.5]), direction)
    np.testing.assert_allclose(actual, [0, midpoint, 1, negative], rtol=0, atol=1e-15)


def make_image(pattern, precision, direction, shape=(13, 17)):
    dtype = np.float16 if precision == "fp16" else np.float32
    h, w = shape
    if pattern == "threshold":
        center = dtype(CUTOFFS[direction])
        lo = np.nextafter(center, dtype(0))
        hi = np.nextafter(center, dtype(1))
        levels = [np.nextafter(lo, dtype(0)), lo, center, hi, np.nextafter(hi, dtype(1))]
    elif pattern == "negative":
        levels = [-2, -1, -0.5, -0.055, -0.001, 0]
    elif pattern == "hdr":
        levels = [-2, 0, 0.25, 1, 1.5, 2, 4]
    elif pattern == "dark":
        levels = [0, 0.0001, 0.0002, 0.0005, 0.001, 0.002]
    elif pattern == "endpoints":
        levels = [0, np.nextafter(dtype(1), dtype(0)), 1, np.nextafter(dtype(1), dtype(2))]
    elif pattern == "ramp":
        levels = np.linspace(0, 1, 256)
    else:
        levels = np.random.default_rng(641).uniform(-0.25, 2, h*w*3)
    image = np.ones((h, w, 4), dtype=dtype)
    # Distinct channel phases exercise per-channel branch selection.
    image[..., :3] = np.resize(levels, (h, w, 3))
    image[..., 3] = np.resize([-2, -0.25, 0, 0.375, 1, 1.25, 4], h*w).reshape(h, w)
    return image


def check_transfer(ctx, stored, precision, direction):
    source = si.from_numpy(stored, device=ctx[0], precision=precision)
    result = getattr(si.ops, direction)(source)
    assert result.device is source.device and result.precision == precision
    actual = result.to_numpy().astype(np.float64)
    assert actual.shape == stored.shape and np.isfinite(actual).all()
    np.testing.assert_array_equal(actual[..., 3], stored[..., 3])
    expected = reference(stored[..., :3].astype(np.float64), direction)
    # Bounded RGB [-2,4], fp32 coefficients/pow and a single output store.
    # Use tighter ULP checks below for the linear branch and inclusive cutoff.
    compute_error = 2e-7 + 2e-6*np.abs(expected)
    error = compute_error + storage_error(expected, precision, compute_error)
    np.testing.assert_array_less(np.abs(actual[..., :3]-expected), np.nextafter(error, np.inf))
    return actual


@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("pattern", ["threshold", "negative", "hdr", "dark", "endpoints", "ramp", "random"])
def test_srgb_independent_reference(ctx, direction, precision, pattern):
    check_transfer(ctx, make_image(pattern, precision, direction), precision, direction)


@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("precision", ["fp16", "fp32"])
def test_srgb_linear_branch_and_inclusive_cutoff(ctx, direction, precision):
    dtype = np.float16 if precision == "fp16" else np.float32
    edge = dtype(CUTOFFS[direction])
    if float(edge) > float(CUTOFFS[direction]):
        edge = np.nextafter(edge, dtype(0))
    levels = np.array([-2, -0.5, -0.001, 0, 0.0001,
                       np.nextafter(edge, dtype(0)), edge], dtype=dtype)
    stored = np.ones((1, len(levels), 4), dtype=dtype)
    stored[..., :3] = levels[None, :, None]
    actual = check_transfer(ctx, stored, precision, direction)
    expected = reference(stored[..., :3].astype(np.float64), direction)
    compute_error = 2*storage_error(expected, "fp32")
    error = compute_error + storage_error(expected, precision, compute_error)
    np.testing.assert_array_less(np.abs(actual[..., :3]-expected), np.nextafter(error, np.inf))
    np.testing.assert_array_equal(actual[0, 3, :3], 0)


@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("shape", [(1, 1), (1, 17), (19, 1)])
def test_srgb_tiny_images(ctx, direction, precision, shape):
    check_transfer(ctx, make_image("hdr", precision, direction, shape), precision, direction)


@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("precision", ["fp16", "fp32"])
def test_srgb_rgb_independent_of_alpha_and_channel_order(ctx, direction, precision):
    stored = make_image("random", precision, direction)
    first = check_transfer(ctx, stored, precision, direction)
    stored[..., 3] = 0
    transparent = check_transfer(ctx, stored, precision, direction)
    np.testing.assert_array_equal(first[..., :3], transparent[..., :3])
    stored[..., :3] = stored[..., [2, 0, 1]]
    permuted = check_transfer(ctx, stored, precision, direction)
    np.testing.assert_array_equal(permuted[..., :3], transparent[..., [2, 0, 1]])
