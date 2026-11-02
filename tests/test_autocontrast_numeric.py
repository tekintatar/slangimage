"""Bounded per-channel autocontrast references; no CLAHE or histogram bins.

Evaluate min/max on stored input values in float64, including negative/HDR
values. Nonfinite pixels, overflow-scale fp32 ranges and performance excluded.
"""

import numpy as np
import pytest

import slangimage as si
from test_core_numeric import storage_error

CUTOFF = float(np.float32(1e-12))


def autocontrast_reference(image):
    rgb = image[..., :3]
    lo, hi = rgb.min(axis=(0, 1)), rgb.max(axis=(0, 1))
    span = hi - lo
    active = span > CUTOFF
    # Ensure fixture branch decisions are not ambiguous after fp32 subtraction.
    np.testing.assert_array_equal(active, (hi.astype(np.float32) - lo.astype(np.float32)) > CUTOFF)
    result = image.copy()
    result[..., :3] = np.divide(rgb - lo, span, out=rgb.copy(), where=active)
    return result, active


def test_autocontrast_reference_hand_calculation():
    image = np.array([[[-2, -0.25, 3, -0.25], [0, -0.25, 1, 0.5], [2, -0.25, -1, 1.25]]])
    reference, active = autocontrast_reference(image)
    np.testing.assert_array_equal(active, [True, False, True])
    np.testing.assert_array_equal(reference,
                                  [[[0, -0.25, 1, -0.25], [0.5, -0.25, 0.5, 0.5],
                                    [1, -0.25, 0, 1.25]]])


def make_image(pattern, precision):
    dtype = np.float16 if precision == "fp16" else np.float32
    h, w = {"pixel": (1, 1), "row": (1, 17), "column": (19, 1)}.get(pattern, (13, 17))
    rng = np.random.default_rng(621)
    image = rng.random((h, w, 4))
    if pattern in ("constant", "pixel"):
        image[..., :3] = (-2, 0.42, 3)
    elif pattern == "negative":
        image[..., :3] = -8 + 6*image[..., :3]
    elif pattern == "hdr":
        image[..., :3] = -8 + 24*image[..., :3]
    elif pattern == "channel-independent":
        image[..., 0] = -2
        image[..., 1] = 2 + image[..., 1]
        image[..., 2] = -4 + 8*image[..., 2]
    elif pattern == "narrow":
        # A few actual input-storage ULPs, not values that collapse on upload.
        delta = float(np.spacing(dtype(0.5)))
        image[..., :3] = 0.5 + delta*np.resize(np.arange(5), (h, w, 3))
    elif pattern == "outliers":
        image[..., :3] = 0.5
        image[0, 0, :3] = (-3, -2, -1)
        image[-1, -1, :3] = (4, 6, 8)
    else:
        ramp = np.linspace(-2, 3, h*w).reshape(h, w)
        image[..., 0], image[..., 1], image[..., 2] = ramp, -ramp, 2*ramp + 1
    image[..., 3] = np.resize([-0.25, 0, 0.375, 1, 1.25], h*w).reshape(h, w)
    return image.astype(dtype)


def check_autocontrast(ctx, stored, precision):
    image = si.from_numpy(stored, precision=precision, device=ctx[0])
    reference, active = autocontrast_reference(stored.astype(np.float64))
    output = si.ops.autocontrast(image)
    assert output.device is image.device and output.precision == precision
    actual = output.to_numpy().astype(np.float64)
    assert actual.shape == stored.shape and np.isfinite(actual).all()
    np.testing.assert_array_equal(actual[..., 3], stored[..., 3])
    # Constant/degenerate channels are exact passthrough, including HDR values.
    np.testing.assert_array_equal(actual[..., :3][..., ~active], stored[..., :3][..., ~active])
    rgb = reference[..., :3]
    error = 5e-7 + storage_error(rgb, precision, 5e-7)
    np.testing.assert_array_less(np.abs(actual[..., :3] - rgb), np.nextafter(error, np.inf))
    # Active channels must reach both endpoints (within arithmetic/store error).
    for channel in np.flatnonzero(active):
        assert abs(actual[..., channel].min()) <= 5e-7
        assert abs(actual[..., channel].max() - 1) <= 5e-7 + storage_error(1, precision)
    return actual


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("pattern", ["constant", "negative", "hdr", "channel-independent",
                                     "narrow", "outliers", "ramp", "pixel", "row", "column"])
def test_autocontrast_matches_stored_minmax(ctx, precision, pattern):
    check_autocontrast(ctx, make_image(pattern, precision), precision)


def test_autocontrast_fp32_degenerate_threshold(ctx):
    edge = np.float32(CUTOFF)
    spans = np.array([np.nextafter(edge, np.float32(0)), edge,
                      np.nextafter(edge, np.float32(1))], dtype=np.float32)
    image = np.ones((1, 5, 4), dtype=np.float32)
    image[..., :3] = np.linspace(0, 1, 5)[:, None]*spans
    actual = check_autocontrast(ctx, image, "fp32")
    np.testing.assert_array_equal(actual[..., :2], image[..., :2])
    # Fast fp32 division need not produce a bit-exact upper endpoint on CUDA.
    np.testing.assert_allclose(actual[0, [0, -1], 2], [0, 1], rtol=0, atol=5e-7)


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
def test_autocontrast_transparent_outliers_and_repeated_calls(ctx, precision):
    stored = make_image("outliers", precision)
    first = check_autocontrast(ctx, stored, precision)
    check_autocontrast(ctx, make_image("constant", precision), precision)
    # Invisible extrema still determine the global range; changing alpha only
    # must not change RGB, and another image's reduction must not leak state.
    stored[..., 3] = 0
    transparent = check_autocontrast(ctx, stored, precision)
    np.testing.assert_array_equal(transparent[..., :3], first[..., :3])
    np.testing.assert_array_equal(transparent[..., 3], 0)
