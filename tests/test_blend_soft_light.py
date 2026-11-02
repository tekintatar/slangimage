"""Soft-light only: normalized straight RGBA, matching storage and image sizes.

Reference interpolates three foreground anchors (black, gray, white), rather
than calling production blend helpers. HDR, tiny alpha and color mode excluded.
"""

import numpy as np
import pytest

import slangimage as si
from test_core_numeric import storage_error


def soft_light_reference(base, foreground):
    white = np.where(base <= 0.25,
                     np.polynomial.polynomial.polyval(base, [0, 4, -12, 16]),
                     np.sqrt(base))
    dark_half = (1 - 2*foreground)*base**2 + 2*foreground*base
    light_half = (2 - 2*foreground)*base + (2*foreground - 1)*white
    return np.where(foreground <= 0.5, dark_half, light_half)


def composite_reference(base, foreground, opacity):
    ab, af = base[..., 3:4], foreground[..., 3:4]*opacity
    weights = ((1-af)*ab, (1-ab)*af, ab*af)
    alpha = sum(weights)
    colors = (base[..., :3], foreground[..., :3],
              soft_light_reference(base[..., :3], foreground[..., :3]))
    premultiplied = sum(w*c for w, c in zip(weights, colors))
    rgb = np.divide(premultiplied, alpha, out=np.zeros_like(premultiplied), where=alpha > 0)
    return np.concatenate((rgb, alpha), axis=-1)


def test_soft_light_reference_anchor_values():
    base = np.array([0, 0.125, 0.25, 0.5625, 1])
    np.testing.assert_array_equal(soft_light_reference(base, 0), base**2)
    np.testing.assert_array_equal(soft_light_reference(base, 0.5), base)
    np.testing.assert_array_equal(soft_light_reference(base, 1), [0, 0.34375, 0.5, 0.75, 1])
    # Interior points on both foreground halves, including the polynomial and
    # square-root branches; rational inputs make these hand results exact.
    np.testing.assert_array_equal(soft_light_reference(np.array([0.5, 0.125, 0.5625]),
                                                      np.array([0.25, 0.75, 0.75])),
                                  [0.375, 0.234375, 0.65625])


def test_soft_light_reference_partial_alpha_hand_calculation():
    # Weights .6 background / .2 foreground / .2 overlap. B(.25,.75)=.375.
    base = np.array([[[0.25, 0.25, 0.25, 0.5]]])
    foreground = np.array([[[0.75, 0.75, 0.75, 0.5]]])
    np.testing.assert_allclose(composite_reference(base, foreground, 0.5),
                               [[[0.375, 0.375, 0.375, 0.625]]], atol=1e-15, rtol=0)


def check_blend(ctx, base, foreground, precision, opacity, reference):
    b = si.from_numpy(base, precision=precision, device=ctx[0])
    f = si.from_numpy(foreground, precision=precision, device=ctx[0])
    result = si.ops.blend(b, f, mode="soft_light", opacity=opacity)
    assert result.device is b.device and result.precision == precision
    actual = result.to_numpy().astype(np.float64)
    assert actual.shape == reference.shape and np.isfinite(actual).all()
    np.testing.assert_array_equal(actual[..., 3], reference[..., 3])
    # Same bounded arithmetic/storage policy as test_blend_alpha. Inputs
    # are already quantized, and all positive output alphas here are >= .5.
    error = 5e-6 + storage_error(reference[..., :3], precision, 5e-6)
    np.testing.assert_array_less(np.abs(actual[..., :3] - reference[..., :3]),
                                 np.nextafter(error, np.inf))


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("ab,af,opacity", [(1, 1, 1), (0.5, 0.75, 0.375),
                                          (0, 0.5, 1), (0.75, 0, 1)])
def test_soft_light_branch_boundaries(ctx, precision, ab, af, opacity):
    dtype = np.float16 if precision == "fp16" else np.float32
    # Immediate representable neighbors stay distinct AFTER input quantization.
    b_nodes = np.array([0, 0.125, np.nextafter(dtype(0.25), dtype(0)), 0.25,
                        np.nextafter(dtype(0.25), dtype(1)), 0.5, 0.75, 1], dtype=dtype)
    f_nodes = np.array([0, 0.25, np.nextafter(dtype(0.5), dtype(0)), 0.5,
                        np.nextafter(dtype(0.5), dtype(1)), 0.75, 1], dtype=dtype)
    b, f = np.meshgrid(b_nodes, f_nodes)
    base = np.stack((b, np.roll(b, 3, axis=1), np.roll(b, 5, axis=1), np.full_like(b, ab)), axis=-1)
    foreground = np.stack((f, np.roll(f, 2, axis=0), np.roll(f, 4, axis=0),
                           np.full_like(f, af)), axis=-1)
    reference = composite_reference(base.astype(np.float64), foreground.astype(np.float64), opacity)
    check_blend(ctx, base, foreground, precision, opacity, reference)


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("opacity", [0, 0.375, 1])
def test_soft_light_neutral_gray_over_opaque_base(ctx, precision, opacity):
    dtype = np.float16 if precision == "fp16" else np.float32
    base = np.random.default_rng(522).random((5, 9, 4)).astype(dtype)
    base[..., 3] = 1
    foreground = np.full_like(base, 0.5)
    foreground[..., 3] = 0.75
    # Neutral gray is an identity over opaque backgrounds, NOT in general
    # over partial alpha (where the foreground-only coverage contributes gray).
    check_blend(ctx, base, foreground, precision, opacity, base.astype(np.float64))
