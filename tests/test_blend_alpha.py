"""Bounded shared alpha/opacity checks: multiply and screen only.

Straight RGBA in [0,1], equal image sizes and matching fp16/fp32 storage.
Other blend formulas, HDR, tiny-alpha cutoff and mixed precision are deferred.
"""

import numpy as np
import pytest

import slangimage as si
from test_core_numeric import storage_error


def source_over_reference(base, foreground, mode, opacity):
    """Partition coverage into background-only, foreground-only and overlap."""
    ab = base[..., 3:4]
    af = foreground[..., 3:4] * np.clip(np.float32(opacity), 0, 1)
    background_only = ab * (1 - af)
    foreground_only = (1 - ab) * af
    overlap = ab * af
    alpha = background_only + foreground_only + overlap
    b, f = base[..., :3], foreground[..., :3]
    blended = b*f if mode == "multiply" else 1 - (1-b)*(1-f)
    premultiplied = background_only*b + foreground_only*f + overlap*blended
    rgb = np.divide(premultiplied, alpha, out=np.zeros_like(b), where=alpha > 0)
    return np.concatenate((rgb, alpha), axis=-1)


@pytest.mark.parametrize("mode,expected", [
    ("multiply", [0.3375, 0.375, 0.625, 0.625]),
    ("screen", [0.4625, 0.475, 0.725, 0.625]),
])
def test_source_over_reference_hand_calculation(mode, expected):
    # Coverage weights after normalizing: .6 background, .2 foreground,
    # .2 blended overlap. Foreground opacity scales its alpha, not its RGB.
    base = np.array([[[0.25, 0.5, 0.75, 0.5]]])
    foreground = np.array([[[0.75, 0.25, 0.5, 0.5]]])
    np.testing.assert_allclose(source_over_reference(base, foreground, mode, 0.5),
                               [[expected]], rtol=0, atol=1e-15)


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("mode", ["multiply", "screen"])
@pytest.mark.parametrize("opacity", [-1, 0, 0.375, 1, 2])
def test_blend_source_over_alpha_and_opacity(ctx, precision, mode, opacity):
    dtype = np.float16 if precision == "fp16" else np.float32
    rng = np.random.default_rng(521)
    base = rng.random((4, 4, 4)).astype(dtype)
    foreground = rng.random((4, 4, 4)).astype(dtype)
    # All 16 base/foreground alpha combinations; transparent pixels deliberately
    # retain nonzero hidden RGB. Nonzero output alpha is >= .09375, keeping the
    # shader's separate tiny-alpha cutoff outside this test's scope.
    base[..., 3] = np.array([0, 0.25, 0.75, 1])[:, None]
    foreground[..., 3] = [0, 0.25, 0.75, 1]
    b = si.from_numpy(base, precision=precision, device=ctx[0])
    f = si.from_numpy(foreground, precision=precision, device=ctx[0])
    result = si.ops.blend(b, f, mode=mode, opacity=opacity)
    assert result.device is b.device and result.precision == precision
    actual = result.to_numpy().astype(np.float64)
    reference = source_over_reference(base.astype(np.float64), foreground.astype(np.float64),
                                      mode, opacity)
    assert actual.shape == base.shape and np.isfinite(actual).all()
    # Binary-exact alphas/opacities make the output alpha representable in both
    # storage formats. Straight RGB includes division, so budget it separately.
    np.testing.assert_array_equal(actual[..., 3], reference[..., 3])
    zero_alpha = reference[..., 3] == 0
    np.testing.assert_array_equal(actual[zero_alpha], 0)
    rgb = reference[..., :3]
    error = 5e-6 + storage_error(rgb, precision, 5e-6)
    np.testing.assert_array_less(np.abs(actual[..., :3] - rgb), np.nextafter(error, np.inf))
