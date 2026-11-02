"""Bounded color-mode references: normalized straight RGBA, matching precision.

Project foreground chroma toward a neutral gray at the background luminosity
until every channel fits [0,1]. This avoids the shader's sequential ClipColor
branches; it is equivalent on this normalized-input corpus, not arbitrary HDR.
"""

import numpy as np
import pytest

import slangimage as si
from test_core_numeric import storage_error

LUMINOSITY = np.array([0.30, 0.59, 0.11])


def color_reference(base, foreground):
    target = (base @ LUMINOSITY)[..., None]
    chroma = foreground - (foreground @ LUMINOSITY)[..., None]
    # Each channel bounds the permitted distance along the neutral->chroma ray.
    limits = np.full_like(chroma, np.inf)
    np.divide(1 - target, chroma, out=limits, where=chroma > 0)
    np.divide(-target, chroma, out=limits, where=chroma < 0)
    scale = np.minimum(1, limits.min(axis=-1, keepdims=True))
    return target + scale*chroma


def composite_reference(base, foreground, opacity):
    ab, af = base[..., 3:4], foreground[..., 3:4]*opacity
    weights = ((1-af)*ab, (1-ab)*af, ab*af)
    alpha = sum(weights)
    colors = (base[..., :3], foreground[..., :3],
              color_reference(base[..., :3], foreground[..., :3]))
    premultiplied = sum(w*c for w, c in zip(weights, colors))
    rgb = np.divide(premultiplied, alpha, out=np.zeros_like(premultiplied), where=alpha > 0)
    return np.concatenate((rgb, alpha), axis=-1)


def test_color_reference_hand_calculations():
    base = np.repeat(np.array([0.15, 0.65, 0.5, 0.4])[:, None], 3, axis=1)
    foreground = np.array([[1, 0, 0], [1, 0, 0], [0.6, 0.3, 0.2], [0.5, 0.5, 0.5]])
    # Lower clipping, upper clipping, no clipping and achromatic foreground.
    expected = np.array([[0.5, 0, 0], [1, 0.5, 0.5], [0.721, 0.421, 0.321], [0.4]*3])
    result = color_reference(base, foreground)
    np.testing.assert_allclose(result, expected, atol=1e-15, rtol=0)
    np.testing.assert_allclose(result @ LUMINOSITY, base @ LUMINOSITY, atol=1e-15, rtol=0)


def test_color_reference_partial_alpha_hand_calculation():
    # Target lum=.65 gives mapped red (1,.5,.5). Coverage weights are .6/.2/.2.
    base = np.array([[[0.65, 0.65, 0.65, 0.5]]])
    foreground = np.array([[[1, 0, 0, 0.5]]])
    np.testing.assert_allclose(composite_reference(base, foreground, 0.5),
                               [[[0.79, 0.49, 0.49, 0.625]]], atol=1e-15, rtol=0)


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("ab,af,opacity", [(1, 1, 1), (0.5, 0.75, 0.375),
                                          (0, 0.5, 1), (0.75, 0, 1),
                                          (0.5, 0.75, 0), (0, 0, 1)])
def test_color_gamut_and_alpha_reference(ctx, precision, ab, af, opacity):
    dtype = np.float16 if precision == "fp16" else np.float32
    palette = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0],
                        [1, 0, 1], [0, 1, 1], [0.75, 0.25, 0.25],
                        [0.25, 0.75, 0.25], [0.25, 0.25, 0.75],
                        [0, 0, 0], [1, 1, 1], [0.5]*3, [0.75, 0.5, 0.25]], dtype=dtype)
    # (.75,.25,.25) reaches the low/high gamut edges at target lum .15/.65.
    # Sample those levels and their adjacent representable stored values.
    levels = [0, 0.125, 0.25, 0.5, 0.75, 0.875, 1]
    for edge in (dtype(0.15), dtype(0.65)):
        levels.extend([np.nextafter(edge, dtype(0)), edge, np.nextafter(edge, dtype(1))])
    backgrounds = np.concatenate((np.repeat(np.array(levels, dtype=dtype)[:, None], 3, axis=1),
                                   palette))
    shape = (len(backgrounds), len(palette), 4)
    base, foreground = np.empty(shape, dtype=dtype), np.empty(shape, dtype=dtype)
    base[..., :3], foreground[..., :3] = backgrounds[:, None, :], palette[None, :, :]
    base[..., 3], foreground[..., 3] = ab, af
    b = si.from_numpy(base, precision=precision, device=ctx[0])
    f = si.from_numpy(foreground, precision=precision, device=ctx[0])
    result = si.ops.blend(b, f, mode="color", opacity=opacity)
    assert result.device is b.device and result.precision == precision
    actual = result.to_numpy().astype(np.float64)
    reference = composite_reference(base.astype(np.float64), foreground.astype(np.float64), opacity)
    assert actual.shape == reference.shape and np.isfinite(actual).all()
    np.testing.assert_array_equal(actual[..., 3], reference[..., 3])
    if ab == 0 and af == 0:
        np.testing.assert_array_equal(actual, 0)
    # Bounded RGB/alpha corpus; positive output alpha >= .5. Input quantization
    # is excluded from this arithmetic + output storage budget, as in the other blend tests.
    error = 5e-6 + storage_error(reference[..., :3], precision, 5e-6)
    np.testing.assert_array_less(np.abs(actual[..., :3] - reference[..., :3]),
                                 np.nextafter(error, np.inf))
    if ab == af == opacity == 1:
        expected_lum = base[..., :3].astype(np.float64) @ LUMINOSITY
        np.testing.assert_array_less(np.abs(actual[..., :3] @ LUMINOSITY - expected_lum),
                                     np.nextafter(error @ LUMINOSITY, np.inf))
