"""Complete bounded references for the twelve public texture blend modes.

The oracle evaluates formulas and source-over coverage independently of the
shader. Inputs are quantized to their own texture formats before evaluation;
the output is stored at the base image's precision.
"""

import numpy as np
import pytest

import slangimage as si
from slangimage.kernels.blend_modes import BLEND_MODES, BlendModesOp
from slangimage.runtime.textures import texture_to_np
from test_core_numeric import storage_error


MODES = tuple(BLEND_MODES)
EPSILON = 1e-6
LUMINOSITY = np.array([.30, .59, .11])


def overlay(base, foreground):
    return np.where(
        base <= .5,
        2*base*foreground,
        1-2*(1-base)*(1-foreground),
    )


def color_dodge(base, foreground):
    quotient = np.divide(
        base,
        1-foreground,
        out=np.zeros_like(base),
        where=foreground < 1,
    )
    return np.where(base <= 0, 0, np.where(foreground >= 1, 1, np.minimum(1, quotient)))


def color_burn(base, foreground):
    quotient = np.divide(
        1-base,
        foreground,
        out=np.zeros_like(base),
        where=foreground > 0,
    )
    return np.where(base >= 1, 1, np.where(foreground <= 0, 0, 1-np.minimum(1, quotient)))


def soft_light(base, foreground):
    d = np.where(
        base <= .25,
        ((16*base-12)*base+4)*base,
        np.sqrt(np.maximum(base, 0)),
    )
    return np.where(
        foreground <= .5,
        base-(1-2*foreground)*base*(1-base),
        base+(2*foreground-1)*(d-base),
    )


def clip_color(color):
    """W3C ClipColor evaluated without calling production helpers."""
    result = color.copy()
    lum = color @ LUMINOSITY
    low = color.min(axis=-1)
    high = color.max(axis=-1)
    scale = np.divide(lum, np.maximum(lum-low, EPSILON))
    result = np.where(
        (low < 0)[..., None],
        lum[..., None]+(result-lum[..., None])*scale[..., None],
        result,
    )
    scale = np.divide(1-lum, np.maximum(high-lum, EPSILON))
    return np.where(
        (high > 1)[..., None],
        lum[..., None]+(result-lum[..., None])*scale[..., None],
        result,
    )


def color_mode(base, foreground):
    shifted = foreground+(base @ LUMINOSITY-foreground @ LUMINOSITY)[..., None]
    return clip_color(shifted)


def blend_formula(mode, base, foreground):
    if mode == "multiply":
        return base*foreground
    if mode == "screen":
        return base+foreground-base*foreground
    if mode == "overlay":
        return overlay(base, foreground)
    if mode == "darken":
        return np.minimum(base, foreground)
    if mode == "lighten":
        return np.maximum(base, foreground)
    if mode == "color_dodge":
        return color_dodge(base, foreground)
    if mode == "color_burn":
        return color_burn(base, foreground)
    if mode == "hard_light":
        return overlay(foreground, base)
    if mode == "soft_light":
        return soft_light(base, foreground)
    if mode == "linear_light":
        return np.clip(base+2*foreground-1, 0, 1)
    if mode == "vivid_light":
        return np.where(
            foreground <= .5,
            color_burn(base, 2*foreground),
            color_dodge(base, 2*(foreground-.5)),
        )
    if mode == "color":
        return color_mode(base, foreground)
    raise AssertionError(f"unhandled mode {mode}")


def extend_foreground(foreground, shape):
    rows = np.minimum(np.arange(shape[0]), foreground.shape[0]-1)
    cols = np.minimum(np.arange(shape[1]), foreground.shape[1]-1)
    return foreground[rows[:, None], cols[None, :]]


def composite_reference(base, foreground, mode, opacity):
    foreground = extend_foreground(foreground, base.shape[:2])
    opacity = float(np.clip(np.float32(opacity), 0, 1))
    alpha_base = base[..., 3:4]
    alpha_foreground = foreground[..., 3:4]*opacity
    alpha = alpha_foreground+alpha_base*(1-alpha_foreground)
    blended = blend_formula(mode, base[..., :3], foreground[..., :3])
    premultiplied = (
        (1-alpha_base)*alpha_foreground*foreground[..., :3]
        +(1-alpha_foreground)*alpha_base*base[..., :3]
        +alpha_foreground*alpha_base*blended
    )
    rgb = np.divide(
        premultiplied,
        alpha,
        out=np.zeros_like(premultiplied),
        where=alpha > EPSILON,
    )
    return np.concatenate((rgb, alpha), axis=-1)


def stored(array, precision):
    return np.asarray(array, dtype=np.float16 if precision == "fp16" else np.float32)


def check_blend(
    ctx,
    base,
    foreground,
    base_precision,
    foreground_precision,
    mode,
    opacity,
    *,
    low_level=False,
    compute_scale=2e-5,
):
    base = stored(base, base_precision)
    foreground = stored(foreground, foreground_precision)
    base_image = si.from_numpy(base, precision=base_precision, device=ctx[0])
    foreground_image = si.from_numpy(
        foreground,
        precision=foreground_precision,
        device=ctx[0],
    )
    if low_level:
        actual = texture_to_np(
            BlendModesOp(*ctx).apply(
                base_image.texture,
                foreground_image.texture,
                mode,
                opacity,
            )
        ).astype(np.float64)
    else:
        result = si.ops.blend(base_image, foreground_image, mode, opacity)
        assert result.device is ctx[0] and result.precision == base_precision
        actual = result.to_numpy().astype(np.float64)
    reference = composite_reference(
        base.astype(np.float64),
        foreground.astype(np.float64),
        mode,
        opacity,
    )
    assert actual.shape == base.shape and np.isfinite(actual).all()
    compute = compute_scale*(1+np.abs(reference))
    error = compute+storage_error(reference, base_precision, compute)
    np.testing.assert_array_less(np.abs(actual-reference), np.nextafter(error, np.inf))
    return actual


def normalized_images():
    rng = np.random.default_rng(1401)
    base = rng.random((7, 9, 4))
    foreground = rng.random((4, 5, 4))
    base[..., 3] = np.resize([0, .125, .5, 1], base.shape[:2])
    foreground[..., 3] = np.resize([0, .25, .75, 1], foreground.shape[:2])
    return base, foreground


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "base_precision,foreground_precision",
    [("fp16", "fp16"), ("fp16", "fp32"), ("fp32", "fp16"), ("fp32", "fp32")],
)
def test_all_modes_source_over_mixed_precision_and_foreground_extent(
    ctx,
    mode,
    base_precision,
    foreground_precision,
):
    base, foreground = normalized_images()
    check_blend(ctx, base, foreground, base_precision, foreground_precision, mode, .375)


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("mode", MODES)
def test_all_modes_opaque_hdr_formula(ctx, precision, mode):
    nodes = np.array([-.5, -.125, 0, .25, .5, .75, 1, 1.25, 1.5])
    base_values, foreground_values = np.meshgrid(nodes, nodes)
    base = np.stack(
        (base_values, np.roll(base_values, 2, 1), np.roll(base_values, 4, 1), np.ones_like(base_values)),
        axis=-1,
    )
    foreground = np.stack(
        (
            foreground_values,
            np.roll(foreground_values, 3, 0),
            np.roll(foreground_values, 5, 0),
            np.ones_like(foreground_values),
        ),
        axis=-1,
    )
    check_blend(
        ctx,
        base,
        foreground,
        precision,
        precision,
        mode,
        1,
        compute_scale=4e-5,
    )


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("mode", ["overlay", "color_dodge", "color_burn", "hard_light", "soft_light", "vivid_light"])
def test_branch_neighbors(ctx, precision, mode):
    dtype = np.float16 if precision == "fp16" else np.float32
    half = dtype(.5)
    zero, one = dtype(0), dtype(1)
    smallest_normal = dtype(np.finfo(dtype).tiny)
    nodes = np.array(
        [
            -smallest_normal,
            zero,
            smallest_normal,
            dtype(.25),
            np.nextafter(half, zero),
            half,
            np.nextafter(half, one),
            np.nextafter(one, zero),
            one,
            np.nextafter(one, dtype(2)),
        ],
        dtype=dtype,
    )
    base_values, foreground_values = np.meshgrid(nodes, nodes)
    base = np.stack((base_values, base_values, base_values, np.ones_like(base_values)), -1)
    foreground = np.stack(
        (foreground_values, foreground_values, foreground_values, np.ones_like(foreground_values)),
        -1,
    )
    check_blend(ctx, base, foreground, precision, precision, mode, 1)


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
def test_tiny_output_alpha_cutoff(ctx, precision):
    dtype = np.float16 if precision == "fp16" else np.float32
    epsilon = dtype(EPSILON)
    alphas = np.array(
        [
            0,
            np.nextafter(epsilon, dtype(0)),
            epsilon,
            np.nextafter(epsilon, dtype(1)),
            dtype(2e-6),
        ],
        dtype=dtype,
    )
    base = np.zeros((1, len(alphas), 4), dtype=dtype)
    base[..., :3] = [.75, .25, .5]
    foreground = np.zeros_like(base)
    foreground[..., :3] = [.2, .4, .8]
    foreground[..., 3] = alphas
    actual = check_blend(
        ctx,
        base,
        foreground,
        precision,
        precision,
        "multiply",
        1,
        compute_scale=2e-7,
    )
    stored_epsilon = float(epsilon)
    below = alphas.astype(np.float64) <= EPSILON
    np.testing.assert_array_equal(actual[0, below, :3], 0)
    expected_rgb = np.broadcast_to(stored([.2, .4, .8], precision), actual[0, ~below, :3].shape)
    np.testing.assert_allclose(actual[0, ~below, :3], expected_rgb, rtol=0, atol=2e-7)
    assert stored_epsilon > 0


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("mode", MODES)
def test_opacity_clamps_repeated_and_low_level(ctx, precision, mode):
    base, foreground = normalized_images()
    zero = check_blend(ctx, base, foreground, precision, precision, mode, -2)
    np.testing.assert_array_equal(
        zero,
        check_blend(ctx, base, foreground, precision, precision, mode, 0),
    )
    one = check_blend(ctx, base, foreground, precision, precision, mode, 2)
    np.testing.assert_array_equal(
        one,
        check_blend(ctx, base, foreground, precision, precision, mode, 1),
    )
    np.testing.assert_array_equal(
        one,
        check_blend(
            ctx,
            base,
            foreground,
            precision,
            precision,
            mode,
            1,
            low_level=True,
        ),
    )
