"""Independent bounded reference for photographic white balance.

The test oracle contains its own chromaticity loci and fixed inverse matrices;
it does not import the production matrix builder. The resulting host matrix is
quantized to fp32 before an independent clipped linear-RGB pipeline is applied.
"""

import numpy as np
import pytest

import slangimage as si
from slangimage.kernels.white_balance import WhiteBalanceOp
from slangimage.runtime.textures import texture_to_np
from test_core_numeric import decode_srgb, encode_srgb, storage_error


BRADFORD = np.array([
    [.8951, .2664, -.1614],
    [-.7502, 1.7135, .0367],
    [.0389, -.0685, 1.0296],
])
BRADFORD_INV = np.array([
    [.9869929, -.1470543, .1599627],
    [.4323053, .5183603, .0492912],
    [-.0085287, .0400428, .9684867],
])
RGB_TO_XYZ = np.array([
    [.4124564, .3575761, .1804375],
    [.2126729, .7151522, .0721750],
    [.0193339, .1191920, .9503041],
])
XYZ_TO_RGB = np.array([
    [3.2404542, -1.5371385, -.4985314],
    [-.9692660, 1.8760108, .0415560],
    [.0556434, -.2040259, 1.0572252],
])


def illuminant_xy_reference(temperature):
    temperature = float(np.clip(temperature, 2000, 25000))
    reciprocal = 1000/temperature
    if temperature >= 4000:
        if temperature <= 7000:
            x = .244063+.09911*reciprocal+2.9678*reciprocal**2-4.6070*reciprocal**3
        else:
            x = .237040+.24748*reciprocal+1.9018*reciprocal**2-2.0064*reciprocal**3
        y = -3*x*x+2.87*x-.275
    else:
        x = .179910+.8776956*reciprocal-.2343589*reciprocal**2-.2661239*reciprocal**3
        if temperature <= 2222:
            y = -.20219683+2.18555832*x-1.34811020*x*x-1.1063814*x**3
        else:
            y = -.16748867+2.09137015*x-1.37418593*x*x-.9549476*x**3
    return x, y


def white_xyz(x, y):
    return np.array([x/y, 1., (1-x-y)/y])


def matrix_reference(temperature, tint):
    sx, sy = illuminant_xy_reference(temperature)
    sy += float(np.clip(tint, -150, 150))*3e-4
    dx, dy = illuminant_xy_reference(6500)
    source = BRADFORD @ white_xyz(sx, sy)
    target = BRADFORD @ white_xyz(dx, dy)
    adaptation = BRADFORD_INV @ np.diag(target/source) @ BRADFORD
    matrix = XYZ_TO_RGB @ adaptation @ RGB_TO_XYZ
    return matrix.astype(np.float32).astype(np.float64)


def reference(stored, temperature, tint):
    rgb = np.clip(stored[..., :3].astype(np.float64), 0, 1)
    linear = decode_srgb(rgb.copy())
    adapted = np.clip(linear @ matrix_reference(temperature, tint).T, 0, 1)
    expected = encode_srgb(adapted.copy())
    # Bounded RGB and adaptation matrices: transfer-function approximation,
    # three-term dots, clamps and final encode, followed by one texture store.
    compute = 2e-5*(1+np.abs(expected))
    assert compute.max() <= 4e-5
    return expected, compute


def make_image(pattern, precision, shape=(9, 13)):
    if pattern == "palette":
        values = [[0, 0, 0], [1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                  [1, 1, 0], [1, 0, 1], [0, 1, 1], [.25, .5, .75]]
    elif pattern == "gray":
        values = np.repeat(np.array([0, .001, .04045, .1, .25, .5, .9, 1])[:, None], 3, axis=1)
    elif pattern == "hdr":
        values = [[-.25, -.1, 0], [1.25, 1.5, 1], [-.25, .5, 1.25],
                  [1.5, .1, -.1], [.1, 1.25, -.25]]
    else:
        values = np.random.default_rng(1301).uniform(-.25, 1.5, (117, 3))
    dtype = np.float16 if precision == "fp16" else np.float32
    stored = np.ones((*shape, 4), dtype=dtype)
    stored[..., :3] = np.resize(values, (*shape, 3))
    stored[..., 3] = np.resize([-1, 0, .125, .5, 1, 2], shape)
    return stored


def check(ctx, stored, precision, temperature, tint, *, low_level=False):
    source = si.from_numpy(stored, precision=precision, device=ctx[0])
    if low_level:
        actual = texture_to_np(
            WhiteBalanceOp(*ctx).apply(source.texture, temperature, tint)
        ).astype(np.float64)
    else:
        result = si.ops.white_balance(source, temperature, tint)
        assert result.device is source.device and result.precision == precision
        actual = result.to_numpy().astype(np.float64)
    assert actual.shape == stored.shape and np.isfinite(actual).all()
    assert actual[..., :3].min() >= 0 and actual[..., :3].max() <= 1
    np.testing.assert_array_equal(actual[..., 3], stored[..., 3])
    expected, compute = reference(stored, temperature, tint)
    error = compute+storage_error(expected, precision, compute)
    np.testing.assert_array_less(
        np.abs(actual[..., :3]-expected), np.nextafter(error, np.inf)
    )
    return actual


def test_white_balance_reference_locus_known_values_and_branches():
    expected = {
        2000: (.52690, .41326),
        4000: (.38234, .38377),
        6500: (.31278, .32918),
        25000: (.24985, .25480),
    }
    for temperature, xy in expected.items():
        np.testing.assert_allclose(illuminant_xy_reference(temperature), xy, atol=5e-5, rtol=0)
    assert illuminant_xy_reference(1000) == illuminant_xy_reference(2000)
    assert illuminant_xy_reference(30000) == illuminant_xy_reference(25000)


def test_white_balance_reference_maps_assumed_white_to_target_white():
    target = white_xyz(*illuminant_xy_reference(6500))
    for temperature, tint in [(2000, 0), (3999, -75), (4000, 0), (7001, 80), (25000, 150)]:
        sx, sy = illuminant_xy_reference(temperature)
        sy += float(np.clip(tint, -150, 150))*3e-4
        source = white_xyz(sx, sy)
        matrix = matrix_reference(temperature, tint)
        source_rgb = XYZ_TO_RGB @ source
        adapted_xyz = RGB_TO_XYZ @ (matrix @ source_rgb)
        np.testing.assert_allclose(adapted_xyz/target, np.ones(3), atol=2e-6, rtol=0)


CONTROLS = [(6500, 0), (2000, 0), (25000, 0), (3200, -150), (3999, 75),
            (4000, -75), (7000, 150), (9000, -40), (5000, 60)]


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("pattern", ["palette", "gray", "hdr", "random"])
@pytest.mark.parametrize("temperature,tint", CONTROLS)
def test_white_balance_independent_reference(ctx, precision, pattern, temperature, tint):
    check(ctx, make_image(pattern, precision), precision, temperature, tint)


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("shape", [(1, 1), (1, 13), (11, 1)])
def test_white_balance_tiny_alpha_repeated_and_low_level(ctx, precision, shape):
    stored = make_image("hdr", precision, shape)
    first = check(ctx, stored, precision, 4500, 20)
    np.testing.assert_array_equal(first, check(ctx, stored, precision, 4500, 20))
    np.testing.assert_array_equal(first, check(ctx, stored, precision, 4500, 20, low_level=True))


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
def test_white_balance_control_clamps_and_alpha_independence(ctx, precision):
    stored = make_image("random", precision)
    np.testing.assert_array_equal(
        check(ctx, stored, precision, 1000, -200),
        check(ctx, stored, precision, 2000, -150),
    )
    np.testing.assert_array_equal(
        check(ctx, stored, precision, 30000, 200),
        check(ctx, stored, precision, 25000, 150),
    )
    first = check(ctx, stored, precision, 5000, 60)
    stored[..., 3] *= -2
    second = check(ctx, stored, precision, 5000, 60)
    np.testing.assert_array_equal(first[..., :3], second[..., :3])

