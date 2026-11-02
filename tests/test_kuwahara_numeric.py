"""Isolated generalized Kuwahara: angular sectors and centered variance.

The oracle normalizes log weights, avoiding raw inverse-variance overflow.
This is the public eight-sector RGB operator, not the composed Lab effect.
"""

import numpy as np
import pytest

import slangimage as si
from test_core_numeric import storage_error


EPS = float(np.float32(1e-5))
ARITHMETIC = 2e-5


def sector_indices(offsets):
    angles = np.arctan2(offsets[:, 0], offsets[:, 1])
    # Angles on axes/diagonals enter the sector that begins at that boundary.
    return (np.searchsorted(np.arange(-4, 5)*np.pi/4, angles, side="right")-1) % 8


def kuwahara_reference(stored, radius=2, sigma_spatial=3, var_power=3):
    rgb = stored[..., :3].astype(np.float64)
    h, w = rgb.shape[:2]
    ss, power = float(np.float32(sigma_spatial)), float(np.float32(var_power))
    dy, dx = np.mgrid[-radius:radius+1, -radius:radius+1]
    offsets = np.stack((dy.ravel(), dx.ravel()), axis=-1)
    sectors = sector_indices(offsets)
    spatial = np.exp(-np.sum(offsets**2, axis=-1)/(ss*radius))
    yy, xx = np.mgrid[:h, :w]
    neighbors = np.stack([rgb[np.clip(yy+y, 0, h-1), np.clip(xx+x, 0, w-1)] for y, x in offsets])
    means, variances, variance_errors = [], [], []
    for sector in range(8):
        mask = sectors == sector
        weights = spatial[mask]
        if weights.sum() <= EPS:
            continue
        weights = weights/weights.sum()
        samples = neighbors[mask]
        mean = np.sum(weights[:, None, None, None]*samples, axis=0)
        signal = np.linalg.norm(samples, axis=-1)
        signal_mean = np.sum(weights[:, None, None]*signal, axis=0)
        # Centered two-pass variance, not the shader's E[x^2]-E[x]^2.
        variance = np.sum(weights[:, None, None]*(signal-signal_mean)**2, axis=0)
        means.append(mean)
        variances.append(variance)
        # Conservative fp32 spatial/sum/cancellation budget on this small
        # corpus. Propagate it through variance weighting; don't use a fixed
        # RGB tolerance that silently admits different low-variance weights.
        moment = np.sum(weights[:, None, None]*signal**2, axis=0)
        variance_errors.append((3*len(weights)+20)*np.finfo(np.float32).eps*moment
                               + 2e-5*variance)
    means, variances = np.array(means), np.array(variances)
    variance_errors = np.array(variance_errors)
    clamped_variances = np.maximum(variances, EPS)
    minimum = clamped_variances.min(axis=0)
    log_weights = -power*np.log(clamped_variances/minimum)
    weights = np.exp(log_weights)
    result = np.sum(weights[..., None]*means, axis=0)/weights.sum(axis=0)[..., None]
    lower_var = np.maximum(variances-variance_errors, EPS)
    upper_var = np.maximum(variances+variance_errors, EPS)
    low_weights = np.exp(-power*np.log(upper_var/minimum))
    high_weights = np.exp(-power*np.log(lower_var/minimum))
    weight_error = np.maximum(weights-low_weights, high_weights-weights)
    propagated = np.sum(weight_error[..., None]*np.abs(means-result), axis=0)/low_weights.sum(axis=0)[..., None]
    return result, propagated


def test_kuwahara_reference_sector_boundaries():
    offsets = np.array([[0, -1], [-1, -1], [-1, 0], [-1, 1],
                        [0, 1], [1, 1], [1, 0], [1, -1], [0, 0]])
    np.testing.assert_array_equal(sector_indices(offsets), [0, 1, 2, 3, 4, 5, 6, 7, 4])


def test_kuwahara_reference_equal_variance_blends_color():
    # Every sample has norm 1, but distinct RGB. All eight variance floors tie;
    # the result is a blend of sector means, not a winning sector or grayscale.
    stored = np.zeros((3, 3, 4))
    stored[..., 0] = 1
    stored[1, 1, :3] = (0, 1, 0)
    stored[1, 2, :3] = (0, 0, 1)
    result, _ = kuwahara_reference(stored, radius=1, sigma_spatial=1, var_power=7)
    right_weight = np.exp(-1)
    expected = np.array([7/8, 1/(8*(1+right_weight)), right_weight/(8*(1+right_weight))])
    np.testing.assert_allclose(result[1, 1], expected, rtol=0, atol=1e-15)


def make_image(pattern, precision, shape=(7, 9)):
    h, w = shape
    yy, xx = np.mgrid[:h, :w]
    stored = np.ones((h, w, 4))
    if pattern == "constant":
        stored[..., :3] = (-0.25, 0.375, 1.25)
    elif pattern == "equal-norm":
        stored[..., :3] = np.eye(3)[(xx+2*yy) % 3]
    elif pattern == "impulse":
        stored[..., :3] = 0.125
        stored[h//2, w//2, :3] = (1, 0.5, 0.25)
    elif pattern == "ramp":
        levels = np.linspace(0.125, 1.125, 5)
        magnitude = levels[np.minimum(np.arange(h*w)*5//(h*w), 4)].reshape(h, w)
        stored[..., :3] = np.eye(3)[(xx+yy) % 3]*magnitude[..., None]
    else:
        # Separated dyadic norms keep nonzero sector variances well away from
        # the floor; equal-norm chromatic ties are covered explicitly above.
        palette = np.array([[-0.25, 0, 0], [0, 0.5, 0], [0, 0, 1],
                            [1.25, 0, 0], [0, -0.25, 0], [0, 0, 0.5]])
        stored[..., :3] = palette[np.random.default_rng(671).integers(0, len(palette), (h, w))]
    stored[..., 3] = np.resize([-0.25, 0, 0.375, 1, 1.25], h*w).reshape(h, w)
    return stored.astype(np.float16 if precision == "fp16" else np.float32)


def check_kuwahara(ctx, stored, precision, **params):
    params = {"radius": 2, "sigma_spatial": 3, "var_power": 3, **params}
    expected, propagated = kuwahara_reference(stored, **params)
    assert propagated.max() < 0.001, "fixture is ill-conditioned: variance-floor error dominates"
    source = si.from_numpy(stored, device=ctx[0], precision=precision)
    result = si.ops.kuwahara(source, **params)
    assert result.device is source.device and result.precision == precision
    actual = result.to_numpy().astype(np.float64)
    assert actual.shape == stored.shape and np.isfinite(actual).all()
    np.testing.assert_array_equal(actual[..., 3], stored[..., 3])
    compute = ARITHMETIC+propagated
    error = compute+storage_error(expected, precision, compute)
    np.testing.assert_array_less(np.abs(actual[..., :3]-expected), np.nextafter(error, np.inf))
    return actual


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("radius", [1, 2, 4])
@pytest.mark.parametrize("pattern", ["constant", "equal-norm", "impulse", "ramp", "random"])
def test_kuwahara_sector_reference(ctx, precision, radius, pattern):
    check_kuwahara(ctx, make_image(pattern, precision), precision, radius=radius)


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("power", [0.5, 1, 3.5, 7])
def test_kuwahara_variance_power(ctx, precision, power):
    check_kuwahara(ctx, make_image("random", precision), precision, var_power=power)


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("sigma", [0.01, 0.1, 1, 12])
def test_kuwahara_spatial_weights_and_empty_sectors(ctx, precision, sigma):
    check_kuwahara(ctx, make_image("equal-norm", precision), precision, radius=1, sigma_spatial=sigma)


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("shape", [(1, 1), (1, 9), (7, 1)])
def test_kuwahara_tiny_borders(ctx, precision, shape):
    check_kuwahara(ctx, make_image("random", precision, shape), precision)


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
def test_kuwahara_default_parameters(ctx, precision):
    stored = make_image("equal-norm", precision)
    expected, _ = kuwahara_reference(stored, radius=7, sigma_spatial=12, var_power=7)
    actual = si.ops.kuwahara(si.from_numpy(stored, device=ctx[0], precision=precision)).to_numpy()
    np.testing.assert_array_equal(actual[..., 3], stored[..., 3])
    error = ARITHMETIC+storage_error(expected, precision, ARITHMETIC)
    np.testing.assert_array_less(np.abs(actual[..., :3]-expected), np.nextafter(error, np.inf))


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
def test_kuwahara_alpha_independence_and_repeated_calls(ctx, precision):
    stored = make_image("random", precision)
    first = check_kuwahara(ctx, stored, precision)
    check_kuwahara(ctx, make_image("constant", precision), precision, radius=4)
    stored[..., 3] = 0
    transparent = check_kuwahara(ctx, stored, precision)
    np.testing.assert_array_equal(first[..., :3], transparent[..., :3])


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("power", [7.5, 8, 12, 64, 1e6])
def test_kuwahara_large_power_constant_regression(ctx, precision, power):
    stored = np.full((3, 3, 4), 0.25, dtype=np.float16 if precision == "fp16" else np.float32)
    check_kuwahara(ctx, stored, precision, radius=7, sigma_spatial=12, var_power=power)


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("power", [8, 12, 64])
def test_kuwahara_large_power_nonconstant(ctx, precision, power):
    check_kuwahara(ctx, make_image("random", precision), precision, radius=4, var_power=power)
