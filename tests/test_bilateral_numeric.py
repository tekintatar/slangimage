"""Independent full/separable bilateral references on small finite images.

Float64 neighborhood tensors; separable H-store uncertainty is propagated
through V range weights and normalization, not treated as an exact fp32 image.
"""

from itertools import product

import numpy as np
import pytest

import slangimage as si
from test_core_numeric import storage_error


ARITHMETIC = 5e-6


def neighborhood(rgb, radius, axis):
    offsets = list(product(range(-radius, radius+1), repeat=2)) if axis is None else [
        (k, 0) if axis == 0 else (0, k) for k in range(-radius, radius+1)]
    h, w = rgb.shape[:2]
    yy, xx = np.mgrid[:h, :w]
    indices = [(np.clip(yy+dy, 0, h-1), np.clip(xx+dx, 0, w-1)) for dy, dx in offsets]
    values = np.stack([rgb[y, x] for y, x in indices])
    same_pixel = np.stack([(y == yy) & (x == xx) for y, x in indices])
    return np.array(offsets), indices, values, same_pixel


def pass_reference(rgb, radius, sigma_spatial, sigma_range, range_type,
                   axis=None, uncertainty=None, require_margin=True):
    ss, sr = float(np.float32(sigma_spatial)), float(np.float32(sigma_range))
    offsets, indices, values, same_pixel = neighborhood(rgb, radius, axis)
    distance = np.sum((values-rgb)**2, axis=-1)
    spatial = np.ones(len(offsets)) if ss <= 0 else np.exp(-np.sum(offsets**2, axis=1)/(2*ss**2))
    spatial = spatial[:, None, None]
    if range_type == "cutoff":
        weights = spatial*(distance <= sr**2)
    else:
        weights = spatial*np.exp(-distance/(2*sr**2))
    denominator = weights.sum(axis=0)
    mean = np.sum(weights[..., None]*values, axis=0)/denominator[..., None]

    error = np.zeros_like(rgb) if uncertainty is None else uncertainty
    neighbor_error = np.stack([error[y, x] for y, x in indices])
    delta = np.abs(values-rgb)
    delta_error = neighbor_error + error
    low_distance = np.sum(np.maximum(delta-delta_error, 0)**2, axis=-1)
    high_distance = np.sum((delta+delta_error)**2, axis=-1)
    # A replicated sample that is the center itself has exactly zero distance,
    # even though the center's stored value is uncertain.
    low_distance[same_pixel] = high_distance[same_pixel] = 0
    if range_type == "cutoff":
        if require_margin:
            # Guard discrete decisions against both storage and fp32 distance/
            # sigma-square arithmetic. Boundary fixtures prove exact binary
            # cases separately rather than accepting a loose RGB tolerance.
            rounding = 16*np.finfo(np.float32).eps*np.maximum(high_distance, sr**2)
            rounding[same_pixel] = 0
            inside = distance <= sr**2
            assert np.all((high_distance+rounding)[inside] <= sr**2), "ambiguous included cutoff sample"
            assert np.all((low_distance-rounding)[~inside] > sr**2), "ambiguous excluded cutoff sample"
        lower = upper = weights
    else:
        lower = spatial*np.exp(-high_distance/(2*sr**2))
        upper = spatial*np.exp(-low_distance/(2*sr**2))
    # Centering around the nominal mean avoids a loose numerator/denominator
    # interval. Bound weight perturbation times residual, plus sample error.
    weight_error = np.maximum(weights-lower, upper-weights)
    numerator_error = np.sum(weight_error[..., None]*np.abs(values-mean)
                             + upper[..., None]*neighbor_error, axis=0)
    propagated = numerator_error/lower.sum(axis=0)[..., None]
    return mean, propagated


def bilateral_reference(stored, precision, radius=2, sigma_spatial=0,
                        sigma_range=0.35, range_type="gaussian", separable=False,
                        require_margin=True):
    rgb = stored[..., :3].astype(np.float64)
    if separable:
        horizontal, _ = pass_reference(rgb, radius, sigma_spatial, sigma_range, range_type,
                                       axis=1, require_margin=require_margin)
        h_error = ARITHMETIC + storage_error(horizontal, precision, ARITHMETIC)
        result, propagated = pass_reference(horizontal, radius, sigma_spatial, sigma_range,
                                             range_type, axis=0, uncertainty=h_error,
                                             require_margin=require_margin)
    else:
        result, propagated = pass_reference(rgb, radius, sigma_spatial, sigma_range, range_type,
                                             require_margin=require_margin)
    compute = ARITHMETIC + propagated
    return result, compute + storage_error(result, precision, compute)


def test_bilateral_reference_corner_box_hand_calculation():
    rgb = np.array([[[0, 0, 0], [1, 0, 0]], [[2, 0, 0], [3, 0, 0]]], dtype=float)
    actual, _ = pass_reference(rgb, 1, 0, 4, "cutoff")
    # Edge replication gives 4*top-left + 2*top-right + 2*bottom-left + bottom-right.
    np.testing.assert_allclose(actual[0, 0], [1, 0, 0], rtol=0, atol=1e-15)


def test_bilateral_reference_gaussian_hand_calculation():
    rgb = np.array([[[0, 0, 0], [0.5, 0, 0], [1, 0, 0]]])
    actual, _ = pass_reference(rgb, 1, 1, 0.5, "gaussian", axis=1)
    # Left border: two zero samples, plus .5 with spatial*range weight exp(-1).
    expected = 0.5*np.exp(-1)/(1 + np.exp(-0.5) + np.exp(-1))
    np.testing.assert_allclose(actual[0, 0, 0], expected, rtol=0, atol=1e-15)


def make_image(pattern, precision, shape=(7, 9)):
    h, w = shape
    yy, xx = np.mgrid[:h, :w]
    image = np.ones((h, w, 4))
    if pattern == "constant":
        image[..., :3] = (-0.25, 0.375, 1.25)
    elif pattern == "impulse":
        image[..., :3] = 0.125
        image[h//2, w//2, :3] = (1, 0.5, 0.25)
    elif pattern in ("clusters", "hdr-clusters"):
        image[..., :3] = (((xx//3+yy//2) % 2)*0.75)[..., None]
        image[..., :3] += (((xx+yy) % 3)/32)[..., None]
        if pattern == "hdr-clusters":
            image[..., :3] += (-0.25, 0.125, 0.5)
    elif pattern == "ramp":
        image[..., :3] = np.linspace(-0.25, 1.25, h*w*3).reshape(h, w, 3)
    else:
        image[..., :3] = np.random.default_rng(651).uniform(-0.25, 1.25, (h, w, 3))
    image[..., 3] = np.resize([-0.25, 0, 0.375, 1, 1.25], h*w).reshape(h, w)
    return image.astype(np.float16 if precision == "fp16" else np.float32)


def check_bilateral(ctx, stored, precision, require_margin=True, **params):
    params = {"radius": 2, "sigma_range": 0.35, **params}
    expected, error = bilateral_reference(stored, precision, require_margin=require_margin, **params)
    source = si.from_numpy(stored, device=ctx[0], precision=precision)
    output = si.ops.bilateral(source, **params)
    assert output.device is source.device and output.precision == precision
    actual = output.to_numpy().astype(np.float64)
    assert actual.shape == stored.shape and np.isfinite(actual).all()
    np.testing.assert_array_equal(actual[..., 3], stored[..., 3])
    np.testing.assert_array_less(np.abs(actual[..., :3]-expected), np.nextafter(error, np.inf))
    return actual


CASES = [("gaussian", p) for p in ["constant", "impulse", "ramp", "random", "hdr-clusters"]]
CASES += [("cutoff", p) for p in ["clusters", "hdr-clusters"]]


def test_bilateral_reference_separable_is_not_full_2d():
    stored = make_image("random", "fp32")
    full, _ = bilateral_reference(stored, "fp32")
    separable, error = bilateral_reference(stored, "fp32", separable=True)
    # Pin a nontrivial fixture: treating separable as mathematically identical
    # to full 2D must not fit within the propagated storage/arithmetic budget.
    assert np.max(np.abs(full-separable)) > 0.1
    assert error.max() < 0.001


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("separable", [False, True], ids=["full", "separable"])
@pytest.mark.parametrize("sigma_spatial", [0, 1.5])
@pytest.mark.parametrize("range_type,pattern", CASES)
def test_bilateral_reference(ctx, precision, separable, sigma_spatial, range_type, pattern):
    check_bilateral(ctx, make_image(pattern, precision), precision, sigma_spatial=sigma_spatial,
                    sigma_range=0.35 if range_type == "gaussian" else 0.25,
                    range_type=range_type, separable=separable)


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("separable", [False, True], ids=["full", "separable"])
@pytest.mark.parametrize("shape", [(1, 1), (1, 9), (7, 1)])
@pytest.mark.parametrize("range_type", ["gaussian", "cutoff"])
def test_bilateral_tiny_borders(ctx, precision, separable, shape, range_type):
    check_bilateral(ctx, make_image("clusters", precision, shape), precision, radius=4,
                    sigma_spatial=1.5, sigma_range=0.25, range_type=range_type, separable=separable)


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("separable", [False, True], ids=["full", "separable"])
def test_bilateral_alpha_independence(ctx, precision, separable):
    stored = make_image("random", precision)
    first = check_bilateral(ctx, stored, precision, sigma_range=0.35, separable=separable)
    stored[..., 3] = 0
    second = check_bilateral(ctx, stored, precision, sigma_range=0.35, separable=separable)
    np.testing.assert_array_equal(first[..., :3], second[..., :3])


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("separable", [False, True], ids=["full", "separable"])
@pytest.mark.parametrize("position", ["below", "at", "above"])
def test_bilateral_cutoff_inclusive_boundary(ctx, precision, separable, position):
    dtype = np.float16 if precision == "fp16" else np.float32
    value = {"below": np.nextafter(dtype(0.5), dtype(0)), "at": dtype(0.5),
             "above": np.nextafter(dtype(0.5), dtype(1))}[position]
    stored = np.zeros((1, 2, 4), dtype=dtype)
    stored[0, 1, 0] = value
    # One nonzero channel makes d^2 <= .5^2 unambiguous for these binary levels.
    # The vertical pass only sees replicated copies, so no threshold ambiguity
    # is introduced by the horizontal storage error in this special case.
    actual = check_bilateral(ctx, stored, precision, radius=1, sigma_range=0.5,
                             range_type="cutoff", separable=separable, require_margin=False)
    if position == "above":
        # The zero pixel cannot receive any of its nonzero neighbor. Repeated
        # averaging of the nonzero pixel can still round by an fp32 ULP.
        np.testing.assert_array_equal(actual[0, 0], 0)


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("separable", [False, True], ids=["full", "separable"])
def test_bilateral_zero_cutoff_preserves_rgb(ctx, precision, separable):
    stored = make_image("clusters", precision)
    image = si.from_numpy(stored, device=ctx[0], precision=precision)
    actual = si.ops.bilateral(image, radius=2, sigma_range=0, range_type="cutoff",
                              separable=separable).to_numpy().astype(np.float64)
    assert np.isfinite(actual).all()
    np.testing.assert_array_equal(actual[..., 3], stored[..., 3])
    rgb = stored[..., :3].astype(np.float64)
    error = (2 if separable else 1)*(ARITHMETIC + storage_error(rgb, precision, ARITHMETIC))
    np.testing.assert_array_less(np.abs(actual[..., :3]-rgb), np.nextafter(error, np.inf))
