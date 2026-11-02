"""Bounded NLM reference via flattened patches on an extended image lattice.

Stored inputs precede float64 evaluation. No GPU or production helper serves
as the oracle; virtual candidate patch centers are not clamped prematurely.
"""

from itertools import product

import numpy as np
import pytest

import slangimage as si
from test_core_numeric import storage_error


ARITHMETIC = 1e-5


def nl_means_reference(stored, search_radius=2, patch_radius=1, h=0.2, sigma=0):
    rgb = stored[..., :3].astype(np.float64)
    height, width = rgb.shape[:2]
    search, patch = search_radius, patch_radius
    h, sigma = float(np.float32(h)), float(np.float32(sigma))
    if search == 0:
        return rgb.copy()
    extended = np.pad(rgb, ((search+patch, search+patch), (search+patch, search+patch), (0, 0)), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(extended, (2*patch+1, 2*patch+1), axis=(0, 1))
    features = windows.reshape(height+2*search, width+2*search, -1)
    center = features[search:search+height, search:search+width]
    offsets = [(dy, dx) for dy, dx in product(range(-search, search+1), repeat=2) if dy or dx]
    neighbors = np.stack([features[search+dy:search+dy+height, search+dx:search+dx+width]
                          for dy, dx in offsets])
    distances = np.mean((neighbors-center)**2, axis=-1)
    h2 = max(h*h, float(np.float32(1e-8)))
    weights = np.exp(-np.maximum(distances-2*sigma*sigma, 0)/h2)
    center_weight = np.maximum(weights.max(axis=0), float(np.float32(1e-6)))
    padded_rgb = np.pad(rgb, ((search, search), (search, search), (0, 0)), mode="edge")
    candidates = np.stack([padded_rgb[search+dy:search+dy+height, search+dx:search+dx+width]
                           for dy, dx in offsets])
    values = np.concatenate((candidates, rgb[None]), axis=0)
    weights = np.concatenate((weights, center_weight[None]), axis=0)
    normalized = weights/weights.sum(axis=0)
    return np.sum(values*normalized[..., None], axis=0)


def test_nlm_reference_uniform_weights_replicated_corner():
    stored = np.ones((2, 2, 4))
    stored[..., :3] = np.array([[0, 1], [2, 3]])[..., None]
    actual = nl_means_reference(stored, search_radius=1, patch_radius=1, sigma=4)
    np.testing.assert_allclose(actual[0, 0], [1, 1, 1], rtol=0, atol=1e-15)


def test_nlm_reference_center_weight_floor_hand_calculation():
    stored = np.full((3, 3, 4), 0.4)
    stored[1, 1, :3] = 0
    actual = nl_means_reference(stored, search_radius=1, patch_radius=0, h=0.1)
    weight = np.exp(-0.4**2/float(np.float32(0.1))**2)
    expected = 0.4*8*weight/(8*weight+float(np.float32(1e-6)))
    assert weight < 1e-6
    np.testing.assert_allclose(actual[1, 1], expected, rtol=0, atol=1e-15)


def test_nlm_reference_virtual_patch_center_hand_calculation():
    stored = np.ones((1, 3, 4))
    stored[..., :3] = np.array([0, 0.5, 1])[None, :, None]
    actual = nl_means_reference(stored, search_radius=1, patch_radius=1, h=0.5)
    # At x=0: patch [0,0,.5]; virtual x=-1 has [0,0,0], NOT [0,0,.5].
    left, right = np.exp(-1/3), np.exp(-2/3)
    expected = 0.5*right/(1+left+right)
    np.testing.assert_allclose(actual[0, 0], expected, rtol=0, atol=1e-15)


def make_image(pattern, precision, shape=(5, 7)):
    height, width = shape
    yy, xx = np.mgrid[:height, :width]
    stored = np.ones((height, width, 4))
    if pattern == "constant":
        stored[..., :3] = (-0.25, 0.375, 1.25)
    elif pattern == "impulse":
        stored[..., :3] = 0.125
        stored[height//2, width//2, :3] = (1, 0.5, 0.25)
    elif pattern == "edge":
        stored[..., :3] = np.where(xx < width//2, 0.125, 0.75)[..., None]
        stored[..., 1] += yy/64
    elif pattern == "texture":
        stored[..., 0] = 0.2 + ((xx+yy) % 3)/16
        stored[..., 1] = 0.4 + ((2*xx+yy) % 3)/16
        stored[..., 2] = 0.6 + ((xx+2*yy) % 3)/16
    elif pattern == "microcontrast":
        stored[..., :3] = 0.125 + np.resize(np.arange(7)-3, (height, width, 3))/16384
    else:
        stored[..., :3] = np.random.default_rng(661).uniform(-0.25, 1.25, (height, width, 3))
    stored[..., 3] = np.resize([-0.25, 0, 0.375, 1, 1.25], height*width).reshape(height, width)
    return stored.astype(np.float16 if precision == "fp16" else np.float32)


def check_nlm(ctx, stored, precision, **params):
    params = {"search_radius": 2, "patch_radius": 1, "h": 0.2, "sigma": 0, **params}
    expected = nl_means_reference(stored, **params)
    source = si.from_numpy(stored, device=ctx[0], precision=precision)
    output = si.ops.nl_means(source, **params)
    assert output.device is source.device and output.precision == precision
    actual = output.to_numpy().astype(np.float64)
    assert actual.shape == stored.shape and np.isfinite(actual).all()
    np.testing.assert_array_equal(actual[..., 3], stored[..., 3])
    error = ARITHMETIC + storage_error(expected, precision, ARITHMETIC)
    np.testing.assert_array_less(np.abs(actual[..., :3]-expected), np.nextafter(error, np.inf))
    # Positive normalized weights cannot invent new channel extrema, apart
    # from the arithmetic/output-store allowance. No clipping is imposed.
    low, high = stored[..., :3].min(axis=(0, 1)), stored[..., :3].max(axis=(0, 1))
    assert np.all(actual[..., :3] >= low-error)
    assert np.all(actual[..., :3] <= high+error)
    return actual


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("patch_radius", [0, 1, 2])
@pytest.mark.parametrize("pattern", ["constant", "impulse", "edge", "texture", "random-hdr"])
def test_nlm_independent_reference(ctx, precision, patch_radius, pattern):
    check_nlm(ctx, make_image(pattern, precision), precision, patch_radius=patch_radius)


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
def test_nlm_public_defaults(ctx, precision):
    stored = make_image("texture", precision)
    expected = nl_means_reference(stored, search_radius=5, patch_radius=1, h=0.1)
    source = si.from_numpy(stored, device=ctx[0], precision=precision)
    result = si.ops.nl_means(source)
    assert result.device is source.device and result.precision == precision
    actual = result.to_numpy().astype(np.float64)
    np.testing.assert_array_equal(actual[..., 3], stored[..., 3])
    error = ARITHMETIC + storage_error(expected, precision, ARITHMETIC)
    np.testing.assert_array_less(np.abs(actual[..., :3]-expected), np.nextafter(error, np.inf))


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("sigma", [0, 0.05, 0.2, 2])
def test_nlm_noise_compensation(ctx, precision, sigma):
    check_nlm(ctx, make_image("texture", precision), precision, sigma=sigma)


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (5, 1)])
def test_nlm_tiny_virtual_patch_borders(ctx, precision, shape):
    check_nlm(ctx, make_image("random-hdr", precision, shape), precision, patch_radius=2)


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("patch_radius", [0, 2])
def test_nlm_zero_search(ctx, precision, patch_radius):
    check_nlm(ctx, make_image("random-hdr", precision), precision, search_radius=0, patch_radius=patch_radius)


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("h", [0, 0.00005, 0.0001, 0.0002])
def test_nlm_strength_floor(ctx, precision, h):
    check_nlm(ctx, make_image("microcontrast", precision), precision, h=h, patch_radius=0)


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("neighbor", [0.2, 0.36, 0.38, 0.5])
def test_nlm_max_neighbor_and_center_floor(ctx, precision, neighbor):
    stored = make_image("constant", precision, (3, 3))
    stored[..., :3] = neighbor
    stored[1, 1, :3] = 0
    check_nlm(ctx, stored, precision, search_radius=1, patch_radius=0, h=0.1)


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
def test_nlm_alpha_independence_and_repeated_calls(ctx, precision):
    stored = make_image("texture", precision)
    first = check_nlm(ctx, stored, precision)
    check_nlm(ctx, make_image("random-hdr", precision), precision, patch_radius=2, sigma=0.1)
    stored[..., 3] = 0
    transparent = check_nlm(ctx, stored, precision)
    np.testing.assert_array_equal(first[..., :3], transparent[..., :3])
