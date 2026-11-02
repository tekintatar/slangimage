"""Bounded CLAHE reference: float-count clipping, luma ratio, tile interpolation.

Not an OpenCV/Lab oracle. Stored inputs precede float64 evaluation. Fixtures
avoid uncertain luminance bins; nonfinite/overflow-scale data and large counts
are excluded. Histogram tiles and interpolation use the same nominal footprint.
"""

import numpy as np
import pytest

import slangimage as si
from test_core_numeric import storage_error


WEIGHTS = np.array([0.299, 0.587, 0.114])
COMPUTE_ATOL = 1e-5


def stable_luma_bins(image):
    rgb = image[..., :3]
    luma = np.clip(rgb @ WEIGHTS, 0, 1)
    bins = np.floor(luma*255 + 0.5).astype(int)
    # Enclose fp32 dot-product/coefficient and bin arithmetic error instead of
    # hiding a different discrete bin assignment with an output tolerance.
    uncertainty = 8*np.finfo(np.float32).eps*(np.abs(rgb) @ WEIGHTS) + 1e-7
    low = np.floor(np.clip(luma - uncertainty, 0, 1)*255 + 0.5).astype(int)
    high = np.floor(np.clip(luma + uncertainty, 0, 1)*255 + 0.5).astype(int)
    np.testing.assert_array_equal(low, bins)
    np.testing.assert_array_equal(high, bins)
    return luma, bins


def clipped_cdf(samples, clip_limit):
    """Weighted empirical CDF plus uniform redistributed mass; no prefix loop."""
    values, counts = np.unique(samples, return_counts=True)
    total = samples.size
    assert total > 0
    cap = max(float(np.float32(clip_limit))*total/256, 1)
    weights = np.minimum(counts, cap)
    uniform_mass = total - weights.sum()
    queries = np.arange(256)
    cumulative = (queries[:, None] >= values) @ weights
    return (cumulative + (queries + 1)*uniform_mass/256)/total


def clahe_reference(image, clip_limit=3, tiles=(8, 8), strength=1):
    h, w = image.shape[:2]
    nx, ny = tiles
    tw, th = (w + nx - 1)//nx, (h + ny - 1)//ny
    # Only occupied tiles participate. The nominal centers are spaced by the
    # same ceil-sized footprints used to collect samples, including edge tiles.
    xstarts, ystarts = np.arange(0, w, tw), np.arange(0, h, th)
    xcenters, ycenters = xstarts + tw/2, ystarts + th/2
    luma, bins = stable_luma_bins(image)
    maps = np.array([[clipped_cdf(bins[y:y+th, x:x+tw], clip_limit)
                      for x in xstarts] for y in ystarts])
    mapped = np.empty((h, w))
    for y in range(h):
        for x in range(w):
            # Independent separable interpolation with constant edge extension.
            rows = [np.interp(x + 0.5, xcenters, row[:, bins[y, x]]) for row in maps]
            mapped[y, x] = np.interp(y + 0.5, ycenters, rows)
    amount = np.clip(float(np.float32(strength)), 0, 1)
    target = (1 - amount)*luma + amount*mapped
    ratio = np.divide(target, luma, out=np.ones_like(luma), where=luma > float(np.float32(1e-4)))
    result = image.copy()
    result[..., :3] = np.clip(image[..., :3]*ratio[..., None], 0, 1)
    return result


def test_clahe_reference_clipped_mass_hand_calculation():
    samples = np.array([32]*6 + [192]*2)
    # cap=1, retained mass=2, excess=6; no CDF-min subtraction.
    mapping = clipped_cdf(samples, 3)
    np.testing.assert_array_equal(mapping[[0, 31, 32, 191, 192, 255]],
                                  np.array([6/256, 32*6/256, 1+33*6/256,
                                            1+192*6/256, 2+193*6/256, 8])/8)
    assert np.all(np.diff(mapping) >= 0)


def test_clahe_reference_fractional_clip_hand_calculation():
    samples = np.full(128, 40)
    # cap=1.5, not integer-truncated; uniform excess=126.5.
    mapping = clipped_cdf(samples, 3)
    np.testing.assert_array_equal(mapping[[39, 40, 255]],
                                  [(40*126.5/256)/128, (1.5+41*126.5/256)/128, 1])


def test_clahe_reference_tile_interpolation_hand_calculation():
    image = np.ones((1, 4, 4))
    image[..., :3] = (np.array([17, 61, 173, 231])/255)[None, :, None]
    result = clahe_reference(image, tiles=(2, 1))
    # Centers at 1 and 3; interior pixels mix the two empirical CDFs at 1/4,3/4.
    np.testing.assert_allclose(result[0, :, 0], [0.5, 0.75, 0.625, 1], rtol=0, atol=1e-15)


def make_image(pattern, precision, shape=(16, 24)):
    h, w = shape
    yy, xx = np.mgrid[:h, :w]
    image = np.ones((h, w, 4))
    if pattern == "constant":
        image[..., :3] = 0.42
    elif pattern == "black":
        image[..., :3] = 0
    elif pattern == "near-black":
        image[..., :3] = np.resize([0, 0.00005, 0.0002, 0.0008], h*w).reshape(h, w, 1)
    elif pattern == "color":
        palette = np.array([[0.13, 0.27, 0.62], [0.83, 0.21, 0.09],
                            [0.05, 0.74, 0.32], [0.8, 0.9, 0.7]])
        image[..., :3] = palette[(xx//3 + yy//2) % 4]
    elif pattern == "hdr":
        palette = np.array([[-0.25, -0.25, -0.25], [1.25, 1.25, 1.25],
                            [-0.2, 0.7, 0.3], [1.4, 0.2, 0.1]])
        image[..., :3] = palette[(xx + yy) % 4]
    else:
        # Spatially distinct populations; centers of bins avoid half-bin ties.
        levels = np.array([17, 61, 109, 173, 231])/255
        image[..., :3] = levels[(xx//2 + 2*(yy//3)) % 5][..., None]
    image[..., 3] = np.resize([-0.25, 0, 0.375, 1, 1.25], h*w).reshape(h, w)
    return image.astype(np.float16 if precision == "fp16" else np.float32)


def check_clahe(ctx, stored, precision, **params):
    source = si.from_numpy(stored, precision=precision, device=ctx[0])
    expected = clahe_reference(stored.astype(np.float64), **params)
    output = si.ops.clahe(source, **params)
    assert output.device is source.device and output.precision == precision
    actual = output.to_numpy().astype(np.float64)
    assert actual.shape == stored.shape and np.isfinite(actual).all()
    np.testing.assert_array_equal(actual[..., 3], stored[..., 3])
    assert actual[..., :3].min() >= 0 and actual[..., :3].max() <= 1
    error = COMPUTE_ATOL + storage_error(expected[..., :3], precision, COMPUTE_ATOL)
    np.testing.assert_array_less(np.abs(actual[..., :3] - expected[..., :3]),
                                 np.nextafter(error, np.inf))
    return actual


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("pattern", ["constant", "black", "near-black", "color", "hdr", "spatial"])
def test_clahe_divisible_grid_reference(ctx, precision, pattern):
    check_clahe(ctx, make_image(pattern, precision), precision, tiles=(3, 2))


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("clip_limit", [0.25, 3, 32, 256])
def test_clahe_clip_reference(ctx, precision, clip_limit):
    # Single tile: fractional and unclipped caps, strongly repeated populations.
    check_clahe(ctx, make_image("spatial", precision), precision,
                tiles=(1, 1), clip_limit=clip_limit)


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("strength", [-1, 0, 0.375, 1, 2])
def test_clahe_strength_and_hdr_clipping(ctx, precision, strength):
    check_clahe(ctx, make_image("hdr", precision), precision, tiles=(3, 2), strength=strength)


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("shape,tiles", [((1, 1), (1, 1)), ((1, 24), (3, 1)), ((16, 1), (1, 2))])
def test_clahe_tiny_divisible_grid(ctx, precision, shape, tiles):
    check_clahe(ctx, make_image("spatial", precision, shape), precision, tiles=tiles)


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
def test_clahe_alpha_and_repeated_calls(ctx, precision):
    stored = make_image("spatial", precision)
    first = check_clahe(ctx, stored, precision, tiles=(3, 2))
    check_clahe(ctx, make_image("color", precision), precision, tiles=(1, 1))
    stored[..., 3] = 0
    transparent = check_clahe(ctx, stored, precision, tiles=(3, 2))
    np.testing.assert_array_equal(first[..., :3], transparent[..., :3])


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("shape,tiles", [((13, 17), (3, 2)), ((1, 1), (8, 8)),
                                        ((3, 5), (8, 8)), ((8, 8), (7, 7))])
def test_clahe_histogram_and_interpolation_grid_agree(ctx, precision, shape, tiles):
    check_clahe(ctx, make_image("spatial", precision, shape), precision, tiles=tiles)
