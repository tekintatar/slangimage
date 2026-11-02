"""Global equalization only: independent rank-based CDF, not CLAHE.

Input quantization precedes reference evaluation. Fixtures pin bin assignments
that agree under ideal, fused-fp32 and unfused-fp32 arithmetic; ambiguous other
half-bin boundaries, nonfinite pixels and very large histograms are excluded.
"""

import numpy as np
import pytest

import slangimage as si
from test_core_numeric import storage_error


def equalize_reference(image):
    rgb = np.clip(image[..., :3], 0, 1)
    bins = np.floor(rgb*255 + 0.5).astype(int)
    # Binning is discontinuous: don't mask a different bin with RGB tolerance.
    # Check that the chosen stored inputs have unambiguous bin assignments for
    # both fp32 multiply/add lowering choices, without using either as oracle.
    fused = (rgb*255 + 0.5).astype(np.float32).astype(int)
    separate = (rgb.astype(np.float32)*np.float32(255) + np.float32(0.5)).astype(int)
    np.testing.assert_array_equal(bins, fused)
    np.testing.assert_array_equal(bins, separate)
    result = image.copy()
    for channel in range(3):
        values = bins[..., channel]
        ordered = np.sort(values.ravel())
        first_count = np.count_nonzero(ordered == ordered[0])
        if first_count == len(ordered):
            result[..., channel] = values / 255
        else:
            # Number of samples <= this bin, computed as an order statistic
            # rather than the shader's 256-bin histogram/prefix-sum pipeline.
            ranks = np.searchsorted(ordered, values, side="right")
            result[..., channel] = (ranks - first_count) / (len(ordered) - first_count)
    return result


def test_equalize_reference_hand_calculated_sparse_cdf():
    levels = np.array([3, 3, 91, 91, 91, 240])/255
    image = np.ones((1, 6, 4))
    image[..., :3] = levels[None, :, None]
    result = equalize_reference(image)
    np.testing.assert_array_equal(result[0, :, 0], [0, 0, 0.75, 0.75, 0.75, 1])
    np.testing.assert_array_equal(result[..., 3], 1)


def test_equalize_reference_single_bin_is_quantized_not_identity():
    image = np.array([[[0.4995, 0.42, 0.75, -0.25], [0.4997, 0.421, 0.75, 1.25]]])
    result = equalize_reference(image)
    np.testing.assert_array_equal(result[..., :3],
                                  np.tile([127/255, 107/255, 191/255], (1, 2, 1)))
    np.testing.assert_array_equal(result[..., 3], image[..., 3])


def make_image(pattern, precision):
    dtype = np.float16 if precision == "fp16" else np.float32
    shape = {"pixel": (1, 1), "row": (1, 17), "column": (19, 1),
             "midpoint": (1, 5)}.get(pattern, (13, 17))
    h, w = shape
    y, x = np.mgrid[:h, :w]
    image = np.random.default_rng(611).random((h, w, 4))
    if pattern in ("constant", "pixel"):
        image[..., :3] = (0.42, 0.5, 0.75)
    elif pattern == "single-bin":
        image[..., :3] = (107 + (image[..., :3] - 0.5)*0.4)/255
    elif pattern in ("ramp", "row", "column"):
        ramp = np.arange(h*w).reshape(h, w)/(h*w - 1)
        # Bin centers, not arbitrary near-half-bin samples.
        ramp = np.floor(ramp*255)/255
        image[..., 0], image[..., 1], image[..., 2] = ramp, 1-ramp, 0.42
    elif pattern in ("sparse", "channel-independent", "hdr"):
        levels = np.array([0, 17, 93, 211, 255])/255
        image[..., 0] = levels[(x + y) % 5]
        image[..., 1] = levels[(2*x + y) % 5]
        image[..., 2] = 0.42 if pattern == "channel-independent" else levels[(x + 3*y) % 5]
        if pattern == "hdr":
            image[::2, ::3, :3] = (-2, 2, 0.42)
    elif pattern == "midpoint":
        levels = [0, np.nextafter(dtype(0.5), dtype(0)), 0.5,
                  np.nextafter(dtype(0.5), dtype(1)), 1]
        image[..., :3] = np.array(levels)[None, :, None]
    image[..., 3] = np.resize([-0.25, 0, 0.375, 1, 1.25], h*w).reshape(h, w)
    return image.astype(dtype)


def check_equalize(ctx, stored, precision):
    source = si.from_numpy(stored, precision=precision, device=ctx[0])
    reference = equalize_reference(stored.astype(np.float64))
    output = si.ops.equalize(source)
    assert output.device is source.device and output.precision == precision
    actual = output.to_numpy().astype(np.float64)
    assert actual.shape == stored.shape and np.isfinite(actual).all()
    np.testing.assert_array_equal(actual[..., 3], stored[..., 3])
    assert actual[..., :3].min() >= 0 and actual[..., :3].max() <= 1
    # Small, integer-exact counts; fp32 mapping division + one texture store.
    # No relative tolerance, input-quantization error or bin-error allowance.
    rgb = reference[..., :3]
    error = 5e-7 + storage_error(rgb, precision, 5e-7)
    np.testing.assert_array_less(np.abs(actual[..., :3] - rgb), np.nextafter(error, np.inf))
    return actual


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("pattern", ["constant", "single-bin", "ramp", "sparse",
                                     "channel-independent", "hdr", "pixel", "row",
                                     "column", "midpoint"])
def test_equalize_matches_rank_reference(ctx, precision, pattern):
    check_equalize(ctx, make_image(pattern, precision), precision)


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
def test_equalize_alpha_does_not_weight_histogram_and_calls_are_independent(ctx, precision):
    stored = make_image("sparse", precision)
    first = check_equalize(ctx, stored, precision)
    check_equalize(ctx, make_image("constant", precision), precision)
    # All pixels still count when fully transparent; alpha itself remains zero.
    stored[..., 3] = 0
    transparent = check_equalize(ctx, stored, precision)
    np.testing.assert_array_equal(transparent[..., :3], first[..., :3])
    np.testing.assert_array_equal(transparent[..., 3], 0)
