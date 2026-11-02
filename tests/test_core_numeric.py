"""Independent texture references for gamma, Gaussian blur, and unsharp mask.

No production math helpers or CUDA tensor implementations are used as oracles.
Inputs are quantized before float64 reference computation. Gaussian's two
texture stores are accounted for by propagating a per-value error bound through
the normalized convolution; they are not incorrectly treated as fp32 scratch.
"""

import numpy as np
import pytest

import slangimage as si


# Small test images, finite RGB in [-0.25, 1.25], radius <= 7, sigma >= 0.1
# (or size-aware sigma below that): conservative fp32 arithmetic allowance per
# pass. One storage ULP permits Metal's observed toward-zero fp16 stores as
# well as nearest rounding. No image-average or relative-error tolerance.
COMPUTE_ATOL = 5e-6
LUMA_LINEAR_ATOL = 2e-6
LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114])


def storage_error(value, precision, prior=0.0):
    dtype = np.float16 if precision == "fp16" else np.float32
    # Use the largest magnitude in the uncertainty interval, so crossing a
    # binade boundary cannot underestimate the output spacing.
    return np.spacing((np.abs(value) + prior).astype(dtype)).astype(np.float64)


def convolution(image, weights, axis):
    """Normalized 1D convolution with independently specified edge replication."""
    radius = len(weights) // 2
    padding = [(0, 0)] * image.ndim
    padding[axis] = (radius, radius)
    padded = np.pad(image, padding, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, len(weights), axis=axis)
    return np.tensordot(windows, weights, axes=([-1], [0]))


def gaussian_reference(image, radius, sigma, precision):
    positions = np.arange(-radius, radius + 1, dtype=np.float64)
    weights = np.exp(-0.5 * (positions / sigma) ** 2)
    weights /= weights.sum()
    horizontal = convolution(image, weights, axis=1)
    h_error = COMPUTE_ATOL + storage_error(horizontal, precision, COMPUTE_ATOL)
    reference = convolution(horizontal, weights, axis=0)
    propagated = convolution(h_error, weights, axis=0) + COMPUTE_ATOL
    error = propagated + storage_error(reference, precision, propagated)
    return reference, error


def decode_srgb(rgb):
    result = rgb / 12.92
    high = rgb > 0.04045
    result[high] = ((rgb[high] + 0.055) / 1.055) ** 2.4
    return result


def encode_srgb(linear):
    result = linear * 12.92
    high = linear > 0.0031308
    result[high] = 1.055 * linear[high] ** (1.0 / 2.4) - 0.055
    return result


def unsharp_reference(image, radius, sigma, amount, color_space, precision):
    blurred, blur_error = gaussian_reference(image, radius, sigma, precision)
    source = image[..., :3]
    if color_space == "rgb":
        reference = np.clip(source + amount * (source - blurred[..., :3]), 0, 1)
        error = abs(amount) * blur_error[..., :3] + COMPUTE_ATOL
    else:
        # Library's rounded, BT.601-style YCbCr coefficients in LINEAR RGB.
        # This is not gamma-encoded YCbCr and not an exact inverse matrix.
        linear = decode_srgb(source)
        y = linear @ LUMA_WEIGHTS
        cb = (linear[..., 2] - y) * 0.564
        cr = (linear[..., 0] - y) * 0.713
        chroma = np.stack((1.403 * cr, -0.714 * cr - 0.344 * cb, 1.770 * cb), axis=-1)
        blur_y = decode_srgb(blurred[..., :3]) @ LUMA_WEIGHTS
        sharpened_y = np.clip(y + amount * (y - blur_y), 0, 1)
        reference = np.clip(encode_srgb(sharpened_y[..., None] + chroma), 0, 1)

        # Positive luma weights and monotone transfer functions let us carry
        # blur-storage uncertainty through the nonlinear path. In particular,
        # don't ignore its amplification near black or use a global loose bound.
        low_y = decode_srgb(blurred[..., :3] - blur_error[..., :3]) @ LUMA_WEIGHTS
        high_y = decode_srgb(blurred[..., :3] + blur_error[..., :3]) @ LUMA_WEIGHTS
        ends = np.stack((y + amount * (y - low_y), y + amount * (y - high_y)))
        arithmetic = (1 + 2 * abs(amount)) * LUMA_LINEAR_ATOL
        low = np.clip(ends.min(axis=0) - arithmetic, 0, 1)
        high = np.clip(ends.max(axis=0) + arithmetic, 0, 1)
        lower_rgb = np.clip(encode_srgb(low[..., None] + chroma - LUMA_LINEAR_ATOL), 0, 1)
        upper_rgb = np.clip(encode_srgb(high[..., None] + chroma + LUMA_LINEAR_ATOL), 0, 1)
        error = np.maximum(reference - lower_rgb, upper_rgb - reference) + COMPUTE_ATOL
    error += storage_error(reference, precision, error)
    return reference, error


def check_output(output, source, reference, error, *, preserve_alpha=False):
    assert isinstance(output, si.Image)
    assert output.device is source.device
    assert output.precision == source.precision
    actual = output.to_numpy().astype(np.float64)
    assert actual.shape == source.to_numpy().shape
    assert np.isfinite(actual).all()
    if preserve_alpha:
        np.testing.assert_array_equal(actual[..., 3], source.to_numpy()[..., 3])
        actual = actual[..., :3]
    assert actual.shape == reference.shape
    np.testing.assert_array_less(np.abs(actual - reference), np.nextafter(error, np.inf))
    return actual


def test_gaussian_reference_has_closed_form_corner_weights():
    # Choose exp(-1/(2*sigma**2)) = 1/2, giving 1D weights [1/4, 1/2, 1/4].
    # Replicating a corner impulse then gives outer([3/4, 1/4], [3/4, 1/4]).
    image = np.zeros((2, 2, 4))
    image[0, 0] = (1, 0.5, 0.25, 1)
    reference, _ = gaussian_reference(image, 1, 1 / np.sqrt(2 * np.log(2)), "fp32")
    expected = np.outer([0.75, 0.25], [0.75, 0.25])[..., None] * image[0, 0]
    np.testing.assert_allclose(reference, expected, rtol=0, atol=1e-15)


@pytest.fixture(scope="module", params=["fp16", "fp32"])
def precision(request):
    return request.param


@pytest.fixture(params=["constant", "ramp", "step", "impulse", "corner", "hdr", "1x1", "1xN", "Nx1"])
def sample(ctx, precision, request):
    pattern = request.param
    h, w = {"1x1": (1, 1), "1xN": (1, 9), "Nx1": (7, 1)}.get(pattern, (13, 17))
    rng = np.random.default_rng(71)
    image = rng.random((h, w, 4)).astype(np.float32)
    if pattern == "constant":
        image[...] = (-0.25, 0.375, 1.25, 0.625)
    elif pattern == "ramp":
        x = np.linspace(0, 1, w)[None, :]
        y = np.linspace(0, 1, h)[:, None]
        image[..., 0], image[..., 1], image[..., 2], image[..., 3] = x, y, (x + y) / 2, y
    elif pattern == "step":
        image[:, :w // 2] = (0.1, 0.8, 0.2, 0.0)
        image[:, w // 2:] = (0.9, 0.2, 0.7, 0.75)
    elif pattern in {"impulse", "corner"}:
        image[...] = (0, 0, 0, 0.25)
        row, col = (h // 2, w // 2) if pattern == "impulse" else (0, 0)
        image[row, col] = (1, 0.5, 0.25, 1)
    elif pattern == "hdr":
        image[..., :3] = image[..., :3] * 1.5 - 0.25
    dtype = np.float16 if precision == "fp16" else np.float32
    stored = image.astype(dtype)
    source = si.from_numpy(stored, precision=precision, device=ctx[0])
    return stored.astype(np.float64), source


@pytest.mark.parametrize("power", [-1.0, 0.0, 0.01, 0.5, 1.0, 2.2, 10.0, 20.0])
def test_gamma_matches_numpy(sample, precision, power):
    image, source = sample
    # Model the scalar argument's fp32 representation separately from pixels.
    exponent = float(np.clip(np.float32(power), np.float32(0.01), np.float32(10.0)))
    reference = np.clip(image[..., :3], 0, 1) ** exponent
    error = COMPUTE_ATOL + storage_error(reference, precision, COMPUTE_ATOL)
    actual = check_output(si.ops.gamma(source, power), source, reference, error, preserve_alpha=True)
    assert actual.min() >= 0 and actual.max() <= 1


def test_gamma_endpoints_and_direction(ctx, precision):
    dtype = np.float16 if precision == "fp16" else np.float32
    image = np.repeat(np.linspace(0, 1, 257, dtype=dtype)[None, :, None], 4, axis=2)
    source = si.from_numpy(image, precision=precision, device=ctx[0])
    bright = si.ops.gamma(source, 0.5).to_numpy()
    dark = si.ops.gamma(source, 2.0).to_numpy()
    np.testing.assert_array_equal(bright[0, [0, -1]], image[0, [0, -1]])
    np.testing.assert_array_equal(dark[0, [0, -1]], image[0, [0, -1]])
    assert np.all(bright[..., :3] >= image[..., :3])
    assert np.all(dark[..., :3] <= image[..., :3])


@pytest.mark.parametrize("radius, sigma", [(1, 0.1), (2, 0.8), (3, 1.5), (7, 4.0)])
def test_gaussian_matches_numpy(sample, precision, radius, sigma):
    image, source = sample
    sigma = float(np.float32(sigma))
    reference, error = gaussian_reference(image, radius, sigma, precision)
    check_output(si.ops.gaussian_blur(source, radius=radius, sigma=sigma), source, reference, error)


@pytest.mark.parametrize("amount", [0.0, 40.0])
def test_gaussian_amount_matches_numpy(sample, precision, amount):
    image, source = sample
    # Independently specify the documented size-aware amount conversion.
    sigma = max(0.1, amount) * np.sqrt(image.shape[0] * image.shape[1] / 1e6)
    radius = max(1, int(np.ceil(3 * sigma)))
    reference, error = gaussian_reference(image, radius, float(np.float32(sigma)), precision)
    check_output(si.ops.gaussian_blur(source, amount=amount), source, reference, error)


@pytest.mark.parametrize("color_space", ["rgb", "luma"])
@pytest.mark.parametrize("amount", [-0.5, 0.0, 0.75, 2.0])
def test_unsharp_matches_numpy(sample, precision, color_space, amount):
    image, source = sample
    radius, sigma = 3, 1.25  # both differ from the public defaults
    reference, error = unsharp_reference(image, radius, sigma, amount, color_space, precision)
    output = si.ops.unsharp_mask(source, radius=radius, sigma=sigma, amount=amount, color_space=color_space)
    actual = check_output(output, source, reference, error, preserve_alpha=True)
    assert actual.min() >= 0 and actual.max() <= 1
