"""Independent float64 texture resize/rotation oracles, not tensor-path tests.

Dense weight matrices specify resize mathematically, independently of the
shader's tap loops. Rotation maps canvas centers directly using trigonometry,
without production matrix construction/inversion or the affine test oracle.
"""

import numpy as np
import pytest

import slangimage as si
from slangimage.kernels.geometry import rotation_matrix
import test_core_numeric as core

precision = core.precision
sample = core.sample

# Bounded inputs [-.25, 1.25], dimensions <= 95, and the ratios below:
# allowance per pass for fp32 coordinates, weights, and accumulation. Storage
# is budgeted separately, with absolute weights for Lanczos' negative lobes.
# No relative tolerance, image-average metric, or clipping of the reference.
COMPUTE_ATOL = 3e-5


def resize_weights(source_size, target_size, mode):
    centers = (np.arange(target_size) + 0.5) * source_size / target_size - 0.5
    if mode == "lanczos3":
        distance = (np.arange(source_size)[None, :] - centers[:, None]) / max(
            source_size / target_size, 1,
        )
        weights = np.sinc(distance) * np.sinc(distance / 3)
        weights[np.abs(distance) >= 3] = 0
    else:
        # Scatter the four cardinal-cubic basis weights onto clamped indices.
        # Unlike Lanczos, repeated edge taps retain their individual weights.
        indices = np.floor(centers).astype(int)[:, None] + np.arange(-1, 3)
        x = np.abs(indices - centers[:, None])
        values = np.where(x <= 1, 1 + x*x*(1.5*x - 2.5), -0.5*(x-1)*(x-2)**2)
        values[x >= 2] = 0
        weights = np.zeros((target_size, source_size))
        np.add.at(weights, (np.arange(target_size)[:, None],
                            np.clip(indices, 0, source_size - 1)), values)
    return weights / weights.sum(axis=1, keepdims=True)


def resize_reference(image, width, height, mode, precision):
    wx = resize_weights(image.shape[1], width, mode)
    wy = resize_weights(image.shape[0], height, mode)
    horizontal = np.einsum("xi,yic->yxc", wx, image)
    reference = np.einsum("yj,jxc->yxc", wy, horizontal)
    if mode == "lanczos3":
        h_error = COMPUTE_ATOL + core.storage_error(horizontal, precision, COMPUTE_ATOL)
        error = np.einsum("yj,jxc->yxc", np.abs(wy), h_error) + COMPUTE_ATOL
    else:
        # Bicubic has one 2D dispatch, NOT an intermediate texture store.
        error = np.full(reference.shape, COMPUTE_ATOL)
    return reference, error + core.storage_error(reference, precision, error)


def check_output(output, source, reference, error=None):
    assert isinstance(output, si.Image)
    assert output.device is source.device
    assert output.precision == source.precision
    actual = output.to_numpy().astype(np.float64)
    assert actual.shape == reference.shape
    assert np.isfinite(actual).all()
    if error is None:
        np.testing.assert_array_equal(actual, reference)
    else:
        np.testing.assert_array_less(np.abs(actual - reference), np.nextafter(error, np.inf))
    return actual


def test_resize_oracle_closed_form_weights():
    for mode in ("lanczos3", "bicubic"):
        np.testing.assert_allclose(resize_weights(7, 7, mode), np.eye(7), atol=1e-15, rtol=0)
        np.testing.assert_allclose(resize_weights(1, 9, mode), 1, atol=1e-15, rtol=0)
    # First output of a 2:1 reduction samples source position .5. The -1 tap
    # repeats source zero: (-1 + 9)/16, 9/16, -1/16.
    np.testing.assert_array_equal(resize_weights(4, 2, "bicubic")[0], [8/16, 9/16, -1/16, 0])
    np.testing.assert_array_equal(
        resize_weights(8, 4, "bicubic")[1, 1:5], np.array([-1, 9, 9, -1])/16,
    )
    # A truncated Lanczos edge is not an edge-replicated footprint.
    assert resize_weights(4, 8, "lanczos3")[0, 0] > 1


@pytest.mark.parametrize("mode", ["lanczos3", "bicubic"])
@pytest.mark.parametrize("size", [(7, 5), (23, 19), (5, 21), (1, 1), None],
                         ids=["down", "up", "mixed", "single-output", "identity"])
def test_resize_matches_reference(sample, precision, mode, size):
    image, source = sample
    width, height = size or (source.width, source.height)
    reference, error = resize_reference(image, width, height, mode, precision)
    check_output(si.ops.resize(source, width=width, height=height, mode=mode),
                 source, reference, error)


@pytest.mark.parametrize("mode", ["lanczos3", "bicubic"])
def test_resize_large_reduction_and_ringing(ctx, precision, mode):
    dtype = np.float16 if precision == "fp16" else np.float32
    y, x = np.mgrid[:63, :95]
    image = np.stack(((x+y) % 2, x % 2, y % 2, (x+y) % 2), axis=-1).astype(dtype)
    source = si.from_numpy(image, precision=precision, device=ctx[0])
    reference, error = resize_reference(image.astype(np.float64), 11, 7, mode, precision)
    actual = check_output(si.ops.resize(source, 11, 7, mode=mode), source, reference, error)
    if mode == "lanczos3":
        # Antialiasing is a Lanczos contract, not a promise for fixed-footprint cubic.
        assert np.max(np.abs(actual - 0.5)) < 0.02

    image = np.zeros((3, 6, 4), dtype=dtype)
    image[:, 3:] = 1  # Includes alpha: filters neither premultiply nor clamp it.
    source = si.from_numpy(image, precision=precision, device=ctx[0])
    reference, error = resize_reference(image.astype(np.float64), 19, 7, mode, precision)
    actual = check_output(si.ops.resize(source, 19, 7, mode=mode), source, reference, error)
    assert actual[..., 3].min() < -0.01 and actual[..., 3].max() > 1.01


@pytest.mark.parametrize("kwargs,size", [
    ({"scale": 0.5}, (8, 6)),  # 17*.5 and 13*.5 use ties-to-even rounding.
    ({"width": 7}, (7, 5)),
    ({"height": 7}, (9, 7)),
    ({"scale": 0.001}, (1, 1)),
])
@pytest.mark.parametrize("mode", ["lanczos3", "bicubic"])
def test_resize_dimension_forms(ctx, precision, kwargs, size, mode):
    dtype = np.float16 if precision == "fp16" else np.float32
    image = np.random.default_rng(317).random((13, 17, 4)).astype(dtype)
    source = si.from_numpy(image, precision=precision, device=ctx[0])
    reference, error = resize_reference(image.astype(np.float64), *size, mode, precision)
    check_output(si.ops.resize(source, mode=mode, **kwargs), source, reference, error)


def rotation_coordinates(width, height, angle, expand):
    radians = angle * np.pi / 180
    c, s = np.cos(radians), np.sin(radians)
    span_x, span_y = (abs(c)*width + abs(s)*height, abs(s)*width + abs(c)*height)
    out_w, out_h = (int(np.ceil(span_x)), int(np.ceil(span_y))) if expand else (width, height)
    y, x = np.mgrid[:out_h, :out_w]
    # Expanded canvas is anchored to the exact bounding-box minimum, NOT
    # centered in its ceil-rounded integer dimensions.
    dx = x + 0.5 - (span_x if expand else width)/2
    dy = y + 0.5 - (span_y if expand else height)/2
    return c*dx - s*dy + width/2, s*dx + c*dy + height/2


def rotate_reference(image, angle, expand, mode):
    h, w, channels = image.shape
    sx, sy = rotation_coordinates(w, h, angle, expand)
    result = np.zeros((*sx.shape, channels))
    if mode == "nearest":
        # These arbitrary-angle cases intentionally avoid ties. Assert that
        # fact: do not hide unstable nearest pixels behind a large tolerance.
        assert np.min(np.abs(sx - np.rint(sx))) > 2e-5
        assert np.min(np.abs(sy - np.rint(sy))) > 2e-5
        xi, yi = np.floor(sx).astype(int), np.floor(sy).astype(int)
        valid = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        result[valid] = image[yi[valid], xi[valid]]
    else:
        # Evaluate tent functions against ALL source pixel centers. This is
        # independent of the shader's floor/fraction/four-tap implementation.
        for row in range(h):
            wy = np.maximum(1 - np.abs(sy - (row + 0.5)), 0)
            for col in range(w):
                wx = np.maximum(1 - np.abs(sx - (col + 0.5)), 0)
                result += (wx*wy)[..., None] * image[row, col]
    return result


def test_rotation_oracle_identity_and_counterclockwise():
    image = np.arange(3*3*4).reshape(3, 3, 4)
    np.testing.assert_array_equal(rotate_reference(image, 0, False, "bilinear"), image)
    np.testing.assert_allclose(rotate_reference(image, 90, False, "bilinear"),
                               np.rot90(image), atol=1e-13, rtol=0)


@pytest.mark.parametrize("angle", [17.0, -33.0, 123.0])
@pytest.mark.parametrize("expand", [False, True])
@pytest.mark.parametrize("mode", ["bilinear", "nearest"])
def test_rotate_matches_reference(sample, precision, angle, expand, mode):
    image, source = sample
    reference = rotate_reference(image, angle, expand, mode)
    error = None if mode == "nearest" else (
        COMPUTE_ATOL + core.storage_error(reference, precision, COMPUTE_ATOL)
    )
    check_output(si.ops.rotate(source, angle, mode=mode, expand=expand), source, reference, error)


@pytest.mark.parametrize("mode", ["bilinear", "nearest"])
@pytest.mark.parametrize("expand", [False, True])
def test_rotate_zero_is_exact(sample, precision, mode, expand):
    image, source = sample
    check_output(si.ops.rotate(source, 0, mode=mode, expand=expand), source, image)


QUARTER_TURN_ANGLES = [0, 90, 180, 270, 360, -90, -180, -270, -360, 450, 720, 360_000_000_090]


@pytest.mark.parametrize("angle", QUARTER_TURN_ANGLES)
@pytest.mark.parametrize("mode", ["bilinear", "nearest"])
@pytest.mark.parametrize("shape", [(13, 17), (30, 50), (1, 1), (1, 9), (7, 1)])
def test_rotate_expanded_quarter_turn_matches_rot90(ctx, precision, angle, mode, shape):
    dtype = np.float16 if precision == "fp16" else np.float32
    image = np.random.default_rng(318).uniform(-0.25, 1.25, (*shape, 4)).astype(dtype)
    source = si.from_numpy(image, precision=precision, device=ctx[0])
    check_output(si.ops.rotate(source, angle, mode=mode, expand=True),
                 source, np.rot90(image, angle // 90))


@pytest.mark.parametrize("angle", QUARTER_TURN_ANGLES)
@pytest.mark.parametrize("mode", ["bilinear", "nearest"])
def test_rotate_square_quarter_turn_without_expand(ctx, precision, angle, mode):
    dtype = np.float16 if precision == "fp16" else np.float32
    image = np.random.default_rng(319).uniform(-0.25, 1.25, (13, 13, 4)).astype(dtype)
    source = si.from_numpy(image, precision=precision, device=ctx[0])
    check_output(si.ops.rotate(source, angle, mode=mode, expand=False),
                 source, np.rot90(image, angle // 90))


@pytest.mark.parametrize("angle", [
    17.0, -33.0, 123.0, 450.25, 90 - 1e-10, 90 + 1e-10,
    np.nextafter(90.0, -np.inf), np.nextafter(90.0, np.inf),
    np.nextafter(-90.0, -np.inf), np.nextafter(-90.0, np.inf),
])
def test_rotation_matrix_keeps_non_quarter_angles_unchanged(angle):
    # Compatibility check against the previous formula, NOT the image oracle.
    # Even the immediate float64 neighbors of a quarter turn must not snap.
    cx, cy = 8.5, 6.5
    c, s = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
    expected = np.array([[c, s, cx - c*cx - s*cy], [-s, c, cy + s*cx - c*cy]])
    np.testing.assert_array_equal(rotation_matrix(angle, cx, cy), expected)
