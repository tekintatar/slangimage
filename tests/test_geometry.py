# tests/test_geometry.py
# Geometric transforms: flip / rot90 / crop against exact numpy index
# semantics; wrapped tiling and affine references at fp16/fp32. Independent
# arbitrary-angle rotate references are in test_resampling_numeric.py.

from __future__ import annotations

import numpy as np
import pytest

from slangimage.runtime.textures import np_to_texture, texture_to_np
from slangimage.kernels.geometry import GeometryOps, rotation_matrix

ATOL = 2e-3
# For these bounded images (RGB in [-0.25, 1.25], alpha in [0, 1]):
# fp32 covers coordinate/coefficient rounding in the small canvases below.
# fp16 additionally allows one output ULP: Metal texture stores were observed
# rounding toward zero, not nearest (1.070068359375 -> 1.0693359375, whereas the
# fp32 path on identical quantized input returns 1.070068359375 exactly).
# References use the ALREADY QUANTIZED input, promoted to float64, so source
# storage error is not confused with sampling/compute/output error. No rtol.
FP32_INTERPOLATION_ATOL = 3e-5


@pytest.fixture(scope="module", params=["fp16", "fp32"])
def precision(request):
    return request.param


def upload(device, img, precision):
    stored = img.astype(np.float16 if precision == "fp16" else np.float32)
    tex = np_to_texture(device, stored, usage="read", precision=precision)
    return stored.astype(np.float64), tex


def assert_sampled_output(output, source, reference, precision, *, exact=False):
    assert output.format == source.format, "geometry must preserve storage precision"
    assert output.device is source.device, "geometry must preserve the device"
    actual = texture_to_np(output)
    assert_geometry_values(actual, reference, precision, exact=exact)


def assert_geometry_values(actual, reference, precision, *, exact=False):
    assert actual.shape == reference.shape
    assert np.isfinite(actual).all()
    if exact:
        np.testing.assert_array_equal(actual, reference)
    else:
        budget = np.full(reference.shape, FP32_INTERPOLATION_ATOL)
        if precision == "fp16":
            budget += np.spacing(np.abs(reference).astype(np.float16)).astype(np.float64)
        # Per-value bound, not an image-wide PSNR or a relative tolerance that
        # can hide a large absolute error. nextafter makes the bound inclusive.
        np.testing.assert_array_less(np.abs(actual - reference), np.nextafter(budget, np.inf))


@pytest.fixture(scope="module")
def img_and_tex(ctx, precision):
    device, _ = ctx
    rng = np.random.default_rng(61)
    img = rng.random((40, 56, 4)).astype(np.float32)  # non-square on purpose
    # Keep varying alpha, including fully transparent pixels, in every oracle.
    img[::3, ::4, 3] = 0.0
    return upload(device, img, precision)


@pytest.fixture(scope="module")
def geo(ctx):
    device, modules = ctx
    return GeometryOps(device, modules)


# ---------------------------------------------------------- exact remaps
def test_flip_matches_numpy(geo, img_and_tex, precision):
    img, t = img_and_tex
    for horizontal, vertical in ((False, False), (True, False), (False, True), (True, True)):
        ref = img[::(-1 if vertical else 1), ::(-1 if horizontal else 1)]
        assert_sampled_output(
            geo.flip(t, horizontal=horizontal, vertical=vertical), t, ref, precision, exact=True,
        )


def test_rot90_matches_numpy(geo, img_and_tex, precision):
    img, t = img_and_tex
    for k in (-1, 0, 1, 2, 3, 4, 5):
        out = geo.rot90(t, k=k)
        ref = np.rot90(img, k=k)
        assert_sampled_output(out, t, ref, precision, exact=True)


def test_crop_matches_slice(geo, img_and_tex, precision):
    img, t = img_and_tex
    assert_sampled_output(geo.crop(t, 7, 5, 21, 18), t, img[5:23, 7:28], precision, exact=True)
    with pytest.raises(ValueError):
        geo.crop(t, 50, 0, 20, 10)  # out of bounds


# -------------------------------------------------------- wrapped tiling
def np_tile(img, out_w, out_h, scale_x, scale_y, offset_x, offset_y, mode):
    """Wrapped pixel-center sampling reference for GeometryOps.tile."""
    h, w, _ = img.shape
    ys, xs = np.mgrid[0:out_h, 0:out_w]
    sx = (xs + 0.5) / scale_x + offset_x
    sy = (ys + 0.5) / scale_y + offset_y
    if mode == "nearest":
        return img[np.floor(sy).astype(int) % h, np.floor(sx).astype(int) % w]

    px = sx - 0.5
    py = sy - 0.5
    x0 = np.floor(px).astype(int)
    y0 = np.floor(py).astype(int)
    fx = (px - x0)[..., None]
    fy = (py - y0)[..., None]
    top = (1 - fx) * img[y0 % h, x0 % w] + fx * img[y0 % h, (x0 + 1) % w]
    bottom = (1 - fx) * img[(y0 + 1) % h, x0 % w] + fx * img[
        (y0 + 1) % h, (x0 + 1) % w
    ]
    return (1 - fy) * top + fy * bottom


@pytest.mark.parametrize("mode", ["bilinear", "nearest"])
def test_tile_matches_numpy_with_scale_and_phase(geo, img_and_tex, precision, mode):
    img, t = img_and_tex
    args = (83, 67, 1.3, 0.7, -2.25, 4.5, mode)
    out = geo.tile(t, *args)
    ref = np_tile(img, *args)
    assert_sampled_output(out, t, ref, precision, exact=mode == "nearest")


@pytest.mark.parametrize("mode", ["bilinear", "nearest"])
def test_tile_is_periodic_at_integer_scale(geo, img_and_tex, precision, mode):
    img, t = img_and_tex
    period_x = img.shape[1] * 2
    period_y = img.shape[0] * 3
    out = texture_to_np(
        geo.tile(
            t,
            period_x + 17,
            period_y + 13,
            scale_x=2.0,
            scale_y=3.0,
            offset_x=-0.375,
            offset_y=1.25,
            mode=mode,
        )
    )
    assert_geometry_values(
        out[:, :17], out[:, period_x:period_x + 17], precision, exact=mode == "nearest",
    )
    assert_geometry_values(
        out[:13], out[period_y:period_y + 13], precision, exact=mode == "nearest",
    )


def test_tile_rejects_invalid_geometry(geo, img_and_tex):
    _, t = img_and_tex
    with pytest.raises(ValueError, match="width/height must be positive"):
        geo.tile(t, 0, 20)
    with pytest.raises(ValueError, match="scale_x and scale_y"):
        geo.tile(t, 20, 20, scale_x=0.0)
    with pytest.raises(ValueError, match="offset_x and offset_y"):
        geo.tile(t, 20, 20, offset_y=np.inf)
    with pytest.raises(ValueError, match="mode must be one of"):
        geo.tile(t, 20, 20, mode="cubic")


# ------------------------------------------------------------- affine
def np_affine(img, matrix, out_w, out_h, mode):
    """Reference inverse-warp with pixel-center coords and zeros outside."""
    H, W, C = img.shape
    # Independently solve the forward 2x2 system instead of calling the
    # production invert_affine helper (which could share a bug with the kernel).
    a, b, tx, c, d, ty = np.asarray(matrix, dtype=np.float64).ravel()
    determinant = a * d - b * c
    ys, xs = np.mgrid[0:out_h, 0:out_w]
    xo = xs + 0.5
    yo = ys + 0.5
    sx = (d * (xo - tx) - b * (yo - ty)) / determinant
    sy = (a * (yo - ty) - c * (xo - tx)) / determinant
    out = np.zeros((out_h, out_w, C), dtype=np.float64)
    if mode == "nearest":
        xi = np.floor(sx).astype(int)
        yi = np.floor(sy).astype(int)
        ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        out[ok] = img[yi[ok], xi[ok]]
        return out
    px = sx - 0.5
    py = sy - 0.5
    x0 = np.floor(px).astype(int)
    y0 = np.floor(py).astype(int)
    fx = (px - x0)[..., None]
    fy = (py - y0)[..., None]
    for dx, dy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                      (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        xi, yi = x0 + dx, y0 + dy
        ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        vals = np.zeros((out_h, out_w, C))
        vals[ok] = img[yi[ok], xi[ok]]
        out += w * vals
    return out


@pytest.mark.parametrize("mode", ["bilinear", "nearest"])
def test_affine_identity_and_translation(geo, img_and_tex, precision, mode):
    img, t = img_and_tex
    ident = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert_sampled_output(geo.affine(t, ident, mode=mode), t, img, precision, exact=True)

    # integer translation: bilinear weights collapse to an exact copy
    shift = np.array([[1.0, 0.0, 7.0], [0.0, 1.0, 3.0]])
    ref = np.zeros_like(img)
    ref[3:, 7:] = img[:-3, :-7]
    assert_sampled_output(geo.affine(t, shift, mode=mode), t, ref, precision, exact=True)


@pytest.mark.parametrize("mode", ["bilinear", "nearest"])
def test_affine_matches_numpy(geo, img_and_tex, precision, mode):
    img, t = img_and_tex
    m = rotation_matrix(23.0, 28.0, 20.0)
    m[0, 2] += 2.5  # add a fractional translation on top
    out = geo.affine(t, m, mode=mode)
    ref = np_affine(img, m, 56, 40, mode)
    assert_sampled_output(out, t, ref, precision, exact=mode == "nearest")


# --------------------------------------------------- small canvases / borders
@pytest.fixture(params=[(1, 1), (1, 9), (7, 1), (5, 9)], ids=["1x1", "1xN", "Nx1", "non-square"])
def boundary_image(ctx, precision, request):
    h, w = request.param
    rng = np.random.default_rng(65)
    img = rng.random((h, w, 4)).astype(np.float32)
    img[..., :3] = img[..., :3] * 1.5 - 0.25  # no clipping, even at borders
    img[0, 0] = (-0.25, 1.25, 0.375, 0.0)
    return upload(ctx[0], img, precision)


def test_exact_remaps_at_small_image_borders(geo, boundary_image, precision):
    img, t = boundary_image
    h, w, _ = img.shape
    assert_sampled_output(
        geo.flip(t, horizontal=True, vertical=True), t, img[::-1, ::-1], precision, exact=True,
    )
    for k in range(4):
        assert_sampled_output(geo.rot90(t, k), t, np.rot90(img, k), precision, exact=True)
    for x, y, cw, ch in ((0, 0, w, h), (w - 1, h - 1, 1, 1)):
        assert_sampled_output(
            geo.crop(t, x, y, cw, ch), t, img[y:y + ch, x:x + cw], precision, exact=True,
        )


@pytest.mark.parametrize("mode", ["bilinear", "nearest"])
@pytest.mark.parametrize("scale, phase", [
    ((1.0, 1.0), (0.0, 0.0)),
    ((2.0, 0.5), (-0.25, 0.375)),
    ((0.75, 1.25), (17.125, -23.375)),
    ((1.5, 0.75), (1048576.125, -1048576.375)),
], ids=["identity", "seam", "multi-period-phase", "large-phase"])
def test_tile_small_images_and_seams(geo, boundary_image, precision, mode, scale, phase):
    img, t = boundary_image
    args = (19, 17, *scale, *phase, mode)
    ref = np_tile(img, *args)
    assert_sampled_output(geo.tile(t, *args), t, ref, precision, exact=mode == "nearest")


@pytest.mark.parametrize("mode", ["bilinear", "nearest"])
@pytest.mark.parametrize("matrix", [
    ((1.0, 0.0, 0.25), (0.0, 1.0, -0.375)),
    ((1.0, 0.0, 0.5), (0.0, 1.0, -0.5)),  # exact nearest-neighbor boundaries
    # Non-binary inverse coefficients: avoid exact nearest-neighbor ties.
    # For this canvas sx is >= 1/60 and sy >= 1/10 from any integer. Integer
    # boundaries are tested separately with exactly representable transforms.
    ((0.75, 0.25, -0.1875), (0.0, 1.25, 0.375)),
    ((-1.0, 0.0, 3.0), (0.0, 1.0, 0.0)),
    ((1.0, 0.0, 30.0), (0.0, 1.0, -30.0)),
], ids=["fractional-border", "half-pixel-boundary", "scale-shear", "reflection", "outside"])
def test_affine_small_images_and_zero_fill(geo, boundary_image, precision, mode, matrix):
    img, t = boundary_image
    ref = np_affine(img, matrix, 13, 11, mode)
    out = geo.affine(t, matrix, width=13, height=11, mode=mode)
    assert_sampled_output(out, t, ref, precision, exact=mode == "nearest")


def test_functional_geometry_matches_reference(img_and_tex, precision):
    import slangimage as si

    arr, t = img_and_tex
    img = si.Image(t)
    matrix = ((1.0, 0.0, 0.25), (0.0, 1.0, -0.375))
    cases = [
        (si.ops.flip(img, horizontal=True), arr[:, ::-1], True),
        (si.ops.rot90(img, -1), np.rot90(arr, -1), True),
        (si.ops.crop(img, 2, 3, 11, 7), arr[3:10, 2:13], True),
    ]
    for mode in ("bilinear", "nearest"):
        cases.extend([
            (si.ops.tile(img, 63, 47, scale_x=1.5, offset_y=-2.25, mode=mode),
             np_tile(arr, 63, 47, 1.5, 1.0, 0.0, -2.25, mode), mode == "nearest"),
            (si.ops.affine(img, matrix, width=59, height=43, mode=mode),
             np_affine(arr, matrix, 59, 43, mode), mode == "nearest"),
        ])
    for out, ref, exact in cases:
        assert isinstance(out, si.Image)
        assert out.precision == precision
        assert_sampled_output(out.texture, t, ref, precision, exact=exact)


def test_rotate_90_matches_rot90(ctx):
    import slangimage as si

    device, _ = ctx
    rng = np.random.default_rng(62)
    arr = rng.random((32, 32, 4)).astype(np.float32)  # square: canvas fits
    arr[..., 3] = 1.0
    img = si.from_numpy(arr, precision="fp32", device=device)
    out = si.ops.rotate(img, 90.0).to_numpy()
    ref = np.rot90(arr, k=1)
    # interior only: the analytic rotation samples half-pixel boundaries
    # at the edges where bilinear taps mix with the zero fill
    np.testing.assert_allclose(out[1:-1, 1:-1], ref[1:-1, 1:-1], atol=ATOL)


def test_rotate_expand(ctx):
    import slangimage as si

    device, _ = ctx
    rng = np.random.default_rng(63)
    arr = rng.random((30, 50, 4)).astype(np.float32)
    arr[..., 3] = 1.0
    img = si.from_numpy(arr, precision="fp32", device=device)

    out = si.ops.rotate(img, 90.0, expand=True)
    assert (out.width, out.height) == (30, 50)  # swapped canvas

    out45 = si.ops.rotate(img, 45.0, expand=True)
    expect = int(np.ceil((30 + 50) / np.sqrt(2)))
    assert abs(out45.width - expect) <= 1 and abs(out45.height - expect) <= 1
    # energy preserved-ish: nothing clipped away by the larger canvas
    assert out45.to_numpy()[..., :3].sum() > 0.95 * arr[..., :3].sum()


def test_functional_wrappers(ctx):
    import slangimage as si

    rng = np.random.default_rng(64)
    arr = rng.random((24, 24, 4)).astype(np.float32)
    arr[..., 3] = 1.0
    img = si.from_numpy(arr, precision="fp32", device=ctx[0])
    assert si.ops.flip(img, horizontal=True).to_numpy().shape == arr.shape
    assert si.ops.rot90(img, 1).to_numpy().shape == arr.shape
    assert si.ops.crop(img, 2, 2, 10, 8).to_numpy().shape == (8, 10, 4)
    assert si.ops.tile(img, 31, 27, scale_x=1.5, offset_y=2.0).to_numpy().shape == (
        27,
        31,
        4,
    )
    assert np.isfinite(si.ops.rotate(img, 33.0).to_numpy()).all()
    assert "tile" in si.ops.__all__

    ident = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    with pytest.raises(ValueError, match="width and height must be >= 0"):
        si.ops.affine(img, ident, width=-1)
    with pytest.raises(ValueError, match="width and height must be >= 0"):
        si.ops.affine(img, ident, height=-1)
