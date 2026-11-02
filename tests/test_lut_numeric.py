"""Texture CLUT references using a logical [blue, green, red, channel] volume.

No production identity builder, atlas-coordinate helper, or tensor LUT oracle.
Both inputs are quantized before reference evaluation, including mixed storage.
"""

import itertools

import numpy as np
import pytest

import slangimage as si
from slangimage.runtime.textures import texture_to_np
from test_core_numeric import storage_error


def pack_cube(cube):
    _, cell, _, channels = cube.shape
    return cube.reshape(8, 8, cell, cell, channels).transpose(0, 2, 1, 3, 4).reshape(
        8*cell, 8*cell, channels,
    )


def make_cube(size, kind):
    cell = size // 8
    b, g, r = np.meshgrid(np.linspace(0, 1, 64), np.linspace(0, 1, cell),
                          np.linspace(0, 1, cell), indexing="ij")
    if kind == "multiaffine":
        rgb = np.stack((-0.25 + r*g + 0.5*b, 1.25 - g*b - 0.5*r,
                        0.125 + r*b - 0.25*g), axis=-1)
    elif kind == "nonlinear":
        rgb = np.stack((r*r - 0.25*b, np.sqrt(g) + 0.25*r,
                        np.sin(np.pi*b)*g + 0.125*r), axis=-1)
    else:
        rgb = np.random.default_rng(411).uniform(-0.25, 1.25, (*r.shape, 3))
    # Deliberately unrelated to the source alpha, including outside [0,1].
    return np.concatenate((rgb, (2*b - g - 0.5)[..., None]), axis=-1)


def source_values(cell, precision):
    dtype = np.float16 if precision == "fp16" else np.float32
    points = list(itertools.product([0.0, 1.0], repeat=3))
    # Every blue slice boundary and midpoint, notably 7->8 and other atlas
    # row wraps, with R/G at a cell edge and in its interior.
    for blue in np.arange(127)/126:
        points.extend([(0, 1, blue), (1, 0, blue), (0.3125, 0.6875, blue)])
    # Immediate representable neighbors of internal R/G and blue nodes.
    for axis, levels in enumerate((cell, cell, 64)):
        for index in sorted({1, levels//2, levels - 2}):
            if not 0 < index < levels - 1:
                continue
            value = dtype(index / (levels - 1))
            for v in (np.nextafter(value, dtype(-np.inf)), value,
                      np.nextafter(value, dtype(np.inf))):
                point = [0.3125, 0.6875, 0.4375]
                point[axis] = float(v)
                points.append(point)
    points.extend(np.random.default_rng(412).uniform(-0.25, 1.25, (41, 3)))
    rgb = np.asarray(points)
    alpha = np.resize([-0.25, 0, 0.3125, 1, 1.25], len(rgb))
    # Exercise the singleton-row upload normalization as part of LUT coverage.
    return np.column_stack((rgb, alpha)).astype(dtype)[None, ...]


def lut_reference(image, cube, intensity):
    source = np.clip(image, 0, 1)
    cell = cube.shape[1]
    coordinates = source[..., [2, 1, 0]] * np.array([63, cell-1, cell-1])
    lower = np.floor(coordinates).astype(int)
    fraction = coordinates - lower
    mapped = np.zeros((*image.shape[:2], 3))
    for corner in itertools.product((0, 1), repeat=3):
        index = np.minimum(lower + corner, [63, cell-1, cell-1])
        weight = np.prod(np.where(corner, fraction, 1 - fraction), axis=-1)
        mapped += weight[..., None] * cube[index[..., 0], index[..., 1], index[..., 2], :3]
    amount = float(np.clip(np.float32(intensity), 0, 1))
    result = source.copy()
    result[..., :3] = (1 - amount)*source[..., :3] + amount*mapped
    return result


def error_budget(reference, cube, precision, intensity):
    # The atlas shader forms normalized coordinates then converts back to
    # texels. Bound this fp32 roundoff in texel units, independently of the
    # ideal 3D oracle. Adjacent-node differences give global per-channel
    # Lipschitz bounds, valid across cell boundaries too. Smooth tables thus
    # get tighter bounds than high-contrast random tables. No relative/PSNR
    # tolerance, and no input/LUT quantization error is counted twice.
    eps = np.finfo(np.float32).eps
    size = cube.shape[1]*8
    coordinate_error = (2*eps*63, 4*eps*size, 4*eps*size)
    rgb_error = np.full(3, 5e-6)  # interpolation/final-mix arithmetic, values <= 1.5
    amount = float(np.clip(np.float32(intensity), 0, 1))
    for axis, delta in enumerate(coordinate_error):
        slope = np.abs(np.diff(cube[..., :3], axis=axis)).max(axis=(0, 1, 2))
        rgb_error += amount * delta * slope
    error = np.broadcast_to(rgb_error, reference[..., :3].shape)
    return error + storage_error(reference[..., :3], precision, error)


@pytest.fixture(scope="module", params=[16, 24, 40, 128, 512])
def size(request):
    return request.param


@pytest.fixture(scope="module", params=["multiaffine", "nonlinear", "random"])
def table_kind(request):
    return request.param


@pytest.fixture(scope="module", params=["fp16", "fp32"])
def lut_precision(request):
    return request.param


@pytest.fixture(scope="module")
def table(ctx, size, table_kind, lut_precision):
    dtype = np.float16 if lut_precision == "fp16" else np.float32
    stored = make_cube(size, table_kind).astype(dtype)
    image = si.from_numpy(pack_cube(stored), precision=lut_precision, device=ctx[0])
    return stored.astype(np.float64), image


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("intensity", [-2, 0, 0.25, 0.7, 1, 2])
def test_lut_matches_volume_reference(ctx, table, precision, intensity):
    cube, lut_image = table
    stored = source_values(cube.shape[1], precision)
    image = si.from_numpy(stored, precision=precision, device=ctx[0])
    reference = lut_reference(stored.astype(np.float64), cube, intensity)
    output = si.ops.lut(image, lut_image, intensity=intensity)
    assert output.device is image.device and output.precision == precision
    actual = output.to_numpy().astype(np.float64)
    assert actual.shape == stored.shape and np.isfinite(actual).all()
    np.testing.assert_array_equal(actual[..., 3], reference[..., 3])
    if intensity <= 0:
        np.testing.assert_array_equal(actual, reference)
    else:
        error = error_budget(reference, cube, precision, intensity)
        np.testing.assert_array_less(np.abs(actual[..., :3] - reference[..., :3]),
                                     np.nextafter(error, np.inf))


def test_lut_oracle_closed_forms():
    cube = make_cube(24, "multiaffine")
    image = np.array([[[0.25, 0.5, 0.75, 0.3125]]])
    expected = np.array([[[0.25, 0.75, 0.1875, 0.3125]]])
    np.testing.assert_allclose(lut_reference(image, cube, 1), expected, atol=1e-15, rtol=0)
    # A 3D corner impulse has product weight .5*.5*.5 = 1/8.
    cube = np.zeros((64, 2, 2, 4))
    cube[1, 1, 1, :3] = (1, 2, 3)
    image = np.array([[[0.5, 0.5, 0.5/63, 1.0]]])
    np.testing.assert_allclose(lut_reference(image, cube, 1),
                               [[[1/8, 2/8, 3/8, 1]]], atol=1e-15, rtol=0)
    atlas = pack_cube(cube)
    np.testing.assert_array_equal(atlas[1, 3], cube[1, 1, 1])
    cube[8, 0, 0] = (7, 6, 5, 4)
    np.testing.assert_array_equal(pack_cube(cube)[2, 0], [7, 6, 5, 4])


@pytest.mark.parametrize("size", [16, 24, 40, 128, 512])
def test_identity_lut_builder_matches_grid_coordinates(size):
    y, x = np.mgrid[:size, :size]
    cell = size//8
    expected = np.stack(((x % cell)/(cell-1), (y % cell)/(cell-1),
                         ((y//cell)*8 + x//cell)/63, np.ones_like(x)), axis=-1)
    np.testing.assert_array_equal(si.ops.identity_lut_np(size), expected.astype(np.float32))


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("shape", [(1, 1), (7, 1), (5, 9)])
def test_lut_nonconstant_small_images(ctx, precision, shape):
    dtype = np.float16 if precision == "fp16" else np.float32
    cube = make_cube(24, "random").astype(dtype)
    image = np.random.default_rng(414).uniform(-0.25, 1.25, (*shape, 4)).astype(dtype)
    source = si.from_numpy(image, precision=precision, device=ctx[0])
    table = si.from_numpy(pack_cube(cube), precision=precision, device=ctx[0])
    cube = cube.astype(np.float64)
    reference = lut_reference(image.astype(np.float64), cube, 0.7)
    result = si.ops.lut(source, table, 0.7)
    assert result.device is source.device and result.precision == precision
    actual = result.to_numpy()
    assert actual.shape == image.shape and np.isfinite(actual).all()
    np.testing.assert_array_equal(actual[..., 3], reference[..., 3])
    np.testing.assert_array_less(np.abs(actual[..., :3] - reference[..., :3]),
                                 np.nextafter(error_budget(reference, cube, precision, 0.7), np.inf))


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("shape", [(1, 1), (7, 1), (5, 9)])
def test_lut_ignores_table_alpha_and_preserves_unclipped_rgb(ctx, precision, shape):
    dtype = np.float16 if precision == "fp16" else np.float32
    cube = np.empty((64, 2, 2, 4), dtype=dtype)
    cube[...] = (-0.25, 1.25, 0.5, -2)
    image = np.random.default_rng(413).random((*shape, 4)).astype(dtype)
    source = si.from_numpy(image, precision=precision, device=ctx[0])
    expected = image.copy()
    expected[..., :3] = (-0.25, 1.25, 0.5)
    outputs = []
    for alpha in (-2, 3):
        cube[..., 3] = alpha
        table = si.from_numpy(pack_cube(cube), precision=precision, device=ctx[0])
        # Exercise the raw-texture wrapper too; the main reference covers Image.
        out = si.ops.lut(source.texture, table.texture)
        assert out.device is source.device and out.format == source.texture.format
        actual = texture_to_np(out)
        assert actual.shape == expected.shape and np.isfinite(actual).all()
        np.testing.assert_array_equal(actual[..., 3], expected[..., 3])
        # Even at intensity=1, CUDA's c + (mapped-c)*1 can round twice;
        # a constant mapped value is not an exact final-output contract.
        # Bound subtraction/addition error, then the output texture store.
        rgb = expected[..., :3].astype(np.float64)
        compute = 2*np.finfo(np.float32).eps * (
            np.abs(rgb - image[..., :3].astype(np.float64)) + np.abs(rgb)
        )
        error = compute + storage_error(rgb, precision, compute)
        np.testing.assert_array_less(np.abs(actual[..., :3] - rgb), np.nextafter(error, np.inf))
        assert np.all(actual[..., 0] < 0) and np.all(actual[..., 1] > 1)
        outputs.append(actual)
    np.testing.assert_array_equal(*outputs)  # Changing LUT alpha has zero effect.
