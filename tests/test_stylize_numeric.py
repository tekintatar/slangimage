"""Bounded references for pointillize and the public normalized-CMY halftone."""

import math

import numpy as np
import pytest

import slangimage as si
from slangimage.kernels.stylize import CmykHalftoneOp, PointillizeOp
from slangimage.runtime.textures import texture_to_np
from test_core_numeric import storage_error


def frac(value):
    return value-np.floor(value)


def point_hash(cell, seed):
    cell = np.asarray([cell[0], cell[1], cell[0]], dtype=np.float32)
    constants = np.asarray([.1031, .1030, .0973], dtype=np.float32)
    seed = np.float32(seed)
    hashed = frac(np.float32(cell*constants+np.float32(seed*np.float32(.113))))
    rotated = np.asarray([hashed[1], hashed[2], hashed[0]], dtype=np.float32)
    hashed = np.float32(hashed+np.float32(np.dot(hashed, rotated+np.float32(33.33))))
    return frac(np.asarray([(hashed[0]+hashed[1])*hashed[2],
                            (hashed[0]+hashed[2])*hashed[1]], dtype=np.float32))


def pointillize_indices(shape, radius, jitter, seed):
    height, width = shape
    cell = float(np.float32(max(np.float32(radius), np.float32(2))))
    jitter = float(np.clip(np.float32(jitter), 0, 1))
    rows = np.empty(shape, dtype=np.int64)
    cols = np.empty(shape, dtype=np.int64)
    min_margin = math.inf
    for row in range(height):
        for col in range(width):
            point = np.array([col+.5, row+.5], dtype=np.float32)
            base = np.floor(point/np.float32(cell)).astype(np.int64)
            candidates = []
            for j in range(-1, 2):
                for i in range(-1, 2):
                    cid = base+np.array([i, j])
                    offset = (point_hash(cid, seed)-np.float32(.5))*np.float32(jitter)
                    center = np.float32((cid.astype(np.float32)+np.float32(.5)+offset)*np.float32(cell))
                    delta = np.float32(center-point)
                    distance = float(np.float32(np.dot(delta, delta)))
                    candidates.append((distance, center))
            candidates.sort(key=lambda item: item[0])
            min_margin = min(min_margin, candidates[1][0]-candidates[0][0])
            center = candidates[0][1]
            cols[row, col] = np.clip(int(center[0]), 0, width-1)
            rows[row, col] = np.clip(int(center[1]), 0, height-1)
    return rows, cols, min_margin


def make_image(pattern, precision, shape=(9, 13)):
    if pattern == "coordinates":
        yy, xx = np.mgrid[:shape[0], :shape[1]]
        rgb = np.stack((xx/(shape[1]+1), yy/(shape[0]+1),
                        (xx+shape[1]*yy)/(shape[0]*shape[1]+1)), axis=-1)
    elif pattern == "hdr":
        rgb = np.random.default_rng(1302).uniform(-.25, 1.5, (*shape, 3))
    else:
        rgb = np.random.default_rng(1303).uniform(0, 1, (*shape, 3))
    dtype = np.float16 if precision == "fp16" else np.float32
    stored = np.ones((*shape, 4), dtype=dtype)
    stored[..., :3] = rgb
    stored[..., 3] = np.resize([-1, 0, .125, .5, 1, 2], shape)
    return stored


POINT_CONTROLS = [(-3, -1, 0), (2, 0, 0), (3, 0, 2), (3.5, .5, -1),
                  (5, 1, .25), (7.25, 2, 3), (20, .75, -4)]


def check_pointillize(ctx, stored, precision, controls):
    radius, jitter, seed = controls
    source = si.from_numpy(stored, precision=precision, device=ctx[0])
    result = si.ops.pointillize(source, radius, jitter, seed)
    assert result.device is source.device and result.precision == precision
    actual = result.to_numpy()
    rows, cols, margin = pointillize_indices(stored.shape[:2], radius, jitter, seed)
    # Reject a corpus where compiler rounding could legitimately select a tie.
    assert margin > 2e-4
    expected = stored[rows, cols].copy()
    expected[..., 3] = stored[..., 3]
    np.testing.assert_array_equal(actual, expected)
    return actual


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("pattern", ["coordinates", "hdr", "random"])
@pytest.mark.parametrize("controls", POINT_CONTROLS)
def test_pointillize_independent_sample_map(ctx, precision, pattern, controls):
    check_pointillize(ctx, make_image(pattern, precision), precision, controls)


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("shape", [(1, 1), (1, 13), (11, 1)])
def test_pointillize_tiny_alpha_repeated_and_low_level(ctx, precision, shape):
    stored = make_image("hdr", precision, shape)
    controls = (3.5, .5, -1)
    first = check_pointillize(ctx, stored, precision, controls)
    np.testing.assert_array_equal(first, check_pointillize(ctx, stored, precision, controls))
    source = si.from_numpy(stored, precision=precision, device=ctx[0])
    direct = PointillizeOp(*ctx).apply(source.texture, *controls)
    np.testing.assert_array_equal(first, texture_to_np(direct))


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
def test_pointillize_control_clamps(ctx, precision):
    stored = make_image("coordinates", precision)
    np.testing.assert_array_equal(
        check_pointillize(ctx, stored, precision, (-3, .5, 2)),
        check_pointillize(ctx, stored, precision, (2, .5, 2)),
    )
    np.testing.assert_array_equal(
        check_pointillize(ctx, stored, precision, (5, -1, 2)),
        check_pointillize(ctx, stored, precision, (5, 0, 2)),
    )
    np.testing.assert_array_equal(
        check_pointillize(ctx, stored, precision, (5, 2, 2)),
        check_pointillize(ctx, stored, precision, (5, 1, 2)),
    )


DEGREES = np.float32(3.14159265/180)


def smoothstep(low, high, value):
    amount = np.clip((value-low)/(high-low), 0, 1)
    return amount*amount*(3-2*amount)


def halftone_channel(x, y, angle, width, value, sharpness):
    sine, cosine = np.sin(angle), np.cos(angle)
    qx, qy = (cosine*x-sine*y)/width, (sine*x+cosine*y)/width
    dx, dy = qx-(np.floor(qx)+.5), qy-(np.floor(qy)+.5)
    distance = np.sqrt(dx*dx+dy*dy)
    radius = np.sqrt(np.clip(value, 0, 1))*.7071
    return 1-smoothstep(radius-sharpness, radius+sharpness, distance)


def halftone_reference(stored, width, angle_deg, sharpness):
    rgb = np.clip(stored[..., :3].astype(np.float64), 0, 1)
    black = 1-np.max(rgb, axis=-1)
    denominator = np.maximum(1-black, 1e-4)
    inks = [(1-rgb[..., channel]-black)/denominator for channel in range(3)]
    inks.append(black)
    height, width_px = stored.shape[:2]
    y, x = np.mgrid[:height, :width_px].astype(np.float64)+.5
    cell = float(max(np.float32(width), np.float32(1)))
    edge = float(max(np.float32(sharpness), np.float32(1e-3)))
    base = float(np.float32(np.float32(angle_deg)*DEGREES))
    angles = [base+float(np.float32(value)*DEGREES) for value in (15, 75, 0, 45)]
    screens = [halftone_channel(x, y, angle, cell, ink, edge)
               for angle, ink in zip(angles, inks)]
    expected = np.stack(((1-screens[0])*(1-screens[3]),
                         (1-screens[1])*(1-screens[3]),
                         (1-screens[2])*(1-screens[3])), axis=-1)
    compute = np.full_like(expected, 4e-4)
    return expected, compute


HALFTONE_CONTROLS = [(1, 0, .08), (3.5, 11, .08), (7, -45, .03),
                     (20, 123, .2), (0, 0, 0), (6, 360, .001)]


def check_halftone(ctx, stored, precision, controls, *, low_level=False):
    width, angle, sharpness = controls
    source = si.from_numpy(stored, precision=precision, device=ctx[0])
    if low_level:
        actual = texture_to_np(CmykHalftoneOp(*ctx).apply(
            source.texture, width, angle, sharpness
        )).astype(np.float64)
    else:
        result = si.ops.cmyk_halftone(source, width, angle, sharpness)
        assert result.device is source.device and result.precision == precision
        actual = result.to_numpy().astype(np.float64)
    assert actual.shape == stored.shape and np.isfinite(actual).all()
    assert actual[..., :3].min() >= 0 and actual[..., :3].max() <= 1
    np.testing.assert_array_equal(actual[..., 3], stored[..., 3])
    expected, compute = halftone_reference(stored, width, angle, sharpness)
    error = compute+storage_error(expected, precision, compute)
    np.testing.assert_array_less(
        np.abs(actual[..., :3]-expected), np.nextafter(error, np.inf)
    )
    return actual


def test_halftone_reference_normalized_cmy_hand_values():
    stored = np.array([[[.2, .5, .8, 1.], [0, 0, 0, 1.], [1, 1, 1, 1.]]])
    rgb = np.clip(stored[..., :3], 0, 1)
    black = 1-rgb.max(axis=-1)
    denominator = np.maximum(1-black, 1e-4)
    inks = np.stack([(1-rgb[..., channel]-black)/denominator for channel in range(3)], -1)
    np.testing.assert_allclose(inks[0, 0], [.75, .375, 0], atol=1e-15, rtol=0)
    np.testing.assert_array_equal(inks[0, 1:], 0)


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("pattern", ["coordinates", "hdr", "random"])
@pytest.mark.parametrize("controls", HALFTONE_CONTROLS)
def test_cmyk_halftone_independent_reference(ctx, precision, pattern, controls):
    check_halftone(ctx, make_image(pattern, precision), precision, controls)


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("shape", [(1, 1), (1, 13), (11, 1)])
def test_cmyk_halftone_tiny_alpha_repeated_and_low_level(ctx, precision, shape):
    stored = make_image("hdr", precision, shape)
    controls = (3.5, 11, .08)
    first = check_halftone(ctx, stored, precision, controls)
    np.testing.assert_array_equal(first, check_halftone(ctx, stored, precision, controls))
    np.testing.assert_array_equal(
        first, check_halftone(ctx, stored, precision, controls, low_level=True)
    )


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
def test_cmyk_halftone_control_and_input_clamps(ctx, precision):
    stored = make_image("hdr", precision)
    np.testing.assert_array_equal(
        check_halftone(ctx, stored, precision, (0, 15, .08)),
        check_halftone(ctx, stored, precision, (1, 15, .08)),
    )
    np.testing.assert_array_equal(
        check_halftone(ctx, stored, precision, (5, 15, 0)),
        check_halftone(ctx, stored, precision, (5, 15, .001)),
    )
