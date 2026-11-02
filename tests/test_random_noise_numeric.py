"""Backend-honest contracts for the fp32 sin/fmod hash noise generator.

The hash intentionally diverges between GPU backends, so this suite checks
repeatability, control quantization, algebraic invariants, precision storage and
broad distribution quality without inventing a cross-backend CPU golden image.
"""

import numpy as np
import pytest

import slangimage as si
from slangimage.kernels.point_ops import PointOps
from slangimage.runtime.textures import texture_to_np
from test_core_numeric import storage_error


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("shape", [(1, 1), (1, 31), (29, 1), (17, 23)])
@pytest.mark.parametrize("seed", [-10, 0, .5, 7.25])
def test_random_noise_repeatable_shape_range_channels_and_origin(ctx, precision, shape, seed):
    height, width = shape
    first = si.ops.random_noise(width, height, seed, precision, ctx[0])
    second = si.ops.random_noise(width, height, seed, precision, ctx[0])
    assert first.device is ctx[0] and first.precision == precision
    actual = first.to_numpy()
    assert actual.shape == (height, width, 4) and np.isfinite(actual).all()
    np.testing.assert_array_equal(actual, second.to_numpy())
    np.testing.assert_array_equal(actual[..., 0], actual[..., 1])
    np.testing.assert_array_equal(actual[..., 1], actual[..., 2])
    np.testing.assert_array_equal(actual[..., 3], 1)
    # Mirror only the fp32 lerp endpoints, then account for destination storage.
    lower = np.float32(.5)+np.float32(.9)*(np.float32(0)-np.float32(.5))
    upper = np.float32(.5)+np.float32(.9)*(np.float32(1)-np.float32(.5))
    storage = np.float16 if precision == "fp16" else np.float32
    lower, upper = float(storage(lower)), float(storage(upper))
    assert actual[..., :3].min() >= lower and actual[..., :3].max() <= upper
    # coord=(0,0) makes the hash dot product zero for every seed.
    expected_origin = storage(np.float32(.5)+np.float32(.9)*(np.float32(0)-np.float32(.5)))
    assert actual[0, 0, 0] == expected_origin


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
def test_random_noise_seed_quantization_and_first_column(ctx, precision):
    seed = float(np.float32(.5))
    same_fp32 = seed+1e-10
    different = seed+.125
    base = si.ops.random_noise(64, 64, seed, precision, ctx[0]).to_numpy()
    same = si.ops.random_noise(64, 64, same_fp32, precision, ctx[0]).to_numpy()
    changed = si.ops.random_noise(64, 64, different, precision, ctx[0]).to_numpy()
    np.testing.assert_array_equal(base, same)
    # With x=0 the seed-dependent coefficient is multiplied by zero.
    np.testing.assert_array_equal(base[:, 0], changed[:, 0])
    assert np.count_nonzero(base[:, 1:, 0] != changed[:, 1:, 0]) > 64


@pytest.mark.parametrize("seed", [-3, 0, .5, 11])
def test_random_noise_fp16_matches_backend_fp32_with_storage_bound(ctx, seed):
    fp32 = si.ops.random_noise(73, 61, seed, "fp32", ctx[0]).to_numpy().astype(np.float64)
    fp16 = si.ops.random_noise(73, 61, seed, "fp16", ctx[0]).to_numpy().astype(np.float64)
    error = storage_error(fp32, "fp16")
    np.testing.assert_array_less(np.abs(fp16-fp32), np.nextafter(error, np.inf))


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("seed", [-7, 0, .5, 9.25])
def test_random_noise_distribution_is_non_degenerate(ctx, precision, seed):
    values = si.ops.random_noise(256, 256, seed, precision, ctx[0]).to_numpy()[..., 0]
    assert .47 < float(values.mean()) < .53
    assert .23 < float(values.std()) < .29
    counts, _ = np.histogram(values, bins=10, range=(.05, .95))
    assert counts.min() > .7*counts.mean()
    assert counts.max() < 1.3*counts.mean()
    horizontal = np.corrcoef(values[:, :-1].ravel(), values[:, 1:].ravel())[0, 1]
    vertical = np.corrcoef(values[:-1].ravel(), values[1:].ravel())[0, 1]
    assert abs(horizontal) < .05 and abs(vertical) < .05


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
def test_random_noise_public_and_low_level_parity(ctx, precision):
    public = si.ops.random_noise(19, 13, 2.5, precision, ctx[0]).to_numpy()
    fmt = si.from_numpy(np.zeros((1, 1, 4), dtype=np.float32),
                        precision=precision, device=ctx[0]).texture.format
    direct = PointOps(*ctx).random_noise(19, 13, 2.5, fmt)
    np.testing.assert_array_equal(public, texture_to_np(direct))
