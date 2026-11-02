# tests/test_textures.py
# Upload/readback contracts: 3-channel expansion, lossless round-trips.

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import slangimage as si
from slangimage.runtime.textures import np_to_texture, texture_to_np

@pytest.mark.parametrize("precision", ["fp16", "fp32"])
def test_three_channel_upload_expands_to_rgba(ctx, precision):
    device, _ = ctx
    rng = np.random.default_rng(1)
    rgb = rng.random((16, 16, 3)).astype(np.float32)
    tex = np_to_texture(device, rgb, usage="read", precision=precision)
    out = texture_to_np(tex)
    assert out.shape == (16, 16, 4)
    atol = 1e-3 if precision == "fp16" else 1e-6
    np.testing.assert_allclose(out[..., :3], rgb, atol=atol)
    np.testing.assert_allclose(out[..., 3], 1.0, atol=atol)


def test_readback_is_lossless_outside_unit_range(ctx):
    device, _ = ctx
    data = np.array([[[-2.5, 0.5, 7.25, 1.0]]], dtype=np.float32)
    tex = np_to_texture(device, data, usage="read", precision="fp32")
    out = texture_to_np(tex)
    np.testing.assert_array_equal(out, data)  # no silent clipping

    clipped = texture_to_np(tex, clip=True)
    np.testing.assert_allclose(clipped, [[[0.0, 0.5, 1.0, 1.0]]])


@pytest.mark.parametrize("shape", [(0, 4, 4), (4, 0, 4)])
def test_upload_rejects_zero_sized_images(ctx, shape):
    device, _ = ctx
    with pytest.raises(ValueError, match="height and width must both be >= 1"):
        si.from_numpy(np.zeros(shape, np.float32), device=device)


def upload_view(layout, dtype, channels):
    base = np.linspace(-2, 3, 6*8*channels, dtype=dtype).reshape(6, 8, channels)
    return {
        "row-newaxis": base[0][None, ...],
        "column-newaxis": base[:, 0].copy()[:, None, :],
        "pixel-newaxis": base[0, 0][None, None, :],
        "row-negative": base[:1][::-1],
        "column-negative": base[:, :1].copy()[:, ::-1],
        "row-oversized": base[:1:3],
        "column-oversized": base[:, :1].copy()[:, ::3],
        "contiguous": base,
        "transpose": base.transpose(1, 0, 2),
        "reverse": base[:, ::-1],
        "slice": base[::2, ::3],
    }[layout]


@pytest.mark.parametrize("layout", [
    "row-newaxis", "column-newaxis", "pixel-newaxis", "row-negative",
    "column-negative", "row-oversized", "column-oversized", "contiguous",
    "transpose", "reverse", "slice",
])
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("channels", [3, 4])
@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("readonly", [False, True])
def test_upload_views_roundtrip_exactly(ctx, layout, dtype, channels, precision, readonly):
    values = upload_view(layout, dtype, channels)
    before = values.copy()
    strides = values.strides
    values.flags.writeable = not readonly
    result = si.from_numpy(values, precision=precision, device=ctx[0])
    assert result.device is ctx[0] and result.precision == precision
    expected = values.astype(np.float16 if precision == "fp16" else np.float32)
    if channels == 3:
        expected = np.concatenate((expected, np.ones((*expected.shape[:2], 1))), axis=-1)
    np.testing.assert_array_equal(result.to_numpy(), expected)
    np.testing.assert_array_equal(values, before)
    assert values.strides == strides and values.flags.writeable == (not readonly)


@pytest.mark.parametrize("layout", [
    "contiguous", "row-newaxis", "column-newaxis", "pixel-newaxis",
    "row-negative", "column-negative", "row-oversized", "column-oversized",
])
@pytest.mark.parametrize("readonly", [False, True])
def test_fp32_upload_normalization_preserves_caller_and_fast_path(layout, readonly):
    values = upload_view(layout, np.float32, 4)
    assert values.flags.c_contiguous
    original_strides = values.strides
    values.flags.writeable = not readonly
    captured = {}

    def create_texture(**kwargs):
        captured.update(kwargs)
        return "texture"

    device = SimpleNamespace(create_texture=create_texture)
    assert np_to_texture(device, values, precision="fp32") == "texture"
    data = captured["data"]
    assert data.strides == (values.shape[1]*16, 16, 4)
    assert data.flags.writeable
    assert np.shares_memory(data, values) == (not readonly)
    np.testing.assert_array_equal(data, values)
    assert values.strides == original_strides and values.flags.writeable == (not readonly)
    if layout == "contiguous" and not readonly:
        assert data is values  # Preserve the ordinary fp32 no-copy fast path.
