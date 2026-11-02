# tests/test_torch_conversion.py
# Image.to_torch / si.from_torch: lossless copies between Images and torch
# tensors (HWC or CHW), on the selected texture backend. Needs torch itself,
# not the CUDA torch bridge, so it also runs against CPU tensors on Metal.

from __future__ import annotations

import numpy as np
import pytest

import slangimage as si

torch = pytest.importorskip("torch")

PRECISIONS = [("fp16", np.float16, torch.float16), ("fp32", np.float32, torch.float32)]


def _pixels(dtype, h=5, w=7, seed=3):
    rng = np.random.default_rng(seed)
    # include negative and HDR values: conversion must not clip
    return (rng.random((h, w, 4)) * 1.5 - 0.25).astype(dtype)


@pytest.mark.parametrize("precision,np_dtype,torch_dtype", PRECISIONS)
def test_to_torch_is_lossless_in_both_layouts(ctx, precision, np_dtype, torch_dtype):
    image = si.from_numpy(_pixels(np_dtype), precision=precision, device=ctx[0])
    expected = image.to_numpy()

    hwc = image.to_torch(alpha=True)
    assert hwc.dtype == torch_dtype and hwc.device.type == "cpu"
    assert tuple(hwc.shape) == (5, 7, 4)
    np.testing.assert_array_equal(hwc.float().numpy(), expected)

    rgb = image.to_torch()
    assert tuple(rgb.shape) == (5, 7, 3)
    np.testing.assert_array_equal(rgb.float().numpy(), expected[..., :3])

    chw = image.to_torch(layout="chw")
    assert tuple(chw.shape) == (3, 5, 7) and chw.is_contiguous()
    np.testing.assert_array_equal(chw.float().numpy(), expected[..., :3].transpose(2, 0, 1))


@pytest.mark.parametrize("precision,np_dtype,torch_dtype", PRECISIONS)
def test_round_trip_through_torch_is_exact(ctx, precision, np_dtype, torch_dtype):
    image = si.from_numpy(_pixels(np_dtype), precision=precision, device=ctx[0])
    for layout in ("hwc", "chw"):
        back = si.from_torch(image.to_torch(layout=layout, alpha=True), layout=layout,
                             device=ctx[0])
        assert back.precision == precision and back.device is image.device
        np.testing.assert_array_equal(back.to_numpy(), image.to_numpy())


def test_from_torch_expands_rgb_to_opaque_rgba(ctx):
    rgb = torch.tensor(_pixels(np.float32)[..., :3])
    image = si.from_torch(rgb, device=ctx[0])
    out = image.to_numpy()
    np.testing.assert_array_equal(out[..., :3], rgb.numpy())
    np.testing.assert_array_equal(out[..., 3], 1.0)
    chw = si.from_torch(rgb.permute(2, 0, 1), layout="chw", device=ctx[0])
    np.testing.assert_array_equal(chw.to_numpy(), out)


@pytest.mark.parametrize("dtype,expected", [
    (torch.float16, "fp16"), (torch.float32, "fp32"),
    (torch.bfloat16, "fp32"), (torch.float64, "fp32"),
])
def test_from_torch_infers_precision(ctx, dtype, expected):
    tensor = torch.full((2, 3, 4), 0.5, dtype=dtype)
    image = si.from_torch(tensor, device=ctx[0])
    assert image.precision == expected
    np.testing.assert_array_equal(image.to_numpy(), 0.5)


def test_from_torch_explicit_precision_and_grad_tensor(ctx):
    tensor = torch.full((2, 3, 3), 0.25, requires_grad=True)
    image = si.from_torch(tensor, precision="fp16", device=ctx[0])
    assert image.precision == "fp16"
    np.testing.assert_array_equal(image.to_numpy()[..., :3], 0.25)


@pytest.mark.parametrize("tensor,kwargs,error,match", [
    (torch.zeros(2, 4, 4, 3), {}, ValueError, "3-D tensor"),
    (torch.zeros(4, 4, 3, dtype=torch.uint8), {}, ValueError, "float tensor"),
    (torch.zeros(3, 4, 5), {}, ValueError, 'layout="chw"'),
    (torch.zeros(4, 4, 2), {}, ValueError, "3 or 4 channels"),
    (torch.zeros(4, 4, 3), {"layout": "nchw"}, ValueError, "layout must be"),
    (np.zeros((4, 4, 3), np.float32), {}, TypeError, "torch.Tensor"),
])
def test_from_torch_rejects_invalid_input(ctx, tensor, kwargs, error, match):
    with pytest.raises(error, match=match):
        si.from_torch(tensor, device=ctx[0], **kwargs)


def test_to_torch_rejects_invalid_layout(ctx):
    image = si.from_numpy(_pixels(np.float32), device=ctx[0])
    with pytest.raises(ValueError, match="layout must be"):
        image.to_torch(layout="nchw")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA torch device")
def test_to_torch_on_cuda_device(ctx):
    image = si.from_numpy(_pixels(np.float16), device=ctx[0])
    tensor = image.to_torch(device="cuda")
    assert tensor.is_cuda and tensor.dtype == torch.float16
    np.testing.assert_array_equal(tensor.cpu().float().numpy(), image.to_numpy()[..., :3])
    back = si.from_torch(tensor, device=ctx[0])
    np.testing.assert_array_equal(back.to_numpy()[..., :3], image.to_numpy()[..., :3])
