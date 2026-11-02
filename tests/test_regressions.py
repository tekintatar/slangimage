# tests/test_regressions.py
# Regression tests for previously fixed defects. These run on any backend;
# the concurrency/CUDA-specific items are covered elsewhere.

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

import slangimage as si


def _img(device, seed=1, alpha=1.0):
    a = np.random.default_rng(seed).random((32, 32, 4)).astype(np.float32)
    a[..., 3] = alpha
    return a, si.from_numpy(a, device=device)


# Input boundary: from_numpy channel, dtype and memory-layout handling
def test_from_numpy_rejects_one_channel(ctx):
    device, _ = ctx
    with pytest.raises(ValueError):
        si.from_numpy(np.random.rand(8, 8, 1).astype(np.float32), device=device)


def test_from_numpy_rejects_integer(ctx):
    device, _ = ctx
    with pytest.raises(ValueError):
        si.from_numpy((np.random.rand(8, 8, 3) * 255).astype(np.uint8), device=device)


def test_from_numpy_accepts_noncontiguous(ctx):
    device, _ = ctx
    view = np.random.rand(8, 8, 3).astype(np.float32).transpose(1, 0, 2)  # non-contiguous
    assert si.from_numpy(view, device=device).to_numpy().shape == (8, 8, 4)


# Cross-device guards
def test_two_input_op_rejects_cross_device(ctx):
    import slangpy as spy

    current, _ = ctx
    backend = {
        spy.DeviceType.metal: si.Backend.METAL,
        spy.DeviceType.cuda: si.Backend.CUDA,
        spy.DeviceType.vulkan: si.Backend.VULKAN,
    }[current.info.type]
    d1 = si.DeviceManager(backend).device
    d2 = si.DeviceManager(backend).device
    a = np.full((4, 4, 4), 0.5, np.float32)
    a[..., 3] = 1.0
    i1, i2 = si.from_numpy(a, device=d1), si.from_numpy(a, device=d2)
    try:
        with pytest.raises(ValueError, match="same device"):
            si.ops.clamp_multiply(i1, i2)

        from slangimage.fx import GaussianBlurEffect

        modules1 = si.ModuleCache(d1)
        effect = GaussianBlurEffect(d1, modules1)
        with pytest.raises(ValueError, match="same device"):
            effect.apply(i2)
        with pytest.raises(ValueError, match="module-cache device"):
            GaussianBlurEffect(d1, si.ModuleCache(d2))
    finally:
        si.release_device(d1)
        si.release_device(d2)


# Releasing a device removes all package-owned lifecycle roots
@pytest.mark.auto_device
def test_release_device_clears_manager_and_pool_state():
    import slangpy as spy
    from slangimage.runtime.device import _default_managers

    manager = si.DeviceManager(si.Backend.AUTO)
    device = manager.device
    old_pool = manager.texture_pool
    tex = old_pool.alloc(4, 4, spy.Format.rgba32_float)
    old_pool.recycle(tex)
    assert old_pool.stats()["free"] == 1

    # CPU cannot create a manager, so this key is safe for an isolated cache
    # entry that exercises removal without disturbing the session fixture.
    _default_managers[si.Backend.CPU] = manager
    try:
        si.release_device(device)
        assert si.Backend.CPU not in _default_managers
        assert old_pool.stats()["free"] == 0
        assert manager.texture_pool is not old_pool
        assert callable(si.release_device)
    finally:
        _default_managers.pop(si.Backend.CPU, None)
        si.release_device(device)


# CPU device rejected
def test_cpu_backend_rejected():
    with pytest.raises(RuntimeError):
        si.get_default_device("cpu")


# Alpha preserved
def test_invert_and_extract_preserve_alpha(ctx):
    device, _ = ctx
    _, im = _img(device, alpha=0.5)
    assert abs(float(si.ops.invert(im).to_numpy()[0, 0, 3]) - 0.5) < 2e-3
    assert abs(float(si.ops.extract_channel(im).to_numpy()[0, 0, 3]) - 0.5) < 2e-3


# JPEG save from RGBA
def test_jpeg_save_from_rgba(ctx):
    device, _ = ctx
    _, im = _img(device)
    p = os.path.join(tempfile.mkdtemp(), "o.jpg")
    im.save(p)
    assert os.path.getsize(p) > 0


# ep_unsharp radius/bsigma affect the result
def test_ep_unsharp_params_have_effect(ctx):
    device, _ = ctx
    _, im = _img(device, seed=5)
    o1 = si.ops.ep_unsharp_mask(im, radius=1, bsigma=0.05).to_numpy()
    o2 = si.ops.ep_unsharp_mask(im, radius=6, bsigma=0.30).to_numpy()
    assert np.abs(o1[..., :3] - o2[..., :3]).max() > 1e-3


# Public contrast is monotonic
def test_color_controls_contrast_monotonic(ctx):
    device, _ = ctx
    _, im = _img(device)
    base = si.ops.color_controls(im).to_numpy()[..., :3].std()
    more = si.ops.color_controls(im, contrast=0.8).to_numpy()[..., :3].std()
    less = si.ops.color_controls(im, contrast=-0.8).to_numpy()[..., :3].std()
    assert more > base > less


# Parameter validation
@pytest.mark.parametrize("call", [
    lambda im: si.ops.gaussian_blur(im, radius=5, sigma=0),
    lambda im: si.ops.gaussian_blur(im, radius=-1, sigma=1),
    lambda im: si.ops.kuwahara(im, radius=0),
    lambda im: si.ops.unsharp_mask(im, color_space="nope"),
    lambda im: si.ops.clahe(im, tiles=(0, 8)),
    lambda im: si.ops.resize(im, mode="typo"),
    lambda im: si.ops.resize(im, scale=-1),
    lambda im: si.ops.random_noise(8, 8, precision="wat"),
])
def test_invalid_params_raise(ctx, call):
    device, _ = ctx
    _, im = _img(device)
    with pytest.raises(ValueError):
        call(im)


def test_resize_tiny_never_degenerate(ctx):
    device, _ = ctx
    tiny = si.from_numpy(np.random.rand(1, 1, 4).astype(np.float32), device=device)
    assert si.ops.resize(tiny, scale=0.5).to_numpy().shape[:2] == (1, 1)


# Only operators whose kernels require aligned coordinates reject
# mixed sizes; blend/clamp retain their documented clamped-sampling behavior.
def test_two_input_spatial_contracts(ctx):
    device, _ = ctx
    large = si.from_numpy(np.full((4, 4, 4), 0.5, np.float32), device=device)
    small = si.from_numpy(np.full((2, 2, 4), 0.25, np.float32), device=device)

    with pytest.raises(ValueError, match="high_pass_contrast.*matching"):
        si.ops.high_pass_contrast(large, small)

    assert si.ops.blend(large, small).to_numpy().shape[:2] == (4, 4)
    assert si.ops.clamp_multiply(large, small).to_numpy().shape[:2] == (4, 4)


# Every public texture entry point enforces RGBA readable storage.
def test_raw_texture_contract(ctx):
    import slangpy as spy
    from slangimage.fx import GaussianBlurEffect

    device, modules = ctx
    with pytest.raises(TypeError, match="slangpy Texture"):
        si.Image(object())

    scalar = device.create_texture(
        width=4,
        height=4,
        format=spy.Format.r32_float,
        usage=spy.TextureUsage.shader_resource,
    )
    for call in (
        lambda: si.Image(scalar),
        lambda: si.ops.invert(scalar),
        lambda: GaussianBlurEffect(device, modules).apply(scalar),
    ):
        with pytest.raises(ValueError, match="rgba16_float or rgba32_float"):
            call()

    write_only = device.create_texture(
        width=4,
        height=4,
        format=spy.Format.rgba32_float,
        usage=spy.TextureUsage.unordered_access,
    )
    with pytest.raises(ValueError, match="shader_resource"):
        si.Image(write_only)


# Exported effects have explicit, validated parameter contracts.
def test_effect_parameter_contracts(ctx):
    import inspect

    from slangimage.fx import (
        GaussianBlurEffect,
        KuwaharaPaintEffect,
        SoftGlowEffect,
    )

    device, modules = ctx
    _, image = _img(device, seed=14)
    gaussian = GaussianBlurEffect(device, modules)
    soft_glow = SoftGlowEffect(device, modules)

    for effect_type in (
        GaussianBlurEffect,
        SoftGlowEffect,
        KuwaharaPaintEffect,
    ):
        parameters = inspect.signature(effect_type.apply).parameters.values()
        assert all(param.kind is not inspect.Parameter.VAR_KEYWORD for param in parameters)
    assert "blur_amount" in inspect.signature(gaussian.apply).parameters
    with pytest.raises(TypeError, match="sigam"):
        gaussian.apply(image, sigam=0.1)
    with pytest.raises(ValueError, match="provided together"):
        gaussian.apply(image, radius=2)
    with pytest.raises(ValueError, match="radius"):
        gaussian.apply(image, radius=-1, sigma=1.0)
    with pytest.raises(ValueError, match="not both"):
        gaussian.apply(image, blur_amount=2.0, radius=2, sigma=1.0)
    with pytest.raises(ValueError, match="provided together"):
        soft_glow.apply(image, sigma=1.0)

    effect_out = gaussian.apply(image, blur_amount=2.0).to_numpy()
    functional_out = si.ops.gaussian_blur(image, amount=2.0).to_numpy()
    np.testing.assert_allclose(effect_out, functional_out, atol=0, rtol=0)


# Every LUT consumer shares the generalized 8x8-grid contract.
def test_functional_lut_contract(ctx):
    device, _ = ctx
    _, image = _img(device, seed=15)

    for shape in ((8, 8), (10, 10), (16, 24)):
        h, w = shape
        bad_lut = si.from_numpy(np.zeros((h, w, 4), np.float32), device=device)
        with pytest.raises(ValueError, match="square, divisible by 8"):
            si.ops.lut(image, bad_lut)

    valid_lut = si.from_numpy(si.ops.identity_lut_np(16), device=device)
    out = si.ops.lut(image, valid_lut).to_numpy()
    np.testing.assert_allclose(out[..., :3], image.to_numpy()[..., :3], atol=2e-2)


def test_identity_lut_size_validation():
    from slangimage.ops import identity_lut_np
    with pytest.raises(ValueError):
        identity_lut_np(7)     # not a multiple of 8
    with pytest.raises(ValueError):
        identity_lut_np(8)     # cell size 1 -> divide-by-zero ramp
    assert identity_lut_np(16).shape == (16, 16, 4)


# Deliberate public namespaces
def test_fx_exports_effects():
    from slangimage.fx import GaussianBlurEffect, KuwaharaPaintEffect, SoftGlowEffect
    assert all(x is not None for x in
               (GaussianBlurEffect, KuwaharaPaintEffect, SoftGlowEffect))


def test_ops_all_is_clean():
    import slangimage.ops as o
    assert "gaussian_blur" in o.__all__
    for leaked in ("Any", "Optional", "spy", "weakref", "GaussianBlurOp"):
        assert leaked not in o.__all__


# EXIF orientation honored + 16-bit input not clipped
def test_exif_orientation_and_16bit():
    import io
    from PIL import Image
    from slangimage.runtime.textures import pil_to_np_rgba32
    img = Image.new("RGB", (2, 3), (255, 0, 0))
    exif = img.getexif()
    exif[0x0112] = 6                       # orientation 6 = rotate
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    buf.seek(0)
    arr = pil_to_np_rgba32(Image.open(buf))
    assert arr.shape[:2] == (2, 3)         # 3x2 stored, rotated to 2x3 (HxW)
    i16 = pil_to_np_rgba32(Image.new("I;16", (4, 4), 40000))
    assert abs(float(i16[0, 0, 0]) - 40000 / 65535) < 1e-2   # not clipped to 1.0


# Separable lanczos3 resize stays correct and anti-aliased
@pytest.mark.parametrize("wh", [(32, 32), (8, 8), (48, 20), (128, 128)])
def test_lanczos3_downscale_constant_preserved(ctx, wh):
    device, _ = ctx
    w, h = wh
    a = np.full((64, 64, 4), 0.37, np.float32)
    a[..., 3] = 1.0
    out = si.ops.resize(si.from_numpy(a, device=device), width=w, height=h,
                        mode="lanczos3").to_numpy()
    assert out.shape[:2] == (h, w)
    assert np.abs(out[..., :3] - 0.37).max() < 5e-3   # flat regions stay flat


def test_lanczos3_downscale_finite_and_energy(ctx):
    device, _ = ctx
    a = np.random.default_rng(7).random((64, 64, 4)).astype(np.float32)
    a[..., 3] = 1.0
    out = si.ops.resize(si.from_numpy(a, device=device), width=16, height=16,
                        mode="lanczos3").to_numpy()
    assert np.isfinite(out).all()
    assert abs(out[..., :3].mean() - a[..., :3].mean()) < 0.02  # ~energy preserving


# Kuwahara effect preserves constant colors (smoothing invariant)
@pytest.mark.parametrize("rgb", [(0.2, 0.8, 0.4), (0.0, 1.0, 1.0), (1.0, 0.0, 0.0)])
def test_kuwahara_effect_preserves_constant_color(ctx, rgb):
    device, modules = ctx
    from slangimage.fx import KuwaharaPaintEffect
    eff = KuwaharaPaintEffect(device, modules)
    a = np.zeros((32, 32, 4), np.float32)
    a[..., :3] = rgb
    a[..., 3] = 1.0
    out = eff.apply(si.from_numpy(a, device=device)).to_numpy()
    assert np.abs(out[..., :3] - np.array(rgb)).max() < 1.5e-2


# Repeated composite ops don't churn the pool (scratch is recycled)
def test_pool_steady_state_no_churn(ctx):
    import gc
    from slangimage.runtime.texture_pool import get_pool
    device, _ = ctx
    pool = get_pool(device)
    _, im = _img(device, seed=2)
    for _ in range(3):        # warm up: first calls legitimately allocate
        si.ops.unsharp_mask(im)
    gc.collect()
    base = pool.stats()["created"]
    for _ in range(8):
        si.ops.unsharp_mask(im)   # composite op (gaussian scratch + combine)
        gc.collect()
    # was +8 (one new texture per call) before the frame/wrap fix
    assert pool.stats()["created"] - base <= 1


# Benchmark reporter rejects incompatible protocols
def _bench_json(protocol, backend, iters=15, slangimage="0.1", slangpy="0.42",
                op="invert", size=64):
    d = {"meta": {"backend": backend, "protocol": protocol, "slangpy": slangpy,
                  "slangimage": slangimage, "platform": "x", "iters": iters,
                  "timestamp": "t", "adapter": "a"},
         "records": [{"backend": backend, "op": op, "size": size,
                      "precision": "fp16", "steady_ms": 1.0, "iters": iters}]}
    p = os.path.join(tempfile.mkdtemp(), f"{backend}.json")
    with open(p, "w") as f:
        json.dump(d, f)
    return p


def test_benchmark_rejects_mixed_protocol():
    from slangimage.bench.report import render
    with pytest.raises(ValueError):
        render([_bench_json(1, "cuda"), _bench_json(2, "vulkan")])
    # same protocol renders fine
    assert render([_bench_json(2, "cuda"), _bench_json(2, "vulkan")]).startswith("#")


def test_benchmark_rejects_duplicate_backend_before_merging():
    from slangimage.bench.report import render

    paths = [_bench_json(2, "cuda", op="invert"),
             _bench_json(2, "cuda", op="gaussian")]
    with pytest.raises(ValueError, match="more than one file.*cuda"):
        render(paths)
    assert render(paths, strict=False).startswith("#")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"slangimage": "0.2"}, "SlangImage versions"),
        ({"slangpy": "0.43"}, "SlangPy versions"),
        ({"iters": 20}, "iteration counts"),
        ({"size": 128}, "suite sizes"),
        ({"op": "gaussian"}, "suite ops"),
    ],
)
def test_benchmark_rejects_other_incompatible_metadata(overrides, message):
    from slangimage.bench.report import render

    first = _bench_json(2, "metal")
    second = _bench_json(2, "cuda", **overrides)
    with pytest.raises(ValueError, match=message):
        render([first, second])
