# tests/test_neighborhood_ops.py
# Property tests for the heavy neighborhood/stylize ops: bilateral, CLAHE,
# NLMeans, pointillize, CMYK halftone.

from __future__ import annotations


import numpy as np
import pytest

from slangimage.runtime.textures import np_to_texture, texture_to_np
from slangimage.kernels.bilateral import BilateralOp
from slangimage.kernels.clahe import ClaheOp
from slangimage.kernels.nl_means import NLMeansOp
from slangimage.kernels.stylize import PointillizeOp, CmykHalftoneOp

ATOL = 2e-3




def to_tex(device, arr):
    return np_to_texture(device, arr.astype(np.float32), usage="read", precision="fp32")


def step_image(size=96, noise=0.0, seed=0):
    """Vertical step edge: left 0.2, right 0.8 (+ optional noise)."""
    rng = np.random.default_rng(seed)
    img = np.full((size, size, 4), 0.2, dtype=np.float32)
    img[:, size // 2 :, :3] = 0.8
    if noise > 0:
        img[..., :3] += rng.normal(0, noise, (size, size, 3)).astype(np.float32)
    img[..., 3] = 1.0
    return np.clip(img, 0, 1)


# ---------------------------------------------------------------- bilateral
def test_bilateral_constant_is_identity(ctx):
    device, modules = ctx
    op = BilateralOp(device, modules)
    img = np.full((32, 32, 4), 0.42, dtype=np.float32)
    out = texture_to_np(op.apply(to_tex(device, img), radius=5, sigma_range=0.1))
    np.testing.assert_allclose(out[..., :3], 0.42, atol=ATOL)


@pytest.mark.parametrize("variant", ["full", "separable"])
def test_bilateral_denoises_but_keeps_edge(ctx, variant):
    device, modules = ctx
    op = BilateralOp(device, modules)
    img = step_image(noise=0.04, seed=3)
    t = to_tex(device, img)
    fn = op.apply if variant == "full" else op.apply_separable
    out = texture_to_np(fn(t, radius=5, sigma_spatial=2.5, sigma_range=0.15))

    # noise reduced on flat regions
    flat_in = img[8:40, 8:32, 0].std()
    flat_out = out[8:40, 8:32, 0].std()
    assert flat_out < flat_in * 0.5, f"noise not reduced: {flat_in} -> {flat_out}"

    # edge preserved: means of the two sides stay far apart
    left, right = out[:, : 96 // 2 - 4, 0].mean(), out[:, 96 // 2 + 4 :, 0].mean()
    assert right - left > 0.5, "edge was blurred away"


def test_bilateral_cutoff_mode_matches_reference_semantics(ctx):
    # box spatial + hard threshold: only colors within threshold participate
    device, modules = ctx
    op = BilateralOp(device, modules)
    img = step_image()
    out = texture_to_np(
        op.apply(to_tex(device, img), radius=7, sigma_spatial=0.0,
                 sigma_range=0.2, range_type="cutoff")
    )
    # 0.6 color distance across edge > 0.2 threshold: sides never mix
    np.testing.assert_allclose(out[:, : 96 // 2 - 8, 0], 0.2, atol=ATOL)
    np.testing.assert_allclose(out[:, 96 // 2 + 8 :, 0], 0.8, atol=ATOL)


# -------------------------------------------------------------------- clahe
def test_clahe_boosts_low_contrast(ctx):
    device, modules = ctx
    op = ClaheOp(device, modules)
    rng = np.random.default_rng(5)
    # low-contrast image compressed into [0.4, 0.6]
    img = np.zeros((128, 128, 4), dtype=np.float32)
    img[..., :3] = 0.4 + 0.2 * rng.random((128, 128, 1)).astype(np.float32)
    img[..., 3] = 1.0
    out = texture_to_np(op.apply(to_tex(device, img), clip_limit=4.0, tiles=(4, 4)))

    def lum(x):
        return x[..., :3].mean(axis=-1)
    assert lum(out).std() > lum(img).std() * 1.5, "contrast not increased"
    assert np.isfinite(out).all()
    assert out.min() >= 0.0 and out.max() <= 1.0 + 1e-3


def test_clahe_strength_zero_is_passthrough(ctx):
    device, modules = ctx
    op = ClaheOp(device, modules)
    rng = np.random.default_rng(6)
    img = rng.random((64, 64, 4)).astype(np.float32)
    img[..., 3] = 1.0
    out = texture_to_np(op.apply(to_tex(device, img), strength=0.0))
    np.testing.assert_allclose(out[..., :3], img[..., :3], atol=5e-3)


def test_clahe_clip_limits_amplification(ctx):
    # a nearly-flat tile: unclipped equalization would explode the tiny
    # variations; clipping must keep the output near the input
    device, modules = ctx
    op = ClaheOp(device, modules)
    rng = np.random.default_rng(7)
    img = np.zeros((128, 128, 4), dtype=np.float32)
    img[..., :3] = 0.5 + 0.01 * rng.standard_normal((128, 128, 1)).astype(np.float32)
    img[..., 3] = 1.0
    out_clipped = texture_to_np(op.apply(to_tex(device, img), clip_limit=1.0, tiles=(4, 4)))
    def lum(x):
        return x[..., :3].mean(axis=-1)
    assert lum(out_clipped).std() < 0.05, "clip_limit=1 should suppress amplification"


# ----------------------------------------------------------------- nlmeans
def test_nl_means_denoises_flat_and_keeps_edge(ctx):
    device, modules = ctx
    op = NLMeansOp(device, modules)
    img = step_image(noise=0.05, seed=8)
    out = texture_to_np(op.apply(to_tex(device, img), search_radius=4, patch_radius=1, h=0.15))

    flat_in = img[8:40, 8:32, 0].std()
    flat_out = out[8:40, 8:32, 0].std()
    assert flat_out < flat_in * 0.45, f"weak denoise: {flat_in} -> {flat_out}"

    left, right = out[:, : 96 // 2 - 4, 0].mean(), out[:, 96 // 2 + 4 :, 0].mean()
    assert right - left > 0.5, "edge destroyed"


def test_nl_means_small_h_near_identity(ctx):
    device, modules = ctx
    op = NLMeansOp(device, modules)
    rng = np.random.default_rng(9)
    img = rng.random((48, 48, 4)).astype(np.float32)
    img[..., 3] = 1.0
    out = texture_to_np(op.apply(to_tex(device, img), search_radius=3, h=0.005))
    assert np.abs(out[..., :3] - img[..., :3]).mean() < 0.01


# ------------------------------------------------------------------ stylize
def legacy_cmyk_halftone_reference(
    image,
    width=6.0,
    angle_deg=0.0,
    sharpness=0.08,
):
    """CPU reference for the normalized-CMY shader."""
    rgb = np.clip(image[..., :3], 0.0, 1.0)
    k = 1.0 - rgb.max(axis=-1)
    denom = np.maximum(1.0 - k, 1e-4)
    inks = (
        (1.0 - rgb[..., 0] - k) / denom,
        (1.0 - rgb[..., 1] - k) / denom,
        (1.0 - rgb[..., 2] - k) / denom,
        k,
    )

    height, width_px = image.shape[:2]
    yy, xx = np.mgrid[:height, :width_px].astype(np.float32)
    px = xx + np.float32(0.5)
    py = yy + np.float32(0.5)
    cell_width = np.float32(max(width, 1.0))
    aa = np.float32(max(sharpness, 1e-3))
    degrees = np.float32(3.14159265 / 180.0)
    base_angle = np.float32(angle_deg) * degrees

    def screen(value, channel_angle):
        angle = base_angle + np.float32(channel_angle) * degrees
        sin_angle = np.sin(angle)
        cos_angle = np.cos(angle)
        qx = (cos_angle * px - sin_angle * py) / cell_width
        qy = (sin_angle * px + cos_angle * py) / cell_width
        dx = qx - (np.floor(qx) + np.float32(0.5))
        dy = qy - (np.floor(qy) + np.float32(0.5))
        distance = np.sqrt(dx * dx + dy * dy)
        radius = np.sqrt(np.clip(value, 0.0, 1.0)) * np.float32(0.7071)
        t = np.clip((distance - (radius - aa)) / (2.0 * aa), 0.0, 1.0)
        return 1.0 - t * t * (3.0 - 2.0 * t)

    cyan = screen(inks[0], 15.0)
    magenta = screen(inks[1], 75.0)
    yellow = screen(inks[2], 0.0)
    black = screen(inks[3], 45.0)
    out = np.empty_like(image)
    out[..., 0] = (1.0 - cyan) * (1.0 - black)
    out[..., 1] = (1.0 - magenta) * (1.0 - black)
    out[..., 2] = (1.0 - yellow) * (1.0 - black)
    out[..., 3] = image[..., 3]
    return out


def test_pointillize_deterministic_and_quantizes(ctx):
    device, modules = ctx
    op = PointillizeOp(device, modules)
    rng = np.random.default_rng(10)
    img = rng.random((128, 128, 4)).astype(np.float32)
    img[..., 3] = 1.0
    t = to_tex(device, img)

    o1 = texture_to_np(op.apply(t, radius=12.0, seed=1.0))
    o2 = texture_to_np(op.apply(t, radius=12.0, seed=1.0))
    np.testing.assert_array_equal(o1, o2)

    # cells collapse colors: far fewer unique values than the input
    uniq = np.unique(o1[..., 0].round(4)).size
    assert uniq < 128 * 128 / 32, f"too many unique colors: {uniq}"

    o3 = texture_to_np(op.apply(t, radius=12.0, seed=2.0))
    assert np.abs(o3 - o1).max() > 0.05, "seed has no effect"


def test_cmyk_halftone_extremes(ctx):
    device, modules = ctx
    op = CmykHalftoneOp(device, modules)

    white = np.ones((64, 64, 4), dtype=np.float32)
    out_w = texture_to_np(op.apply(to_tex(device, white), width=6.0))
    assert out_w[..., :3].mean() > 0.95, "white should stay paper-white"

    black = np.zeros((64, 64, 4), dtype=np.float32)
    black[..., 3] = 1.0
    out_b = texture_to_np(op.apply(to_tex(device, black), width=6.0))
    assert out_b[..., :3].mean() < 0.2, "black should be mostly ink"

    gray = np.full((64, 64, 4), 0.5, dtype=np.float32)
    gray[..., 3] = 1.0
    out_g = texture_to_np(op.apply(to_tex(device, gray), width=6.0))
    m = out_g[..., :3].mean()
    assert 0.2 < m < 0.85, f"mid gray coverage off: {m}"
    # screening creates structure: both ink and paper present
    assert out_g[..., :3].std() > 0.1


def test_cmyk_halftone_matches_normalized_cmy_reference(ctx):
    device, modules = ctx
    op = CmykHalftoneOp(device, modules)
    img = np.empty((64, 64, 4), dtype=np.float32)
    img[...] = (0.2, 0.5, 0.8, 1.0)
    src = to_tex(device, img)

    output = texture_to_np(op.apply(src, width=7.0, angle_deg=11.0))
    expected = legacy_cmyk_halftone_reference(img, width=7.0, angle_deg=11.0)
    np.testing.assert_allclose(output, expected, rtol=0.0, atol=2e-4)


# ------------------------------------------------------- fp16 smoke for all
def test_heavy_ops_fp16_smoke(ctx):
    device, modules = ctx
    rng = np.random.default_rng(11)
    img = rng.random((96, 96, 4)).astype(np.float16)
    img[..., 3] = 1.0
    t = np_to_texture(device, img, usage="read", precision="fp16")

    outs = [
        BilateralOp(device, modules).apply(t, radius=4),
        BilateralOp(device, modules).apply_separable(t, radius=4),
        ClaheOp(device, modules).apply(t, tiles=(4, 4)),
        NLMeansOp(device, modules).apply(t, search_radius=3),
        PointillizeOp(device, modules).apply(t, radius=8.0),
        CmykHalftoneOp(device, modules).apply(t),
    ]
    for out in outs:
        arr = texture_to_np(out)
        assert np.isfinite(arr).all()
        assert arr.min() >= -1.0 and arr.max() <= 2.0  # unclipped readback
