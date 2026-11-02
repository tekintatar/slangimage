# tests/test_blend_and_point_ops.py
# Semantic tests for blend modes, smooth threshold, point ops and the LUT:
# each op is checked against a mathematical property or a NumPy reference,
# not just "no NaN".

from __future__ import annotations


import numpy as np
import pytest

from slangimage.runtime.textures import np_to_texture, texture_to_np
from slangimage.kernels.blend_modes import BlendModesOp, BLEND_MODES
from slangimage.kernels.smooth_threshold import SmoothThresholdOp
from slangimage.kernels.point_ops import PointOps
from slangimage.kernels.lut import LutOp, identity_lut_np

SIZE = 64
ATOL = 2e-3  # fp32 pipeline round-trips through 8-bit-free float textures




def rand_img(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = rng.random((SIZE, SIZE, 4)).astype(np.float32)
    img[..., 3] = 1.0
    return img


@pytest.fixture(scope="module")
def imgs(ctx):
    device, _ = ctx
    a_np, b_np = rand_img(1), rand_img(2)
    a = np_to_texture(device, a_np, usage="read", precision="fp32")
    b = np_to_texture(device, b_np, usage="read", precision="fp32")
    return a_np, b_np, a, b


# ------------------------------------------------------------- blend modes
def blend_ref(mode: str, b: np.ndarray, s: np.ndarray) -> np.ndarray:
    """NumPy reference for opaque inputs (reduces to plain formulas)."""
    if mode == "multiply":
        return b * s
    if mode == "screen":
        return b + s - b * s
    if mode == "overlay":
        return np.where(b <= 0.5, 2 * b * s, 1 - 2 * (1 - b) * (1 - s))
    if mode == "darken":
        return np.minimum(b, s)
    if mode == "lighten":
        return np.maximum(b, s)
    if mode == "color_dodge":
        return np.where(b <= 0, 0.0, np.where(s >= 1, 1.0, np.minimum(1.0, b / (1 - s))))
    if mode == "color_burn":
        return np.where(b >= 1, 1.0, np.where(s <= 0, 0.0, 1 - np.minimum(1.0, (1 - b) / np.maximum(s, 1e-9))))
    if mode == "hard_light":
        return blend_ref("overlay", s, b)
    if mode == "linear_light":
        return np.clip(b + 2 * s - 1, 0, 1)
    if mode == "vivid_light":
        return np.where(s <= 0.5, blend_ref("color_burn", b, 2 * s), blend_ref("color_dodge", b, 2 * (s - 0.5)))
    raise KeyError(mode)


NUMERIC_MODES = [
    "multiply", "screen", "overlay", "darken", "lighten",
    "color_dodge", "color_burn", "hard_light", "linear_light", "vivid_light",
]


@pytest.mark.parametrize("mode", NUMERIC_MODES)
def test_blend_matches_reference(ctx, imgs, mode):
    device, modules = ctx
    a_np, b_np, a, b = imgs
    op = BlendModesOp(device, modules)
    out = texture_to_np(op.apply(a, b, mode=mode))
    ref = np.clip(blend_ref(mode, a_np[..., :3], b_np[..., :3]), 0, 1)
    np.testing.assert_allclose(out[..., :3], ref, atol=ATOL)


def test_blend_soft_light_neutral_and_bounds(ctx, imgs):
    device, modules = ctx
    a_np, _, a, _ = imgs
    op = BlendModesOp(device, modules)
    # soft light with 0.5 gray is identity
    gray = np.full((SIZE, SIZE, 4), 0.5, dtype=np.float32)
    gray[..., 3] = 1.0
    g = np_to_texture(device, gray, usage="read", precision="fp32")
    out = texture_to_np(op.apply(a, g, mode="soft_light"))
    np.testing.assert_allclose(out[..., :3], a_np[..., :3], atol=ATOL)


def test_blend_color_mode_preserves_base_luminosity(ctx, imgs):
    device, modules = ctx
    a_np, b_np, a, b = imgs
    op = BlendModesOp(device, modules)
    out = texture_to_np(op.apply(a, b, mode="color"))
    def lum(x):
        return x[..., 0] * 0.3 + x[..., 1] * 0.59 + x[..., 2] * 0.11
    # luminosity of result == luminosity of base (up to clip-color adjustments)
    np.testing.assert_allclose(lum(out), lum(a_np), atol=2e-2)


def test_blend_opacity_zero_is_base(ctx, imgs):
    device, modules = ctx
    a_np, _, a, b = imgs
    op = BlendModesOp(device, modules)
    out = texture_to_np(op.apply(a, b, mode="multiply", opacity=0.0))
    np.testing.assert_allclose(out[..., :3], a_np[..., :3], atol=ATOL)


def test_blend_mode_names_cover_shader_switch(ctx):
    assert sorted(BLEND_MODES.values()) == list(range(12))


# --------------------------------------------------------- smooth threshold
def test_smooth_threshold_extremes_and_monotonic(ctx):
    device, modules = ctx
    ramp = np.zeros((1, 256, 4), dtype=np.float32)
    ramp[..., :3] = np.linspace(0, 1, 256, dtype=np.float32)[None, :, None]
    ramp[..., 3] = 1.0
    t = np_to_texture(device, ramp, usage="read", precision="fp32")
    op = SmoothThresholdOp(device, modules)

    out = texture_to_np(op.apply(t, threshold=0.5, smooth_amount=0.1))[0, :, 0]
    assert out[0] == 0.0 and out[-1] == 1.0
    assert (np.diff(out) >= -1e-6).all(), "not monotonic"
    # below t-s -> 0, above t+s -> 1
    assert out[np.linspace(0, 1, 256) < 0.39].max() == 0.0
    assert out[np.linspace(0, 1, 256) > 0.61].min() == 1.0

    out_bw = texture_to_np(op.apply_bw(t, 0.25, 0.75))[0, :, 0]
    x = np.linspace(0, 1, 256)
    ref = np.clip((x - 0.25) / 0.5, 0, 1)
    np.testing.assert_allclose(out_bw, ref, atol=5e-3)


# --------------------------------------------------------------- point ops
def test_invert_involution(ctx, imgs):
    device, modules = ctx
    a_np, _, a, _ = imgs
    ops = PointOps(device, modules)
    out = texture_to_np(ops.invert(ops.invert(a)))
    np.testing.assert_allclose(out[..., :3], a_np[..., :3], atol=ATOL)


def test_extract_channel_matches_numpy(ctx, imgs):
    device, modules = ctx
    a_np, _, a, _ = imgs
    ops = PointOps(device, modules)
    for i, ch in enumerate("rgb"):
        out = texture_to_np(ops.extract_channel(a, ch))
        for c in range(3):
            np.testing.assert_allclose(out[..., c], a_np[..., i], atol=ATOL)


def test_clamp_multiply_white_is_identity(ctx, imgs):
    device, modules = ctx
    a_np, _, a, _ = imgs
    ops = PointOps(device, modules)
    white = np.ones((SIZE, SIZE, 4), dtype=np.float32)
    w = np_to_texture(device, white, usage="read", precision="fp32")
    out = texture_to_np(ops.clamp_multiply(a, w))
    np.testing.assert_allclose(out[..., :3], a_np[..., :3], atol=ATOL)


def test_high_pass_contrast_self_is_gray(ctx, imgs):
    device, modules = ctx
    _, _, a, _ = imgs
    ops = PointOps(device, modules)
    out = texture_to_np(ops.high_pass_contrast(a, a, strength=2.0))
    np.testing.assert_allclose(out[..., :3], 0.5, atol=ATOL)


def test_random_noise_deterministic_and_bounded(ctx):
    device, modules = ctx
    ops = PointOps(device, modules)
    n1 = texture_to_np(ops.random_noise(128, 128, seed=3.0))
    n2 = texture_to_np(ops.random_noise(128, 128, seed=3.0))
    np.testing.assert_array_equal(n1, n2)
    # mix(0.5, noise, 0.9) -> [0.05, 0.95]
    assert n1.min() >= 0.05 - 1e-3 and n1[..., :3].max() <= 0.95 + 1e-3
    assert n1[..., 0].std() > 0.1, "noise has no variance"
    n3 = texture_to_np(ops.random_noise(128, 128, seed=4.0))
    assert np.abs(n3 - n1).max() > 0.1, "seed has no effect"


# --------------------------------------------------------------------- lut
def test_identity_lut_is_identity(ctx, imgs):
    device, modules = ctx
    a_np, _, a, _ = imgs
    lut_np = identity_lut_np()
    lut = np_to_texture(device, lut_np, usage="read", precision="fp32")
    op = LutOp(device, modules)
    out = texture_to_np(op.apply(a, lut))
    np.testing.assert_allclose(out[..., :3], a_np[..., :3], atol=1.5e-2)


def test_lut_intensity_zero_is_passthrough(ctx, imgs):
    device, modules = ctx
    a_np, _, a, _ = imgs
    lut_np = identity_lut_np()
    lut_np[..., :3] = 1.0 - lut_np[..., :3]  # some non-identity LUT
    lut = np_to_texture(device, lut_np, usage="read", precision="fp32")
    op = LutOp(device, modules)
    out = texture_to_np(op.apply(a, lut, intensity=0.0))
    np.testing.assert_allclose(out[..., :3], a_np[..., :3], atol=ATOL)


def test_inverting_lut_inverts(ctx, imgs):
    device, modules = ctx
    a_np, _, a, _ = imgs
    lut_np = identity_lut_np()
    lut_np[..., :3] = 1.0 - lut_np[..., :3]
    lut = np_to_texture(device, lut_np, usage="read", precision="fp32")
    op = LutOp(device, modules)
    out = texture_to_np(op.apply(a, lut))
    np.testing.assert_allclose(out[..., :3], 1.0 - a_np[..., :3], atol=1.5e-2)


# --------------------------------------------------- fp16 smoke for new ops
def test_new_ops_fp16_smoke(ctx):
    device, modules = ctx
    img = rand_img(9).astype(np.float16)
    a = np_to_texture(device, img, usage="read", precision="fp16")
    b = np_to_texture(device, rand_img(10).astype(np.float16), usage="read", precision="fp16")

    blend = BlendModesOp(device, modules)
    thresh = SmoothThresholdOp(device, modules)
    pops = PointOps(device, modules)
    lut = LutOp(device, modules)
    lut_tex = np_to_texture(device, identity_lut_np().astype(np.float16), usage="read", precision="fp16")

    outs = [
        blend.apply(a, b, mode="overlay"),
        thresh.apply(a),
        pops.invert(a),
        pops.extract_channel(a, "g"),
        pops.clamp_multiply(a, b),
        pops.high_pass_contrast(a, b, 1.5),
        pops.random_noise(SIZE, SIZE, seed=1.0, fmt=a.format),
        lut.apply(a, lut_tex),
    ]
    for out in outs:
        arr = texture_to_np(out)
        assert np.isfinite(arr).all()
        # unclipped readback; high_pass_contrast legitimately leaves [0,1]
        assert arr.min() >= -1.0 and arr.max() <= 2.0
