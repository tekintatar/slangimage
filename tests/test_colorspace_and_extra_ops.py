# tests/test_colorspace_and_extra_ops.py
# Dedicated property tests for ops that were previously exercised only by the
# prewarm() smoke pass: sRGB<->linear roundtrip, color_controls identity,
# smooth_threshold_bw ramp, ep_unsharp_mask.

from __future__ import annotations

import numpy as np
import pytest

import slangimage as si

ATOL = 3e-3


def _rand(device, seed=7, size=64, lo=0.05, hi=0.95):
    rng = np.random.default_rng(seed)
    a = rng.random((size, size, 4)).astype(np.float32) * (hi - lo) + lo
    a[..., 3] = 1.0
    return a, si.from_numpy(a, device=device)


def _gradient(device, size=64):
    g = np.linspace(0.0, 1.0, size, dtype=np.float32)
    a = np.empty((size, size, 4), dtype=np.float32)
    a[..., 0] = a[..., 1] = a[..., 2] = np.tile(g, (size, 1))  # left->right ramp
    a[..., 3] = 1.0
    return g, a, si.from_numpy(a, device=device)


# --------------------------------------------- sRGB <-> linear are inverses
def test_srgb_linear_roundtrip(ctx):
    device, _ = ctx
    a, im = _rand(device)
    back = si.ops.linear_to_srgb(si.ops.srgb_to_linear(im)).to_numpy()
    np.testing.assert_allclose(back[..., :3], a[..., :3], atol=ATOL)
    back2 = si.ops.srgb_to_linear(si.ops.linear_to_srgb(im)).to_numpy()
    np.testing.assert_allclose(back2[..., :3], a[..., :3], atol=ATOL)
    # decoding to linear darkens midtones (linear <= srgb on [0,1])
    lin = si.ops.srgb_to_linear(im).to_numpy()
    assert (lin[..., :3] <= a[..., :3] + 1e-3).all()


# --------------------------------------------- color_controls neutral = identity
def test_color_controls_identity(ctx):
    device, _ = ctx
    # neutral controls are identity on LOW-CHROMA pixels; saturated pixels get
    # chroma-limited (max_chroma ~0.6) even at neutral, so test near gray — the
    # same restriction the hue test documents.
    rng = np.random.default_rng(3)
    a = np.full((64, 64, 4), 0.5, dtype=np.float32)
    a[..., :3] += (rng.random((64, 64, 3)).astype(np.float32) - 0.5) * 0.2
    a[..., 3] = 1.0
    im = si.from_numpy(a, device=device)
    out = si.ops.color_controls(
        im, hue=0.0, saturation=0.0, contrast=0.0, brightness=0.0
    ).to_numpy()
    np.testing.assert_allclose(out[..., :3], a[..., :3], atol=ATOL)


# --------------------------------------------- smooth_threshold_bw ramp
def test_smooth_threshold_bw_ramp(ctx):
    device, _ = ctx
    g, _, im = _gradient(device)
    out = si.ops.smooth_threshold_bw(
        im, black_threshold=0.3, white_threshold=0.7
    ).to_numpy()
    lum = out[..., :3].mean(-1)
    assert lum[:, g < 0.25].max() < 0.15   # below black -> ~black
    assert lum[:, g > 0.75].min() > 0.85   # above white -> ~white
    assert (np.diff(lum[32]) >= -1e-3).all()  # monotone non-decreasing


def test_smooth_threshold_bw_preserves_alpha_and_validates_range(ctx):
    device, _ = ctx
    a = np.full((8, 8, 4), 0.5, dtype=np.float32)
    a[..., 3] = 0.25
    image = si.from_numpy(a, device=device)
    out = si.ops.smooth_threshold_bw(image, 0.2, 0.8).to_numpy()
    np.testing.assert_allclose(out[..., 3], 0.25, atol=1e-6)

    for black, white in ((0.5, 0.5), (0.7, 0.3)):
        with pytest.raises(ValueError, match="white_threshold"):
            si.ops.smooth_threshold_bw(image, black, white)


# --------------------------------------------- ep_unsharp_mask (surface-blur path)
def test_ep_unsharp_mask_finite_and_active(ctx):
    device, _ = ctx
    # edge-preserving unsharp leaves piecewise-constant regions alone, so probe
    # it on sub-bsigma texture (fine noise) where it demonstrably sharpens.
    a, im = _rand(device, seed=5, lo=0.0, hi=1.0)
    out = si.ops.ep_unsharp_mask(im, amount=1.0).to_numpy()
    assert np.isfinite(out).all()
    assert np.abs(out[..., :3] - a[..., :3]).max() > 1e-3  # actually modifies
