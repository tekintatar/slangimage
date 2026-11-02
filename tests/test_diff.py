# tests/test_diff.py
# Differentiable ops (slangimage.diff): forward vs numpy references,
# gradients vs closed form / explicit fp32 finite differences.
# Requires torch + CUDA — skipped on machines without them (e.g. Metal dev
# machines); runs on a CUDA host, e.g. via scripts/cuda_box/run_tests.sh.

from __future__ import annotations

import os

import numpy as np
import pytest

if os.environ.get("SLANGIMAGE_BACKEND", "auto") not in {"auto", "cuda"}:
    pytest.skip("CUDA autograd tests excluded by SLANGIMAGE_BACKEND", allow_module_level=True)

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("slangimage.diff requires CUDA", allow_module_level=True)

import slangimage.diff as sid  # noqa: E402
from _fp32_gradcheck import assert_fp32_finite_differences  # noqa: E402

R, SIGMA = 5, 2.0


def np_gaussian_hwc(img: np.ndarray, radius: int, sigma: float) -> np.ndarray:
    dx = np.arange(-radius, radius + 1)
    w = np.exp(-(dx * dx) / (2.0 * sigma * sigma))
    w /= w.sum()

    def conv(a, axis):
        pad_width = [(0, 0)] * a.ndim
        pad_width[axis] = (radius, radius)
        ap = np.pad(a, pad_width, mode="edge")
        out = np.zeros_like(a)
        for i, wi in enumerate(w):
            sl = [slice(None)] * a.ndim
            sl[axis] = slice(i, i + a.shape[axis])
            out += wi * ap[tuple(sl)]
        return out

    return conv(conv(img, 1), 0)


@pytest.fixture(scope="module")
def img_np():
    rng = np.random.default_rng(17)
    return rng.random((48, 40, 3)).astype(np.float32)


@pytest.fixture(scope="module")
def img(img_np):
    return torch.tensor(img_np, device="cuda")


# ---------------------------------------------------------------- gaussian
def test_gaussian_forward_matches_numpy(img, img_np):
    out = sid.gaussian_blur(img, radius=R, sigma=SIGMA)
    ref = np_gaussian_hwc(img_np.astype(np.float64), R, SIGMA)
    err = np.abs(out.cpu().numpy() - ref).max()
    assert err < 1e-5, f"forward mismatch: {err}"


def test_gaussian_backward_closed_form(img_np):
    src = torch.tensor(img_np, device="cuda", requires_grad=True)
    out = sid.gaussian_blur(src, radius=R, sigma=SIGMA)

    rng = np.random.default_rng(18)
    wgt_np = rng.random(img_np.shape).astype(np.float32)
    (out * torch.tensor(wgt_np, device="cuda")).sum().backward()
    grad = src.grad.cpu().numpy()

    # interior: transpose of a symmetric separable blur == the same blur
    ref = np_gaussian_hwc(wgt_np.astype(np.float64), R, SIGMA)
    m = R + 1
    err = np.abs(grad[m:-m, m:-m] - ref[m:-m, m:-m]).max()
    assert err < 1e-4, f"backward interior mismatch: {err}"


def test_gaussian_fp32_finite_differences():
    small = torch.rand(10, 9, 2, device="cuda", requires_grad=True)
    assert_fp32_finite_differences(
        lambda t: sid.gaussian_blur(t, radius=3, sigma=1.5),
        (small,),
    )


def test_gaussian_2d_and_channels(img):
    gray = img[..., 0]
    out2d = sid.gaussian_blur(gray, radius=R, sigma=SIGMA)
    assert out2d.shape == gray.shape
    out3d = sid.gaussian_blur(img, radius=R, sigma=SIGMA)
    torch.testing.assert_close(out2d, out3d[..., 0], atol=1e-6, rtol=0)


@pytest.mark.parametrize("kwargs", [
    {"radius": 0, "sigma": 1.0},
    {"radius": -1, "sigma": 1.0},
    {"radius": 2, "sigma": 0.0},
])
def test_gaussian_invalid_parameters_raise(img, kwargs):
    with pytest.raises(ValueError):
        sid.gaussian_blur(img, **kwargs)


# --------------------------------------------------------------- color ops
def test_exposure_forward_and_grad(img):
    src = img.clone().requires_grad_(True)
    out = sid.exposure(src, ev=1.5)
    torch.testing.assert_close(out, src.detach() * 2.0**1.5, atol=1e-5, rtol=0)
    out.sum().backward()
    # d(out)/d(src) = 2^ev everywhere
    torch.testing.assert_close(
        src.grad, torch.full_like(src, 2.0**1.5), atol=1e-5, rtol=0
    )


def test_sepia_forward_matches_numpy(img, img_np):
    out = sid.sepia(img, intensity=0.7).cpu().numpy()
    m = np.array([[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]])
    sep = img_np @ m.T  # unclamped by design
    ref = img_np + 0.7 * (sep - img_np)
    assert np.abs(out - ref).max() < 1e-5


def test_sepia_fp32_finite_differences():
    small = torch.rand(6, 7, 3, device="cuda", requires_grad=True)
    assert_fp32_finite_differences(
        lambda t: sid.sepia(t, intensity=0.5),
        (small,),
    )


def test_hue_rotate_luma_exact_and_fp32_finite_differences(img, img_np):
    out = sid.hue_rotate(img, angle=1.1).cpu().numpy()
    lum = np.array([0.299, 0.587, 0.114])
    # unclamped -> luminance preserved to the precision of the (3-decimal)
    # YIQ matrix constants (~1e-4 residual per unit chroma)
    np.testing.assert_allclose(out @ lum, img_np @ lum, atol=2e-4)

    small = torch.rand(6, 6, 3, device="cuda", requires_grad=True)
    assert_fp32_finite_differences(
        lambda t: sid.hue_rotate(t, angle=0.7),
        (small,),
    )


# ----------------------------------------------------------------- chains
def test_chained_ops_backward():
    src = torch.rand(24, 24, 3, device="cuda", requires_grad=True)
    out = sid.sepia(sid.gaussian_blur(sid.exposure(src, 0.5), radius=3, sigma=1.5), 0.6)
    out.mean().backward()
    g = src.grad
    assert g is not None and torch.isfinite(g).all()
    assert float(g.abs().sum()) > 0


# --------------------------------------------------------------- bilateral
def np_bilateral(img, radius, ss, sr):
    # border: out-of-bounds taps skipped (truncated window), matching the
    # kernel — that symmetry is what enables its gather backward
    H, W, _ = img.shape
    out = np.zeros_like(img, dtype=np.float64)
    for y in range(H):
        for x in range(W):
            c = img[y, x].astype(np.float64)
            acc = np.zeros(3)
            wsum = 0.0
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    yy, xx = y + dy, x + dx
                    if not (0 <= yy < H and 0 <= xx < W):
                        continue
                    s = img[yy, xx].astype(np.float64)
                    d = s - c
                    w = np.exp(-(dx * dx + dy * dy) / (2 * ss * ss)) * np.exp(-(d @ d) / (2 * sr * sr))
                    acc += w * s
                    wsum += w
            out[y, x] = acc / wsum
    return out


def test_bilateral_forward_matches_numpy():
    rng = np.random.default_rng(23)
    img_np = rng.random((14, 12, 3)).astype(np.float32)
    img_t = torch.tensor(img_np, device="cuda")
    out = sid.bilateral(img_t, radius=3, sigma_spatial=1.5, sigma_range=0.3)
    ref = np_bilateral(img_np, 3, 1.5, 0.3)
    assert np.abs(out.cpu().numpy() - ref).max() < 1e-5


def test_bilateral_fp32_finite_differences():
    # The gather backward is deterministic (no atomics).
    small = torch.rand(8, 7, 3, device="cuda", requires_grad=True)
    assert_fp32_finite_differences(
        lambda t: sid.bilateral(t, radius=2, sigma_spatial=1.5, sigma_range=0.5),
        (small,),
    )


def test_bilateral_in_chain():
    # the manual autograd.Function bridge must compose with the automatic
    # slangpy-autograd ops on both sides
    src = torch.rand(20, 18, 3, device="cuda", requires_grad=True)
    out = sid.sepia(sid.bilateral(sid.exposure(src, 0.5), radius=2), 0.6)
    out.mean().backward()
    g = src.grad
    assert g is not None and torch.isfinite(g).all()
    assert float(g.abs().sum()) > 0


@pytest.mark.parametrize("kwargs", [
    {"radius": 0},
    {"radius": -1},
    {"sigma_spatial": 0.0},
    {"sigma_range": 0.0},
])
def test_bilateral_invalid_parameters_raise(img, kwargs):
    with pytest.raises(ValueError):
        sid.bilateral(img, **kwargs)


# ------------------------------------------------------------------- blend
def test_blend_forward_and_fp32_finite_differences():
    rng = np.random.default_rng(24)
    a_np = rng.random((10, 11, 3)).astype(np.float32)
    b_np = rng.random((10, 11, 3)).astype(np.float32)
    a = torch.tensor(a_np, device="cuda")
    b = torch.tensor(b_np, device="cuda")

    out = sid.blend(a, b, mode="multiply").cpu().numpy()
    assert np.abs(out - a_np * b_np).max() < 1e-6
    out = sid.blend(a, b, mode="screen").cpu().numpy()
    assert np.abs(out - (a_np + b_np - a_np * b_np)).max() < 1e-6
    out = sid.blend(a, b, mode="overlay", opacity=0.7).cpu().numpy()
    ov = np.where(a_np <= 0.5, 2 * a_np * b_np, 1 - 2 * (1 - a_np) * (1 - b_np))
    assert np.abs(out - (a_np + 0.7 * (ov - a_np))).max() < 1e-6

    # Check both inputs, biased off the overlay 0.5 branch point.
    ga = (torch.rand(6, 5, 3, device="cuda") * 0.4 + 0.05).requires_grad_(True)
    gb = torch.rand(6, 5, 3, device="cuda", requires_grad=True)
    assert_fp32_finite_differences(
        lambda x, y: sid.blend(x, y, mode="overlay"),
        (ga, gb),
    )


# ----------------------------------------------------------------- unsharp
def test_unsharp_forward_matches_numpy(img, img_np):
    out = sid.unsharp_mask(img, radius=3, sigma=1.5, amount=0.8)
    blur = np_gaussian_hwc(img_np.astype(np.float64), 3, 1.5)
    ref = img_np + 0.8 * (img_np - blur)
    assert np.abs(out.cpu().numpy() - ref).max() < 1e-4


def test_unsharp_fp32_finite_differences():
    small = torch.rand(9, 8, 3, device="cuda", requires_grad=True)
    assert_fp32_finite_differences(
        lambda t: sid.unsharp_mask(t, radius=2, sigma=1.2, amount=1.5),
        (small,),
    )


# ------------------------------------------------------------------ resize
@pytest.mark.parametrize("mode", ["bilinear", "bicubic"])
@pytest.mark.parametrize("shapes", [((16, 12), (7, 5)), ((7, 9), (13, 17))])
def test_resize_matches_torch(mode, shapes):
    # torch F.interpolate(align_corners=False, antialias=False) is the
    # reference for BOTH forward and backward
    (H, W), (oh, ow) = shapes
    torch.manual_seed(3)
    src = torch.rand(H, W, 3, device="cuda")

    def torch_resize(t):
        return torch.nn.functional.interpolate(
            t.permute(2, 0, 1)[None], size=(oh, ow), mode=mode,
            align_corners=False,
        )[0].permute(1, 2, 0)

    out = sid.resize(src, width=ow, height=oh, mode=mode)
    torch.testing.assert_close(out, torch_resize(src), atol=1e-4, rtol=1e-4)

    wgt = torch.rand(oh, ow, 3, device="cuda")
    a = src.clone().requires_grad_(True)
    (sid.resize(a, width=ow, height=oh, mode=mode) * wgt).sum().backward()
    b = src.clone().requires_grad_(True)
    (torch_resize(b) * wgt).sum().backward()
    torch.testing.assert_close(a.grad, b.grad, atol=1e-4, rtol=1e-4)


def test_resize_fp32_finite_differences_and_aspect():
    small = torch.rand(6, 7, 2, device="cuda", requires_grad=True)
    for mode in ("bilinear", "bicubic"):
        assert_fp32_finite_differences(
            lambda t: sid.resize(t, width=9, height=4, mode=mode),
            (small,),
        )
    # scale= and single-dimension aspect logic
    src = torch.rand(20, 10, 3, device="cuda")
    assert sid.resize(src, scale=0.5).shape == (10, 5, 3)
    assert sid.resize(src, width=5).shape == (10, 5, 3)
    assert sid.resize(src, height=40).shape == (40, 20, 3)
    tiny = torch.rand(1, 1, 3, device="cuda")
    assert sid.resize(tiny, scale=0.1).shape == (1, 1, 3)


@pytest.mark.parametrize("kwargs", [
    {"scale": 0.0},
    {"scale": -1.0},
    {"width": 0, "height": 2},
    {"width": 2, "height": -1},
])
def test_resize_invalid_geometry_raises(img, kwargs):
    with pytest.raises(ValueError):
        sid.resize(img, **kwargs)


# ----------------------------------------------------------- learnable LUT
def _identity_lut_torch():
    from slangimage.ops import identity_lut_np
    return torch.tensor(identity_lut_np()[..., :3].copy(), device="cuda")


def test_lut_identity_is_identity(img, img_np):
    out = sid.lut(img, _identity_lut_torch())
    assert np.abs(out.cpu().numpy() - img_np).max() < 1.5e-2


@pytest.mark.parametrize("shape", [(8, 8, 3), (10, 10, 3), (16, 24, 3)])
def test_lut_invalid_layout_raises(img, shape):
    bad_lut = torch.rand(*shape, device="cuda")
    with pytest.raises(ValueError, match="square, divisible by 8"):
        sid.lut(img, bad_lut)


@pytest.mark.parametrize("channels", [1, 2, 4, 5])
def test_lut_invalid_image_channels_raise(channels):
    image = torch.rand(8, 8, channels, device="cuda")
    lut = torch.rand(16, 16, 3, device="cuda")
    with pytest.raises(ValueError, match=r"\(H, W, 3\) image"):
        sid.lut(image, lut)


def test_lut_fp32_finite_differences_wrt_both():
    # Keep image samples in the interior [0.2, 0.8]: uniform [0,1] with eps=1e-2
    # regularly lands within eps of the 0/1 clamp, where the one-sided central
    # difference straddles a non-smooth point and the check spuriously fails
    # (interior gradients are correct). Seeded for determinism.
    torch.manual_seed(0)
    img_s = (0.2 + 0.6 * torch.rand(5, 4, 3, device="cuda")).requires_grad_(True)
    lut_s = _identity_lut_torch().requires_grad_(True)
    assert_fp32_finite_differences(
        lambda i, lt: sid.lut(i, lt),
        (img_s, lut_s), samples_per_input=12,
    )


def test_lut_learns_sepia():
    """The headline: recover a color grade as a LUT by gradient descent."""
    rng = np.random.default_rng(25)
    img_np = rng.random((64, 64, 3)).astype(np.float32)
    img_t = torch.tensor(img_np, device="cuda")
    target = sid.sepia(img_t, intensity=1.0).detach()

    lut_p = torch.nn.Parameter(_identity_lut_torch())
    optim = torch.optim.Adam([lut_p], lr=5e-2)

    loss0 = None
    for step in range(120):
        out = sid.lut(img_t, lut_p)
        loss = (out - target).pow(2).mean()
        if loss0 is None:
            loss0 = float(loss.detach())
        optim.zero_grad()
        loss.backward()
        optim.step()

    final = float((sid.lut(img_t, lut_p) - target).pow(2).mean().detach())
    assert final < loss0 * 0.05, f"LUT failed to learn sepia: {loss0} -> {final}"
