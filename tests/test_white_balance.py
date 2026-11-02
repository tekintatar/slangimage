# tests/test_white_balance.py
# White balance: the Python-side color science (illuminant locus, Bradford
# matrix) plus the kernel against a full numpy reference of the pipeline.

from __future__ import annotations

import numpy as np
import pytest

from slangimage.runtime.textures import np_to_texture, texture_to_np
from slangimage.kernels.white_balance import (
    WhiteBalanceOp,
    illuminant_xy,
    white_balance_matrix,
)

ATOL = 3e-3


# ------------------------------------------------------- matrix properties
def test_neutral_is_identity_matrix():
    m = white_balance_matrix(6500.0, 0.0)
    np.testing.assert_allclose(m, np.eye(3), atol=1e-12)


def test_illuminant_locus_sane():
    # D65 (0.3127, 0.3290) at 6500K; monotonically bluer (x falls) with T
    x65, y65 = illuminant_xy(6500.0)
    assert abs(x65 - 0.3127) < 2e-3 and abs(y65 - 0.3290) < 2e-3
    xs = [illuminant_xy(t)[0] for t in (2000, 3000, 4500, 6500, 10000, 20000)]
    assert all(a > b for a, b in zip(xs, xs[1:])), "x must fall as T rises"


def test_source_illuminant_maps_to_neutral():
    # the defining property: a pixel with the assumed illuminant's color
    # becomes achromatic after adaptation
    for temp in (3000.0, 5000.0, 9000.0):
        m = white_balance_matrix(temp, 0.0)
        from slangimage.kernels.white_balance import _white_xyz, _XYZ_TO_RGB
        sx, sy = illuminant_xy(temp)
        illum_rgb = _XYZ_TO_RGB @ _white_xyz(sx, sy)
        out = m @ illum_rgb
        assert np.abs(out - out.mean()).max() < 1e-3, f"T={temp}: {out}"


# --------------------------------------------------------------- kernel
def np_white_balance(img_srgb, temperature, tint):
    m = white_balance_matrix(temperature, tint)
    c = np.clip(img_srgb, 0.0, 1.0)
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    adapted = np.clip(lin @ m.T, 0.0, 1.0)
    return np.where(adapted <= 0.0031308,
                    adapted * 12.92,
                    1.055 * adapted ** (1.0 / 2.4) - 0.055)


@pytest.fixture(scope="module")
def img_and_tex(ctx):
    device, _ = ctx
    rng = np.random.default_rng(41)
    img = rng.random((48, 64, 4)).astype(np.float32)
    img[..., 3] = 1.0
    return img, np_to_texture(device, img, usage="read", precision="fp32")


@pytest.fixture(scope="module")
def wb(ctx):
    device, modules = ctx
    return WhiteBalanceOp(device, modules)


def test_kernel_matches_numpy(wb, img_and_tex):
    img, t = img_and_tex
    for temp, tint in ((6500.0, 0.0), (3200.0, 0.0), (9000.0, -40.0), (5000.0, 60.0)):
        out = texture_to_np(wb.apply(t, temperature=temp, tint=tint))
        ref = np_white_balance(img[..., :3], temp, tint)
        np.testing.assert_allclose(out[..., :3], ref, atol=ATOL,
                                   err_msg=f"T={temp} tint={tint}")


def test_warm_cool_and_tint_directions(wb, img_and_tex):
    img, t = img_and_tex
    base = texture_to_np(wb.apply(t))[..., :3]

    warm = texture_to_np(wb.apply(t, temperature=9000.0))[..., :3]
    assert warm[..., 0].mean() > base[..., 0].mean()  # red up
    assert warm[..., 2].mean() < base[..., 2].mean()  # blue down

    cool = texture_to_np(wb.apply(t, temperature=4000.0))[..., :3]
    assert cool[..., 2].mean() > base[..., 2].mean()  # blue up

    magenta = texture_to_np(wb.apply(t, tint=80.0))[..., :3]
    assert magenta[..., 1].mean() < base[..., 1].mean()  # green down


def test_functional_wrapper(ctx):
    import slangimage as si

    rng = np.random.default_rng(42)
    arr = rng.random((32, 32, 4)).astype(np.float32)
    arr[..., 3] = 1.0
    img = si.from_numpy(arr, precision="fp32", device=ctx[0])
    out = si.ops.white_balance(img, temperature=4500.0, tint=20.0)
    a = out.to_numpy()
    assert a.shape[:2] == (32, 32) and np.isfinite(a).all()
