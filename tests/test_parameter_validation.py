"""Reject bad host parameters before allocation/dispatch; preserve valid sentinels.

Dangerous/extreme inputs are never submitted to a GPU as a way to test failure.
Host stubs prove the boundary checks, while bounded GPU cases check compatibility.
"""

from types import SimpleNamespace
import inspect

import numpy as np
import pytest

import slangimage as si
from slangimage import _validation as validate
from slangimage.kernels.geometry import GeometryOps, invert_affine


# Public scalar controls: one valid base call plus every numeric/boolean field.
SCALARS = [
    ("gaussian_blur", {"radius": 3, "sigma": 1}, "radius sigma"),
    ("gaussian_blur", {"amount": 1}, "amount"),
    ("surface_blur", {}, "radius bsigma"),
    ("bilateral", {}, "radius sigma_spatial sigma_range separable"),
    ("nl_means", {}, "search_radius patch_radius h sigma"),
    ("unsharp_mask", {}, "radius sigma amount"),
    ("ep_unsharp_mask", {}, "radius bsigma amount"),
    ("sharpen_luma", {}, "radius sharpness sharpen_rgb"),
    ("vibrance", {}, "amount"), ("gamma", {}, "power"),
    ("color_controls", {}, "hue saturation contrast brightness"),
    ("exposure", {}, "ev"), ("sepia", {}, "intensity"), ("hue_rotate", {}, "angle"),
    ("highlight_shadow", {}, "shadow_amount highlight_amount"),
    ("solarize", {}, "threshold"), ("posterize", {}, "bits"),
    ("white_balance", {}, "temperature tint"), ("clahe", {}, "clip_limit strength"),
    ("resize", {"width": 5, "height": 4}, "width height"),
    ("resize", {"scale": 1}, "scale"),
    ("flip", {}, "horizontal vertical"), ("rot90", {}, "k"),
    ("crop", {"x": 0, "y": 0, "width": 2, "height": 2}, "x y width height"),
    ("tile", {"width": 5, "height": 4}, "width height scale_x scale_y offset_x offset_y"),
    ("affine", {"matrix": [[1, 0, 0], [0, 1, 0]]}, "width height"),
    ("rotate", {"angle": 10}, "angle expand"),
    ("kuwahara", {}, "radius sigma_spatial var_power"),
    ("pointillize", {}, "radius jitter seed"),
    ("cmyk_halftone", {}, "width angle_deg sharpness"),
    ("smooth_threshold", {}, "threshold smooth_amount"),
    ("smooth_threshold_bw", {"black_threshold": 0.2, "white_threshold": 0.8},
     "black_threshold white_threshold"),
    ("blend", {"blend_image": object()}, "opacity"),
    ("high_pass_contrast", {"blur_image": object()}, "strength"),
    ("lut", {"lut_image": object()}, "intensity"),
    ("random_noise", {"width": 4, "height": 3}, "width height seed"),
    ("prewarm", {}, "size"),
]
SCALAR_CASES = [(name, kwargs, field) for name, kwargs, fields in SCALARS for field in fields.split()]


def test_scalar_inventory_covers_public_numeric_controls():
    covered = {(name, field) for name, _, field in SCALAR_CASES}
    expected = set()
    for name in si.ops.__all__:
        if name in {"BLEND_MODES", "identity_lut_np"}:
            continue
        for field, parameter in inspect.signature(getattr(si.ops, name)).parameters.items():
            if parameter.annotation in {"float", "int", "bool", "Optional[int]", "Optional[float]"}:
                expected.add((name, field))
    assert covered == expected


def invoke(name, kwargs):
    function = getattr(si.ops, name)
    return function(**kwargs) if name in {"random_noise", "prewarm"} else function(object(), **kwargs)


@pytest.fixture
def no_gpu(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid input reached device/dispatch/allocation")

    device = SimpleNamespace(info=SimpleNamespace(limits=SimpleNamespace(max_texture_dimension_2d=16384)))
    texture = SimpleNamespace(width=9, height=7, device=device)
    monkeypatch.setattr(si.ops, "_unwrap", lambda image: (texture, False))
    for name in ("_run1", "_run2", "get_default_device", "_op_for_device", "get_pool"):
        monkeypatch.setattr(si.ops, name, forbidden)
    return texture


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, 1e100, -1e100])
@pytest.mark.parametrize("name,base,field", SCALAR_CASES,
                         ids=[f"{name}-{field}" for name, _, field in SCALAR_CASES])
def test_nonfinite_and_unrepresentable_scalars_rejected(no_gpu, name, base, field, value):
    with pytest.raises(ValueError, match=field):
        invoke(name, {**base, field: value})


INVALID = [
    ("gaussian_blur", {"radius": 0, "sigma": 1}),
    ("gaussian_blur", {"radius": 1.5, "sigma": 1}),
    ("gaussian_blur", {"radius": 2, "sigma": 0}),
    ("gaussian_blur", {"radius": 2, "sigma": 1e-30}),
    ("gaussian_blur", {"radius": 2, "sigma": 1e30}),
    ("gaussian_blur", {"radius": validate.MAX_RADIUS + 1, "sigma": 1}),
    ("gaussian_blur", {"amount": -1}),
    ("gaussian_blur", {"amount": 1, "radius": 3, "sigma": 1}),
    ("gaussian_blur", {}),
    ("surface_blur", {"radius": 0}), ("surface_blur", {"radius": 2.9}),
    ("surface_blur", {"bsigma": 0}), ("surface_blur", {"bsigma": -0.1}),
    ("surface_blur", {"bsigma": 1e-30}),
    ("bilateral", {"radius": -1}), ("bilateral", {"radius": 1.5}),
    ("bilateral", {"sigma_range": 0}), ("bilateral", {"sigma_range": -1}),
    ("bilateral", {"sigma_spatial": 1e-30}),
    ("bilateral", {"range_type": "mystery"}),
    ("nl_means", {"search_radius": -1}), ("nl_means", {"patch_radius": -1}),
    ("nl_means", {"h": -1}), ("nl_means", {"sigma": -1}),
    ("nl_means", {"search_radius": validate.MAX_RADIUS + 1}),
    ("unsharp_mask", {"radius": 0.9}), ("unsharp_mask", {"sigma": 0}),
    ("unsharp_mask", {"color_space": "bad"}),
    ("ep_unsharp_mask", {"radius": 0.49}), ("ep_unsharp_mask", {"bsigma": 0}),
    ("ep_unsharp_mask", {"radius": validate.MAX_RADIUS + 1}),
    ("posterize", {"bits": 4.5}), ("posterize", {"bits": 0}), ("posterize", {"bits": 9}),
    ("normalize", {"mean": (0, 0, 0), "std": (1, 0, 1)}),
    ("normalize", {"mean": (0, 0, 0), "std": (1, 1e-40, 1)}),
    ("clahe", {"tiles": (1.5, 2)}), ("clahe", {"tiles": (0, 2)}),
    ("clahe", {"tiles": (65536, 65536)}), ("clahe", {"clip_limit": 0}),
    ("resize", {"width": 3.5}), ("resize", {"scale": 0}),
    ("resize", {"width": 4, "scale": 2}), ("resize", {"width": 10**8}),
    ("resize", {"scale": 1e20}), ("resize", {"mode": "bad", "width": 2}),
    ("rot90", {"k": 0.5}),
    ("crop", {"x": -1, "y": 0, "width": 1, "height": 1}),
    ("crop", {"x": 0, "y": 0, "width": 1.5, "height": 1}),
    ("tile", {"width": 0, "height": 2}),
    ("tile", {"width": 1, "height": 2, "scale_x": 0}),
    ("affine", {"matrix": [[1, 0, 0], [0, 1, 0]], "width": -1}),
    ("affine", {"matrix": [[1, 0, 0], [0, 1, 0]], "width": 0.5}),
    ("rotate", {"angle": 0, "mode": "bad"}),
    ("kuwahara", {"radius": 0}), ("kuwahara", {"sigma_spatial": 0}),
    ("kuwahara", {"var_power": 0}),
    ("pointillize", {"radius": 1e20}),
    ("smooth_threshold", {"smooth_amount": 0}),
    ("smooth_threshold", {"smooth_amount": -0.2}),
    ("smooth_threshold", {"smooth_amount": 1e-20}),
    ("smooth_threshold", {"threshold": 2, "smooth_amount": 0.1}),
    ("smooth_threshold_bw", {"black_threshold": 0.5, "white_threshold": 0.5 + 1e-10}),
    ("smooth_threshold_bw", {"black_threshold": 2**-126,
                             "white_threshold": float(np.nextafter(
                                 np.float32(2**-126), np.float32(np.inf)))}),
    ("smooth_threshold_bw", {"black_threshold": -validate.FLOAT32_MAX,
                             "white_threshold": validate.FLOAT32_MAX}),
    ("extract_channel", {"channel": "a"}),
    ("blend", {"blend_image": object(), "mode": "bad"}),
    ("random_noise", {"width": 4.5, "height": 2}),
    ("random_noise", {"width": 4, "height": 2, "precision": "fp8"}),
    ("prewarm", {"size": 20.5}), ("prewarm", {"size": 0}),
    ("prewarm", {"precision": "fp8"}),
    ("exposure", {"ev": 128}), ("exposure", {"ev": -127}),
]


@pytest.mark.parametrize("name,kwargs", INVALID, ids=[f"{n}-{i}" for i, (n, _) in enumerate(INVALID)])
def test_invalid_domains_rejected_before_dispatch(no_gpu, name, kwargs):
    with pytest.raises(ValueError):
        invoke(name, kwargs)


@pytest.mark.parametrize("value", [(), (1,), (1, 2), (1, 2, 3, 4), None,
                                  (np.nan, 0, 0), (0, np.inf, 0), (0, 0, 1e100)])
@pytest.mark.parametrize("name,field", [("normalize", "mean"), ("normalize", "std"),
                                      ("false_color", "color0"), ("false_color", "color1")])
def test_invalid_vectors_rejected(no_gpu, name, field, value):
    kwargs = {"mean": (0, 0, 0), "std": (1, 1, 1)} if name == "normalize" else {}
    kwargs[field] = value
    with pytest.raises(ValueError, match=field):
        invoke(name, kwargs)


@pytest.mark.parametrize("tiles", [(), (1,), (1, 2, 3), None,
                                   (np.nan, 2), (2, np.inf), (-np.inf, 2), (2, 1e100)])
def test_invalid_clahe_grid(no_gpu, tiles):
    with pytest.raises(ValueError, match="tiles"):
        invoke("clahe", {"tiles": tiles})


def test_generator_device_limits_before_allocation(no_gpu):
    with pytest.raises(ValueError, match="texture limit"):
        si.ops.random_noise(100000, 1, device=no_gpu.device)
    with pytest.raises(ValueError, match="texture limit"):
        si.ops.prewarm(size=100000, device=no_gpu.device)


@pytest.mark.parametrize("size", [np.nan, np.inf, -np.inf, 16.5, 0, 1e100])
def test_identity_lut_dimensions_reject_invalid_before_allocation(size):
    with pytest.raises(ValueError):
        si.ops.identity_lut_np(size)


@pytest.mark.parametrize("matrix", [np.zeros((2, 3)), [[1, 0, np.nan], [0, 1, 0]],
                                    [[1, 0, 0], [0, np.inf, 0]], np.eye(3),
                                    [[1e-100, 0, 0], [0, 1, 0]]])
def test_invalid_affine_inverse(matrix):
    with pytest.raises(ValueError, match="affine"):
        invert_affine(matrix)


@pytest.mark.parametrize("method,args,kwargs", [
    ("tile", (2, 3), {"scale_x": 1e-20}),
    ("tile", (2, 3), {"scale_y": 1e-100}),
    ("tile", (10**8, 3), {}),
    ("affine", ([[1e-20, 0, 0], [0, 1, 0]],), {}),
    ("affine", ([[1, 0, 1e20], [0, 1, 0]],), {}),
    ("affine", ([[1, 0, 0], [0, 1, 0]],), {"width": 1.5}),
])
def test_geometry_bounds_before_allocation(no_gpu, monkeypatch, method, args, kwargs):
    import slangimage.kernels.geometry as geometry

    def forbidden(*args, **kwargs):
        raise AssertionError("geometry allocated for invalid coordinates")

    monkeypatch.setattr(geometry, "alloc_output", forbidden)
    op = object.__new__(GeometryOps)
    op.device = no_gpu.device
    with pytest.raises(ValueError):
        getattr(op, method)(no_gpu, *args, **kwargs)


def test_size_aware_gaussian_bound_before_dispatch(no_gpu):
    from slangimage.kernels.gaussian import blur_params_from_amount

    with pytest.raises(ValueError, match="radius"):
        blur_params_from_amount(1e30, 1024, 1024)


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
def test_valid_sentinels_and_small_images(ctx, precision):
    source = si.from_numpy(np.full((1, 3, 4), 0.5, dtype=np.float32), precision=precision, device=ctx[0])
    checks = [
        si.ops.surface_blur(source, radius=1),
        si.ops.bilateral(source, radius=1, sigma_spatial=-1),
        si.ops.bilateral(source, radius=1, sigma_spatial=0, sigma_range=0, range_type="cutoff"),
        si.ops.nl_means(source, search_radius=0, patch_radius=0, h=0, sigma=0),
        si.ops.sharpen_luma(source, radius=-1, sharpness=-3),
        si.ops.gamma(source, power=-1), si.ops.vibrance(source, amount=3),
        si.ops.pointillize(source, radius=0, jitter=-1),
        si.ops.cmyk_halftone(source, width=0, sharpness=0),
        si.ops.normalize(source, mean=(0, 0, 0), std=(-1, 1, 1)),
        si.ops.resize(source, scale=1e-6),
        si.ops.affine(source, [[1, 0, 0], [0, 1, 0]], width=0, height=0),
    ]
    for output in checks:
        assert output.device is source.device and output.precision == precision
        assert np.isfinite(output.to_numpy()).all()
    assert checks[-2].to_numpy().shape == (1, 1, 4)
    np.testing.assert_array_equal(checks[-1].to_numpy(), source.to_numpy())


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
def test_safe_weight_and_threshold_boundaries(ctx, precision):
    source = si.from_numpy(np.full((3, 5, 4), 0.5, dtype=np.float32), precision=precision, device=ctx[0])
    for sigma in (validate.MIN_SIGMA, validate.MAX_SIGMA):
        for output in (si.ops.gaussian_blur(source, radius=1, sigma=sigma),
                       si.ops.surface_blur(source, radius=3, bsigma=sigma),
                       si.ops.bilateral(source, radius=1, sigma_range=sigma)):
            np.testing.assert_allclose(output.to_numpy(), source.to_numpy(), rtol=0, atol=1e-3)
    # Smallest simple symmetric interval around 0.5 used here stays ordered in fp32.
    result = si.ops.smooth_threshold(source, threshold=0.5, smooth_amount=2**-23)
    assert np.isfinite(result.to_numpy()).all()
