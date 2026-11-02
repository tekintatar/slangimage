"""Independent CUDA forward/VJP and autograd lifecycle contracts."""

from __future__ import annotations

import math
import os
import gc
import weakref
from collections import Counter
from collections.abc import Callable

import pytest

if os.environ.get("SLANGIMAGE_BACKEND", "auto") not in {"auto", "cuda"}:
    pytest.skip(
        "CUDA autograd tests excluded by SLANGIMAGE_BACKEND",
        allow_module_level=True,
    )

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("slangimage.diff requires CUDA", allow_module_level=True)

import slangimage.diff as sid  # noqa: E402
import slangimage.nn as snn  # noqa: E402
from slangimage.ops import identity_lut_np  # noqa: E402


TensorFn = Callable[..., "torch.Tensor"]


def _clone_inputs(
    inputs: tuple[torch.Tensor, ...], requires: tuple[bool, ...]
) -> tuple[torch.Tensor, ...]:
    return tuple(
        tensor.detach().clone().requires_grad_(needs_grad)
        for tensor, needs_grad in zip(inputs, requires, strict=True)
    )


def _assert_forward_and_vjp(
    slang_fn: TensorFn,
    reference_fn: TensorFn,
    inputs: tuple[torch.Tensor, ...],
    requires: tuple[bool, ...],
    *,
    atol: float = 2e-5,
    rtol: float = 2e-5,
) -> None:
    actual_inputs = _clone_inputs(inputs, requires)
    reference_inputs = _clone_inputs(inputs, requires)
    actual = slang_fn(*actual_inputs)
    reference = reference_fn(*reference_inputs)
    torch.testing.assert_close(actual, reference, atol=atol, rtol=rtol)

    generator = torch.Generator(device="cuda")
    generator.manual_seed(612)
    projection = torch.randn(
        actual.shape, generator=generator, device="cuda", dtype=torch.float32
    ) / math.sqrt(actual.numel())
    actual_leaves = tuple(
        tensor for tensor, needed in zip(actual_inputs, requires, strict=True) if needed
    )
    reference_leaves = tuple(
        tensor
        for tensor, needed in zip(reference_inputs, requires, strict=True)
        if needed
    )
    actual_grads = torch.autograd.grad((actual * projection).sum(), actual_leaves)
    reference_grads = torch.autograd.grad(
        (reference * projection).sum(), reference_leaves
    )
    for actual_grad, reference_grad in zip(
        actual_grads, reference_grads, strict=True
    ):
        assert torch.isfinite(actual_grad).all()
        torch.testing.assert_close(
            actual_grad, reference_grad, atol=atol * 5, rtol=rtol * 5
        )


def _gaussian_reference(
    image: torch.Tensor, radius: int, sigma: float
) -> torch.Tensor:
    squeeze = image.ndim == 2
    x = image.unsqueeze(-1) if squeeze else image
    height, width, _ = x.shape
    offsets = torch.arange(-radius, radius + 1, device=x.device)
    weights = torch.exp(-(offsets.float() ** 2) / (2.0 * sigma * sigma))
    weights = weights / weights.sum()
    columns = torch.arange(width, device=x.device)
    rows = torch.arange(height, device=x.device)
    horizontal = sum(
        weight * x.index_select(1, (columns + int(offset)).clamp(0, width - 1))
        for offset, weight in zip(offsets, weights, strict=True)
    )
    output = sum(
        weight
        * horizontal.index_select(0, (rows + int(offset)).clamp(0, height - 1))
        for offset, weight in zip(offsets, weights, strict=True)
    )
    return output.squeeze(-1) if squeeze else output


def _exposure_reference(image: torch.Tensor, ev: float) -> torch.Tensor:
    return image * 2.0**ev


_SEPIA = torch.tensor(
    [[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]],
    device="cuda",
)


def _sepia_reference(image: torch.Tensor, intensity: float) -> torch.Tensor:
    sepia = image @ _SEPIA.T
    return image + intensity * (sepia - image)


def _hue_reference(image: torch.Tensor, angle: float) -> torch.Tensor:
    y = image @ torch.tensor([0.299, 0.587, 0.114], device="cuda")
    i = image @ torch.tensor([0.596, -0.274, -0.322], device="cuda")
    q = image @ torch.tensor([0.211, -0.523, 0.312], device="cuda")
    sine, cosine = math.sin(angle), math.cos(angle)
    i2, q2 = i * cosine - q * sine, i * sine + q * cosine
    return torch.stack(
        (
            y + 0.956 * i2 + 0.621 * q2,
            y - 0.272 * i2 - 0.647 * q2,
            y - 1.106 * i2 + 1.703 * q2,
        ),
        dim=-1,
    )


def _blend_reference(
    base: torch.Tensor, layer: torch.Tensor, mode: str, opacity: float
) -> torch.Tensor:
    if mode == "multiply":
        mixed = base * layer
    elif mode == "screen":
        mixed = base + layer - base * layer
    else:
        mixed = torch.where(
            base <= 0.5,
            2.0 * base * layer,
            1.0 - 2.0 * (1.0 - base) * (1.0 - layer),
        )
    return base + opacity * (mixed - base)


@pytest.fixture
def small_rgb() -> torch.Tensor:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(611)
    return torch.rand((7, 6, 3), generator=generator, device="cuda")


@pytest.mark.parametrize("name", ("exposure", "sepia", "hue"))
def test_linear_color_forward_and_vjp_match_torch(name, small_rgb) -> None:
    if name == "exposure":
        def slang_fn(image):
            return sid.exposure(image, -1.25)

        def reference_fn(image):
            return _exposure_reference(image, -1.25)
    elif name == "sepia":
        def slang_fn(image):
            return sid.sepia(image, 0.65)

        def reference_fn(image):
            return _sepia_reference(image, 0.65)
    else:
        def slang_fn(image):
            return sid.hue_rotate(image, -0.9)

        def reference_fn(image):
            return _hue_reference(image, -0.9)
    _assert_forward_and_vjp(slang_fn, reference_fn, (small_rgb,), (True,))


@pytest.mark.parametrize("name", ("gaussian", "unsharp"))
def test_gaussian_family_full_forward_and_vjp_match_torch(name, small_rgb) -> None:
    if name == "gaussian":
        def slang_fn(image):
            return sid.gaussian_blur(image, radius=3, sigma=1.4)

        def reference_fn(image):
            return _gaussian_reference(image, 3, 1.4)
    else:
        def slang_fn(image):
            return sid.unsharp_mask(image, radius=2, sigma=1.2, amount=1.35)

        def reference_fn(image):
            return image + 1.35 * (
                image - _gaussian_reference(image, 2, 1.2)
            )
    _assert_forward_and_vjp(
        slang_fn, reference_fn, (small_rgb,), (True,), atol=4e-5, rtol=4e-5
    )


@pytest.mark.parametrize("mode", ("multiply", "screen", "overlay"))
@pytest.mark.parametrize("requires", ((True, True), (True, False), (False, True)))
def test_blend_mixed_gradness_forward_and_vjp_match_torch(
    mode, requires, small_rgb
) -> None:
    base = small_rgb * 0.35 + 0.05  # stay away from overlay's 0.5 branch
    layer = small_rgb.flip(1) * 0.8 + 0.1
    _assert_forward_and_vjp(
        lambda a, b: sid.blend(a, b, mode=mode, opacity=0.7),
        lambda a, b: _blend_reference(a, b, mode, 0.7),
        (base, layer),
        requires,
    )


def test_composed_generated_chain_matches_torch_forward_and_vjp(small_rgb) -> None:
    def actual(image):
        return sid.sepia(
            sid.gaussian_blur(
                sid.hue_rotate(sid.exposure(image, 0.4), -0.35),
                radius=2,
                sigma=1.1,
            ),
            0.55,
        )

    def reference(image):
        return _sepia_reference(
            _gaussian_reference(
                _hue_reference(_exposure_reference(image, 0.4), -0.35),
                2,
                1.1,
            ),
            0.55,
        )

    _assert_forward_and_vjp(
        actual, reference, (small_rgb,), (True,), atol=6e-5, rtol=6e-5
    )


def test_nn_sequential_matches_torch_forward_and_vjp(small_rgb) -> None:
    pipeline = torch.nn.Sequential(
        snn.Exposure(0.4),
        snn.HueRotate(-0.35),
        snn.GaussianBlur(radius=2, sigma=1.1),
        snn.Sepia(0.55),
    )

    def reference(image):
        return _sepia_reference(
            _gaussian_reference(
                _hue_reference(_exposure_reference(image, 0.4), -0.35),
                2,
                1.1,
            ),
            0.55,
        )

    _assert_forward_and_vjp(
        pipeline, reference, (small_rgb,), (True,), atol=6e-5, rtol=6e-5
    )


def _identity_lut() -> torch.Tensor:
    return torch.tensor(identity_lut_np()[..., :3].copy(), device="cuda")


def _transition_case(name: str, image: torch.Tensor) -> tuple[TensorFn, tuple]:
    if name == "gaussian":
        return lambda x: sid.gaussian_blur(x, radius=2, sigma=1.2), (image,)
    if name == "exposure":
        return lambda x: sid.exposure(x, 0.3), (image,)
    if name == "sepia":
        return lambda x: sid.sepia(x, 0.4), (image,)
    if name == "hue":
        return lambda x: sid.hue_rotate(x, 0.2), (image,)
    if name == "blend":
        return lambda x, y: sid.blend(x, y, mode="screen"), (image, image.flip(0))
    return lambda x, lut: sid.lut(x, lut), (image, _identity_lut())


@pytest.mark.parametrize(
    "name", ("gaussian", "exposure", "sepia", "hue", "blend", "lut")
)
def test_nograd_grad_nograd_grad_dispatch_transition(name, small_rgb) -> None:
    fn, inputs = _transition_case(name, small_rgb[:3, :4])
    with torch.no_grad():
        inference = fn(*inputs)
    assert not inference.requires_grad

    grad_inputs = tuple(t.detach().clone().requires_grad_(True) for t in inputs)
    trained = fn(*grad_inputs)
    assert trained.requires_grad
    torch.autograd.grad(trained.sum(), grad_inputs)

    with torch.no_grad():
        inference_after_grad = fn(*grad_inputs)
    assert not inference_after_grad.requires_grad

    final_inputs = tuple(t.detach().clone().requires_grad_(True) for t in inputs)
    final = fn(*final_inputs)
    gradients = torch.autograd.grad(final.sum(), final_inputs)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def _repeated_case(name: str, image: torch.Tensor) -> torch.Tensor:
    if name == "gaussian":
        return sid.gaussian_blur(image, radius=2, sigma=1.2)
    if name == "bilateral":
        return sid.bilateral(image, radius=1, sigma_spatial=1.2, sigma_range=0.4)
    if name == "unsharp":
        return sid.unsharp_mask(image, radius=2, sigma=1.2, amount=0.8)
    if name == "resize":
        return sid.resize(image, width=5, height=4, mode="bicubic")
    if name == "exposure":
        return sid.exposure(image, 0.3)
    if name == "sepia":
        return sid.sepia(image, 0.4)
    if name == "hue":
        return sid.hue_rotate(image, 0.2)
    if name == "blend":
        return sid.blend(image, image.detach().flip(0), mode="multiply")
    return sid.lut(image, _identity_lut())


@pytest.mark.parametrize(
    "name",
    (
        "gaussian",
        "bilateral",
        "unsharp",
        "resize",
        "exposure",
        "sepia",
        "hue",
        "blend",
        "lut",
    ),
)
def test_same_graph_repeated_backward_accumulates_exactly(name, small_rgb) -> None:
    source = small_rgb.detach().clone().requires_grad_(True)
    output = _repeated_case(name, source)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(613)
    projection = torch.randn(
        output.shape, generator=generator, device="cuda", dtype=torch.float32
    )
    objective = (output * projection).sum()
    objective.backward(retain_graph=True)
    first = source.grad.detach().clone()
    objective.backward()
    assert torch.isfinite(source.grad).all()
    torch.testing.assert_close(source.grad, first * 2.0, atol=2e-5, rtol=2e-5)


# ------------------------------------------------ dtype and layout boundaries
_ALL_OPS = (
    "gaussian",
    "bilateral",
    "unsharp",
    "resize",
    "exposure",
    "sepia",
    "hue",
    "blend",
    "lut",
)


def _case_inputs(name: str, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
    if name == "blend":
        return image, image.flip(0) * 0.7 + 0.1
    if name == "lut":
        return image * 0.6 + 0.2, _identity_lut()
    return (image,)


def _apply_case(name: str, *inputs: torch.Tensor) -> torch.Tensor:
    image = inputs[0]
    if name == "gaussian":
        return sid.gaussian_blur(image, radius=2, sigma=1.2)
    if name == "bilateral":
        return sid.bilateral(
            image, radius=1, sigma_spatial=1.2, sigma_range=0.4
        )
    if name == "unsharp":
        return sid.unsharp_mask(image, radius=2, sigma=1.2, amount=0.8)
    if name == "resize":
        return sid.resize(image, width=5, height=4, mode="bicubic")
    if name == "exposure":
        return sid.exposure(image, 0.3)
    if name == "sepia":
        return sid.sepia(image, 0.4)
    if name == "hue":
        return sid.hue_rotate(image, 0.2)
    if name == "blend":
        return sid.blend(image, inputs[1], mode="overlay", opacity=0.7)
    return sid.lut(image, inputs[1])


@pytest.mark.parametrize("name", _ALL_OPS)
@pytest.mark.parametrize(
    "dtype",
    (torch.float16, torch.bfloat16),
    ids=("float16", "bfloat16"),
)
def test_low_precision_matches_quantized_fp32_forward_and_vjp(
    name, dtype, small_rgb
) -> None:
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support bfloat16")

    values = _case_inputs(name, small_rgb)
    low_inputs = tuple(
        value.to(dtype).detach().requires_grad_(True) for value in values
    )
    fp32_inputs = tuple(
        value.detach().to(dtype).float().requires_grad_(True) for value in values
    )
    actual = _apply_case(name, *low_inputs)
    reference = _apply_case(name, *fp32_inputs)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, reference, atol=3e-5, rtol=3e-5)

    generator = torch.Generator(device="cuda")
    generator.manual_seed(614)
    projection = torch.randn(
        actual.shape, generator=generator, device="cuda", dtype=torch.float32
    ) / math.sqrt(actual.numel())
    actual_grads = torch.autograd.grad((actual * projection).sum(), low_inputs)
    reference_grads = torch.autograd.grad(
        (reference * projection).sum(), fp32_inputs
    )
    tolerance = 2e-3 if dtype == torch.float16 else 2e-2
    for actual_grad, reference_grad in zip(
        actual_grads, reference_grads, strict=True
    ):
        assert actual_grad.dtype == dtype
        assert torch.isfinite(actual_grad).all()
        torch.testing.assert_close(
            actual_grad.float(),
            reference_grad.to(dtype).float(),
            atol=tolerance,
            rtol=tolerance,
        )


def _noncontiguous(value: torch.Tensor, layout: str) -> torch.Tensor:
    if layout == "spatial-stride":
        backing = torch.empty(
            value.shape[0] * 2,
            value.shape[1] * 2,
            value.shape[2],
            device=value.device,
            dtype=value.dtype,
        )
        view = backing[::2, ::2]
        view.copy_(value)
    else:
        view = value.permute(2, 0, 1).contiguous().permute(1, 2, 0)
    assert not view.is_contiguous()
    return view.detach().requires_grad_(True)


@pytest.mark.parametrize("name", _ALL_OPS)
@pytest.mark.parametrize("layout", ("spatial-stride", "chw-permute"))
def test_noncontiguous_inputs_match_contiguous_forward_and_vjp(
    name, layout, small_rgb
) -> None:
    values = _case_inputs(name, small_rgb)
    strided_inputs = tuple(_noncontiguous(value, layout) for value in values)
    contiguous_inputs = tuple(
        value.detach().contiguous().requires_grad_(True) for value in strided_inputs
    )
    actual = _apply_case(name, *strided_inputs)
    reference = _apply_case(name, *contiguous_inputs)
    torch.testing.assert_close(actual, reference, atol=3e-5, rtol=3e-5)

    generator = torch.Generator(device="cuda")
    generator.manual_seed(615)
    projection = torch.randn(
        actual.shape, generator=generator, device="cuda", dtype=torch.float32
    ) / math.sqrt(actual.numel())
    actual_grads = torch.autograd.grad((actual * projection).sum(), strided_inputs)
    reference_grads = torch.autograd.grad(
        (reference * projection).sum(), contiguous_inputs
    )
    for actual_grad, reference_grad in zip(
        actual_grads, reference_grads, strict=True
    ):
        assert torch.isfinite(actual_grad).all()
        torch.testing.assert_close(
            actual_grad, reference_grad, atol=3e-5, rtol=3e-5
        )


# ---------------------------------------- manual backward saved-tensor lifetime
def test_bilateral_saved_tensors_survive_until_backward(small_rgb) -> None:
    packed: list[tuple[tuple[int, ...], weakref.ReferenceType]] = []
    unpacked: list[tuple[int, ...]] = []
    captured_grad: list[torch.Tensor] = []

    def pack(tensor):
        packed.append((tuple(tensor.shape), weakref.ref(tensor)))
        return tensor

    def unpack(tensor):
        unpacked.append(tuple(tensor.shape))
        return tensor

    source = small_rgb.detach().clone().requires_grad_(True)
    source.register_hook(lambda grad: captured_grad.append(grad.detach().clone()))
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        output = sid.bilateral(
            source, radius=1, sigma_spatial=1.2, sigma_range=0.4
        )

    assert Counter(shape for shape, _ in packed) == Counter(
        (tuple(source.shape), (source.shape[0], source.shape[1], 4))
    )
    objective = output.square().mean()
    del output, source
    gc.collect()
    assert all(reference() is not None for _, reference in packed)

    objective.backward()
    assert Counter(unpacked) == Counter(shape for shape, _ in packed)
    assert len(captured_grad) == 1
    assert torch.isfinite(captured_grad[0]).all()


def test_bilateral_detects_saved_input_mutation(small_rgb) -> None:
    source = small_rgb.detach().clone().requires_grad_(True)
    output = sid.bilateral(
        source, radius=1, sigma_spatial=1.2, sigma_range=0.4
    )
    with torch.no_grad():
        source.add_(0.1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        output.sum().backward()


def test_resize_saves_no_activation_tensors(small_rgb) -> None:
    packed: list[tuple[int, ...]] = []

    def pack(tensor):
        packed.append(tuple(tensor.shape))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        source = small_rgb.detach().clone().requires_grad_(True)
        output = sid.resize(source, width=5, height=4, mode="bicubic")
    assert packed == []
    output.sum().backward()
    assert source.grad is not None and torch.isfinite(source.grad).all()


def test_resize_backward_is_independent_of_input_mutation(small_rgb) -> None:
    changed = small_rgb.detach().clone().requires_grad_(True)
    baseline = small_rgb.detach().clone().requires_grad_(True)
    changed_output = sid.resize(changed, width=5, height=4, mode="bicubic")
    baseline_output = sid.resize(baseline, width=5, height=4, mode="bicubic")
    generator = torch.Generator(device="cuda")
    generator.manual_seed(616)
    projection = torch.randn(
        changed_output.shape,
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )
    with torch.no_grad():
        changed.add_(100.0)
    changed_grad = torch.autograd.grad(
        (changed_output * projection).sum(), changed
    )[0]
    baseline_grad = torch.autograd.grad(
        (baseline_output * projection).sum(), baseline
    )[0]
    torch.testing.assert_close(changed_grad, baseline_grad, atol=0, rtol=0)


# ------------------------------------------------------ native bridge boundaries
def test_cpu_tensor_is_rejected_before_cuda_dispatch() -> None:
    with pytest.raises(RuntimeError, match="CUDA"):
        sid.exposure(torch.rand(3, 4, 3), 0.1)


def test_float3_color_op_rejects_wrong_channel_count() -> None:
    image = torch.rand(3, 4, 4, device="cuda")
    with pytest.raises(ValueError, match="expected shape"):
        sid.sepia(image, 0.5)


def test_blend_rejects_mismatched_spatial_shapes(small_rgb) -> None:
    layer = torch.rand(2, small_rgb.shape[1], 3, device="cuda")
    with pytest.raises(ValueError, match="Shape mismatch"):
        sid.blend(small_rgb, layer)
