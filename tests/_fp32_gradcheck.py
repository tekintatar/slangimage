"""Finite-difference checks for CUDA kernels whose public dtype is fp32.

PyTorch's ``gradcheck`` is designed around double-precision inputs. SlangImage's
torch bridge intentionally computes in fp32, so these tests instead compare
autograd against central differences with a stated fp32 error budget. A seeded
output projection keeps the check scalar, and each input samples both its
largest analytical gradients and coordinates spread across the tensor.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import torch


def assert_fp32_finite_differences(
    fn: Callable[..., torch.Tensor],
    inputs: Sequence[torch.Tensor],
    *,
    eps: float = 1e-2,
    atol: float = 2e-3,
    rtol: float = 2e-2,
    samples_per_input: int = 10,
    seed: int = 20260713,
) -> None:
    """Check sampled fp32 gradients against central finite differences.

    The default budget is an absolute error of 0.002 plus 2% relative error at
    a perturbation of 0.01. This is deliberately looser than double-precision
    gradcheck and deliberately explicit about that tradeoff.
    """
    if not inputs:
        raise ValueError("at least one input tensor is required")
    if samples_per_input < 1:
        raise ValueError("samples_per_input must be >= 1")

    checked = tuple(t.detach().clone().requires_grad_(True) for t in inputs)
    output = fn(*checked)
    if output.dtype != torch.float32:
        raise AssertionError(f"expected an fp32 output, got {output.dtype}")
    if not torch.isfinite(output).all():
        raise AssertionError("forward output contains non-finite values")

    generator = torch.Generator(device=output.device)
    generator.manual_seed(seed)
    projection = torch.randint(
        0, 2, output.shape, generator=generator, device=output.device
    ).to(output.dtype)
    projection = (projection * 2.0 - 1.0) / math.sqrt(output.numel())
    objective = (output * projection).sum()
    gradients = torch.autograd.grad(objective, checked)
    base_inputs = tuple(t.detach() for t in checked)

    for input_index, gradient in enumerate(gradients):
        if gradient is None or not torch.isfinite(gradient).all():
            raise AssertionError(f"input {input_index} has a missing or non-finite gradient")

        flat_gradient = gradient.reshape(-1)
        count = min(samples_per_input, flat_gradient.numel())
        top_count = max(1, (count + 1) // 2)
        candidates = torch.topk(flat_gradient.abs(), k=top_count).indices.tolist()
        landmark_count = count - top_count
        if landmark_count:
            landmarks = torch.linspace(
                0, flat_gradient.numel() - 1, landmark_count,
                device=flat_gradient.device,
            ).round().to(torch.long).tolist()
            candidates.extend(landmarks)
        indices = list(dict.fromkeys(candidates))

        # Ties (especially all-zero analytical regions) may deduplicate the
        # landmarks. Fill deterministically so the promised sample count holds.
        candidate = 0
        while len(indices) < count:
            if candidate not in indices:
                indices.append(candidate)
            candidate += 1

        for flat_index in indices:
            positive = [t.clone() for t in base_inputs]
            negative = [t.clone() for t in base_inputs]
            positive[input_index].reshape(-1)[flat_index] += eps
            negative[input_index].reshape(-1)[flat_index] -= eps
            with torch.no_grad():
                high = (fn(*positive) * projection).sum()
                low = (fn(*negative) * projection).sum()
            numerical = (high - low) / (2.0 * eps)
            analytical = flat_gradient[flat_index]
            error = float((numerical - analytical).abs())
            tolerance = atol + rtol * abs(float(numerical))
            if error > tolerance:
                raise AssertionError(
                    f"input {input_index}, flat index {flat_index}: analytical "
                    f"{float(analytical):.7g}, numerical {float(numerical):.7g}, "
                    f"error {error:.3g} exceeds fp32 budget {tolerance:.3g} "
                    f"(eps={eps}, atol={atol}, rtol={rtol})"
                )
