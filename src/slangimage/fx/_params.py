from __future__ import annotations

import math

import slangpy as spy

from ..kernels.gaussian import GaussianBlurOp, validate_gaussian_params, validate_blur_amount


def apply_gaussian(
    op: GaussianBlurOp,
    src_tex: spy.Texture,
    *,
    blur_amount: float | None,
    default_amount: float,
    radius: int | None,
    sigma: float | None,
) -> spy.Texture:
    """Apply the shared effect Gaussian contract.

    Explicit radius/sigma must be supplied together. Otherwise blur_amount
    follows the same image-size-aware conversion as ``si.ops.gaussian_blur``.
    """
    if (radius is None) != (sigma is None):
        raise ValueError("radius and sigma must be provided together, or both omitted")
    if radius is None:
        amount = default_amount if blur_amount is None else blur_amount
        return op.apply_with_amount(src_tex, validate_blur_amount(amount))
    if blur_amount is not None:
        raise ValueError("choose blur_amount or radius/sigma, not both")
    assert sigma is not None  # paired-presence check above narrows the runtime contract
    radius, sigma = validate_gaussian_params(radius, sigma)
    return op.apply(src_tex, radius=radius, sigma=sigma)


def positive_float(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite value > 0")
    return value


def finite_float(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value
