# runtime/resize_utils.py

from __future__ import annotations

from typing import Tuple
import slangpy as spy

from ..kernels.resize import ResizeOp


def compute_scaled_size(
    width: int,
    height: int,
    max_side: int,
) -> Tuple[int, int]:
    """
    Compute (new_w, new_h) such that:
      - aspect ratio is preserved
      - max(new_w, new_h) == max_side
      - if image is already smaller, returns original size
    """
    longest = max(width, height)
    if longest <= max_side:
        return width, height

    if width >= height:
        new_w = max_side
        new_h = int(round(height * (max_side / float(width))))
    else:
        new_h = max_side
        new_w = int(round(width * (max_side / float(height))))

    return new_w, new_h


def resize_to_max_side(
    resize_op: ResizeOp,
    tex: spy.Texture,
    max_side: int,
    mode="lanczos3",
) -> tuple[spy.Texture, tuple[int, int]]:
    """
    Resize a texture so that its longest side is max_side.
    Returns (resized_tex, (orig_w, orig_h)).

    If the texture is already within that size it is returned as is.
    """
    orig_w, orig_h = tex.width, tex.height
    new_w, new_h = compute_scaled_size(orig_w, orig_h, max_side)

    if (new_w, new_h) == (orig_w, orig_h):
        return tex, (orig_w, orig_h)

    resized = resize_op.apply(tex, new_w, new_h, mode=mode)
    return resized, (orig_w, orig_h)


def resize_to_exact_size(
    resize_op: ResizeOp,
    tex: spy.Texture,
    target_w: int,
    target_h: int,
    mode="lanczos3",
) -> spy.Texture:
    """
    Convenience helper: resize a texture to an exact output size.
    """
    if tex.width == target_w and tex.height == target_h:
        return tex
    return resize_op.apply(tex, target_w, target_h, mode=mode)