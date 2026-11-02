# tests/test_resize_utils.py
# resize_utils is public API; these tests pin its contract.

from __future__ import annotations

import numpy as np

from slangimage.runtime.textures import np_to_texture
from slangimage.runtime.resize_utils import (
    compute_scaled_size,
    resize_to_max_side,
    resize_to_exact_size,
)
from slangimage.runtime.modules import get_module_cache
from slangimage.kernels.resize import ResizeOp


def test_compute_scaled_size():
    assert compute_scaled_size(4000, 3000, 2000) == (2000, 1500)
    assert compute_scaled_size(3000, 4000, 2000) == (1500, 2000)
    assert compute_scaled_size(800, 600, 2000) == (800, 600)  # already smaller


def test_resize_helpers(ctx):
    device, _ = ctx
    op = ResizeOp(device, get_module_cache(device))
    rng = np.random.default_rng(41)
    img = rng.random((300, 400, 4)).astype(np.float32)
    tex = np_to_texture(device, img, usage="read", precision="fp32")

    out, orig = resize_to_max_side(op, tex, 200)
    assert orig == (400, 300)
    assert (out.width, out.height) == (200, 150)

    same, _ = resize_to_max_side(op, tex, 4096)  # already smaller: unchanged
    assert same is tex

    exact = resize_to_exact_size(op, tex, 123, 77)
    assert (exact.width, exact.height) == (123, 77)
