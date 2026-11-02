# tests/test_pool.py
# Texture pool: frame reuse, escape semantics, eager recycle.

from __future__ import annotations


import numpy as np
import pytest
import slangpy as spy

from slangimage.fx.base import Effect
from slangimage.fx.soft_glow import SoftGlowEffect
from slangimage.runtime.texture_pool import get_pool
from slangimage.runtime.texture_pool import alloc_output
from slangimage.runtime.textures import np_to_texture, texture_to_np


class _AllocatingEffect(Effect):
    def _apply_impl(self, src_tex: spy.Texture, **params: object) -> spy.Texture:
        return alloc_output(self.device, src_tex.width, src_tex.height, src_tex.format)


class _NestedAllocatingEffect(Effect):
    def __init__(self, device: spy.Device, modules: object) -> None:
        super().__init__(device, modules)
        self.inner = _AllocatingEffect(device, modules)
        self.intermediate_id: int | None = None

    def _apply_impl(self, src_tex: spy.Texture, **params: object) -> spy.Texture:
        intermediate = self.inner.apply(src_tex)
        self.intermediate_id = id(intermediate)
        return alloc_output(self.device, src_tex.width, src_tex.height, src_tex.format)


class _FailingAllocatingEffect(Effect):
    def __init__(self, device: spy.Device, modules: object) -> None:
        super().__init__(device, modules)
        self.allocated_ids: set[int] = set()

    def _apply_impl(self, src_tex: spy.Texture, **params: object) -> spy.Texture:
        for _ in range(2):
            tex = alloc_output(
                self.device, src_tex.width, src_tex.height, src_tex.format
            )
            self.allocated_ids.add(id(tex))
        raise RuntimeError("deliberate effect failure")


@pytest.fixture()
def src(ctx):
    device, _ = ctx
    rng = np.random.default_rng(11)
    img = rng.random((128, 128, 4)).astype(np.float32)
    img[..., 3] = 1.0
    return np_to_texture(device, img, usage="read", precision="fp32")


def test_effect_reuses_pool_across_calls(ctx, src):
    device, modules = ctx
    pool = get_pool(device)
    effect = SoftGlowEffect(device, modules)

    out1 = effect.apply(src)
    created_after_first = pool.created
    out2 = effect.apply(src)

    # steady state: all intermediates reused; only the escaped output slot
    # may allocate (run 1's output is still alive out here)
    assert pool.created - created_after_first <= 1, "intermediates were not reused"
    assert pool.reused > 0

    # escaped outputs are distinct live textures with identical content
    assert out1 is not out2
    a, b = texture_to_np(out1), texture_to_np(out2)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    np.testing.assert_allclose(a, b, atol=1e-5)

    # once the caller recycles a previous output, runs are fully allocation-free
    pool.recycle(out1)
    created_before_third = pool.created
    out3 = effect.apply(src)
    assert pool.created == created_before_third, "run after output recycle still allocated"
    assert np.isfinite(texture_to_np(out3)).all()


def test_effect_pool_allocations_plateau_under_sustained_workload(ctx, src):
    device, modules = ctx
    pool = get_pool(device)
    pool.clear()
    effect = SoftGlowEffect(device, modules)

    for _ in range(4):
        warm = effect.apply(src)
        pool.recycle(warm)
    created_after_warmup = pool.created
    reused_after_warmup = pool.reused

    for iteration in range(64):
        out = effect.apply(src)
        if iteration in {0, 63}:
            assert np.isfinite(texture_to_np(out)).all()
        pool.recycle(out)

    device.wait_for_idle()
    assert pool.created == created_after_warmup
    assert pool.reused >= reused_after_warmup + 64
    assert pool.stats()["free"] <= pool.MAX_FREE_PER_KEY


def test_output_not_recycled_into_next_run(ctx, src):
    device, modules = ctx
    effect = SoftGlowEffect(device, modules)

    out1 = effect.apply(src)
    before = texture_to_np(out1).copy()
    _ = effect.apply(src)  # would corrupt out1 if it had been recycled
    after = texture_to_np(out1)
    np.testing.assert_array_equal(before, after)


def test_recycle_roundtrip(ctx):
    device, _ = ctx
    pool = get_pool(device)
    t = pool.alloc(64, 64, __import__("slangpy").Format.rgba32_float)
    key_created = pool.created
    pool.recycle(t)
    t2 = pool.alloc(64, 64, __import__("slangpy").Format.rgba32_float)
    assert t2 is t, "recycled texture was not reused"
    assert pool.created == key_created


def test_double_recycle_cannot_lend_one_texture_twice(ctx):
    device, _ = ctx
    pool = get_pool(device)
    pool.clear()
    tex = pool.alloc(37, 31, spy.Format.rgba32_float)

    pool.recycle(tex)
    pool.recycle(tex)
    first = pool.alloc(37, 31, spy.Format.rgba32_float)
    second = pool.alloc(37, 31, spy.Format.rgba32_float)

    assert first is tex
    assert second is not tex
    pool.recycle(first)
    pool.recycle(second)


def test_nested_effect_intermediate_returns_to_outer_frame(ctx, src):
    device, modules = ctx
    pool = get_pool(device)
    pool.clear()
    effect = _NestedAllocatingEffect(device, modules)

    out = effect.apply(src)
    assert effect.intermediate_id is not None
    borrowed = pool.alloc(src.width, src.height, src.format)

    assert id(borrowed) == effect.intermediate_id
    assert borrowed is not out
    pool.recycle(borrowed)
    pool.recycle(out)


def test_effect_exception_recycles_allocated_textures(ctx, src):
    device, modules = ctx
    pool = get_pool(device)
    pool.clear()
    effect = _FailingAllocatingEffect(device, modules)

    with pytest.raises(RuntimeError, match="deliberate effect failure"):
        effect.apply(src)

    borrowed = {
        id(pool.alloc(src.width, src.height, src.format)),
        id(pool.alloc(src.width, src.height, src.format)),
    }
    assert borrowed == effect.allocated_ids


# ------------------------------------------------- out-of-frame auto-recycle
def test_functional_output_recycles_when_dropped(ctx):
    import gc

    import slangimage as si
    from slangimage.runtime.texture_pool import get_pool

    device, _ = ctx
    rng = np.random.default_rng(5)
    arr = rng.random((96, 96, 4)).astype(np.float32)
    arr[..., 3] = 1.0
    img = si.from_numpy(arr, precision="fp32", device=device)
    pool = get_pool(device)

    # recycling lags one call (slangpy's call-data cache holds the previous
    # dispatch's output until the next call replaces it), so steady state
    # alternates two textures — warm up past the transient first
    for _ in range(3):
        _ = si.ops.invert(img)
        gc.collect()
    created0, reused0 = pool.created, pool.reused
    for _ in range(6):
        _ = si.ops.invert(img)
        gc.collect()
    assert pool.created == created0, "dropped outputs must be reused, not re-created"
    assert pool.reused >= reused0 + 6


def test_held_texture_is_never_recycled(ctx):
    import gc

    import slangimage as si

    device, _ = ctx
    rng = np.random.default_rng(6)
    arr = rng.random((80, 80, 4)).astype(np.float32)
    arr[..., 3] = 1.0
    img = si.from_numpy(arr, precision="fp32", device=device)

    out = si.ops.invert(img)
    keep_tex = out.texture       # user takes the raw texture
    expected = out.to_numpy()
    del out                      # Image dies, but keep_tex must stay ours
    gc.collect()
    for _ in range(8):           # would reuse keep_tex if wrongly recycled
        _ = si.ops.gamma(img, 2.0)
        gc.collect()
    device.wait_for_idle()
    from slangimage.runtime.textures import texture_to_np
    np.testing.assert_array_equal(texture_to_np(keep_tex), expected)


def test_held_texture_recycles_after_alias_is_released(ctx):
    import gc

    import slangimage as si

    device, _ = ctx
    pool = get_pool(device)
    pool.clear()
    arr = np.zeros((29, 41, 4), dtype=np.float32)
    arr[..., 3] = 1.0
    img = si.from_numpy(arr, precision="fp32", device=device)

    out = si.ops.invert(img)
    keep_tex = out.texture
    kept_id = id(keep_tex)
    fmt = keep_tex.format
    del out
    gc.collect()

    # The dead Image queues its output, but the retained raw alias prevents
    # the next same-key allocation from borrowing it.
    other = pool.alloc(41, 29, fmt)
    assert other is not keep_tex
    pool.recycle(other)
    del other

    # Once the alias is gone, the next allocation drains pending ownership
    # and makes that exact texture reusable.
    del keep_tex
    gc.collect()
    reused = pool.alloc(41, 29, fmt)
    assert id(reused) == kept_id
    pool.recycle(reused)


def test_recycling_chain_produces_correct_values(ctx):
    import gc

    import slangimage as si

    device, _ = ctx
    rng = np.random.default_rng(7)
    arr = rng.random((64, 64, 4)).astype(np.float32)
    arr[..., 3] = 1.0
    img = si.from_numpy(arr, precision="fp32", device=device)

    gc.collect()
    # interleave drops and reads: recycled textures must never alias a
    # result that is still being read
    for _ in range(5):
        a = si.ops.invert(img)
        b = si.ops.invert(a)
        del a
        gc.collect()
        back = b.to_numpy()
        np.testing.assert_allclose(back[..., :3], arr[..., :3], atol=2e-3)
        del b
        gc.collect()
