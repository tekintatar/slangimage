"""Host-side ownership invariants for the texture pool.

These tests use texture-shaped fakes so queue state can be asserted directly
without allocating a second GPU device.  ``tests/test_pool.py`` provides the
corresponding real-dispatch integration coverage.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass

import pytest
import slangpy as spy

from slangimage.runtime.texture_pool import TexturePool, _tex_bytes
from slangimage.runtime.textures import TEX_USAGE_RW


@dataclass(frozen=True)
class _Desc:
    format: spy.Format
    usage: spy.TextureUsage


class _Texture:
    def __init__(
        self,
        width: int,
        height: int,
        fmt: spy.Format,
        usage: spy.TextureUsage,
    ) -> None:
        self.width = width
        self.height = height
        self.format = fmt
        self.desc = _Desc(fmt, usage)


class _Device:
    def create_texture(
        self,
        *,
        width: int,
        height: int,
        format: spy.Format,
        usage: spy.TextureUsage,
    ) -> _Texture:
        return _Texture(width, height, format, usage)


class _Owner:
    pass


def _pool() -> TexturePool:
    return TexturePool(_Device())  # type: ignore[arg-type]


def _assert_disjoint_queues(pool: TexturePool) -> None:
    free_ids = {id(tex) for textures in pool._free.values() for tex in textures}
    pending_ids = {id(tex) for tex in pool._pending}
    assert free_ids == pool._free_ids
    assert pending_ids == pool._pending_ids
    assert free_ids.isdisjoint(pending_ids)


def _free_textures(pool: TexturePool) -> list[spy.Texture]:
    return [tex for textures in pool._free.values() for tex in textures]


def _assert_free_accounting(pool: TexturePool) -> None:
    free = _free_textures(pool)
    assert pool.stats()["free"] == len(free)
    assert pool._free_bytes == sum(_tex_bytes(tex) for tex in free)
    _assert_disjoint_queues(pool)


def test_explicit_recycle_is_idempotent_and_cannot_lend_twice() -> None:
    pool = _pool()
    tex = pool.alloc(8, 6, spy.Format.rgba32_float)

    pool.recycle(tex)
    free_bytes = pool._free_bytes
    pool.recycle(tex)

    assert pool.stats()["free"] == 1
    assert pool._free_bytes == free_bytes
    _assert_disjoint_queues(pool)

    first = pool.alloc(8, 6, spy.Format.rgba32_float)
    second = pool.alloc(8, 6, spy.Format.rgba32_float)
    assert first is tex
    assert second is not tex
    _assert_disjoint_queues(pool)


def test_explicit_recycle_and_frame_exit_enqueue_only_once() -> None:
    pool = _pool()

    with pool.frame():
        tex = pool.alloc(9, 7, spy.Format.rgba16_float)
        pool.recycle(tex)

    assert pool.stats()["free"] == 1
    assert pool._frames == []
    _assert_disjoint_queues(pool)


def test_explicit_recycle_wins_over_later_owner_finalizer() -> None:
    pool = _pool()
    tex = pool.alloc(10, 8, spy.Format.rgba32_float)
    owner = _Owner()
    pool.auto_recycle(owner, tex)

    pool.recycle(tex)
    del owner
    gc.collect()

    assert pool.stats()["free"] == 1
    assert pool._pending == []
    _assert_disjoint_queues(pool)


def test_retained_texture_alias_blocks_reuse_until_released() -> None:
    pool = _pool()
    tex = pool.alloc(11, 9, spy.Format.rgba32_float)
    tex_id = id(tex)
    owner = _Owner()
    pool.auto_recycle(owner, tex)

    del owner
    gc.collect()
    assert [id(candidate) for candidate in pool._pending] == [tex_id]

    other = pool.alloc(11, 9, spy.Format.rgba32_float)
    assert other is not tex
    _assert_disjoint_queues(pool)

    del other
    del tex
    reused = pool.alloc(11, 9, spy.Format.rgba32_float)
    assert id(reused) == tex_id
    assert pool._pending == []
    _assert_disjoint_queues(pool)


def test_multiple_finalizers_for_one_texture_do_not_duplicate_pending() -> None:
    pool = _pool()
    tex = pool.alloc(12, 10, spy.Format.rgba32_float)
    tex_id = id(tex)
    first_owner = _Owner()
    second_owner = _Owner()
    pool.auto_recycle(first_owner, tex)
    pool.auto_recycle(second_owner, tex)

    del tex
    del first_owner
    gc.collect()
    assert [id(candidate) for candidate in pool._pending] == [tex_id]

    del second_owner
    gc.collect()
    assert [id(candidate) for candidate in pool._pending] == [tex_id]
    _assert_disjoint_queues(pool)

    reused = pool.alloc(12, 10, spy.Format.rgba32_float)
    assert id(reused) == tex_id
    assert pool._pending == []
    _assert_disjoint_queues(pool)


def test_clear_resets_free_and_pending_ownership_state() -> None:
    pool = _pool()
    free_tex = pool.alloc(13, 11, spy.Format.rgba16_float, TEX_USAGE_RW)
    held_tex = pool.alloc(14, 12, spy.Format.rgba32_float, TEX_USAGE_RW)
    owner = _Owner()
    pool.auto_recycle(owner, held_tex)
    pool.recycle(free_tex)
    del owner
    gc.collect()
    assert pool._free_ids
    assert pool._pending_ids

    pool.clear()

    assert pool.stats()["free"] == 0
    assert pool._free_bytes == 0
    assert pool._free_ids == set()
    assert pool._pending == []
    assert pool._pending_ids == set()


@pytest.mark.parametrize(
    ("fmt", "bytes_per_pixel"),
    [
        (spy.Format.rgba32_float, 16),
        (spy.Format.rgba16_float, 8),
        (spy.Format.rg32_float, 8),
        (spy.Format.rg16_float, 4),
        (spy.Format.r32_float, 4),
        (spy.Format.r16_float, 2),
    ],
)
def test_free_byte_accounting_for_supported_formats(
    fmt: spy.Format, bytes_per_pixel: int
) -> None:
    pool = _pool()
    tex = pool.alloc(7, 5, fmt)

    pool.recycle(tex)

    assert pool._free_bytes == 7 * 5 * bytes_per_pixel
    _assert_free_accounting(pool)


def test_budget_evicts_older_size_keys_before_hot_key() -> None:
    pool = _pool()
    pool.FREE_BYTE_BUDGET = 500
    old_rgba32 = pool.alloc(4, 4, spy.Format.rgba32_float)  # 256 bytes
    old_rgba16 = pool.alloc(4, 4, spy.Format.rgba16_float)  # 128 bytes
    hot = pool.alloc(5, 5, spy.Format.rgba32_float)  # 400 bytes

    pool.recycle(old_rgba32)
    pool.recycle(old_rgba16)
    pool.recycle(hot)

    assert _free_textures(pool) == [hot]
    assert pool._free_bytes == 400
    _assert_free_accounting(pool)


def test_budget_trims_hot_key_to_one_most_recent_texture() -> None:
    pool = _pool()
    pool.FREE_BYTE_BUDGET = 300
    textures = [pool.alloc(4, 4, spy.Format.rgba32_float) for _ in range(3)]

    for tex in textures:
        pool.recycle(tex)

    assert _free_textures(pool) == [textures[-1]]
    assert pool._free_bytes == 256
    _assert_free_accounting(pool)


def test_single_hot_texture_is_retained_when_larger_than_budget() -> None:
    pool = _pool()
    pool.FREE_BYTE_BUDGET = 128
    oversized = pool.alloc(4, 4, spy.Format.rgba32_float)

    pool.recycle(oversized)

    assert _free_textures(pool) == [oversized]
    assert pool._free_bytes == 256
    _assert_free_accounting(pool)


def test_per_key_cap_drops_excess_textures_and_accounts_exactly() -> None:
    pool = _pool()
    pool.MAX_FREE_PER_KEY = 3
    textures = [pool.alloc(6, 5, spy.Format.rgba16_float) for _ in range(5)]

    for tex in textures:
        pool.recycle(tex)

    assert _free_textures(pool) == textures[:3]
    assert pool._free_bytes == 3 * 6 * 5 * 8
    _assert_free_accounting(pool)


def test_pending_aliases_are_capped_and_accounted_by_identity() -> None:
    pool = _pool()
    pool.MAX_PENDING = 3
    held: list[spy.Texture] = []

    for width in range(5, 10):
        tex = pool.alloc(width, 4, spy.Format.rgba32_float)
        held.append(tex)
        owner = _Owner()
        pool.auto_recycle(owner, tex)
        del owner
        gc.collect()

    assert [id(tex) for tex in pool._pending] == [id(tex) for tex in held[-3:]]
    assert len(pool._pending_ids) == pool.MAX_PENDING
    _assert_disjoint_queues(pool)


def test_changing_size_workload_stays_within_budget_accounting() -> None:
    pool = _pool()
    pool.FREE_BYTE_BUDGET = 4096

    for iteration in range(100):
        width = 4 + iteration % 5
        height = 5 + iteration % 4
        fmt = (
            spy.Format.rgba16_float
            if iteration % 2
            else spy.Format.rgba32_float
        )
        pool.recycle(pool.alloc(width, height, fmt))
        _assert_free_accounting(pool)
        assert pool._free_bytes <= pool.FREE_BYTE_BUDGET

    assert 1 < pool.stats()["free"] < 20


def test_frame_exception_recycles_every_owned_texture() -> None:
    pool = _pool()
    allocated_ids: set[int] = set()

    with pytest.raises(RuntimeError, match="deliberate"):
        with pool.frame():
            allocated_ids.add(id(pool.alloc(15, 13, spy.Format.rgba32_float)))
            allocated_ids.add(id(pool.alloc(15, 13, spy.Format.rgba32_float)))
            raise RuntimeError("deliberate")

    assert pool._frames == []
    assert pool._frame_owner is None
    assert pool.stats()["free"] == 2
    borrowed_ids = {
        id(pool.alloc(15, 13, spy.Format.rgba32_float)),
        id(pool.alloc(15, 13, spy.Format.rgba32_float)),
    }
    assert borrowed_ids == allocated_ids


def test_top_level_escape_keeps_only_result_out_of_free_list() -> None:
    pool = _pool()

    with pool.frame() as frame:
        scratch = pool.alloc(16, 14, spy.Format.rgba16_float)
        result = pool.alloc(16, 14, spy.Format.rgba16_float)
        assert frame.escape(result) is result

    free_ids = {id(tex) for textures in pool._free.values() for tex in textures}
    assert free_ids == {id(scratch)}
    assert id(result) not in free_ids
    _assert_disjoint_queues(pool)


def test_retained_top_level_escape_survives_exception() -> None:
    pool = _pool()
    retained: list[spy.Texture] = []

    with pytest.raises(RuntimeError, match="after escape"):
        with pool.frame() as frame:
            result = pool.alloc(17, 14, spy.Format.rgba32_float)
            retained.append(frame.escape(result))
            raise RuntimeError("after escape")

    assert id(retained[0]) not in pool._free_ids
    assert pool._frames == []
    assert pool._frame_owner is None
    pool.recycle(retained.pop())
    _assert_disjoint_queues(pool)


def test_nested_escape_transfers_owned_texture_to_parent_frame() -> None:
    pool = _pool()

    with pool.frame() as outer:
        with pool.frame() as inner:
            intermediate = pool.alloc(17, 15, spy.Format.rgba32_float)
            inner.escape(intermediate)

        assert id(intermediate) in outer._records
        assert id(intermediate) not in pool._free_ids

    assert pool._free_ids == {id(intermediate)}
    _assert_disjoint_queues(pool)


def test_nested_escape_does_not_claim_passthrough_texture() -> None:
    pool = _pool()
    external = pool.alloc(18, 16, spy.Format.rgba32_float)

    with pool.frame() as outer:
        with pool.frame() as inner:
            inner.escape(external)
        assert id(external) not in outer._records

    assert id(external) not in pool._free_ids
    pool.recycle(external)
    _assert_disjoint_queues(pool)


def test_nested_exception_recycles_both_frame_levels() -> None:
    pool = _pool()
    allocated_ids: set[int] = set()

    with pytest.raises(RuntimeError, match="inner failed"):
        with pool.frame():
            outer_tex = pool.alloc(19, 16, spy.Format.rgba16_float)
            allocated_ids.add(id(outer_tex))
            del outer_tex
            with pool.frame():
                inner_tex = pool.alloc(19, 16, spy.Format.rgba16_float)
                allocated_ids.add(id(inner_tex))
                del inner_tex
                raise RuntimeError("inner failed")

    assert pool._free_ids == allocated_ids
    assert pool._frames == []
    assert pool._frame_owner is None
    _assert_disjoint_queues(pool)


def test_outer_exception_recycles_nested_escaped_intermediate() -> None:
    pool = _pool()
    intermediate_id = -1

    with pytest.raises(RuntimeError, match="outer failed"):
        with pool.frame():
            with pool.frame() as inner:
                intermediate = pool.alloc(19, 17, spy.Format.rgba32_float)
                intermediate_id = id(intermediate)
                inner.escape(intermediate)
            del intermediate
            raise RuntimeError("outer failed")

    assert intermediate_id in pool._free_ids
    assert pool._frames == []
    assert pool._frame_owner is None
    _assert_disjoint_queues(pool)
