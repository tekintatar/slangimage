# runtime/texture_pool.py

from __future__ import annotations

import os
import sys
import threading
import weakref
from contextlib import contextmanager
from typing import Dict, List, Tuple

import slangpy as spy

from .textures import TEX_USAGE_RW


def _calibrate_finalize_refcount() -> int:
    """Refcount a finalize callback sees for its argument when the callback
    is the ONLY remaining owner (args tuple + parameter + getrefcount temp).
    Measured, not assumed — it is a CPython implementation detail."""
    class _Anchor:  # bare object() is not weakref-able
        pass

    probe: list = []
    anchor = _Anchor()
    weakref.finalize(anchor, lambda t: probe.append(sys.getrefcount(t)), _Anchor())
    del anchor
    return probe[0] if probe else 3


_FINALIZE_BASELINE = _calibrate_finalize_refcount()

# bytes per pixel by format id, for the free-list byte budget
_BPP = {}
for _name, _bpp in (
    # no 3-channel formats exist on the GPU (RGB auto-expands to RGBA)
    ("rgba32_float", 16), ("rgba16_float", 8),
    ("rg32_float", 8), ("rg16_float", 4),
    ("r32_float", 4), ("r16_float", 2),
):
    _fmt = getattr(spy.Format, _name, None)
    if _fmt is not None:
        _BPP[int(_fmt.value) if hasattr(_fmt, "value") else int(_fmt)] = _bpp


def _tex_bytes(tex: spy.Texture) -> int:
    key = _tex_key(tex)
    return key[0] * key[1] * _BPP.get(key[2], 16)

# One pool per device. Weak keys prevent stale id reuse, but each pool owns its
# device, so explicit release_device() is required to remove the cache root.
_pools: "weakref.WeakKeyDictionary[spy.Device, TexturePool]" = weakref.WeakKeyDictionary()


def get_pool(device: spy.Device) -> "TexturePool":
    from .device import _claim_device_thread

    _claim_device_thread(device)
    pool = _pools.get(device)
    if pool is None:
        pool = TexturePool(device)
        _pools[device] = pool
    return pool


def alloc_output(
    device: spy.Device,
    width: int,
    height: int,
    fmt: spy.Format,
    usage: spy.TextureUsage = TEX_USAGE_RW,
) -> spy.Texture:
    """Allocate an op output texture through the device's pool.

    Inside a pool frame the texture is returned to the free list when the
    frame ends (unless escaped); outside a frame this is equivalent to
    device.create_texture except that it may reuse a previously recycled
    texture of the same exact size/format/usage.
    """
    return get_pool(device).alloc(width, height, fmt, usage)


def _tex_key(tex: spy.Texture) -> Tuple[int, int, int, int]:
    d = tex.desc
    fmt = d.format
    fmt_id = int(fmt.value) if hasattr(fmt, "value") else int(fmt)
    return (int(tex.width), int(tex.height), fmt_id, int(d.usage))


class _Frame:
    """Tracks textures allocated here and transfers escapes to its parent."""

    __slots__ = ("_pool", "_parent", "_records")

    def __init__(self, pool: TexturePool, parent: _Frame | None = None) -> None:
        self._pool = pool
        self._parent = parent
        self._records: Dict[int, spy.Texture] = {}

    def escape(self, tex: spy.Texture) -> spy.Texture:
        """Transfer an owned result to the parent frame or external caller."""
        self._pool._check_thread()
        tex_id = id(tex)
        owned = self._records.pop(tex_id, None)
        if owned is not None and self._parent is not None:
            self._parent._records[tex_id] = owned
        return tex


class TexturePool:
    """
    Exact-size texture pool with frame scopes.

    - alloc() pops a free texture of the exact (w, h, format, usage) or
      creates one. GPU work on a single queue is ordered, so reusing a
      texture whose previous consumer was already dispatched is safe.
    - frame(): textures allocated inside are returned to the free list when
      the frame exits, except those escape()d. Effects wrap _apply_impl in a
      frame, so effect chains stop allocating once the pool is warm. An escape
      from a nested frame transfers ownership to its parent; each enclosing
      frame must escape a result that continues outward.
    - recycle(): eager return for internal scratch textures (ping-pong).
    """

    # Effect chains park all their same-size intermediates under one key at
    # frame end, so the cap must cover deep multi-op effect chains.
    MAX_FREE_PER_KEY = 16

    # Total bytes kept in free lists. When exceeded, oldest keys are evicted
    # first — mixed-size workloads shed stale sizes instead of hoarding them.
    # One texture under the current hot key is retained even if it alone exceeds
    # the budget, so the effective bound is max(budget, one hot texture).
    FREE_BYTE_BUDGET = int(os.environ.get("SLANGIMAGE_POOL_BUDGET_MB", "1536")) * 1024 * 1024

    # Textures whose Image died while something transient still referenced
    # them (slangpy's call-data cache keeps the last dispatch's args until
    # the next call). Re-checked on every alloc; capped so held-forever
    # textures fall out instead of pinning VRAM.
    MAX_PENDING = 8

    def __init__(self, device: spy.Device) -> None:
        from .device import _claim_device_thread

        _claim_device_thread(device)
        self.device = device
        self._free: Dict[Tuple[int, int, int, int], List[spy.Texture]] = {}
        self._free_ids: set[int] = set()  # mirrors free-list contents exactly
        self._free_bytes = 0
        self._pending: List[spy.Texture] = []
        self._pending_ids: set[int] = set()  # mirrors pending contents exactly
        self._frames: List[_Frame] = []
        self._frame_owner: int | None = None  # thread that opened the outer frame
        self.created = 0
        self.reused = 0

    def _check_thread(self) -> None:
        from .device import _claim_device_thread

        _claim_device_thread(self.device)

    # ---------------- allocation ----------------

    def alloc(
        self,
        width: int,
        height: int,
        fmt: spy.Format,
        usage: spy.TextureUsage = TEX_USAGE_RW,
    ) -> spy.Texture:
        self._check_thread()
        self._drain_pending()
        fmt_id = int(fmt.value) if hasattr(fmt, "value") else int(fmt)
        key = (int(width), int(height), fmt_id, int(usage))
        lst = self._free.get(key)
        if lst:
            tex = lst.pop()
            self._free_ids.discard(id(tex))
            self._free_bytes -= _tex_bytes(tex)
            self.reused += 1
        else:
            tex = self.device.create_texture(
                width=width, height=height, format=fmt, usage=usage
            )
            self.created += 1
        if self._frames:
            self._frames[-1]._records[id(tex)] = tex
        return tex

    def recycle(self, tex: spy.Texture) -> None:
        """Return a texture to the free list immediately. The caller must not
        use it afterwards; contents are undefined for the next borrower."""
        self._check_thread()
        for f in self._frames:
            f._records.pop(id(tex), None)
        self._push_free(tex)

    def auto_recycle(self, owner: object, tex: spy.Texture) -> None:
        """Return tex to the free list when `owner` (the Image wrapping it)
        is garbage collected — makes out-of-frame functional calls reuse
        instead of re-creating (device.create_texture costs ~0.3 ms on
        CUDA; see benchmarks/compare_augmentation.py dispatch-floor notes).
        Conservative: recycles only if the finalizer holds the last
        reference, so user code that kept img.texture keeps sole use."""
        self._check_thread()
        weakref.finalize(owner, self._finalize_recycle, tex)

    def _finalize_recycle(self, tex: spy.Texture) -> None:
        from .device import _is_device_thread

        # A weakref callback runs in whichever thread drops the last owner.
        # Cross-thread Image transfer is unsupported; don't compound it by
        # mutating the original thread's pool from an unraisable finalizer.
        if not _is_device_thread(self.device):
            return
        tex_id = id(tex)
        # A texture may have more than one wrapper/finalizer, or a caller may
        # explicitly recycle it before its wrapper dies. Keep the two owning
        # queues mutually exclusive and make every recycle path idempotent.
        if tex_id in self._free_ids or tex_id in self._pending_ids:
            return
        if sys.getrefcount(tex) <= _FINALIZE_BASELINE:
            self._push_free(tex)
        else:
            # something transient still references it — retry at next alloc
            self._pending.append(tex)
            self._pending_ids.add(tex_id)
            if len(self._pending) > self.MAX_PENDING:
                dropped = self._pending.pop(0)  # genuinely held: let it go to GC
                self._pending_ids.discard(id(dropped))

    def _drain_pending(self) -> None:
        if not self._pending:
            return
        still_held = []
        still_held_ids = set()
        for tex in self._pending:
            tex_id = id(tex)
            if tex_id in self._free_ids:
                continue
            # refs here: this list + loop var + getrefcount temp = baseline
            if sys.getrefcount(tex) <= _FINALIZE_BASELINE:
                self._push_free(tex)
            else:
                still_held.append(tex)
                still_held_ids.add(tex_id)
        self._pending = still_held
        self._pending_ids = still_held_ids

    def _push_free(self, tex: spy.Texture) -> None:
        tex_id = id(tex)
        if tex_id in self._free_ids:
            return  # already pooled (finalizer/frame overlap) — never twice
        if tex_id in self._pending_ids:
            self._pending = [
                candidate for candidate in self._pending if candidate is not tex
            ]
            self._pending_ids.discard(tex_id)
        key = _tex_key(tex)
        lst = self._free.setdefault(key, [])
        if len(lst) < self.MAX_FREE_PER_KEY:
            lst.append(tex)
            self._free_ids.add(tex_id)
            self._free_bytes += _tex_bytes(tex)
            self._evict(keep=key)
        # else: drop the reference; GC frees the GPU memory

    def _evict(self, keep: Tuple[int, int, int, int]) -> None:
        """Shed oldest free keys until the byte budget holds. The key being
        pushed right now is evicted last and retains one texture as the hot
        working set, even when that texture alone is larger than the budget."""
        if self._free_bytes <= self.FREE_BYTE_BUDGET:
            return
        for key in list(self._free.keys()):
            if key == keep:
                continue
            lst = self._free[key]
            while lst and self._free_bytes > self.FREE_BYTE_BUDGET:
                dropped = lst.pop(0)
                self._free_ids.discard(id(dropped))
                self._free_bytes -= _tex_bytes(dropped)
            if not lst:
                del self._free[key]
            if self._free_bytes <= self.FREE_BYTE_BUDGET:
                return
        # only the hot key remains; trim it too if still over budget
        lst = self._free.get(keep, [])
        while len(lst) > 1 and self._free_bytes > self.FREE_BYTE_BUDGET:
            dropped = lst.pop(0)
            self._free_ids.discard(id(dropped))
            self._free_bytes -= _tex_bytes(dropped)

    # ---------------- frames ----------------

    @contextmanager
    def frame(self):
        self._check_thread()
        # A device/pool is not thread-safe. Frames are a shared LIFO, so two
        # threads opening frames on one device would interleave records and
        # recycle each other's live outputs. Detect that and fail loudly rather
        # than corrupt results. (Best-effort GIL-level check; the contract is
        # one device per thread — see get_default_device.)
        tid = threading.get_ident()
        if self._frames and self._frame_owner != tid:
            raise RuntimeError(
                "texture pool used from two threads at once — a slangimage "
                "device is not thread-safe. Use one device per thread."
            )
        if not self._frames:
            self._frame_owner = tid
        f = _Frame(self, self._frames[-1] if self._frames else None)
        self._frames.append(f)
        try:
            yield f
        finally:
            self._frames.pop()
            if not self._frames:
                self._frame_owner = None
            for tex in f._records.values():
                self._push_free(tex)

    # ---------------- maintenance ----------------

    def clear(self) -> None:
        """Drop all pooled textures (frees GPU memory on next GC)."""
        self._check_thread()
        self._free.clear()
        self._free_ids.clear()
        self._free_bytes = 0
        self._pending.clear()
        self._pending_ids.clear()

    def stats(self) -> dict:
        self._check_thread()
        free = sum(len(v) for v in self._free.values())
        return {
            "created": self.created,
            "reused": self.reused,
            "free": free,
            "free_mb": round(self._free_bytes / 1e6, 1),
        }
