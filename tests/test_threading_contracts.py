"""Thread-affinity contracts for devices and all package-owned runtime state."""

from __future__ import annotations

import gc
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import slangpy as spy

import slangimage as si
import slangimage.runtime.device as device_runtime
from slangimage.runtime.modules import ModuleCache
from slangimage.runtime.texture_pool import TexturePool, get_pool
from slangimage.fx.base import Effect


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


class _Effect(Effect):
    def _apply_impl(self, src_tex, **params):
        return src_tex


def _pool() -> TexturePool:
    return TexturePool(_Device())  # type: ignore[arg-type]


def _thread_error(callable_) -> str:
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(callable_)
        with pytest.raises(RuntimeError, match="owned by") as caught:
            future.result(timeout=5)
    return str(caught.value)


def test_process_default_rejects_another_thread(monkeypatch) -> None:
    class _Manager:
        def __init__(self, backend: si.Backend) -> None:
            assert backend is si.Backend.CPU
            self.device = _Device()

    monkeypatch.setattr(device_runtime, "DeviceManager", _Manager)
    device = device_runtime.get_default_device(si.Backend.CPU)
    try:
        message = _thread_error(
            lambda: device_runtime.get_default_device(si.Backend.CPU)
        )
        assert "process-wide default device is single-thread only" in message
    finally:
        si.release_device(device)  # type: ignore[arg-type]


def test_concurrent_first_default_use_builds_once_and_rejects_sharing(
    monkeypatch,
) -> None:
    created: list[_Device] = []
    created_lock = threading.Lock()
    start = threading.Barrier(3)
    release_owner = threading.Event()
    results: queue.Queue[tuple[str, object]] = queue.Queue()

    class _Manager:
        def __init__(self, backend: si.Backend) -> None:
            assert backend is si.Backend.CPU
            with created_lock:
                self.device = _Device()
                created.append(self.device)

    monkeypatch.setattr(device_runtime, "DeviceManager", _Manager)

    def worker() -> None:
        start.wait()
        try:
            device = device_runtime.get_default_device(si.Backend.CPU)
        except RuntimeError as exc:
            results.put(("rejected", str(exc)))
            return
        results.put(("owner", device))
        assert release_owner.wait(timeout=5)
        si.release_device(device)  # type: ignore[arg-type]

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(worker) for _ in range(2)]
        start.wait()
        outcomes = [results.get(timeout=5), results.get(timeout=5)]
        release_owner.set()
        for future in futures:
            future.result(timeout=5)

    assert len(created) == 1
    assert sorted(kind for kind, _ in outcomes) == ["owner", "rejected"]
    rejected = next(value for kind, value in outcomes if kind == "rejected")
    assert "single-thread only" in str(rejected)


def test_explicit_device_construction_is_serialized(monkeypatch) -> None:
    state_lock = threading.Lock()
    start = threading.Barrier(3)
    active = 0
    max_active = 0

    def create(_self, _backend):
        nonlocal active, max_active
        with state_lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.03)
        with state_lock:
            active -= 1
        return _Device()

    monkeypatch.setattr(device_runtime.DeviceManager, "_create_device", create)

    def worker() -> _Device:
        start.wait()
        device = device_runtime.DeviceManager(si.Backend.CPU).device
        si.release_device(device)  # type: ignore[arg-type]
        return device  # type: ignore[return-value]

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(worker) for _ in range(2)]
        start.wait()
        devices = [future.result(timeout=5) for future in futures]

    assert max_active == 1
    assert devices[0] is not devices[1]


def test_shared_pool_rejects_sequential_cross_thread_use() -> None:
    pool = _pool()

    message = _thread_error(
        lambda: pool.alloc(8, 6, spy.Format.rgba32_float)
    )

    assert "Create a DeviceManager inside each worker" in message
    assert pool.stats()["created"] == 0


def test_shared_pool_overlap_rejection_preserves_owner_frame() -> None:
    pool = _pool()

    with pool.frame() as frame:
        first = pool.alloc(9, 7, spy.Format.rgba16_float)
        _thread_error(lambda: pool.frame().__enter__())
        _thread_error(lambda: frame.escape(first))
        second = pool.alloc(9, 7, spy.Format.rgba16_float)
        assert first is not second

    assert pool._frames == []
    assert pool._frame_owner is None
    assert pool.stats()["free"] == 2


def test_distinct_thread_owned_pools_can_overlap() -> None:
    start = threading.Barrier(2)

    def worker(width: int) -> tuple[int, int]:
        pool = _pool()
        with pool.frame():
            start.wait()
            tex = pool.alloc(width, 5, spy.Format.rgba32_float)
            result = (tex.width, pool.stats()["created"])
        return result

    with ThreadPoolExecutor(max_workers=2) as executor:
        assert sorted(executor.map(worker, (7, 8))) == [(7, 1), (8, 1)]


def test_cross_thread_finalizer_does_not_mutate_owner_pool() -> None:
    pool = _pool()
    tex = pool.alloc(10, 8, spy.Format.rgba32_float)
    owner = _Owner()
    pool.auto_recycle(owner, tex)
    holder = [owner]
    del owner

    def drop_owner() -> None:
        holder.clear()
        gc.collect()

    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(drop_owner).result(timeout=5)

    assert pool._pending == []
    assert pool.stats()["free"] == 0
    pool.recycle(tex)
    assert pool.stats()["free"] == 1


def test_module_cache_and_effect_reject_another_thread() -> None:
    device = _Device()
    cache = ModuleCache(device)  # type: ignore[arg-type]
    _thread_error(lambda: cache.get_op("invert"))
    _thread_error(
        lambda: _Effect(device, SimpleNamespace(device=device))  # type: ignore[arg-type]
    )
    assert cache._dispatchers == {}


def test_public_image_and_runtime_boundaries_reject_another_thread(ctx) -> None:
    device, _ = ctx
    pixels = np.zeros((8, 9, 4), dtype=np.float32)
    pixels[..., 3] = 1
    image = si.from_numpy(pixels, device=device)

    calls = (
        lambda: image.width,
        image.to_numpy,
        lambda: si.ops.invert(image),
        lambda: si.ops.random_noise(4, 4, device=device),
        lambda: si.prewarm(device=device, size=20),
        lambda: get_pool(device).stats(),
        lambda: si.release_device(device),
    )
    for call in calls:
        _thread_error(call)

    np.testing.assert_array_equal(image.to_numpy(), pixels)
