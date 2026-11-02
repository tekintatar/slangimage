"""Device/cache teardown contracts, including surviving caller-owned objects."""

from __future__ import annotations

import gc
import weakref
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import slangpy as spy

import slangimage as si
import slangimage.runtime.device as device_runtime
from slangimage.ops import _instances, _op_for_device
from slangimage.runtime.modules import _caches, get_module_cache
from slangimage.runtime.texture_pool import _pools, get_pool
from slangimage.runtime.textures import TEX_USAGE_RW


@dataclass(frozen=True)
class _Desc:
    format: spy.Format
    usage: spy.TextureUsage


class _Texture:
    def __init__(
        self,
        device: _Device,
        width: int,
        height: int,
        fmt: spy.Format,
        usage: spy.TextureUsage,
    ) -> None:
        self.device = device
        self.width = width
        self.height = height
        self.format = fmt
        self.desc = _Desc(fmt, usage)


class _Device:
    def __init__(self) -> None:
        self.closed = False

    def create_texture(
        self,
        *,
        width: int,
        height: int,
        format: spy.Format,
        usage: spy.TextureUsage,
    ) -> _Texture:
        return _Texture(self, width, height, format, usage)

    def close(self) -> None:
        self.closed = True


class _Op:
    def __init__(self, device: _Device, modules: object) -> None:
        self.device = device
        self.modules = modules


class _Owner:
    pass


def _populate(device: _Device):
    pool = get_pool(device)  # type: ignore[arg-type]
    cache = get_module_cache(device)  # type: ignore[arg-type]
    op = _op_for_device(device, _Op)  # type: ignore[arg-type]
    return pool, cache, op


def _assert_not_globally_cached(device: object) -> None:
    assert device not in _pools
    assert device not in _caches
    assert device not in _instances
    assert all(manager.device is not device for manager in device_runtime._default_managers.values())


def test_release_removes_all_cache_roots_and_clears_pool() -> None:
    device = _Device()
    pool, cache, op = _populate(device)
    tex = pool.alloc(7, 5, spy.Format.rgba32_float)
    pool.recycle(tex)
    pending_tex = pool.alloc(8, 6, spy.Format.rgba32_float)
    owner = _Owner()
    pool.auto_recycle(owner, pending_tex)
    del owner
    gc.collect()
    assert pool._pending
    device_runtime._default_managers[si.Backend.CPU] = SimpleNamespace(device=device)

    si.release_device(device)  # type: ignore[arg-type]

    _assert_not_globally_cached(device)
    assert not device.closed
    assert pool.stats()["free"] == 0
    assert pool._pending == []
    assert get_pool(device) is not pool  # type: ignore[arg-type]
    assert get_module_cache(device) is not cache  # type: ignore[arg-type]
    assert _op_for_device(device, _Op) is not op  # type: ignore[arg-type]
    si.release_device(device)  # idempotent  # type: ignore[arg-type]


def test_release_isolated_to_exact_device() -> None:
    first = _Device()
    second = _Device()
    _populate(first)
    second_pool, second_cache, second_op = _populate(second)

    try:
        si.release_device(first)  # type: ignore[arg-type]
        _assert_not_globally_cached(first)
        assert _pools.get(second) is second_pool
        assert _caches.get(second) is second_cache
        assert _instances[second][_Op] is second_op
    finally:
        si.release_device(first)  # type: ignore[arg-type]
        si.release_device(second)  # type: ignore[arg-type]


def test_package_cache_roots_drop_after_explicit_release() -> None:
    device = _Device()
    _populate(device)
    device_ref = weakref.ref(device)

    del device
    gc.collect()
    assert device_ref() is not None

    si.release_device(device_ref())  # type: ignore[arg-type]
    gc.collect()
    assert device_ref() is None


def test_default_manager_recreates_device_after_release(monkeypatch) -> None:
    assert si.Backend.CPU not in device_runtime._default_managers
    created: list[_Device] = []

    class _Manager:
        def __init__(self, backend: si.Backend) -> None:
            assert backend is si.Backend.CPU
            self.device = _Device()
            created.append(self.device)

    monkeypatch.setattr(device_runtime, "DeviceManager", _Manager)
    first = device_runtime.get_default_device(si.Backend.CPU)
    si.release_device(first)  # type: ignore[arg-type]
    second = device_runtime.get_default_device(si.Backend.CPU)

    try:
        assert second is not first
        assert created == [first, second]
    finally:
        si.release_device(second)  # type: ignore[arg-type]


def test_surviving_image_remains_valid_and_recreates_caches(ctx) -> None:
    backend_by_type = {
        spy.DeviceType.metal: si.Backend.METAL,
        spy.DeviceType.cuda: si.Backend.CUDA,
        spy.DeviceType.vulkan: si.Backend.VULKAN,
    }
    selected, _ = ctx
    manager = si.DeviceManager(backend_by_type[selected.info.type])
    device = manager.device
    pixels = np.random.default_rng(31).random((16, 18, 4)).astype(np.float32)
    pixels[..., 3] = 1.0
    image = si.from_numpy(pixels, precision="fp32", device=device)
    output = si.ops.invert(image)
    expected = output.to_numpy()
    old_pool = get_pool(device)
    old_cache = get_module_cache(device)
    spare = old_pool.alloc(7, 5, spy.Format.rgba32_float, TEX_USAGE_RW)
    old_pool.recycle(spare)
    assert device in _pools and device in _caches and device in _instances

    try:
        si.release_device(device)
        _assert_not_globally_cached(device)
        assert old_pool.stats()["free"] == 0
        np.testing.assert_array_equal(output.to_numpy(), expected)

        restored = si.ops.invert(output)
        np.testing.assert_allclose(restored.to_numpy(), pixels, atol=2e-6)
        assert get_pool(device) is not old_pool
        assert get_module_cache(device) is not old_cache
        assert device in _instances
    finally:
        del output, image, spare
        if "restored" in locals():
            del restored
        gc.collect()
        si.release_device(device)
