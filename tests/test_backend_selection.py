"""Collection must respect backend selection before probing CUDA."""

import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import slangpy as spy

import conftest


@pytest.mark.parametrize(
    "module", ["test_diff.py", "test_diff_core_contracts.py", "test_nn.py"]
)
@pytest.mark.parametrize("backend", [None, "auto", "cuda", "metal", "vulkan"])
def test_autograd_collection_backend(monkeypatch, module, backend):
    if backend is None:
        monkeypatch.delenv("SLANGIMAGE_BACKEND", raising=False)
    else:
        monkeypatch.setenv("SLANGIMAGE_BACKEND", backend)

    class CudaProbeReached(Exception):
        pass

    def probe():
        raise CudaProbeReached

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_available=probe),
    ))
    path = Path(__file__).with_name(module)
    if backend in {"metal", "vulkan"}:
        with pytest.raises(pytest.skip.Exception, match="excluded by SLANGIMAGE_BACKEND"):
            runpy.run_path(str(path))
    else:
        with pytest.raises(CudaProbeReached):
            runpy.run_path(str(path))


@pytest.mark.parametrize("backend", ["auto", "metal", "cuda", "vulkan"])
def test_texture_fixture_uses_requested_backend(monkeypatch, backend):
    monkeypatch.setenv("SLANGIMAGE_BACKEND", backend)
    device = SimpleNamespace(info=SimpleNamespace(
        type=spy.DeviceType.metal if backend == "auto" else getattr(spy.DeviceType, backend),
        adapter_name="test adapter",
    ))
    requested = []

    def get_device(selection):
        requested.append(selection)
        return device

    cache = object()
    monkeypatch.setattr(conftest.si, "get_default_device", get_device)
    monkeypatch.setattr(conftest, "get_module_cache", lambda selected: cache)
    config = SimpleNamespace()
    assert conftest.ctx.__wrapped__(SimpleNamespace(config=config)) == (device, cache)
    assert requested == [backend]
    assert "test adapter" in config._slangimage_texture_device


def test_texture_fixture_rejects_wrong_backend(monkeypatch):
    monkeypatch.setenv("SLANGIMAGE_BACKEND", "vulkan")
    monkeypatch.setattr(conftest.si, "get_default_device", lambda backend: SimpleNamespace(
        info=SimpleNamespace(type=spy.DeviceType.cuda),
    ))
    with pytest.raises(AssertionError, match="Requested vulkan"):
        conftest.ctx.__wrapped__(SimpleNamespace(config=SimpleNamespace()))
