"""Device creation: AUTO keeps a CUDA device only if it can compile, and a
backend that cannot start explains what is missing.

The host cases replace native device creation and the compile probe with
recorders; the last cases run the real probe and the image constructors on the
selected device.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image as PILImage

import slangimage as si
from slangimage.runtime import device as device_runtime
from slangimage.runtime.device import Backend, DeviceManager

# What the first operation raised on a CUDA device when CUDA_PATH was set but
# NVRTC was not on the dynamic loader path (SlangPy 0.43.1, Ubuntu 24.04).
E52002 = (
    "m_rhi_command_encoder->finish(rhi_command_buffer.writeRef()) failed with error: "
    "-2147467259 (SLANG_FAIL)\nRHI messages:\n"
    "slang: error[E52002]: pass-through compiler not found\n"
    "/project/src/sgl/device/command.cpp:963 in function finish"
)
CREATE_FAILED = (
    "Failed to create device!\n"
    "/project/src/sgl/device/device.cpp:264 in function Device"
)


@pytest.fixture
def fake_backends(monkeypatch):
    """Record device creation, probing and closing instead of touching a GPU."""
    state = SimpleNamespace(
        system="Linux", unavailable=set(), probe_error=None,
        created=[], probed=[], closed=[],
    )

    def create(self, dtype):
        if dtype.name in state.unavailable:
            raise RuntimeError(CREATE_FAILED)
        state.created.append(dtype.name)
        return SimpleNamespace(kind=dtype.name, close=lambda: state.closed.append(dtype.name))

    def probe(device):
        state.probed.append(device.kind)
        if state.probe_error is not None:
            raise RuntimeError(state.probe_error)

    monkeypatch.setattr(device_runtime.platform, "system", lambda: state.system)
    monkeypatch.setattr(DeviceManager, "_create_device_with_type", create)
    monkeypatch.setattr(device_runtime, "_verify_runtime_compiler", probe)
    return state


def test_auto_keeps_cuda_that_compiles(fake_backends, recwarn):
    assert DeviceManager(Backend.AUTO).device.kind == "cuda"
    assert fake_backends.probed == ["cuda"]
    assert fake_backends.closed == []
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


def test_auto_falls_back_to_vulkan_when_cuda_cannot_compile(fake_backends):
    fake_backends.probe_error = E52002
    with pytest.warns(RuntimeWarning, match=r"E52002.*trying Vulkan instead.*NVRTC"):
        manager = DeviceManager(Backend.AUTO)
    assert manager.device.kind == "vulkan"
    assert fake_backends.created == ["cuda", "vulkan"]
    assert fake_backends.closed == ["cuda"]


def test_auto_does_not_probe_metal_or_vulkan(fake_backends):
    fake_backends.system = "Darwin"
    fake_backends.probe_error = "the probe must not run"
    assert DeviceManager(Backend.AUTO).device.kind == "metal"
    fake_backends.unavailable = {"metal"}
    assert DeviceManager(Backend.AUTO).device.kind == "vulkan"
    assert fake_backends.probed == []


def test_auto_failure_lists_every_attempt(fake_backends):
    fake_backends.unavailable = {"vulkan"}
    fake_backends.probe_error = E52002
    with pytest.warns(RuntimeWarning):
        with pytest.raises(RuntimeError) as info:
            DeviceManager(Backend.AUTO)
    message = str(info.value)
    assert message.startswith("no supported GPU device found")
    assert "cuda: " in message and "E52002" in message
    assert "vulkan: Failed to create device!" in message
    assert "in function" not in message
    assert fake_backends.closed == ["cuda"]


def test_explicit_cuda_that_cannot_compile_raises_setup_guidance(fake_backends):
    fake_backends.probe_error = E52002
    with pytest.raises(RuntimeError) as info:
        DeviceManager(Backend.CUDA)
    message = str(info.value)
    assert message.startswith("created a CUDA device but could not compile a test kernel")
    assert "E52002" in message and "NVRTC" in message
    assert 'si.device("vulkan")' in message
    assert E52002 in str(info.value.__cause__)
    assert fake_backends.closed == ["cuda"]


@pytest.mark.parametrize(
    ("system", "backend", "hint"),
    [
        ("Linux", Backend.CUDA, "NVRTC"),
        ("Darwin", Backend.CUDA, "CUDA is not available on macOS"),
        ("Linux", Backend.VULKAN, "libvulkan"),
        ("Darwin", Backend.VULKAN, "use Metal"),
        ("Linux", Backend.METAL, "only on macOS"),
    ],
)
def test_explicit_backend_that_cannot_start_explains_why(fake_backends, system, backend, hint):
    fake_backends.system = system
    fake_backends.unavailable = {backend.value}
    with pytest.raises(RuntimeError) as info:
        DeviceManager(backend)
    message = str(info.value)
    assert message.startswith(f"could not create a {backend.value} device (Failed to create device!)")
    assert hint in message
    assert "in function" not in message
    assert str(info.value.__cause__) == CREATE_FAILED
    assert fake_backends.probed == []


@pytest.mark.parametrize("backend", [Backend.CPU, "cpu"])
def test_cpu_backend_is_still_rejected_up_front(fake_backends, backend):
    with pytest.raises(RuntimeError, match="the CPU backend is not supported"):
        DeviceManager(backend)
    assert fake_backends.created == []


@pytest.mark.parametrize(("name", "kind"), [("auto", "cuda"), ("cuda", "cuda"), ("vulkan", "vulkan")])
def test_backend_names_are_accepted_as_strings(fake_backends, name, kind):
    manager = DeviceManager(name)
    assert manager.backend is Backend(name)
    assert manager.device.kind == kind


def test_unknown_backend_name_is_rejected(fake_backends):
    with pytest.raises(ValueError, match="not a valid Backend"):
        DeviceManager("directx")
    assert fake_backends.created == []


def test_runtime_probe_compiles_on_the_selected_device(ctx):
    device, _ = ctx
    device_runtime._verify_runtime_compiler(device)


@pytest.mark.parametrize("as_member", [False, True], ids=["name", "member"])
def test_image_constructors_accept_a_backend_name(ctx, tmp_path, as_member):
    device, _ = ctx
    name = os.environ.get("SLANGIMAGE_BACKEND", "auto")
    backend = Backend(name) if as_member else name
    pil = PILImage.fromarray(np.full((4, 4, 3), 64, dtype=np.uint8), "RGB")
    path = tmp_path / "in.png"
    pil.save(path)

    images = [
        si.from_numpy(np.full((4, 4, 4), 0.25, dtype=np.float32), device=backend),
        si.from_pil(pil, device=backend),
        si.load(path, device=backend),
    ]
    assert all(image.device is device for image in images)
    np.testing.assert_allclose(images[0].to_numpy(), 0.25)
    np.testing.assert_allclose(images[2].to_numpy()[..., :3], 64 / 255, atol=1e-3)


def test_diff_device_failure_explains_the_cuda_runtime_compiler():
    import slangimage.diff as sid

    def fail(**kwargs):
        raise RuntimeError(CREATE_FAILED)

    fake_spy = SimpleNamespace(DeviceType=SimpleNamespace(cuda="cuda"), create_torch_device=fail)
    with pytest.raises(RuntimeError) as info:
        sid._create_torch_cuda_device(fake_spy)
    message = str(info.value)
    assert message.startswith(
        "slangimage.diff could not create SlangPy's CUDA device (Failed to create device!)"
    )
    assert "NVRTC" in message and "PyTorch extra" in message
    assert "in function" not in message
    assert str(info.value.__cause__) == CREATE_FAILED


def test_image_constructors_reject_an_unknown_backend_name():
    with pytest.raises(ValueError, match="not a valid Backend"):
        si.from_numpy(np.zeros((2, 2, 4), dtype=np.float32), device="directx")
