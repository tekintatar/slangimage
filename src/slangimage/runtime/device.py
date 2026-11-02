# runtime/device.py

from __future__ import annotations

import enum
import platform
import re
import threading
import warnings
import weakref
from pathlib import Path

import numpy as np
import slangpy as spy
from .texture_pool import get_pool


_device_threads: "weakref.WeakKeyDictionary[spy.Device, weakref.ReferenceType[threading.Thread]]" = (
    weakref.WeakKeyDictionary()
)
_device_thread_lock = threading.RLock()
_device_creation_lock = threading.Lock()


def _claim_device_thread(device: spy.Device) -> None:
    """Bind a device to its first SlangImage thread and reject later sharing."""
    try:
        weakref.ref(device)
    except TypeError:
        # A few host-only tests intentionally use duck-typed SimpleNamespace
        # devices. Real slangpy Devices are weak-referenceable and always take
        # the affinity path below.
        return
    current = threading.current_thread()
    with _device_thread_lock:
        owner_ref = _device_threads.get(device)
        if owner_ref is None:
            _device_threads[device] = weakref.ref(current)
            return
        owner = owner_ref()
        if owner is current:
            return

    owner_label = (
        f"'{owner.name}' ({owner.ident})"
        if owner is not None
        else "its original thread (which has ended)"
    )
    raise RuntimeError(
        f"slangimage device is owned by {owner_label} and cannot be used from "
        f"thread '{current.name}' ({current.ident}). Devices, Images, effects, "
        "texture pools, and module caches are not thread-safe. Create a "
        "DeviceManager inside each worker and keep all derived objects on that "
        "thread; the process-wide default device is single-thread only."
    )


def _is_device_thread(device: spy.Device) -> bool:
    """Non-raising ownership check for weakref finalizers."""
    try:
        weakref.ref(device)
    except TypeError:
        return True
    current = threading.current_thread()
    with _device_thread_lock:
        owner_ref = _device_threads.get(device)
        return owner_ref is not None and owner_ref() is current


class Backend(str, enum.Enum):
    AUTO = "auto"
    CUDA = "cuda"
    METAL = "metal"
    CPU = "cpu"      # accepted for a clear error; kernels cannot run on CPU
    VULKAN = "vulkan"


def _backend_to_device_type(backend: Backend) -> spy.DeviceType:
    if backend is Backend.CUDA:
        return spy.DeviceType.cuda
    if backend is Backend.METAL:
        return spy.DeviceType.metal
    if backend is Backend.VULKAN:
        return spy.DeviceType.vulkan
    if backend is Backend.CPU:
        # the CPU device constructs but every op fails at first dispatch
        # (SLANG_FAIL); reject up front with an actionable message instead.
        raise RuntimeError(
            "the CPU backend is not supported — slangimage kernels require a "
            "Metal, CUDA, or Vulkan GPU."
        )
    raise ValueError(f"Unknown backend {backend}")


# Creating a CUDA device only requires slang-rhi to find the driver and NVRTC.
# The Slang compiler loads NVRTC separately, through the dynamic loader, when it
# first compiles a kernel — so a partially configured toolkit (for example
# CUDA_PATH set without a matching LD_LIBRARY_PATH) yields a device whose first
# operation fails. Compiling and running a one-pixel kernel when a CUDA device is
# created surfaces that failure while another backend can still be chosen. The
# probe writes a texture, like the operators do, and never allocates a buffer:
# the first CUDA buffer allocation reserves a 256 MiB heap for the rest of the
# process, which texture-only workloads otherwise never pay.
_PROBE_NAME = "slangimage_runtime_probe"
_PROBE_SOURCE = """
void slangimage_runtime_probe(uint2 coord, [format("rgba32f")] RWTexture2D<float4> dst)
{
    dst[coord] = float4(0.25, 0.5, 0.75, 1.0);
}
"""
_PROBE_VALUE = (0.25, 0.5, 0.75, 1.0)

_NVRTC_HINT = (
    "The CUDA backend compiles kernels at run time with NVRTC and needs the CUDA "
    "runtime headers, both no newer than the CUDA version your NVIDIA driver "
    "reports (nvidia-smi). After installing them with apt, run `sudo ldconfig` so "
    "NVRTC can be found; with the pip packages, set both CUDA_PATH and "
    "LD_LIBRARY_PATH. See the README section 'CUDA on Linux: install NVRTC'"
)
_CUDA_SETUP_HINT = _NVRTC_HINT + ", or pass si.device(\"vulkan\") to use Vulkan."


# SlangPy appends the native source location ("…/device.cpp:264 in function
# Device") to its errors; it says nothing actionable to a user.
_NATIVE_LOCATION = re.compile(r"^\S+\.(?:c|cc|cpp|h|hpp):\d+ in function \S+$")


def _summarize(err: BaseException, limit: int = 300) -> str:
    """One-line, length-bounded description of a possibly multi-line error."""
    lines = (line.strip() for line in str(err).splitlines())
    kept = [line for line in lines if line and not _NATIVE_LOCATION.match(line)]
    text = " ".join(" ".join(kept).split()) or type(err).__name__
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _backend_hint(backend: Backend) -> str:
    on_macos = platform.system() == "Darwin"
    if backend is Backend.CUDA:
        if on_macos:
            return "CUDA is not available on macOS; use Metal (the default there)."
        return "It needs an NVIDIA GPU and driver. " + _CUDA_SETUP_HINT
    if backend is Backend.VULKAN:
        if on_macos:
            return "Vulkan is usually unavailable on macOS; use Metal (the default there)."
        return (
            "It needs a GPU driver with a Vulkan ICD and the Vulkan loader "
            "(libvulkan.so.1, e.g. the libvulkan1 package on Debian/Ubuntu)."
        )
    if backend is Backend.METAL:
        return "Metal is available only on macOS with Apple Silicon."
    return ""


def _verify_runtime_compiler(device: spy.Device) -> None:
    """Compile and run a one-pixel texture kernel on ``device``; raise if that fails."""
    module = spy.Module.load_from_source(device, _PROBE_NAME, _PROBE_SOURCE)
    # A raw 1x1 texture rather than the pool: this runs inside device creation,
    # before the device is registered with any SlangImage cache.
    texture = device.create_texture(
        width=1,
        height=1,
        format=spy.Format.rgba32_float,
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
    )
    module.slangimage_runtime_probe(spy.grid((1, 1)), texture)
    value = np.asarray(texture.to_numpy(), dtype=np.float32).reshape(-1)[:4]
    if not np.allclose(value, _PROBE_VALUE):
        raise RuntimeError(f"test kernel wrote {value.tolist()} instead of {list(_PROBE_VALUE)}")


def _close_quietly(device: spy.Device) -> None:
    try:
        device.close()
    except Exception:
        pass


class DeviceManager:
    """Create and own one SlangPy GPU device for the calling thread.

    ``DeviceManager(backend).device`` is the device to pass as ``device=`` to
    ``load``/``from_numpy``/``from_pil`` or to effects. ``backend`` is a
    :class:`Backend` member or its value (``"auto"``, ``"metal"``, ``"cuda"``,
    ``"vulkan"``);
    ``Backend.AUTO`` tries Metal then Vulkan on macOS, and CUDA then Vulkan
    elsewhere; a new CUDA device is kept only if it can compile and run a
    one-pixel test kernel, otherwise a ``RuntimeWarning`` explains the fallback.
    An explicitly requested backend that cannot start raises ``RuntimeError``
    with setup guidance. The device and everything derived from it belong to
    the constructing thread; tear down with ``release_device(device)`` and then
    ``device.close()`` on that thread.
    """

    def __init__(self, backend: Backend | str = Backend.AUTO):
        backend = Backend(backend)
        self.backend = backend
        # Slang's global compiler session is not thread-safe. Workers may each
        # own a device, but their construction must not overlap.
        with _device_creation_lock:
            self.device = self._create_device(backend)
        _claim_device_thread(self.device)

    @property
    def texture_pool(self):
        """The current pool for this device (re-created after release)."""
        return get_pool(self.device)

    def _create_device(self, backend: Backend) -> spy.Device:
        if backend is not Backend.AUTO:
            dtype = _backend_to_device_type(backend)
            try:
                dev = self._create_device_with_type(dtype)
            except Exception as err:
                raise RuntimeError(
                    f"could not create a {backend.value} device "
                    f"({_summarize(err)}). {_backend_hint(backend)}"
                ) from err
            if dtype == spy.DeviceType.cuda:
                try:
                    _verify_runtime_compiler(dev)
                except Exception as err:
                    _close_quietly(dev)
                    raise RuntimeError(
                        "created a CUDA device but could not compile a test kernel "
                        f"on it ({_summarize(err)}). {_CUDA_SETUP_HINT}"
                    ) from err
            return dev

        # AUTO: prefer the platform's native backend first.
        # CPU is intentionally NOT a fallback — the device constructs but no op
        # can run on it, so falling back to CPU only defers a cryptic failure.
        if platform.system() == "Darwin":
            order = (spy.DeviceType.metal, spy.DeviceType.vulkan)
        else:
            order = (spy.DeviceType.cuda, spy.DeviceType.vulkan)
        failures = []
        for candidate in order:
            try:
                dev = self._create_device_with_type(candidate)
            except Exception as err:
                failures.append(f"{candidate.name}: {_summarize(err, 160)}")
                continue
            if candidate == spy.DeviceType.cuda:
                try:
                    _verify_runtime_compiler(dev)
                except Exception as err:
                    _close_quietly(dev)
                    failures.append(f"cuda: {_summarize(err, 160)}")
                    warnings.warn(
                        "slangimage: the CUDA device could not compile a test kernel "
                        f"({_summarize(err)}); trying Vulkan instead. {_CUDA_SETUP_HINT}",
                        RuntimeWarning,
                        stacklevel=3,
                    )
                    continue
            return dev
        raise RuntimeError(
            "no supported GPU device found — slangimage needs a Metal, CUDA, "
            "or Vulkan GPU. Tried " + "; ".join(failures) + "."
        )

    def _create_device_with_type(self, dtype: spy.DeviceType) -> spy.Device:
        # repo_root = src/slangimage
        repo_root = Path(__file__).resolve().parents[1]
        shaders_root = repo_root / "shaders"

        # SlangPy’s own shader path is required so built in helpers work
        # (call_id, texture helpers etc)
        include_paths = [
            Path(spy.SHADER_PATH),   # SlangPy internal shaders
            shaders_root,
            shaders_root / "common",
            shaders_root / "ops",
        ]

        compiler_options: spy.SlangCompilerOptionsDict = {
            "include_paths": [str(p) for p in include_paths],
            # optional, but matches their helpers.py example
            "debug_info": spy.SlangDebugInfoLevel.none,
            "optimization": spy.SlangOptimizationLevel.default,
            # fast float mode: Vulkan and Metal drivers ALREADY compile
            # transcendentals to fast hardware approximations by default;
            # only NVRTC defaults to precise software routines (pow measured
            # 4.4x slower). Opting in makes CUDA consistent with the other
            # backends (experiments/nvrtc_fastmath).
            "floating_point_mode": spy.SlangFloatingPointMode.fast,
        }
        dev = spy.Device(
            type=dtype,
            compiler_options=compiler_options,
            enable_debug_layers=False,
            adapter_luid=None,
            enable_cuda_interop=False,
            enable_print=False,
            enable_hot_reload=False,
            enable_compilation_reports=False,
            existing_device_handles=None,
            bindless_options=None,
        )
        return dev


_default_managers: dict[Backend, DeviceManager] = {}
_manager_lock = threading.Lock()


def get_default_device(backend: Backend | str = Backend.AUTO) -> spy.Device:
    """Return a process-wide device for the requested backend.

    One device is cached per backend. It belongs to the thread that first
    creates it; another thread gets a clear error rather than sharing the
    process-wide default. Note AUTO is its own cache key: mixing
    get_default_device() and
    get_default_device("metal") creates two devices even if AUTO resolves to
    metal.

    A device (and everything derived from it) is NOT thread-safe. Multithreaded
    callers must construct a DeviceManager inside each worker and keep that
    device, its Images/effects and final teardown on the same thread.

    Known upstream issue (slangpy/RHI on Linux + NVIDIA): creating a Vulkan
    device — explicitly or via the AUTO fallback — can segfault when CUDA is
    hidden (e.g. ``CUDA_VISIBLE_DEVICES=""``). Don't mask CUDA if you rely on the
    Vulkan fallback there.
    """
    if isinstance(backend, str):
        backend = Backend(backend)
    mgr = _default_managers.get(backend)
    if mgr is None:
        with _manager_lock:
            mgr = _default_managers.get(backend)  # re-check under the lock
            if mgr is None:
                mgr = DeviceManager(backend)
                _default_managers[backend] = mgr
    _claim_device_thread(mgr.device)
    return mgr.device


def release_device(device: spy.Device) -> None:
    """Drop slangimage's cached state (texture pool, module cache, op instances)
    for a device so no package-global cache keeps it alive.

    slangimage caches per-device state keyed by the device, and those caches
    reference the device — so without this call the device is retained for the
    process lifetime. That is fine for the usual single-device app; call this
    when you create many explicit DeviceManager/devices and want to release the
    ones you're done with.

    This function deliberately does not close the caller-owned device or
    invalidate surviving same-thread Images/effects. It must run on the device's
    owner thread. For final teardown, first drop all derived objects, then call
    ``release_device(device)`` followed by
    ``device.close()``. On the validated SlangPy 0.42 stack, native wrappers and
    driver memory can still remain resident until process exit after ``close``;
    applications should reuse a long-lived device rather than repeatedly create
    and destroy devices in one process.
    """
    _claim_device_thread(device)

    from .texture_pool import _pools
    from .modules import _caches

    with _manager_lock:
        for backend, manager in list(_default_managers.items()):
            if manager.device is device:
                del _default_managers[backend]

    pool = _pools.pop(device, None)
    if pool is not None:
        pool.clear()
    _caches.pop(device, None)
    try:
        from ..ops import _instances

        _instances.pop(device, None)
    except Exception:
        pass
