# Platform notes and troubleshooting

Practical notes for installing and running SlangImage. The
[README](../README.md) covers installation and the API overview; this page
collects platform status, first-run behavior, threading and common errors.

## Tested platforms

| Platform | Backends | What has been run |
|---|---|---|
| macOS 26.5, Apple M3 Max (arm64) | Metal | Full test suite (Python 3.12); clean wheel installs with a GPU smoke check on Python 3.10, 3.11, 3.12 and 3.13; examples 01–05; benchmark suite |
| Ubuntu 24.04 (x86_64), NVIDIA L40S, driver 580.126.16 | CUDA, Vulkan | Full test suites from the installed wheel on Python 3.12, including the CUDA autograd tests with the `torch` extra (PyTorch 2.13, SlangPy 0.42.0); examples 01–06; benchmark suite |
| Ubuntu 24.04 (x86_64), NVIDIA RTX PRO 4500 Blackwell (compute capability 12.0), driver 580.126.16 | CUDA, Vulkan | Full test suites from the installed wheel on Python 3.12, including the CUDA autograd tests with the `torch` extra (PyTorch 2.13, SlangPy 0.42.0); examples 01–06; kornia/torchvision comparison scripts (the standard benchmark suite was not run) |

Details and exact counts are in [TEST_COVERAGE.md](TEST_COVERAGE.md)
(*Installed-artifact checks* and *Performance, service and shipped examples*).

Installable platforms are limited by the SlangPy wheels (see *Platform
requirements* in the README): Apple Silicon macOS 15 or later (macOS 15
resolves SlangPy 0.42; SlangPy 0.43 needs macOS 26), Linux with glibc 2.34 or
later on x86_64 or aarch64, and Windows. Intel Macs and macOS 14 or earlier
cannot install SlangPy.

Not tested: Windows, macOS 15, Linux on aarch64, AMD or Intel GPUs, and Linux
distributions other than Ubuntu 24.04. On Linux, the full GPU suites ran on
Python 3.12 only; the GPU-free tests run on 3.10 and 3.13 in CI, and the
`torch` extra install was verified on 3.10, 3.12 and 3.13. On Windows, automatic selection would try CUDA and then Vulkan, but no
Windows configuration has been validated.

Linux + NVIDIA prerequisites are described in the README sections *CUDA on
Linux: install NVRTC* (the runtime compiler the CUDA backend needs) and
*PyTorch extra*.

## Choosing and checking the backend

- `python -m slangimage` prints a setup report: versions, the NVIDIA driver,
  and for each of Metal, CUDA and Vulkan whether a test kernel compiles and
  returns the right values, with the reason for every failure, plus which
  backend automatic selection picks and whether the PyTorch differentiable ops
  work. Each backend is checked in its own child process, so a driver crash is
  reported rather than ending the report, and a software Vulkan renderer such
  as llvmpipe (a CPU fallback, far too slow for real use) is flagged. It exits
  with status 1 when no backend works. `--json` gives the same report as JSON;
  include it in bug reports.
- Automatic selection tries Metal and then Vulkan on macOS, and CUDA and then
  Vulkan on other systems. The CPU is never used: kernels require a GPU, and
  requesting the CPU backend raises an error.
- Choose explicitly with `si.device("metal" | "cuda" | "vulkan")` and pass the
  result as `device=` to `si.load()`, `si.from_numpy()` or `si.from_pil()`.
  Operators run on the device of their input image.
- `print(img)` shows the backend, e.g. `Image(640x480, fp16, metal)`;
  `img.device.info.type`, `img.device.info.api_name` and
  `img.device.info.adapter_name` give the details. Check this on Linux: if the
  CUDA device cannot be created (for example because NVRTC is missing),
  automatic selection logs an `[ERROR]` line such as `Cannot find CUDA nvrtc`
  and continues on Vulkan. A CUDA device that is created but cannot compile a
  one-pixel test kernel (for example when `CUDA_PATH` points at pip-installed
  NVRTC but `LD_LIBRARY_PATH` does not) is closed, and a `RuntimeWarning`
  explains why automatic selection continued on Vulkan.
- `si.DeviceManager` takes a `si.Backend` member (`si.Backend.AUTO`,
  `si.Backend.CUDA`, ...) or its value as a string (`"auto"`, `"metal"`,
  `"cuda"`, `"vulkan"`).
- Requesting a backend that is not available raises `RuntimeError` naming the
  backend, SlangPy's reason and what the backend needs, for example
  `could not create a vulkan device (Failed to create device!). Vulkan is
  usually unavailable on macOS; use Metal (the default there).` On the tested
  macOS machine, Vulkan was not available; use Metal there.
- Known upstream issue on Linux + NVIDIA: creating a Vulkan device (explicitly
  or through the automatic fallback) can segfault when CUDA is hidden, for
  example with `CUDA_VISIBLE_DEVICES=""`. Do not mask CUDA if you rely on the
  Vulkan fallback.
- `SLANGIMAGE_BACKEND` selects the backend for the test suite only; the library
  does not read it.

## First-call compilation and `si.prewarm()`

Kernels are compiled the first time an operator runs on a device at a given
storage precision (fp16 or fp32). In the reference runs, the first call
(compile plus first dispatch) took 46–301 ms per benchmark case; the very first
kernel in each process also paid one-time session setup (362–887 ms in total
for that case). See the first-call table in [BENCHMARKS.md](BENCHMARKS.md).
SlangImage does not yet configure an on-disk shader cache, so every new
process compiles again.

On CUDA, creating the device also compiles and runs a one-pixel test kernel
(so a CUDA setup that cannot compile is caught before your first operation).
The one-time session setup therefore happens during device creation, which
takes about 1.5–3 s on the tested L40S host; the test kernel itself adds about
0.3–0.5 s per process and no lasting GPU memory. With SlangPy 0.43 the one-time
setup is about 1 s longer than with 0.42.

Long-running services should call `si.prewarm(device, precision)` once at
startup for each precision they serve. It runs every texture operator once on
a small image (44 cases), waits for the GPU and returns the number of cases.
Not every mode is prewarmed (for example alternate blend and resize modes), so
those can still compile on first use. The differentiable ops in
`slangimage.diff` compile their own module on first use and are not covered by
`prewarm()`.

## Harmless log output

On first use you may see output such as:

```text
[WARN] No supported shader model found, pretending to support sm_6_0.
... warning :  unused variable '_S13' [-Wunused-variable]
```

The first line comes from SlangPy (seen on Metal, and on CUDA at device
creation); the second kind are Metal compiler warnings about generated shader
code (seen, for example, when the CLAHE, equalize and autocontrast kernels
compile). Neither indicates a failure.

## Threads

Devices and everything derived from them (Images, effects, module caches and
texture pools) are thread-affine. The process-wide default device from
`si.device()` belongs to the first thread that uses it; any other thread gets a
`RuntimeError`. Give each worker its own device, created inside the worker, and
keep all of that device's objects and its teardown on the same thread. Device
construction is serialized internally because Slang's global compiler session
is not thread-safe.

```python
import threading

import numpy as np
import slangimage as si

results = {}


def worker(name, arr):
    device = si.DeviceManager(si.Backend.AUTO).device    # construct inside the worker
    img = si.from_numpy(arr, device=device)
    results[name] = si.ops.invert(img).to_numpy()        # NumPy results may cross threads
    del img
    si.release_device(device)                            # clear SlangImage caches, then
    device.close()                                       # close, on the same thread


arr = np.full((64, 64, 4), 0.25, dtype=np.float32)
threads = [threading.Thread(target=worker, args=(f"w{i}", arr)) for i in range(2)]
for t in threads:
    t.start()
for t in threads:
    t.join()
```

Prefer one long-lived device per worker. On the validated SlangPy 0.42 stack,
native device wrappers and driver memory can stay resident until the process
exits even after `release_device()` and `close()`, so creating devices
repeatedly is not a way to bound memory (see *Conventions* in the README).

## API reference

There is no generated API reference yet. The functional operators, the
differentiable ops and the effects carry docstrings with parameters, defaults
and clipping behavior; use `help()`:

```python
import slangimage as si
import slangimage.diff as sid

help(si.ops.clahe)
help(si.fx.SoftGlowEffect.apply)
help(sid.bilateral)
```

The README lists every operator by category;
[PARAMETER_CONTRACTS.md](PARAMETER_CONTRACTS.md) documents parameter
validation and [TEST_COVERAGE.md](TEST_COVERAGE.md) the test evidence for each
operator.

## Common errors

| Error message (beginning) | Meaning |
|---|---|
| `No matching distribution found for slangpy`, or pip installs `slangimage` 0.0.1 | No SlangImage/SlangPy wheel matches this interpreter or platform (Python outside 3.10–3.13, Intel Mac, macOS 14 or earlier, or Linux with glibc older than 2.34). pip then fails or falls back to 0.0.1, an early placeholder release. Use a supported platform and interpreter (for example `python3.12 -m venv .venv`). |
| `no supported GPU device found — slangimage needs a Metal, CUDA, or Vulkan GPU.` | Automatic selection could not create any GPU device. Run `python -m slangimage` to see why each backend failed. On Linux + NVIDIA, check the CUDA prerequisites and the Vulkan driver. |
| `ImportError: libatomic.so.1` or `libX11.so.6` on `import slangimage` | A minimal Linux image lacks system libraries SlangPy's native module needs: `sudo apt install libatomic1 libx11-6` (plus `libvulkan1` for Vulkan). |
| `E52002` / `pass-through compiler not found`, or `sid.available()` is `False` right after a system-wide NVRTC install | The Slang compiler cannot load NVRTC. After installing the apt packages, run `sudo ldconfig` so the shared-library cache sees the CUDA libraries; with the pip route, set both `CUDA_PATH` and `LD_LIBRARY_PATH` as the README shows. |
| `[ERROR] (rhi) layer: Cannot find CUDA nvrtc` | NVRTC is not installed or not found; automatic selection continues on Vulkan. Install NVRTC as described in the README (*CUDA on Linux: install NVRTC*). |
| `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` | The installed NVRTC is newer than the "CUDA Version" supported by the driver (see `nvidia-smi`); install a matching NVRTC. |
| `could not create a <backend> device (Failed to create device!)` | The explicitly requested backend is not available on this machine; the rest of the message says what that backend needs (on Linux, `si.device("cuda")` raises this when NVRTC is missing). |
| `created a CUDA device but could not compile a test kernel` | The CUDA device exists, but the Slang compiler cannot load NVRTC — typically `CUDA_PATH` is set without the matching `LD_LIBRARY_PATH` for pip-installed NVRTC. Follow the README setup, or use `si.device("vulkan")`. |
| `RuntimeWarning: slangimage: the CUDA device could not compile a test kernel ...; trying Vulkan instead` | The same problem during automatic selection: SlangImage closed the CUDA device and continued on Vulkan. |
| `the CPU backend is not supported` | The CPU backend is rejected by design; a GPU is required. |
| `slangimage device is owned by ... and cannot be used from thread ...` | A device or Image was used from another thread; see [Threads](#threads). |
| `'...' is not a valid Backend` | The backend name is not one of `auto`, `metal`, `cuda` or `vulkan`. |
| `slangimage.diff requires PyTorch and SlangPy's torch bridge` / `slangimage.diff requires a CUDA device` | The differentiable ops need the `torch` extra (a three-step install, see [INSTALL_TORCH.md](INSTALL_TORCH.md)) and a CUDA GPU; `slangimage.diff.available()` checks this without raising. |
| `slangimage.diff could not create SlangPy's CUDA device` | torch and the bridge are installed, but SlangPy cannot start CUDA — usually NVRTC is missing or newer than the driver's CUDA version. See the README's NVRTC section. |
| `ValueError` from an operator | A control failed validation before dispatch; see [PARAMETER_CONTRACTS.md](PARAMETER_CONTRACTS.md). |
