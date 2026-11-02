# slangimage/__main__.py
# `python -m slangimage`: a setup report for troubleshooting and bug reports.
# Each backend counts as working only if a kernel compiles, runs and returns
# the right pixels — a device that merely constructs is not enough (a CUDA
# device can be created without a usable runtime compiler). Each backend is
# checked in its own child process, so a native crash in one driver cannot
# take the whole report down.

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
from typing import Any, Iterator

BACKENDS = ("metal", "cuda", "vulkan")
ENV_VARS = ("SLANGIMAGE_BACKEND", "SLANGIMAGE_POOL_BUDGET_MB", "CUDA_PATH", "LD_LIBRARY_PATH")
CHECK_TIMEOUT_S = 300
# CPU implementations of Vulkan: they work, but far too slowly to be the GPU.
_SOFTWARE_ADAPTER = re.compile(r"llvmpipe|lavapipe|swiftshader|softpipe", re.IGNORECASE)


def _versions() -> dict[str, str | None]:
    from importlib import metadata

    versions: dict[str, str | None] = {"python": platform.python_version()}
    for dist in ("slangimage", "slangpy", "numpy", "pillow", "torch", "slangpy-torch"):
        try:
            versions[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            versions[dist] = None
    return versions


def _platform() -> dict[str, str]:
    info = {"system": platform.platform(), "machine": platform.machine()}
    if platform.system() == "Darwin":
        info["macos"] = platform.mac_ver()[0]
    elif platform.system() == "Linux":
        libc, version = platform.libc_ver()
        if libc:
            info["libc"] = f"{libc} {version}"
    return info


def _nvidia_driver() -> str | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return "nvidia-smi failed"
    return "; ".join(line.strip() for line in out.stdout.splitlines() if line.strip()) or None


def check_backend(backend: str) -> dict[str, Any]:
    """Create a device for ``backend``, run one kernel, verify it, release it."""
    import numpy as np

    import slangimage as si

    try:
        manager = si.DeviceManager(backend)
    except Exception as err:  # noqa: BLE001 - every failure is reported, not raised
        return {"ok": False, "created": False, "error": str(err)}
    device = manager.device
    adapter = device.info.adapter_name
    result: dict[str, Any] = {"created": True, "adapter": adapter,
                              "software": bool(_SOFTWARE_ADAPTER.search(adapter))}
    try:
        pixels = np.linspace(0, 1, 4 * 4 * 4, dtype=np.float32).reshape(4, 4, 4)
        out = si.ops.invert(si.from_numpy(pixels, device=device)).to_numpy()
        if not np.allclose(out[..., :3], 1 - pixels[..., :3], atol=1e-6):
            return {**result, "ok": False,
                    "error": "device created, but a test kernel returned wrong values"}
        return {**result, "ok": True}
    except Exception as err:  # noqa: BLE001
        return {**result, "ok": False, "error": f"device created, but a test kernel failed: {err}"}
    finally:
        si.release_device(device)
        device.close()


def run_check(backend: str, command: list[str] | None = None) -> dict[str, Any]:
    """Run :func:`check_backend` in a child process and read its JSON result."""
    if command is None:
        command = [sys.executable, "-m", "slangimage", "--check-backend", backend]
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=CHECK_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return {"ok": False, "created": False,
                "error": f"the check did not finish within {CHECK_TIMEOUT_S} s"}
    # Native libraries may print around the result line, before or after it.
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("{"):
            try:
                result = json.loads(line)
            except ValueError:
                continue
            if isinstance(result, dict) and "ok" in result:
                return result
    if proc.returncode < 0:
        try:
            name = signal.Signals(-proc.returncode).name
        except ValueError:
            name = f"signal {-proc.returncode}"
        return {"ok": False, "created": False,
                "error": f"the {backend} check crashed ({name}) inside the GPU driver or runtime"}
    detail = (proc.stderr.strip().splitlines() or ["no output"])[-1]
    return {"ok": False, "created": False,
            "error": f"the {backend} check exited with status {proc.returncode}: {detail}"}


def _auto_choice(results: dict[str, dict[str, Any]]) -> str | None:
    """The backend ``Backend.AUTO`` would pick: the first one whose device can be
    created, except that a CUDA device must also run a test kernel (the
    library's CUDA probe); Metal and Vulkan are kept even if a kernel fails."""
    order = ("metal", "vulkan") if platform.system() == "Darwin" else ("cuda", "vulkan")
    for backend in order:
        result = results[backend]
        if result["ok"] if backend == "cuda" else result.get("created", result["ok"]):
            return backend
    return None


def _torch_status() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"installed": False}
    except Exception as err:  # noqa: BLE001 - a broken install is what this reports
        return {"installed": True, "cuda": False, "error": f"import torch failed: {err}"}
    status: dict[str, Any] = {"installed": True, "cuda": False}
    try:
        status["cuda"] = torch.cuda.is_available()
        if status["cuda"]:
            status["cuda_device"] = torch.cuda.get_device_name(0)
            import slangimage.diff as sid

            status["differentiable_ops"] = sid.available()
            if not status["differentiable_ops"]:
                try:  # name the setup problem (e.g. NVRTC not found), if any
                    sid._ctx()
                except Exception as err:  # noqa: BLE001
                    status["error"] = str(err)
    except Exception as err:  # noqa: BLE001
        status["differentiable_ops"] = False
        status["error"] = str(err)
    return status


@contextlib.contextmanager
def _native_output_to_stderr() -> Iterator[None]:
    """Native libraries print diagnostics straight to file descriptor 1 (for
    example a missing CUDA driver symbol when no NVIDIA driver is installed);
    send them to stderr so stdout carries only the report."""
    if sys.stdout is not None:
        sys.stdout.flush()
    _flush_c_stdio()
    try:
        os.fstat(2)  # nowhere to send it if stderr is closed
        saved = os.dup(1)
    except OSError:  # no usable fd 1 or 2 (embedded interpreters): leave it alone
        yield
        return
    try:
        os.dup2(2, 1)
        yield
    finally:
        if sys.stdout is not None:
            sys.stdout.flush()
        _flush_c_stdio()  # native text sits in C's stdio buffer until flushed
        os.dup2(saved, 1)
        os.close(saved)


def _flush_c_stdio() -> None:
    try:
        import ctypes

        ctypes.CDLL(None).fflush(None)
    except (OSError, AttributeError, TypeError):  # no C runtime handle (e.g. Windows)
        pass


def collect() -> dict[str, Any]:
    with _native_output_to_stderr():
        return _collect()


def _collect() -> dict[str, Any]:
    results = {backend: run_check(backend) for backend in BACKENDS}
    return {
        "versions": _versions(),
        "platform": _platform(),
        "nvidia_driver": _nvidia_driver(),
        "environment": {name: ("set" if os.environ.get(name) else "not set") for name in ENV_VARS},
        "backends": results,
        "auto_backend": _auto_choice(results),
        "torch": _torch_status(),
    }


def _format(report: dict[str, Any]) -> str:
    lines = ["SlangImage setup report", ""]
    for name, version in report["versions"].items():
        lines.append(f"  {name:14} {version or 'not installed'}")
    for name, value in report["platform"].items():
        lines.append(f"  {name:14} {value}")
    lines.append(f"  {'nvidia driver':14} {report['nvidia_driver'] or 'not found'}")
    lines += ["", "Environment:"]
    lines += [f"  {name:26} {value}" for name, value in report["environment"].items()]
    lines += ["", "Backends (each runs a test kernel):"]
    for backend, result in report["backends"].items():
        if result["ok"]:
            note = "  (software renderer on the CPU: very slow; check the GPU driver)"
            lines.append(f"  {backend:7} OK      {result['adapter']}{note if result.get('software') else ''}")
        else:
            lines.append(f"  {backend:7} FAILED  {result['error']}")
    auto = report["auto_backend"]
    lines += ["", f"Automatic selection: {auto or 'none — no working GPU backend'}"]
    torch = report["torch"]
    lines += ["", "PyTorch (slangimage.diff / slangimage.nn):"]
    if not torch["installed"]:
        lines.append("  torch not installed (needed only for to_torch/from_torch and diff/nn)")
    elif torch.get("error") and not torch["cuda"]:
        lines.append(f"  torch is installed but failed: {torch['error']}")
    elif not torch["cuda"]:
        lines.append("  torch installed, no CUDA device: differentiable ops unavailable (CUDA only)")
    else:
        state = "available" if torch["differentiable_ops"] else "NOT available"
        lines.append(f"  CUDA device {torch['cuda_device']}: differentiable ops {state}")
        if not torch["differentiable_ops"]:
            if torch.get("error"):
                lines.append(f"  {torch['error']}")
            lines.append("  see the README section 'PyTorch extra' for the three-step install")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m slangimage",
        description="Report the SlangImage setup: versions, GPU backends and PyTorch support.",
    )
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    parser.add_argument("--check-backend", choices=BACKENDS, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.check_backend:  # internal: one isolated backend check (see run_check)
        print(json.dumps(check_backend(args.check_backend)), flush=True)
        return 0
    report = collect()
    print(json.dumps(report, indent=2) if args.json else _format(report))
    return 0 if report["auto_backend"] else 1


if __name__ == "__main__":
    sys.exit(main())
