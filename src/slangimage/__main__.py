# slangimage/__main__.py
# `python -m slangimage`: a setup report for troubleshooting and bug reports.
# Each backend counts as working only if a kernel compiles, runs and returns
# the right pixels — a device that merely constructs is not enough (a CUDA
# device can be created without a usable runtime compiler).

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import shutil
import subprocess
import sys
from typing import Any, Iterator

BACKENDS = ("metal", "cuda", "vulkan")
ENV_VARS = ("SLANGIMAGE_BACKEND", "SLANGIMAGE_POOL_BUDGET_MB", "CUDA_PATH", "LD_LIBRARY_PATH")


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
        return {"ok": False, "error": str(err)}
    device = manager.device
    try:
        pixels = np.linspace(0, 1, 4 * 4 * 4, dtype=np.float32).reshape(4, 4, 4)
        out = si.ops.invert(si.from_numpy(pixels, device=device)).to_numpy()
        if not np.allclose(out[..., :3], 1 - pixels[..., :3], atol=1e-6):
            return {"ok": False, "adapter": device.info.adapter_name,
                    "error": "device created, but a test kernel returned wrong values"}
        return {"ok": True, "adapter": device.info.adapter_name}
    except Exception as err:  # noqa: BLE001
        return {"ok": False, "adapter": device.info.adapter_name,
                "error": f"device created, but a test kernel failed: {err}"}
    finally:
        si.release_device(device)
        device.close()


def _auto_choice(results: dict[str, dict[str, Any]]) -> str | None:
    order = ("metal", "vulkan") if platform.system() == "Darwin" else ("cuda", "vulkan")
    return next((b for b in order if results[b]["ok"]), None)


def _torch_status() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"installed": False}
    status: dict[str, Any] = {"installed": True, "cuda": torch.cuda.is_available()}
    if status["cuda"]:
        status["cuda_device"] = torch.cuda.get_device_name(0)
        try:
            import slangimage.diff as sid

            status["differentiable_ops"] = sid.available()
        except Exception as err:  # noqa: BLE001
            status["differentiable_ops"] = False
            status["error"] = str(err)
    return status


@contextlib.contextmanager
def _native_output_to_stderr() -> Iterator[None]:
    """Native libraries print diagnostics straight to file descriptor 1 (for
    example a missing CUDA driver symbol when no NVIDIA driver is installed);
    send them to stderr so stdout carries only the report."""
    sys.stdout.flush()
    try:
        saved = os.dup(1)
    except OSError:  # no usable fd 1 (embedded interpreters): leave it alone
        yield
        return
    try:
        os.dup2(2, 1)
        yield
    finally:
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
    results = {backend: check_backend(backend) for backend in BACKENDS}
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
            lines.append(f"  {backend:7} OK      {result['adapter']}")
        else:
            lines.append(f"  {backend:7} FAILED  {result['error']}")
    auto = report["auto_backend"]
    lines += ["", f"Automatic selection: {auto or 'none — no working GPU backend'}"]
    torch = report["torch"]
    lines += ["", "PyTorch (slangimage.diff / slangimage.nn):"]
    if not torch["installed"]:
        lines.append("  torch not installed (needed only for to_torch/from_torch and diff/nn)")
    elif not torch["cuda"]:
        lines.append("  torch installed, no CUDA device: differentiable ops unavailable (CUDA only)")
    else:
        state = "available" if torch["differentiable_ops"] else "NOT available"
        lines.append(f"  CUDA device {torch['cuda_device']}: differentiable ops {state}")
        if not torch["differentiable_ops"]:
            lines.append("  see the README section 'PyTorch extra' for the three-step install")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m slangimage",
        description="Report the SlangImage setup: versions, GPU backends and PyTorch support.",
    )
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args(argv)
    report = collect()
    print(json.dumps(report, indent=2) if args.json else _format(report))
    return 0 if report["auto_backend"] else 1


if __name__ == "__main__":
    sys.exit(main())
