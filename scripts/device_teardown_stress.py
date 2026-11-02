#!/usr/bin/env python3
"""Reproduce in-process device/cache teardown and native memory residency."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import resource
import weakref
from typing import Any

import numpy as np


def _rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / 1e6 if platform.system() == "Darwin" else value * 1024 / 1e6


def _device_memory_reader() -> tuple[str | None, Any]:
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return "nvml_device_total", lambda: round(
            float(pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1e6), 3
        )
    except Exception:
        return None, lambda: None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("metal", "cuda", "vulkan"), required=True)
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument(
        "--close",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="call device.close() after clearing SlangImage caches (default: true)",
    )
    args = parser.parse_args()
    if args.cycles < 1:
        parser.error("cycles must be positive")
    return args


def main() -> int:
    args = _parse_args()

    import slangimage as si
    import slangpy as spy
    from slangimage.ops import _instances
    from slangimage.runtime.modules import _caches
    from slangimage.runtime.texture_pool import _pools

    memory_kind, device_memory_mb = _device_memory_reader()
    rows = []
    failures = []
    baseline_device_mb = device_memory_mb()

    for cycle in range(1, args.cycles + 1):
        manager = si.DeviceManager(si.Backend(args.backend))
        device = manager.device
        pixels = np.zeros((64, 64, 4), dtype=np.float32)
        pixels[..., 3] = 1
        image = si.from_numpy(pixels, device=device)
        output = si.ops.invert(image)
        output.to_numpy()
        device.wait_for_idle()
        active_device_mb = device_memory_mb()
        device_ref = weakref.ref(device)

        del output, image
        gc.collect()
        si.release_device(device)
        package_roots_cleared = (
            device not in _pools and device not in _caches and device not in _instances
        )
        if not package_roots_cleared:
            failures.append(f"cycle {cycle}: SlangImage cache root remained")
        if args.close:
            device.close()
        del device, manager
        gc.collect()

        rows.append(
            {
                "cycle": cycle,
                "active_device_memory_mb": active_device_mb,
                "after_teardown_device_memory_mb": device_memory_mb(),
                "rss_high_water_mb": round(_rss_mb(), 3),
                "package_roots_cleared": package_roots_cleared,
                "device_wrapper_alive_after_gc": device_ref() is not None,
                "slangpy_created_device_registry_count": len(
                    spy.Device.get_created_devices()
                ),
            }
        )

    report = {
        "backend": args.backend,
        "cycles": args.cycles,
        "called_device_close": args.close,
        "device_memory_kind": memory_kind,
        "baseline_device_memory_mb": baseline_device_mb,
        "measurements": rows,
        "failures": failures,
        "note": (
            "Device memory is device-wide when NVML is available. Native/driver "
            "retention after package roots clear is reported, not treated as a "
            "SlangImage cache failure."
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
