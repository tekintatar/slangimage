#!/usr/bin/env python3
"""Bounded sustained-workload diagnostic for texture-pool residency.

This is deliberately separate from the versioned performance benchmark: it
checks allocation plateau and reports process/device memory across two equal
steady-state windows; it does not publish timing numbers.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import resource
from typing import Any

import numpy as np


def _rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux reports KiB.
    return value / 1e6 if platform.system() == "Darwin" else value * 1024 / 1e6


def _device_memory_reader() -> tuple[str | None, Any]:
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return "nvml_device_total", lambda: float(
            pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1e6
        )
    except Exception:
        return None, lambda: None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("auto", "metal", "cuda", "vulkan"), default="auto")
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=64)
    parser.add_argument("--rss-growth-limit-mb", type=float, default=64.0)
    parser.add_argument("--device-growth-limit-mb", type=float, default=64.0)
    args = parser.parse_args()
    if args.size < 1 or args.warmup < 1 or args.iterations < 1:
        parser.error("size, warmup and iterations must be positive")
    return args


def main() -> int:
    args = _parse_args()

    import slangimage as si
    from slangimage.fx.soft_glow import SoftGlowEffect
    from slangimage.runtime.modules import get_module_cache
    from slangimage.runtime.texture_pool import get_pool

    device = si.get_default_device(args.backend)
    modules = get_module_cache(device)
    pool = get_pool(device)
    pool.clear()
    dtype = np.float16 if args.precision == "fp16" else np.float32
    pixels = np.zeros((args.size, args.size, 4), dtype=dtype)
    pixels[..., 3] = 1
    src = si.from_numpy(pixels, precision=args.precision, device=device).texture
    effect = SoftGlowEffect(device, modules)
    memory_kind, device_memory_mb = _device_memory_reader()

    def run(count: int) -> None:
        for _ in range(count):
            out = effect.apply(src)
            pool.recycle(out)
        device.wait_for_idle()
        gc.collect()

    def snapshot(label: str) -> dict[str, Any]:
        return {
            "label": label,
            "pool": pool.stats(),
            "rss_high_water_mb": round(_rss_mb(), 3),
            "device_memory_mb": (
                round(value, 3) if (value := device_memory_mb()) is not None else None
            ),
        }

    run(args.warmup)
    warm = snapshot("warm")
    run(args.iterations)
    first = snapshot("steady_1")
    run(args.iterations)
    second = snapshot("steady_2")

    created_values = [phase["pool"]["created"] for phase in (warm, first, second)]
    rss_growth = second["rss_high_water_mb"] - first["rss_high_water_mb"]
    device_growth = None
    if first["device_memory_mb"] is not None and second["device_memory_mb"] is not None:
        device_growth = second["device_memory_mb"] - first["device_memory_mb"]
    failures = []
    if len(set(created_values)) != 1:
        failures.append(f"pool creations changed across steady phases: {created_values}")
    if rss_growth > args.rss_growth_limit_mb:
        failures.append(
            f"RSS high-water grew {rss_growth:.3f} MB, limit {args.rss_growth_limit_mb:.3f} MB"
        )
    if device_growth is not None and device_growth > args.device_growth_limit_mb:
        failures.append(
            "device memory grew "
            f"{device_growth:.3f} MB, limit {args.device_growth_limit_mb:.3f} MB"
        )

    report = {
        "backend": str(device.info.type).split(".")[-1],
        "adapter": device.info.adapter_name,
        "precision": args.precision,
        "size": args.size,
        "warmup": args.warmup,
        "iterations_per_steady_window": args.iterations,
        "device_memory_kind": memory_kind,
        "rss_steady_growth_mb": round(rss_growth, 3),
        "device_steady_growth_mb": (
            round(device_growth, 3) if device_growth is not None else None
        ),
        "phases": [warm, first, second],
        "failures": failures,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
