#!/usr/bin/env python3
"""Exercise the supported one-explicit-device-per-worker threading pattern."""

from __future__ import annotations

import argparse
import gc
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("metal", "cuda", "vulkan"), required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--size", type=int, default=48)
    args = parser.parse_args()
    if args.workers < 2:
        parser.error("workers must be at least 2")
    if args.iterations < 1 or args.size < 1:
        parser.error("iterations and size must be positive")
    return args


def main() -> int:
    args = _parse_args()

    import slangimage as si
    from slangimage.runtime.texture_pool import get_pool

    start = threading.Barrier(args.workers)

    def worker(index: int) -> dict[str, object]:
        manager = si.DeviceManager(si.Backend(args.backend))
        device = manager.device
        image = None
        try:
            pixels = np.random.default_rng(1000 + index).random(
                (args.size, args.size + index, 4), dtype=np.float32
            )
            pixels[..., 3] = 1
            image = si.from_numpy(pixels, precision="fp32", device=device)
            start.wait(timeout=30)
            for _ in range(args.iterations):
                image = si.ops.invert(image)
            actual = image.to_numpy()
            expected = pixels if args.iterations % 2 == 0 else 1 - pixels
            expected[..., 3] = 1
            device.wait_for_idle()
            return {
                "worker": index,
                "thread": threading.current_thread().name,
                "device_type": device.info.type.name,
                "max_abs_error": float(np.max(np.abs(actual - expected))),
                "pool": get_pool(device).stats(),
            }
        finally:
            del image
            gc.collect()
            si.release_device(device)
            device.close()

    failures: list[str] = []
    rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(worker, index) for index in range(args.workers)]
        for index, future in enumerate(futures):
            try:
                row = future.result(timeout=120)
                rows.append(row)
                if float(row["max_abs_error"]) > 2e-6:
                    failures.append(f"worker {index}: numerical mismatch")
            except Exception as exc:
                failures.append(f"worker {index}: {type(exc).__name__}: {exc}")

    report = {
        "backend": args.backend,
        "workers": args.workers,
        "iterations": args.iterations,
        "measurements": rows,
        "failures": failures,
        "note": "Each worker creates, uses, releases and closes its own device.",
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
