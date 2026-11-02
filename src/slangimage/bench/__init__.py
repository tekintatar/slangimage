# slangimage.bench — the standard, versioned benchmark suite.
#
# Run:     slangimage-bench run --backend auto --out results_metal.json
# Report:  slangimage-bench report results_*.json -o BENCHMARKS.md
#
# PROTOCOL_VERSION identifies the methodology + op suite. Results are only
# comparable within the same protocol version; bump it whenever ops, params,
# sizes, or timing methodology change.

from __future__ import annotations

import json
import platform
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np

PROTOCOL_VERSION = 3

SIZES = [1024, 2048, 3456]  # ~1.0 / 4.2 / 11.9 MP
PRECISIONS = ["fp16", "fp32"]
DEFAULT_ITERS = 15


# ---------------------------------------------------------------- input image
def make_image(size: int) -> np.ndarray:
    """Deterministic synthetic photo-ish RGBA image in [0,1]."""
    rng = np.random.default_rng(3)
    y, x = np.mgrid[0:size, 0:size].astype(np.float32) / size
    img = np.stack(
        [
            0.5 + 0.5 * np.sin(8.0 * x) * np.cos(5.0 * y),
            np.clip(1.2 * y - 0.1, 0, 1),
            x,
            np.ones_like(x),
        ],
        axis=-1,
    )
    img[..., :3] += rng.normal(0.0, 0.02, size=(size, size, 3)).astype(np.float32)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


# ------------------------------------------------------------------ op suite
def build_cases(device, modules) -> list[tuple[str, Callable]]:
    """The standard op suite for PROTOCOL_VERSION. Do not edit params without
    bumping the protocol version."""
    from slangimage.kernels.vibrance import VibranceOp
    from slangimage.kernels.gaussian import GaussianBlurOp
    from slangimage.kernels.unsharp_mask import UnsharpMaskOp
    from slangimage.kernels.surface_blur import SurfaceBlurOp
    from slangimage.kernels.kuwahara import KuwaharaOp
    from slangimage.kernels.resize import ResizeOp

    vib = VibranceOp(device, modules)
    gau = GaussianBlurOp(device, modules)
    usm = UnsharpMaskOp(device, modules)
    sur = SurfaceBlurOp(device, modules)
    kuw = KuwaharaOp(device, modules)
    rsz = ResizeOp(device, modules)

    return [
        ("vibrance", lambda t: vib.apply(t, amount=0.6)),
        ("gaussian_r9", lambda t: gau.apply(t, radius=9, sigma=3.0)),
        ("unsharp_mask_r8", lambda t: usm.apply(t, amount=1.5, radius=8, sigma=3.0)),
        ("surface_blur_r9", lambda t: sur.apply(t, radius=9, bsigma=0.1)),
        ("kuwahara_r6", lambda t: kuw.apply(t, radius=6)),
        ("resize_half_lanczos3", lambda t: rsz.apply(t, out_w=t.width // 2, out_h=t.height // 2)),
    ]


# -------------------------------------------------------------- vram sampler
class VramSampler:
    """Peak device-memory sampler via NVML. Silently unavailable elsewhere."""

    def __init__(self):
        self.ok = False
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nv = pynvml
            self._h = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.driver = pynvml.nvmlSystemGetDriverVersion()
            self.ok = True
        except Exception:
            self.driver = None

    def used_mb(self) -> float:
        return self._nv.nvmlDeviceGetMemoryInfo(self._h).used / 1e6

    def watch(self) -> dict | None:
        if not self.ok:
            return None
        state: dict[str, Any] = {"stop": threading.Event(), "base": self.used_mb()}
        state["peak"] = state["base"]

        def loop():
            while not state["stop"].is_set():
                state["peak"] = max(state["peak"], self.used_mb())
                time.sleep(0.005)

        t = threading.Thread(target=loop, daemon=True)
        t.start()
        state["thread"] = t
        return state


# --------------------------------------------------------------------- run
def run(backend: str = "auto", iters: int = DEFAULT_ITERS, sizes: list[int] | None = None) -> dict:
    import slangpy as spy
    import slangimage as si
    from slangimage.runtime.textures import np_to_texture

    sizes = sizes or SIZES

    manager = si.DeviceManager(si.Backend(backend))
    device = manager.device
    modules = si.ModuleCache(device)
    backend_name = str(device.info.type).split(".")[-1]
    adapter = device.info.adapter_name
    print(f"[bench] protocol v{PROTOCOL_VERSION} | {backend_name} on {adapter}")

    vram = VramSampler() if backend_name == "cuda" else None
    cases = build_cases(device, modules)

    records: list[dict] = []
    first_call_seen: set[tuple[str, str]] = set()

    for precision in PRECISIONS:
        for size in sizes:
            img = make_image(size).astype(np.float16 if precision == "fp16" else np.float32)
            src: spy.Texture | None = np_to_texture(device, img, usage="read", precision=precision)
            device.wait_for_idle()

            for name, fn in cases:
                first_call_ms = None
                if (name, precision) not in first_call_seen:
                    t0 = time.perf_counter()
                    out = fn(src)
                    device.wait_for_idle()
                    first_call_ms = (time.perf_counter() - t0) * 1000.0
                    first_call_seen.add((name, precision))
                    out = None

                out = fn(src)  # warmup at this size
                device.wait_for_idle()
                out = None

                watch = vram.watch() if vram else None

                times = []
                for _ in range(iters):
                    t0 = time.perf_counter()
                    out = fn(src)
                    device.wait_for_idle()
                    times.append((time.perf_counter() - t0) * 1000.0)
                    del out

                rec = {
                    "backend": backend_name,
                    "adapter": adapter,
                    "op": name,
                    "size": size,
                    "mp": round(size * size / 1e6, 1),
                    "precision": precision,
                    "steady_ms": round(float(np.median(times)), 3),
                    "steady_ms_min": round(float(np.min(times)), 3),
                    "iters": iters,
                }
                if first_call_ms is not None:
                    rec["first_call_ms"] = round(first_call_ms, 1)
                if watch:
                    watch["stop"].set()
                    watch["thread"].join()
                    rec["vram_peak_mb"] = round(watch["peak"], 0)
                    rec["vram_base_mb"] = round(watch["base"], 0)
                records.append(rec)
                fc = f" first={first_call_ms:8.1f}ms" if first_call_ms is not None else ""
                print(f"  {precision} {size:4d} {name:24s} {rec['steady_ms']:9.3f} ms{fc}")

            src = None
            device.wait_for_idle()

    import slangimage

    return {
        "meta": {
            "protocol": PROTOCOL_VERSION,
            "backend": backend_name,
            "adapter": adapter,
            "gpu_driver": vram.driver if vram else None,
            "platform": platform.platform(),
            "cpu": platform.processor() or platform.machine(),
            "python": sys.version.split()[0],
            "slangpy": spy.__version__,  # type: ignore[attr-defined]  # missing from the stubs
            "slangimage": slangimage.__version__,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "iters": iters,
        },
        "records": records,
    }


def save(result: dict, out_path: str) -> None:
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1)
    print(f"[bench] wrote {out_path} ({len(result['records'])} records)")
