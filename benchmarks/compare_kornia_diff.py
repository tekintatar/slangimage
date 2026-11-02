# benchmarks/compare_kornia_diff.py
# Differentiable-path head-to-head vs kornia AND torchvision v2 on CUDA:
# forward latency, forward+backward latency (one training step: zero
# grad -> forward -> .sum().backward()), and peak torch-allocator VRAM
# per training step. Matched ops/params: gaussian 19x19 sigma 3;
# bilateral 11x11, sigma_color 0.15, sigma_space 2.5; sepia (rescale
# off); hue rotate 1.0 rad (torchvision takes turns: 1/(2pi)); resize
# bilinear x0.5 (align_corners=False, antialias off everywhere).
# Batch 1, fp32 (the training dtype), slangimage HWC vs NCHW others.
#
#   pip install torch kornia torchvision
#   python benchmarks/compare_kornia_diff.py [--sizes 512 1024 2048]
#       [--iters 20] [--json out.json]
#
# Coverage gaps are real gaps: torchvision has NO bilateral and NO
# sepia; nobody else has blend modes or a learnable LUT (not compared).
# VRAM caveat: peaks are torch-allocator deltas; slangpy's internal
# call-data buffers (small, size-independent) are outside the torch
# allocator, so slangimage peaks are floors — competitors' unfold/HSV
# activation blowups are fully captured.
#
# Numbers below are one run on the reference host; sub-millisecond cells and
# their ratios vary by up to about +-50% across hosts with the same GPU.
# Reference run (2026-09-12, NVIDIA L40S, slangimage 0.1.0.dev20260821,
# slangpy 0.42.0, torch 2.13.0 / kornia 0.8.3 / torchvision 0.28.0,
# fp32, batch 1; full data:
# comparisons/results_diff_kornia_l40s.json). slangimage / kornia / torchvision:
#
#   1024^2            step ms                peak MB/step
#   gaussian 19x19    0.81 / 1.18 / 1.20     36 / 61 / 37
#   bilateral 11x11   0.91 / 84.7 / —        40 / 7760 / —
#   sepia             0.28 / 0.49 / —        24 / 32 / —
#   hue rotate        0.28 / 1.61 / ERR      24 / 253 / ERR
#   resize bil x0.5   0.45 / 0.18 / 0.21     21 / 15 / 15
#
#   2048^2: bilateral step 3.72 / 359 / — ms, peak 160 MB / 31 GB / —;
#   hue step 0.34 / 10.4 / ERR ms; gaussian step 2.81 / 2.60 / 4.57 ms.
#
# Notes (the honest ones included):
# - bilateral is the structural win: kornia's unfold path needs 31 GB
#   per training step at 2048^2 (ONE image) vs our 160 MB — batch
#   training doesn't fit on their path; torchvision lacks the op. Runs
#   our hand-written gather backward (diff.py bridge): Slang's generated
#   scatter backward was 261 ms/step at 1024^2, the gather is 0.91 ms.
# - torchvision's adjust_hue RAISES on grad-requiring input (in-place
#   view ops in its HSV round-trip) — augmentation-only, cannot be
#   trained through. Ours and kornia's train; ours steps 5-31x faster
#   (one fused YIQ rotation vs an HSV round-trip graph).
# - resize: kornia/torchvision wrap torch's native fused F.interpolate,
#   which beats our two-pass kernels ~2-3x per step. sid.resize exists
#   for pipeline completeness with exactly-matching semantics (verified
#   against F.interpolate in tests), not for speed.
# - gaussian: near-parity territory. We win the 1024^2 training step
#   against both and beat torchvision at 2048^2; torchvision edges ahead
#   at 512^2, and kornia is near parity at 2048^2.
# - Gaussian and resize use matched tested interpolation/filter semantics.
#   Bilateral border behavior and YIQ-vs-HSV hue semantics differ, so those
#   rows compare the stated operator class and parameters, not pixel parity.

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch
import kornia
import kornia.color as KC
import kornia.enhance as KE
import kornia.filters as KF
from kornia.geometry.transform import resize as k_resize

import slangimage.diff as sid
import slangimage as si
import slangpy as spy

try:
    import torchvision
    from torchvision.transforms.v2 import InterpolationMode
    from torchvision.transforms.v2 import functional as TVF
except ImportError:
    torchvision = None
    TVF = None

LIBS = ("slangimage", "kornia", "torchvision")
HUE_RAD = 1.0

OPS = [
    ("gaussian 19x19", {
        "slangimage": lambda x: sid.gaussian_blur(x, radius=9, sigma=3.0),
        "kornia": lambda x: KF.gaussian_blur2d(x, (19, 19), (3.0, 3.0)),
        "torchvision": None if TVF is None else
            lambda x: TVF.gaussian_blur(x, [19, 19], [3.0, 3.0]),
    }),
    ("bilateral 11x11", {
        "slangimage": lambda x: sid.bilateral(
            x, radius=5, sigma_spatial=2.5, sigma_range=0.15),
        "kornia": lambda x: KF.bilateral_blur(x, (11, 11), 0.15, (2.5, 2.5)),
        "torchvision": None,          # torchvision has no bilateral
    }),
    ("sepia", {
        "slangimage": lambda x: sid.sepia(x, intensity=1.0),
        "kornia": lambda x: KC.sepia(x, rescale=False),
        "torchvision": None,          # torchvision has no sepia
    }),
    ("hue rotate", {
        "slangimage": lambda x: sid.hue_rotate(x, angle=HUE_RAD),
        "kornia": lambda x: KE.adjust_hue(x, HUE_RAD),
        "torchvision": None if TVF is None else
            lambda x: TVF.adjust_hue(x, HUE_RAD / (2.0 * math.pi)),
    }),
    ("resize bil x0.5", {
        "slangimage": lambda x: sid.resize(x, scale=0.5),
        "kornia": lambda x: k_resize(
            x, (x.shape[-2] // 2, x.shape[-1] // 2),
            interpolation="bilinear", antialias=False),
        "torchvision": None if TVF is None else
            lambda x: TVF.resize(
                x, [x.shape[-2] // 2, x.shape[-1] // 2],
                interpolation=InterpolationMode.BILINEAR, antialias=False),
    }),
]


def median_ms(fn, iters: int) -> float:
    fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def bench_op(fwd, src_plain, iters: int) -> dict | None:
    """One op x one library: fwd ms, fwd+bwd ms, peak train-step MB."""
    src_grad = src_plain.clone().requires_grad_(True)

    def step():
        src_grad.grad = None
        fwd(src_grad).sum().backward()

    try:
        step()  # warm up: compiles/caches the backward too
        torch.cuda.synchronize()
    except (RuntimeError, torch.AcceleratorError) as e:
        print(f"    ERR: {str(e)[:100]}")
        return None

    fwd_ms = median_ms(lambda: fwd(src_plain), iters)
    step_ms = median_ms(step, iters)

    src_grad.grad = None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    step()
    torch.cuda.synchronize()
    peak_mb = (torch.cuda.max_memory_allocated() - base) / 2**20

    return {"fwd_ms": round(fwd_ms, 3), "step_ms": round(step_ms, 3),
            "peak_mb": round(peak_mb, 1)}


def _ratios(r: dict, base: dict | None) -> str:
    if base is None:
        return ""
    parts = []
    for key in ("fwd_ms", "step_ms", "peak_mb"):
        parts.append(f"{r[key] / base[key]:.1f}x" if base[key] > 0 else "—")
    return "   (" + " / ".join(parts) + ")"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[512, 1024, 2048])
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("slangimage.diff requires torch CUDA")
    if TVF is None:
        print("torchvision not installed — skipping its column")

    results: dict = {
        "device": torch.cuda.get_device_name(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "slangimage": si.__version__,
        "slangpy": spy.__version__,
        "torch": torch.__version__,
        "kornia": kornia.__version__,
        "torchvision": torchvision.__version__ if torchvision is not None else None,
        "iters": args.iters,
        "dtype": "fp32",
        "cases": [],
    }
    print(f"device: {results['device']}  torch {results['torch']}  "
          f"fp32, batch 1, median of {args.iters}")

    for size in args.sizes:
        torch.manual_seed(7)
        hwc = torch.rand(size, size, 3, device="cuda")
        nchw = hwc.permute(2, 0, 1)[None].contiguous()
        print(f"\n=== {size}x{size} ===")
        print(f"  {'op':16s} {'':11s} {'fwd ms':>8s} {'fwd+bwd ms':>11s} "
              f"{'peak MB':>8s}   (ratios vs slangimage)")

        for name, fns in OPS:
            row: dict = {"size": size, "op": name}
            base = None
            for lib in LIBS:
                fn = fns[lib]
                if fn is None:
                    row[lib] = None
                    if not (lib == "torchvision" and TVF is None):
                        print(f"  {name:16s} {lib:11s} {'—':>8s} {'—':>11s} "
                              f"{'—':>8s}   (no equivalent)")
                    continue
                src = hwc if lib == "slangimage" else nchw
                r = bench_op(fn, src, args.iters)
                row[lib] = r
                if r is None:
                    print(f"  {name:16s} {lib:11s}      ERR         ERR      ERR")
                    continue
                print(f"  {name:16s} {lib:11s} {r['fwd_ms']:8.2f} "
                      f"{r['step_ms']:11.2f} {r['peak_mb']:8.1f}"
                      f"{_ratios(r, base)}")
                if lib == "slangimage":
                    base = r
            results["cases"].append(row)

    results["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
