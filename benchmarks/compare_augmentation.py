# benchmarks/compare_augmentation.py
# NON-differentiable (augmentation-path) head-to-head: si.ops textures vs
# kornia vs torchvision v2 functionals. Forward latency only, fp16 and
# fp32, batch 1, median of synced iterations, steady state (all inputs
# already on the GPU; caches hot). Runs on CUDA or Metal-vs-MPS,
# auto-detected.
#
#   pip install torch kornia torchvision
#   python benchmarks/compare_augmentation.py [--sizes 1024 2048]
#       [--iters 15] [--json out.json]
#
# Matched ops (params aligned where semantics allow; noted where they
# only rhyme):
#   gaussian 19x19 s3 | bilateral 11x11 (torchvision: none) |
#   unsharp r2 s1.5 (torchvision adjust_sharpness is a fixed 3x3 —
#   same class, lighter kernel) | clahe 8x8 tiles (torchvision: global
#   equalize only, uint8-oriented — skipped) | hue rotate 1 rad |
#   gamma 1.8 | saturation (ours is an offset, theirs a factor — same
#   cost class) | invert | resize x0.5 antialiased (ours lanczos3 —
#   the quality path — theirs bicubic antialias=True) | Track C parity
#   set: solarize, posterize, equalize, autocontrast (kornia: none),
#   grayscale, normalize.
#
# Dtype note: augmentation stacks often run uint8; all three libraries
# accept float, and float is what our texture formats store, so the
# comparison is fp16/fp32 tensors in [0,1].
#
# Numbers below are one run per backend on the reference hosts; sub-millisecond
# cells and their ratios vary by up to about +-50% across hosts with the same GPU.
# Reference runs (2026-09-12, slangimage 0.1.0.dev20260821, slangpy 0.42,
# torch 2.13 / kornia 0.8.3 / torchvision 0.28; full data:
# comparisons/results_aug_{cuda,metal}.json).
# Selected fp16 ms, slangimage / kornia / torchvision:
#
#   1024^2, NVIDIA L40S (CUDA):
#     bilateral  0.23 / 15.59 / —    clahe       0.20 / 5.10 / —
#     gaussian   0.13 / 0.92 / 0.58  hue         0.08 / 0.50 / 0.43
#     gamma      0.08 / 0.14 / 0.02  saturation  0.08 / 0.49 / 0.05
#     invert     0.07 / 0.03 / 0.01  resize AA   0.11 / 0.85 / 0.09
#   1024^2, Apple M3 Max (Metal vs MPS):
#     bilateral  2.12 / 489 / —      clahe       0.53 / 8130 / —
#     hue        0.32 / 12.2 / 7.86  saturation  0.32 / 12.2 / 0.56
#     gaussian   2.03 / 4.10 / 1.63  resize AA   0.48 / 3.99 / 1.16
#   (2048^2 MPS: kornia hue/saturation ~155-159 ms vs our 0.61-0.63.)
#
# Honest read:
# - The specialized neighborhood/histogram paths are the structural wins:
#   versus kornia, bilateral is 45-230x, clahe 11-15253x, and hue/
#   saturation 6-261x across these sizes/precisions/backends. Gaussian,
#   unsharp and antialiased resize are generally 1.4-8.4x faster.
# - Blanket "wins everything" wording would be false. Kornia's tiny CUDA
#   invert/grayscale/normalize/solarize cells can be 1.4-3.6x faster.
#   Torchvision is the strongest baseline and wins several native CUDA
#   elementwise/unsharp cells plus a few small Metal cells; slangimage wins
#   the heavier operations it implements, while torchvision has no bilateral
#   or CLAHE equivalent.
# - DISPATCH FLOOR, FIXED 2026-07-12: out-of-frame op outputs now
#   auto-recycle through the texture pool when their Image handle is
#   dropped (device.create_texture cost ~0.3 ms/call on CUDA — that
#   WAS the floor). CUDA elementwise cells went 0.34-0.37 ms ->
#   0.08 ms: gamma/saturation/posterize now BEAT kornia (1.8-13x),
#   torchvision's edge shrank from ~20x to 2-4x — the remainder is
#   slangpy per-call overhead (~0.03 ms) + sync, upstream territory.
#   On Metal the floor is the command-buffer round trip (~0.3 ms,
#   which MPS pays identically), so Metal cells were already parity.
# - Track C parity set: equalize is 214-303x faster than kornia and 5-11x
#   faster than torchvision on Metal; on CUDA it ranges from near parity
#   to 2.5x versus kornia and is 1.4-1.6x faster than torchvision.
#   Autocontrast is 12-18x faster on Metal, but torchvision is 2.6-5.3x
#   faster on CUDA (our two-dispatch histogram/min-max path).
#
# These are latency/cost-class comparisons, not claims of identical output:
# border handling, color models and resize filters differ where noted above,
# and slangimage processes an RGBA texture while the tensor baselines use RGB.

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

import slangimage as si
import slangpy as spy

try:
    import torchvision
    from torchvision.transforms.v2 import InterpolationMode
    from torchvision.transforms.v2 import functional as TVF
except ImportError:
    torchvision = None
    TVF = None

if torch.cuda.is_available():
    TORCH_DEV, SYNC = "cuda", torch.cuda.synchronize
elif torch.backends.mps.is_available():
    TORCH_DEV, SYNC = "mps", torch.mps.synchronize
else:
    raise SystemExit("no torch GPU backend available")

LIBS = ("slangimage", "kornia", "torchvision")
HUE_RAD = 1.0


def _ops():
    """op name -> {lib -> fn(img_or_tensor)}; si fns take si.Image."""
    tv = TVF  # None if torchvision is missing
    return [
        ("gaussian 19x19", {
            "slangimage": lambda x: si.ops.gaussian_blur(x, radius=9, sigma=3.0),
            "kornia": lambda t: KF.gaussian_blur2d(t, (19, 19), (3.0, 3.0)),
            "torchvision": tv and (lambda t: tv.gaussian_blur(t, [19, 19], [3.0, 3.0])),
        }),
        ("bilateral 11x11", {
            "slangimage": lambda x: si.ops.bilateral(
                x, radius=5, sigma_spatial=2.5, sigma_range=0.15),
            "kornia": lambda t: KF.bilateral_blur(t, (11, 11), 0.15, (2.5, 2.5)),
            "torchvision": None,
        }),
        ("unsharp r2", {
            "slangimage": lambda x: si.ops.unsharp_mask(x, radius=2.0, sigma=1.5),
            "kornia": lambda t: KF.unsharp_mask(t, (5, 5), (1.5, 1.5)),
            "torchvision": tv and (lambda t: tv.adjust_sharpness(t, 2.0)),
        }),
        ("clahe 8x8", {
            "slangimage": lambda x: si.ops.clahe(x, tiles=(8, 8)),
            "kornia": lambda t: KE.equalize_clahe(t, grid_size=(8, 8)),
            "torchvision": None,      # global equalize only, uint8-oriented
        }),
        ("hue rotate", {
            "slangimage": lambda x: si.ops.hue_rotate(x, angle=HUE_RAD),
            "kornia": lambda t: KE.adjust_hue(t, HUE_RAD),
            "torchvision": tv and (lambda t: tv.adjust_hue(t, HUE_RAD / (2 * math.pi))),
        }),
        ("gamma 1.8", {
            "slangimage": lambda x: si.ops.gamma(x, power=1.8),
            "kornia": lambda t: KE.adjust_gamma(t, 1.8),
            "torchvision": tv and (lambda t: tv.adjust_gamma(t, 1.8)),
        }),
        ("saturation", {
            "slangimage": lambda x: si.ops.color_controls(x, saturation=0.5),
            "kornia": lambda t: KE.adjust_saturation(t, 1.5),
            "torchvision": tv and (lambda t: tv.adjust_saturation(t, 1.5)),
        }),
        ("invert", {
            "slangimage": lambda x: si.ops.invert(x),
            "kornia": lambda t: KE.invert(t),
            "torchvision": tv and (lambda t: tv.invert(t)),
        }),
        ("resize x0.5 AA", {
            "slangimage": lambda x: si.ops.resize(x, scale=0.5, mode="lanczos3"),
            "kornia": lambda t: k_resize(
                t, (t.shape[-2] // 2, t.shape[-1] // 2),
                interpolation="bicubic", antialias=True),
            "torchvision": tv and (lambda t: tv.resize(
                t, [t.shape[-2] // 2, t.shape[-1] // 2],
                interpolation=InterpolationMode.BICUBIC, antialias=True)),
        }),
        ("solarize 0.5", {
            "slangimage": lambda x: si.ops.solarize(x, 0.5),
            "kornia": lambda t: KE.solarize(t, 0.5),
            "torchvision": tv and (lambda t: tv.solarize(t, 0.5)),
        }),
        ("posterize 3", {
            "slangimage": lambda x: si.ops.posterize(x, 3),
            "kornia": lambda t: KE.posterize(t, 3),
            "torchvision": tv and (lambda t: tv.posterize(t, 3)),
        }),
        ("equalize", {
            "slangimage": lambda x: si.ops.equalize(x),
            "kornia": lambda t: KE.equalize(t),
            "torchvision": tv and (lambda t: tv.equalize(t)),
        }),
        ("autocontrast", {
            "slangimage": lambda x: si.ops.autocontrast(x),
            "kornia": None,           # kornia has no autocontrast
            "torchvision": tv and (lambda t: tv.autocontrast(t)),
        }),
        ("grayscale", {
            "slangimage": lambda x: si.ops.grayscale(x),
            "kornia": lambda t: KC.rgb_to_grayscale(t),
            "torchvision": tv and (lambda t: tv.rgb_to_grayscale(t)),
        }),
        ("normalize", {
            "slangimage": lambda x: si.ops.normalize(
                x, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            "kornia": lambda t: KE.normalize(
                t, torch.tensor([0.485, 0.456, 0.406], device=t.device, dtype=t.dtype),
                torch.tensor([0.229, 0.224, 0.225], device=t.device, dtype=t.dtype)),
            "torchvision": tv and (lambda t: tv.normalize(
                t, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])),
        }),
    ]


def median_ms(fn, sync, iters: int) -> float | None:
    try:
        t0 = time.perf_counter()
        fn()
        sync()
        warm = time.perf_counter() - t0
    except Exception as e:  # unsupported dtype/backend combos happen
        print(f"    ERR: {type(e).__name__}: {str(e)[:90]}")
        return None
    if warm > 2.0:
        iters = 2  # pathologically slow cell: variance is irrelevant here
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        sync()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[1024, 2048])
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    if TVF is None:
        print("torchvision not installed — skipping its column")
    si.prewarm(precision="fp16", size=64)
    si.prewarm(precision="fp32", size=64)

    results: dict = {
        "torch_device": TORCH_DEV,
        "adapter": si.device().info.adapter_name,
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "slangimage": si.__version__,
        "slangpy": spy.__version__,
        "torch": torch.__version__,
        "kornia": kornia.__version__,
        "torchvision": torchvision.__version__ if torchvision is not None else None,
        "iters": args.iters,
        "cases": [],
    }
    print(f"torch device: {TORCH_DEV}  torch {torch.__version__}  "
          f"batch 1, forward only, median of {args.iters}")

    for size in args.sizes:
        rng = np.random.default_rng(3)
        arr = rng.random((size, size, 4)).astype(np.float32)
        arr[..., 3] = 1.0

        print(f"\n=== {size}x{size} ===")
        print(f"  {'op':16s} {'':11s} {'fp16 ms':>8s} {'fp32 ms':>8s}"
              f"   (ratios vs slangimage)")

        per_prec: dict = {}
        for prec, tdtype in (("fp16", torch.float16), ("fp32", torch.float32)):
            img = si.from_numpy(
                arr.astype(np.float16 if prec == "fp16" else np.float32),
                precision=prec)
            t = torch.from_numpy(arr[..., :3].copy()).permute(2, 0, 1)[None]
            per_prec[prec] = (img, img.device, t.to(TORCH_DEV, tdtype))

        for name, fns in _ops():
            row: dict = {"size": size, "op": name}
            base: dict = {}
            for lib in LIBS:
                fn = fns[lib]
                if fn is None:
                    row[lib] = None
                    if not (lib == "torchvision" and TVF is None):
                        print(f"  {name:16s} {lib:11s} {'—':>8s} {'—':>8s}"
                              f"   (no equivalent)")
                    continue
                ms: dict = {}
                for prec in ("fp16", "fp32"):
                    img, dev, t = per_prec[prec]
                    if lib == "slangimage":
                        ms[prec] = median_ms(lambda: fn(img), dev.wait_for_idle,
                                             args.iters)
                    else:
                        ms[prec] = median_ms(lambda: fn(t), SYNC, args.iters)
                row[lib] = ms
                cells = "".join(
                    f" {ms[p]:8.2f}" if ms[p] is not None else f" {'ERR':>8s}"
                    for p in ("fp16", "fp32"))
                ratio = ""
                if lib == "slangimage":
                    base = ms
                elif base:
                    parts = [
                        f"{ms[p] / base[p]:.1f}x"
                        if (ms[p] is not None and base.get(p)) else "—"
                        for p in ("fp16", "fp32")]
                    ratio = f"   ({' / '.join(parts)})"
                print(f"  {name:16s} {lib:11s}{cells}{ratio}")
            results["cases"].append(row)

    results["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
