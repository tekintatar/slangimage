#!/usr/bin/env python3
"""Capture and compare a bounded same-input corpus across texture backends.

Capture archives on each machine/backend, then compare them together. Arrays
are stored losslessly in NPZ files; no archive is a permanent golden baseline.
Independent per-operator correctness remains the responsibility of pytest's
numeric references—this tool detects backend disagreement on identical inputs.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import slangpy as spy

import slangimage as si


NON_OPERATORS = {"prewarm", "BLEND_MODES", "identity_lut_np"}
COVERED_EXPORTS = {
    "affine",
    "autocontrast",
    "bilateral",
    "blend",
    "clahe",
    "clamp_multiply",
    "cmyk_halftone",
    "color_controls",
    "crop",
    "ep_unsharp_mask",
    "equalize",
    "exposure",
    "extract_channel",
    "false_color",
    "flip",
    "gamma",
    "gaussian_blur",
    "grayscale",
    "high_pass_contrast",
    "highlight_shadow",
    "hue_rotate",
    "invert",
    "kuwahara",
    "linear_to_srgb",
    "lut",
    "nl_means",
    "normalize",
    "pointillize",
    "posterize",
    "random_noise",
    "resize",
    "rot90",
    "rotate",
    "sepia",
    "sharpen_luma",
    "smooth_threshold",
    "smooth_threshold_bw",
    "solarize",
    "srgb_to_linear",
    "surface_blur",
    "tile",
    "unsharp_mask",
    "vibrance",
    "white_balance",
}
EXACT_CASES = {"flip", "rot90", "crop", "invert", "extract_channel"}
LOOSE_CASE_PREFIXES = (
    "bilateral_",
    "clahe",
    "cmyk_halftone",
    "kuwahara",
    "nl_means",
    "pointillize",
    "resize_",
    "rotate",
)


def _input_array(size: int, precision: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:size, :size].astype(np.float64)
    x /= max(size-1, 1)
    y /= max(size-1, 1)
    rgb = np.stack(
        (
            .1+.8*x,
            .15+.7*y,
            .5+.22*np.sin(5*x+3*y),
        ),
        axis=-1,
    )
    rgb += rng.normal(0, .015, rgb.shape)
    alpha = np.resize([0, .125, .5, .875, 1], (size, size, 1))
    array = np.concatenate((np.clip(rgb, 0, 1), alpha), axis=-1)
    return array.astype(np.float16 if precision == "fp16" else np.float32)


def _nonlinear_lut(precision: str) -> np.ndarray:
    lut = si.ops.identity_lut_np().astype(np.float64)
    red, green, blue = np.moveaxis(lut[..., :3], -1, 0)
    lut[..., :3] = np.stack(
        (
            .08+.84*red**1.2,
            np.clip(.05+.9*green+.04*blue, 0, 1),
            np.sqrt(np.clip(.85*blue+.1*red, 0, 1)),
        ),
        axis=-1,
    )
    return lut.astype(np.float16 if precision == "fp16" else np.float32)


def _cases(image, second, lut_image):
    cases = {
        "gaussian_blur": lambda: si.ops.gaussian_blur(image, radius=3, sigma=1.4),
        "surface_blur": lambda: si.ops.surface_blur(image, radius=5, bsigma=.12),
        "bilateral_full": lambda: si.ops.bilateral(image, 2, 1.2, .18),
        "bilateral_separable": lambda: si.ops.bilateral(image, 2, 1.2, .18, separable=True),
        "nl_means": lambda: si.ops.nl_means(image, 2, 1, .15, .02),
        "unsharp_mask_rgb": lambda: si.ops.unsharp_mask(image, 2, 1.1, 1.1, "rgb"),
        "unsharp_mask_luma": lambda: si.ops.unsharp_mask(image, 2, 1.1, 1.1, "luma"),
        "ep_unsharp_mask": lambda: si.ops.ep_unsharp_mask(image, 3, .1, .7),
        "sharpen_luma": lambda: si.ops.sharpen_luma(image, .6),
        "vibrance": lambda: si.ops.vibrance(image, .55),
        "gamma": lambda: si.ops.gamma(image, 1.2),
        "color_controls": lambda: si.ops.color_controls(image, .35, -.25, .15, .4),
        "srgb_to_linear": lambda: si.ops.srgb_to_linear(image),
        "linear_to_srgb": lambda: si.ops.linear_to_srgb(image),
        "exposure": lambda: si.ops.exposure(image, .7),
        "sepia": lambda: si.ops.sepia(image, .65),
        "hue_rotate": lambda: si.ops.hue_rotate(image, .8),
        "false_color": lambda: si.ops.false_color(image, (.1, .25, .7), (.95, .6, .1)),
        "highlight_shadow": lambda: si.ops.highlight_shadow(image, .35, .7),
        "solarize": lambda: si.ops.solarize(image, .43),
        "posterize": lambda: si.ops.posterize(image, 5),
        "grayscale": lambda: si.ops.grayscale(image),
        "normalize": lambda: si.ops.normalize(image, (.4, .5, .6), (.7, 1.1, -.8)),
        "white_balance": lambda: si.ops.white_balance(image, 4300, 35),
        "equalize": lambda: si.ops.equalize(image),
        "autocontrast": lambda: si.ops.autocontrast(image),
        "clahe": lambda: si.ops.clahe(image, 2.5, (3, 2), .8),
        "resize_lanczos3": lambda: si.ops.resize(image, width=17, height=19),
        "resize_bicubic": lambda: si.ops.resize(image, width=29, height=21, mode="bicubic"),
        "flip": lambda: si.ops.flip(image, True, True),
        "rot90": lambda: si.ops.rot90(image, -1),
        "crop": lambda: si.ops.crop(image, 3, 4, 16, 15),
        "tile_bilinear": lambda: si.ops.tile(image, 27, 29, 1.2, .8, .35, -.4),
        "tile_nearest": lambda: si.ops.tile(image, 27, 29, 1.2, .8, .35, -.4, "nearest"),
        "affine_bilinear": lambda: si.ops.affine(image, [[1, .08, .4], [-.05, 1, .7]]),
        "affine_nearest": lambda: si.ops.affine(
            image,
            [[1, .08, .4], [-.05, 1, .7]],
            mode="nearest",
        ),
        "rotate": lambda: si.ops.rotate(image, 23, expand=True),
        "kuwahara": lambda: si.ops.kuwahara(image, 2, 3.5, 3),
        "pointillize": lambda: si.ops.pointillize(image, 5, .7, 2.5),
        "cmyk_halftone": lambda: si.ops.cmyk_halftone(image, 5, 13, .06),
        "smooth_threshold": lambda: si.ops.smooth_threshold(image, .47, .16),
        "smooth_threshold_bw": lambda: si.ops.smooth_threshold_bw(image, .2, .78),
        "invert": lambda: si.ops.invert(image),
        "extract_channel": lambda: si.ops.extract_channel(image, "g"),
        "clamp_multiply": lambda: si.ops.clamp_multiply(image, second),
        "high_pass_contrast": lambda: si.ops.high_pass_contrast(image, second, 1.3),
        "lut": lambda: si.ops.lut(image, lut_image, .8),
    }
    for mode in si.ops.BLEND_MODES:
        cases[f"blend_{mode}"] = lambda mode=mode: si.ops.blend(image, second, mode, .65)
    return cases


def _case_export(name: str) -> str:
    for prefix, export in (
        ("affine_", "affine"),
        ("bilateral_", "bilateral"),
        ("blend_", "blend"),
        ("resize_", "resize"),
        ("tile_", "tile"),
        ("unsharp_mask_", "unsharp_mask"),
    ):
        if name.startswith(prefix):
            return export
    return name


def capture(backend: str, output: Path, size: int = 24) -> dict[str, object]:
    if size < 20:
        raise ValueError("size must be >= 20")
    device = si.get_default_device(backend)
    expected_type = getattr(spy.DeviceType, backend)
    if device.info.type != expected_type:
        raise RuntimeError(f"requested {backend}, got {device.info.type}")
    arrays: dict[str, np.ndarray] = {}
    covered = set()
    for precision in ("fp16", "fp32"):
        image = si.from_numpy(
            _input_array(size, precision, 1501),
            precision=precision,
            device=device,
        )
        second = si.from_numpy(
            _input_array(size, precision, 1502),
            precision=precision,
            device=device,
        )
        lut_image = si.from_numpy(
            _nonlinear_lut(precision),
            precision=precision,
            device=device,
        )
        for name, operation in _cases(image, second, lut_image).items():
            result = operation()
            values = result.to_numpy()
            if not np.isfinite(values).all():
                raise RuntimeError(f"{precision}/{name} produced nonfinite output")
            arrays[f"{precision}__{name}"] = values
            covered.add(_case_export(name))
        noise = si.ops.random_noise(size, size, 3.25, precision, device)
        arrays[f"{precision}__random_noise"] = noise.to_numpy()
        covered.add("random_noise")
    if covered != COVERED_EXPORTS:
        raise AssertionError(
            f"capture coverage mismatch: missing={sorted(COVERED_EXPORTS-covered)}, "
            f"extra={sorted(covered-COVERED_EXPORTS)}"
        )
    device.wait_for_idle()
    metadata = {
        "schema": 1,
        "backend": backend,
        "device_type": str(device.info.type),
        "adapter": device.info.adapter_name,
        "size": size,
        "covered_exports": sorted(covered),
    }
    arrays["__metadata__"] = np.frombuffer(
        json.dumps(metadata, sort_keys=True).encode(),
        dtype=np.uint8,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    return metadata


def load_archive(path: Path) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(bytes(archive["__metadata__"]).decode())
        arrays = {key: archive[key] for key in archive.files if key != "__metadata__"}
    if metadata.get("schema") != 1:
        raise ValueError(f"{path} has unsupported schema {metadata.get('schema')!r}")
    if set(metadata.get("covered_exports", ())) != COVERED_EXPORTS:
        raise ValueError(f"{path} does not cover the current texture-export manifest")
    return metadata, arrays


def _noise_metrics(values: np.ndarray) -> tuple[float, float]:
    if not np.array_equal(values[..., 0], values[..., 1]) or not np.array_equal(
        values[..., 1],
        values[..., 2],
    ):
        raise AssertionError("random_noise capture is not grayscale")
    if not np.array_equal(values[..., 3], np.ones(values.shape[:2], dtype=values.dtype)):
        raise AssertionError("random_noise capture is not opaque")
    if values[..., :3].min() < .049 or values[..., :3].max() > .951:
        raise AssertionError("random_noise capture escaped its storage-aware range")
    channel = values[..., 0].astype(np.float64)
    return float(channel.mean()), float(channel.std())


def comparison_bound(key: str) -> tuple[float, float]:
    precision, name = key.split("__", 1)
    if name in EXACT_CASES:
        return 0, 0
    if name == "kuwahara":
        # Each backend passes the independent propagated-variance oracle. Two
        # valid results can differ by the sum of their bounded error envelopes.
        return (.005 if precision == "fp16" else .0021), 0
    loose = name.startswith(LOOSE_CASE_PREFIXES)
    if precision == "fp16":
        return (.005 if loose else .002), 0
    return (.0008 if loose else .00008), 2e-5


def compare_archives(paths: list[Path]) -> list[dict[str, object]]:
    loaded = [(path, *load_archive(path)) for path in paths]
    backends = [metadata["backend"] for _, metadata, _ in loaded]
    if len(set(backends)) != len(backends):
        raise AssertionError("capture archives must come from distinct backends")
    key_sets = [set(arrays) for _, _, arrays in loaded]
    if not key_sets or any(keys != key_sets[0] for keys in key_sets[1:]):
        raise AssertionError("capture archives do not contain identical case keys")
    reports = []
    failures = []
    for (left_path, left_meta, left), (right_path, right_meta, right) in combinations(loaded, 2):
        pair = f"{left_meta['backend']}:{right_meta['backend']}"
        if left_meta["size"] != right_meta["size"]:
            raise AssertionError(f"{pair} used different input sizes")
        for key in sorted(key_sets[0]):
            a, b = left[key], right[key]
            if a.shape != b.shape:
                failures.append(f"{pair} {key}: shape {a.shape} != {b.shape}")
                continue
            _, name = key.split("__", 1)
            max_abs = float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64))))
            if name == "random_noise":
                mean_a, std_a = _noise_metrics(a)
                mean_b, std_b = _noise_metrics(b)
                passed = abs(mean_a-mean_b) <= .04 and abs(std_a-std_b) <= .04
                detail = {"mean_delta": abs(mean_a-mean_b), "std_delta": abs(std_a-std_b)}
            elif name == "pointillize":
                rgb_changed = np.any(a[..., :3] != b[..., :3], axis=-1)
                mismatch_rate = float(rgb_changed.mean())
                alpha_exact = bool(np.array_equal(a[..., 3], b[..., 3]))
                passed = mismatch_rate <= .01 and alpha_exact
                detail = {
                    "pixel_mismatch_rate": mismatch_rate,
                    "pixel_mismatch_limit": .01,
                    "alpha_exact": alpha_exact,
                }
            else:
                atol, rtol = comparison_bound(key)
                scale = np.maximum(np.abs(a), np.abs(b)).astype(np.float64)
                passed = bool(np.all(np.abs(a.astype(np.float64)-b.astype(np.float64)) <= atol+rtol*scale))
                detail = {"atol": atol, "rtol": rtol}
            report = {
                "pair": pair,
                "case": key,
                "max_abs": max_abs,
                "passed": passed,
                **detail,
            }
            reports.append(report)
            if not passed:
                failures.append(f"{pair} {key}: max_abs={max_abs:.9g} ({left_path}, {right_path})")
    if failures:
        raise AssertionError("cross-backend comparison failures:\n"+"\n".join(failures))
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture_parser = subparsers.add_parser("capture")
    capture_parser.add_argument("--backend", choices=("metal", "cuda", "vulkan"), required=True)
    capture_parser.add_argument("--output", type=Path, required=True)
    capture_parser.add_argument("--size", type=int, default=24)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("archives", nargs="+", type=Path)
    compare_parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.command == "capture":
        print(json.dumps(capture(args.backend, args.output, args.size), indent=2))
        return
    if len(args.archives) < 2:
        parser.error("compare needs at least two capture archives")
    reports = compare_archives(args.archives)
    regular = [
        item
        for item in reports
        if not item["case"].endswith(("random_noise", "pointillize"))
    ]
    summary = {
        "pairs": sorted({item["pair"] for item in reports}),
        "comparisons": len(reports),
        "max_abs_including_backend_local_cases": max(item["max_abs"] for item in reports),
        "max_bounded_numeric_abs": max(item["max_abs"] for item in regular),
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"summary": summary, "cases": reports}, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
