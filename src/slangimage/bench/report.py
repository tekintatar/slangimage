# slangimage.bench.report — merge result JSONs into a markdown report.

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BACKEND_ORDER = ["metal", "cuda", "vulkan", "cpu"]
PRECISIONS = ["fp16", "fp32"]


def _fmt(v, nd=1):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "—"


def _load_run(path: str) -> dict[str, Any]:
    with open(path) as f:
        run = json.load(f)
    if not isinstance(run, dict) or not isinstance(run.get("meta"), dict):
        raise ValueError(f"{path}: expected a JSON object with a 'meta' object")
    if not isinstance(run.get("records"), list):
        raise ValueError(f"{path}: expected a 'records' list")

    meta = run["meta"]
    for key in ("backend", "adapter", "slangpy", "platform", "iters", "timestamp"):
        if key not in meta:
            raise ValueError(f"{path}: missing required metadata field meta.{key}")

    seen: set[tuple[str, int, str]] = set()
    for index, record in enumerate(run["records"]):
        if not isinstance(record, dict):
            raise ValueError(f"{path}: record {index} is not an object")
        for key in ("backend", "op", "size", "precision", "steady_ms"):
            if key not in record:
                raise ValueError(f"{path}: record {index} is missing {key!r}")
        if record["backend"] != meta["backend"]:
            raise ValueError(
                f"{path}: record {index} names backend {record['backend']!r}, "
                f"but meta.backend is {meta['backend']!r}"
            )
        if record.get("iters", meta["iters"]) != meta["iters"]:
            raise ValueError(
                f"{path}: record {index} uses {record['iters']} iterations, "
                f"but meta.iters is {meta['iters']}"
            )
        cell = (record["op"], record["size"], record["precision"])
        if cell in seen:
            raise ValueError(f"{path}: duplicate benchmark cell {cell!r}")
        seen.add(cell)
    return run


def _compatibility_error(detail: str) -> ValueError:
    return ValueError(
        f"refusing to merge incompatible benchmark results: {detail}. "
        "Pass --allow-incompatible to override."
    )


def _validate_compatible(runs: list[tuple[str, dict[str, Any]]]) -> None:
    backends = [run["meta"]["backend"] for _, run in runs]
    duplicates = sorted({backend for backend in backends if backends.count(backend) > 1})
    if duplicates:
        raise _compatibility_error(
            f"more than one file was supplied for backend(s) {', '.join(duplicates)}"
        )

    for field, label in (
        ("protocol", "protocol versions"),
        ("iters", "iteration counts"),
        ("slangimage", "SlangImage versions"),
        ("slangpy", "SlangPy versions"),
    ):
        values = {run["meta"].get(field) for _, run in runs}
        if None in values:
            missing = [Path(path).name for path, run in runs if run["meta"].get(field) is None]
            raise _compatibility_error(
                f"{label} cannot be checked because meta.{field} is missing from "
                f"{', '.join(missing)}"
            )
        if len(values) > 1:
            raise _compatibility_error(f"files use different {label} {sorted(values)}")

    signatures: list[dict[str, set[Any]]] = []
    for _, run in runs:
        records = run["records"]
        signatures.append(
            {
                "ops": {record["op"] for record in records},
                "sizes": {record["size"] for record in records},
                "precisions": {record["precision"] for record in records},
                "cells": {
                    (record["op"], record["size"], record["precision"])
                    for record in records
                },
                "first-call cells": {
                    (record["op"], record["precision"])
                    for record in records
                    if "first_call_ms" in record
                },
            }
        )
    for label in ("ops", "sizes", "precisions", "cells", "first-call cells"):
        expected = signatures[0][label]
        if any(signature[label] != expected for signature in signatures[1:]):
            raise _compatibility_error(f"files contain different benchmark suite {label}")


def render(paths: list[str], strict: bool = True) -> str:
    if not paths:
        raise ValueError("at least one benchmark result file is required")

    runs = [(path, _load_run(path)) for path in sorted(paths)]
    if strict:
        _validate_compatible(runs)

    data: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    metas: dict[str, dict[str, Any]] = {}
    for _, run in runs:
        backend = run["meta"]["backend"]
        metas[backend] = run["meta"]
        for record in run["records"]:
            data[(backend, record["op"], record["size"], record["precision"])] = record

    protocols = {meta.get("protocol", 1) for meta in metas.values()}

    backends = [b for b in BACKEND_ORDER if b in metas] + [
        b for b in metas if b not in BACKEND_ORDER
    ]
    ops = list(dict.fromkeys(k[1] for k in data))
    sizes = sorted({k[2] for k in data})

    lines: list[str] = []
    lines.append("# SlangImage Benchmarks\n")
    lines.append(
        f"Protocol v{'/'.join(map(str, sorted(protocols)))} — see docs/BENCHMARKING.md "
        "for methodology and how to reproduce. "
        f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}.\n"
    )
    lines.append(
        "Steady-state = median of synced iterations after warmup (shader/module caches hot). "
        "First-call = module load + slang/backend compile + first dispatch, size-independent. "
        "Bare op calls allocate their output per call.\n"
    )
    for b in backends:
        m = metas[b]
        drv = f", driver {m['gpu_driver']}" if m.get("gpu_driver") else ""
        lines.append(
            f"- **{b}**: {m['adapter']}{drv} — slangimage {m.get('slangimage', '?')}, "
            f"slangpy {m['slangpy']}, python {m.get('python', '?')}, {m['platform']}, "
            f"{m['iters']} iters, {m['timestamp']}"
        )
    lines.append("")

    hdr = "| op | " + " | ".join(f"{b} {p}" for b in backends for p in PRECISIONS) + " |"
    sep = "|---" * (1 + len(backends) * len(PRECISIONS)) + "|"

    lines.append("## Steady-state (ms, median)\n")
    for size in sizes:
        lines.append(f"### {size}×{size} (~{size * size / 1e6:.1f} MP)\n")
        lines.append(hdr)
        lines.append(sep)
        for op in ops:
            row = [op]
            for b in backends:
                for p in PRECISIONS:
                    r = data.get((b, op, size, p))
                    row.append(_fmt(r["steady_ms"], 2) if r else "—")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    lines.append("## First call: compile + dispatch (ms)\n")
    lines.append(
        "Note: the first kernel compiled in the process additionally absorbs "
        "one-time session init — the first suite entry overstates that op's compile cost.\n"
    )
    lines.append(hdr)
    lines.append(sep)
    for op in ops:
        row = [op]
        for b in backends:
            for p in PRECISIONS:
                r = next(
                    (
                        data[k]
                        for k in data
                        if k[0] == b and k[1] == op and k[3] == p and "first_call_ms" in data[k]
                    ),
                    None,
                )
                row.append(_fmt(r["first_call_ms"], 0) if r else "—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    if any("vram_peak_mb" in r for r in data.values()):
        lines.append("## CUDA peak VRAM (MB, absolute device usage during steady loop)\n")
        lines.append("| op | " + " | ".join(f"{s}px {p}" for s in sizes for p in PRECISIONS) + " |")
        lines.append("|---" * (1 + len(sizes) * len(PRECISIONS)) + "|")
        for op in ops:
            row = [op]
            for s in sizes:
                for p in PRECISIONS:
                    r = data.get(("cuda", op, s, p))
                    row.append(_fmt(r.get("vram_peak_mb"), 0) if r else "—")
            lines.append("| " + " | ".join(row) + " |")
        base = next((r["vram_base_mb"] for r in data.values() if "vram_base_mb" in r), None)
        if base is not None:
            lines.append(
                f"\nBaseline device usage before first sampled loop: ~{base:.0f} MB "
                "(CUDA context + caches)."
            )
        lines.append("")

    return "\n".join(lines)
