#!/usr/bin/env python3
"""Validate an installed SlangImage artifact from outside its checkout."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import os
from pathlib import Path


def _source_payload(source_root: Path) -> dict[str, bytes]:
    package_root = source_root / "src" / "slangimage"
    payload: dict[str, bytes] = {}
    for path in package_root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(package_root)
        if path.suffix not in {".py", ".slang"} and relative.as_posix() != "py.typed":
            continue
        payload[relative.as_posix()] = path.read_bytes()
    payload["playground.py"] = (source_root / "playground" / "app.py").read_bytes()
    return payload


def _entry_points(group: str):
    points = importlib.metadata.entry_points()
    return points.select(group=group) if hasattr(points, "select") else points.get(group, ())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--gpu-smoke", action="store_true")
    parser.add_argument("--backend", default="auto")
    args = parser.parse_args()

    os.environ["SLANGIMAGE_BACKEND"] = args.backend
    import numpy as np
    import slangimage as si

    source_root = args.source_root.resolve()
    package_root = Path(si.__file__).resolve().parent
    if package_root.is_relative_to(source_root):
        raise RuntimeError(f"import resolved to checkout instead of artifact: {package_root}")

    expected = _source_payload(source_root)
    actual = {
        path.relative_to(package_root).as_posix(): path.read_bytes()
        for path in package_root.rglob("*")
        if path.is_file()
        and (path.suffix in {".py", ".slang"} or path.name == "py.typed")
        and "__pycache__" not in path.parts
    }
    if actual.keys() != expected.keys():
        raise RuntimeError(
            f"installed payload mismatch: missing={expected.keys() - actual.keys()}, "
            f"extra={actual.keys() - expected.keys()}"
        )
    changed = [name for name in expected if actual[name] != expected[name]]
    if changed:
        raise RuntimeError(f"installed artifact contains stale payload bytes: {changed}")

    installed_version = importlib.metadata.version("slangimage")
    if installed_version != si.__version__:
        raise RuntimeError(
            f"installed version mismatch: metadata={installed_version}, "
            f"module/source={si.__version__}"
        )
    if importlib.util.find_spec("slangimage.playground") is None:
        raise RuntimeError("installed artifact is missing slangimage.playground")
    scripts = {point.name: point.value for point in _entry_points("console_scripts")}
    if scripts.get("slangimage-playground") != "slangimage.playground:main":
        raise RuntimeError("installed artifact is missing the slangimage-playground entry point")
    if scripts.get("slangimage-bench") != "slangimage.bench.__main__:main":
        raise RuntimeError("installed artifact is missing the slangimage-bench entry point")

    if args.gpu_smoke:
        arr = np.full((8, 8, 4), 0.25, dtype=np.float32)
        arr[..., 3] = 0.4
        image = si.from_numpy(arr, device=si.device(args.backend))
        out = si.ops.invert(image).to_numpy()
        np.testing.assert_allclose(out[..., :3], 0.75, atol=1e-3)
        np.testing.assert_allclose(out[..., 3], 0.4, atol=1e-3)

        tiled = si.ops.tile(image, 13, 11, scale_x=1.5, offset_y=-2.0).to_numpy()
        if tiled.shape != (11, 13, 4) or not np.isfinite(tiled).all():
            raise RuntimeError("wrapped-tile GPU smoke failed")
        halftone = si.ops.cmyk_halftone(image, width=4.0, sharpness=0.25).to_numpy()
        if halftone.shape != arr.shape or not np.isfinite(halftone).all():
            raise RuntimeError("CMYK halftone GPU smoke failed")
        clahe = si.ops.clahe(image, clip_limit=1.0, tiles=(2, 2)).to_numpy()
        if clahe.shape != arr.shape or not np.isfinite(clahe).all():
            raise RuntimeError("CLAHE GPU smoke failed")

    print(
        f"installed artifact OK: {package_root} | {len(actual)} runtime files"
        + (f" | {args.backend} GPU smoke" if args.gpu_smoke else "")
    )


if __name__ == "__main__":
    main()
