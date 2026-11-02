# examples/_common.py
# Shared helpers for example scripts. Looks for images in <repo>/test_images
# (override with SLANGIMAGE_TEST_IMAGES); on a fresh clone, where that
# directory doesn't exist, a deterministic synthetic photo-like image is
# generated so every example runs out of the box.

import os
from pathlib import Path

TEST_IMAGE_DIR = Path(
    os.environ.get(
        "SLANGIMAGE_TEST_IMAGES",
        Path(__file__).resolve().parent.parent / "test_images",
    )
)

_SYNTHETIC = "synthetic_1024.png"

# where examples write their output renders (gitignored)
OUT_DIR = Path(__file__).resolve().parent / "out"


def backend_arg() -> str:
    """The backend chosen with ``--backend`` on the command line (default auto)."""
    import argparse

    parser = argparse.ArgumentParser(description="Run a SlangImage example.")
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "metal", "cuda", "vulkan"],
        help="GPU backend (default: auto — Metal on macOS, CUDA elsewhere, "
        "falling back to Vulkan)",
    )
    return parser.parse_args().backend


def pick_device():
    """The device for this example, honoring ``--backend``."""
    import slangimage as si

    return si.device(backend_arg())


def out_path(name: str) -> Path:
    """Return a path under examples/out/ (gitignored), creating the dir."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR / name


def _generate_synthetic(path: Path, size: int = 1024) -> None:
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(7)
    y, x = np.mgrid[0:size, 0:size].astype(np.float32) / size

    # gradients + soft blobs + fine noise: enough structure for every op
    img = np.stack(
        [
            0.55 + 0.35 * np.sin(7.0 * x + 1.0) * np.cos(4.0 * y),
            0.45 + 0.4 * y,
            0.35 + 0.5 * x,
        ],
        axis=-1,
    )
    for _ in range(24):
        cx, cy, r = rng.random(3)
        r = 0.03 + 0.12 * r
        mask = np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / (2 * r * r)))
        img += (rng.random(3) - 0.5)[None, None, :] * 0.5 * mask[..., None]
    img += rng.normal(0, 0.015, img.shape).astype(np.float32)

    arr = (np.clip(img, 0, 1) * 255).round().astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, "RGB").save(path)


def demo_image(name: str = _SYNTHETIC) -> Path:
    p = TEST_IMAGE_DIR / name
    if p.exists():
        return p
    fallback = TEST_IMAGE_DIR / _SYNTHETIC
    if not fallback.exists():
        _generate_synthetic(fallback)
    if name != _SYNTHETIC:
        print(f"[examples] {p} not found — using generated {fallback}")
    return fallback
