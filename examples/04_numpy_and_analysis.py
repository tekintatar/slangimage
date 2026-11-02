# examples/04_numpy_and_analysis.py
# NumPy in / NumPy out, and outputs that are data, not color.
# Readback is lossless: values outside [0, 1] survive exactly.

import numpy as np
from _common import pick_device

import slangimage as si


def main():
    rng = np.random.default_rng(3)
    arr = rng.random((512, 512, 3)).astype(np.float16)  # 3ch -> RGBA on upload
    img = si.from_numpy(arr, device=pick_device())  # --backend to override auto
    print("uploaded:", img)

    # training-tensor normalization: negative values are the point
    norm = si.ops.normalize(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    n = norm.to_numpy()  # lossless — nothing clipped to [0, 1]
    print(f"normalized range: [{n.min():.4f}, {n.max():.4f}] (not clipped to [0,1])")

    # strong contrast pushes RGB past both ends; readback keeps the overshoot
    punchy = si.ops.color_controls(img, contrast=1.5)
    c = punchy.to_numpy()
    print(f"contrast range:   [{c.min():.4f}, {c.max():.4f}]")

    result = punchy.to_numpy(clip=True)  # clip only when you want display data
    print("display:", result.shape, result.dtype, f"[{result.min()}, {result.max()}]")


if __name__ == "__main__":
    main()
