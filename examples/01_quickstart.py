# examples/01_quickstart.py
# Load an image onto the GPU, chain a few operators, save the result.

from _common import demo_image, out_path, pick_device

import slangimage as si


def main():
    # fp16 on the GPU; --backend metal|cuda|vulkan overrides automatic selection
    img = si.load(demo_image(), device=pick_device())
    print("loaded:", img)

    out = si.ops.surface_blur(img, radius=9, bsigma=0.05)
    out = si.ops.vibrance(out, amount=0.6)
    out = si.ops.clahe(out, clip_limit=2.5)

    # equivalent chaining style
    out2 = (
        img.pipe(si.ops.kuwahara, radius=6)
        .pipe(si.ops.sharpen_luma, sharpness=0.8)
    )

    p1 = out.save(out_path("quickstart_enhanced.png"))
    p2 = out2.save(out_path("quickstart_painterly.png"))
    print("saved:", p1, "and", p2)


if __name__ == "__main__":
    main()
