# examples/02_operator_tour.py
# One output image per operator family — a quick visual tour.

from _common import demo_image, OUT_DIR, pick_device

import slangimage as si


def main():
    img = si.load(demo_image(), device=pick_device())  # --backend to override auto

    looks = {
        "gaussian": si.ops.gaussian_blur(img, radius=9, sigma=3.0),
        "bilateral": si.ops.bilateral(img, radius=6, sigma_spatial=3.0, sigma_range=0.15),
        "nl_means": si.ops.nl_means(img, search_radius=4, h=0.12),
        "unsharp": si.ops.unsharp_mask(img, radius=6, sigma=2.5, amount=1.2),
        "sepia": si.ops.sepia(img, intensity=0.8),
        "duotone": si.ops.false_color(img, color0=(0.05, 0.0, 0.2), color1=(1.0, 0.95, 0.8)),
        "clahe": si.ops.clahe(img, clip_limit=3.0),
        "kuwahara": si.ops.kuwahara(img, radius=6),
        "pointillize": si.ops.pointillize(img, radius=10.0),
        "halftone": si.ops.cmyk_halftone(img, width=7.0),
        "threshold": si.ops.smooth_threshold(img, threshold=0.5, smooth_amount=0.15),
        "blend_overlay": si.ops.blend(img, si.ops.invert(img), mode="overlay", opacity=0.6),
    }

    out_dir = OUT_DIR / "tour"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, result in looks.items():
        result.save(out_dir / f"{name}.png")
        print("saved", out_dir / f"{name}.png")


if __name__ == "__main__":
    main()
