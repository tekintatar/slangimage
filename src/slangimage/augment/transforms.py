# augment/transforms.py — the built-in transforms. Names, parameters and
# results follow torchvision.transforms.v2 on float images; random values are
# drawn per image unless same_on_batch=True. RandomGrayscale and
# RandomSolarize are written with the public ColorOp API.

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Sequence

from ._base import ST, Binding, ColorOp, P, Transform, Uniform, _probability, lit

if TYPE_CHECKING:
    import torch


def _size2(size: Any, name: str) -> tuple[int, int]:
    """torchvision size argument -> (height, width)."""
    if isinstance(size, int):
        h = w = size
    elif isinstance(size, Sequence) and len(size) == 1:
        h = w = int(size[0])
    elif isinstance(size, Sequence) and len(size) == 2:
        h, w = int(size[0]), int(size[1])
    else:
        raise ValueError(f"{name}: size must be an int or (height, width), got {size!r}")
    if h <= 0 or w <= 0:
        raise ValueError(f"{name}: size must be positive, got {size!r}")
    return h, w


def _range(value: Any, name: str, center: float, bound: tuple[float, float],
           around_center: bool) -> tuple[float, float] | None:
    """torchvision ColorJitter range checking; None when the op is a no-op."""
    if isinstance(value, (int, float)):
        if value < 0:
            raise ValueError(f"ColorJitter: {name} must be non-negative, got {value}")
        lo, hi = center - value, center + value
        if around_center:
            lo = max(lo, 0.0)
    elif isinstance(value, Sequence) and len(value) == 2:
        lo, hi = float(value[0]), float(value[1])
    else:
        raise ValueError(f"ColorJitter: {name} must be a number or (min, max), got {value!r}")
    if not (math.isfinite(hi) and bound[0] <= lo <= hi <= bound[1]):
        raise ValueError(f"ColorJitter: {name} range ({lo}, {hi}) must be finite and lie "
                         f"in {bound}")
    return None if lo == hi == center else (lo, hi)


# ---------------------------------------------------------------- geometry
# Geometry transforms edit sampler locals: the source box (gl, gt, gw, gh)
# resampled to (RW, RH), then an offset (gcx, gcy) into that grid. The
# source image is W x H.
class RandomResizedCrop(Transform):
    """torchvision RandomResizedCrop (bilinear), the box drawn per image."""

    _kind = "geometry"
    _resamples = True

    def __init__(self, size: Any, scale: tuple[float, float] = (0.08, 1.0),
                 ratio: tuple[float, float] = (3 / 4, 4 / 3), antialias: bool = True,
                 same_on_batch: bool = False):
        self.size = _size2(size, "RandomResizedCrop")
        self.scale = (float(scale[0]), float(scale[1]))
        self.ratio = (float(ratio[0]), float(ratio[1]))
        if not 0 < self.scale[0] <= self.scale[1]:
            raise ValueError(f"RandomResizedCrop: invalid scale {scale}")
        if not 0 < self.ratio[0] <= self.ratio[1]:
            raise ValueError(f"RandomResizedCrop: invalid ratio {ratio}")
        self.antialias = bool(antialias)
        self.same_on_batch = bool(same_on_batch)

    def _geometry(self, rng: str) -> list[str]:
        h, w = self.size
        return [f"sample_rrc({rng}, W, H, {lit(self.scale[0])}, {lit(self.scale[1])}, "
                f"{lit(self.ratio[0])}, {lit(self.ratio[1])}, gl, gt, gw, gh);",
                f"RW = {w}; RH = {h};"]

    def _reference(self, b: Binding, img: "torch.Tensor", row: list[float]) -> "torch.Tensor":
        import torchvision.transforms.v2.functional as F

        left, top, w, h = (int(v) for v in row[b.geo:b.geo + 4])
        return F.resized_crop(img, top, left, h, w, list(self.size), antialias=self.antialias)

    def __repr__(self) -> str:
        return f"RandomResizedCrop(size={self.size}, scale={self.scale}, ratio={self.ratio})"


class Resize(Transform):
    """torchvision Resize (bilinear): an int resizes the shorter side and
    keeps the aspect ratio; (height, width) resizes to exactly that size."""

    _kind = "geometry"
    _resamples = True

    def __init__(self, size: Any, antialias: bool = True):
        self.shorter = isinstance(size, int) or (isinstance(size, Sequence) and len(size) == 1)
        self.size = _size2(size, "Resize")
        self.antialias = bool(antialias)

    def _geometry(self, rng: str) -> list[str]:
        h, w = self.size
        return [f"resize_shorter(W, H, {h}, RW, RH);" if self.shorter else f"RW = {w}; RH = {h};"]

    def _reference(self, b: Binding, img: "torch.Tensor", row: list[float]) -> "torch.Tensor":
        import torchvision.transforms.v2.functional as F

        size = [self.size[0]] if self.shorter else list(self.size)
        return F.resize(img, size, antialias=self.antialias)

    def __repr__(self) -> str:
        return f"Resize(size={self.size[0] if self.shorter else self.size})"


class CenterCrop(Transform):
    """torchvision CenterCrop; a crop larger than the image pads with zeros."""

    _kind = "geometry"
    _resamples = False

    def __init__(self, size: Any):
        self.size = _size2(size, "CenterCrop")

    def _geometry(self, rng: str) -> list[str]:
        h, w = self.size
        return [f"gcx = center_offset(RW, {w}); gcy = center_offset(RH, {h});"]

    def _reference(self, b: Binding, img: "torch.Tensor", row: list[float]) -> "torch.Tensor":
        import torchvision.transforms.v2.functional as F

        return F.center_crop(img, list(self.size))

    def __repr__(self) -> str:
        return f"CenterCrop(size={self.size})"


class RandomHorizontalFlip(Transform):
    """torchvision RandomHorizontalFlip, decided per image."""

    _kind = "flip"

    def __init__(self, p: float = 0.5, same_on_batch: bool = False):
        self.p = _probability(p)
        self.same_on_batch = bool(same_on_batch)

    def _slot_count(self) -> int:
        return 1

    def _sample(self, b: Binding, rng: str) -> list[str]:
        return [f"p[uint2({b.k(self)}, b)] = rnd({rng}) < {lit(self.p)} ? 1.0 : 0.0;"]

    def _reference(self, b: Binding, img: "torch.Tensor", row: list[float]) -> "torch.Tensor":
        import torchvision.transforms.v2.functional as F

        return F.horizontal_flip(img) if row[b.k(self)] > 0.5 else img

    def __repr__(self) -> str:
        return f"RandomHorizontalFlip(p={self.p})"


# ------------------------------------------------------------ colour ops
class ColorJitter(Transform):
    """torchvision ColorJitter: brightness, contrast, saturation and hue in a
    random order, with factors and order drawn per image. Contrast blends with
    each image's exact mean gray value, as torchvision does."""

    def __init__(self, brightness: Any = 0, contrast: Any = 0, saturation: Any = 0,
                 hue: Any = 0, same_on_batch: bool = False):
        self.ranges = [
            _range(brightness, "brightness", 1.0, (0.0, math.inf), True),
            _range(contrast, "contrast", 1.0, (0.0, math.inf), True),
            _range(saturation, "saturation", 1.0, (0.0, math.inf), True),
            _range(hue, "hue", 0.0, (-0.5, 0.5), False),
        ]
        self.same_on_batch = bool(same_on_batch)

    def _slot_count(self) -> int:
        return 5        # four factors and the order

    def _stat_count(self) -> int:
        return 1 if self.ranges[1] else 0

    def _sample(self, b: Binding, rng: str) -> list[str]:
        k = b.k(self)
        lines = [f"p[uint2({k + i}, b)] = urand({rng}, {lit(r[0])}, {lit(r[1])});"
                 for i, r in enumerate(self.ranges) if r]
        return lines + [f"p[uint2({k + 4}, b)] = float(random_order({rng}));"]

    def _apply(self, b: Binding) -> list[str]:
        k = b.k(self)
        cases = []
        if self.ranges[0]:
            cases.append(f"if (op == 0) c = brightness(c, {P(k)});")
        if self.ranges[1]:
            st = b.stats(self)[0]
            cases.append(f"if (op == 1) {{ if (upto == {st}) return c; "
                         f"c = contrast(c, {P(k + 1)}, {ST(st)}); }}")
        if self.ranges[2]:
            cases.append(f"if (op == 2) c = saturation(c, {P(k + 2)});")
        if self.ranges[3]:
            cases.append(f"if (op == 3) c = hue(c, {P(k + 3)});")
        if not cases:
            return []
        return [f"{{ int order = int({P(k + 4)});",
                "  for (int j = 0; j < 4; ++j) { int op = (order >> (2 * j)) & 3;",
                *[f"    {c}" for c in cases], "  } }"]

    def _reference(self, b: Binding, img: "torch.Tensor", row: list[float]) -> "torch.Tensor":
        import torchvision.transforms.v2.functional as F

        k = b.k(self)
        fns = [F.adjust_brightness, F.adjust_contrast, F.adjust_saturation, F.adjust_hue]
        order = int(row[k + 4])
        for j in range(4):
            op = (order >> (2 * j)) & 3
            if self.ranges[op]:
                img = fns[op](img, row[k + op])
        return img

    def __repr__(self) -> str:
        return f"ColorJitter(ranges={self.ranges})"


class RandomGrayscale(ColorOp):
    """torchvision RandomGrayscale (three equal output channels)."""

    slang = "float4 apply(float4 c) { float g = gray(c); return float4(g, g, g, 0.0); }"

    def __init__(self, p: float = 0.1, same_on_batch: bool = False):
        super().__init__(p, same_on_batch)

    def reference(self, img: "torch.Tensor", **params: float) -> "torch.Tensor":
        import torchvision.transforms.v2.functional as F

        return F.rgb_to_grayscale(img, num_output_channels=3)


class RandomSolarize(ColorOp):
    """torchvision RandomSolarize. ``threshold`` is on the 0-1 scale (the
    pipeline works on floats): use 0.5, not 128."""

    slang = """
    float4 apply(float4 c, float t)
    {
        return float4(c.x >= t ? 1.0 - c.x : c.x, c.y >= t ? 1.0 - c.y : c.y,
                      c.z >= t ? 1.0 - c.z : c.z, 0.0);
    }"""

    def __init__(self, threshold: float, p: float = 0.5, same_on_batch: bool = False):
        super().__init__(p, same_on_batch)
        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError(f"RandomSolarize: threshold is on the 0-1 scale, got {threshold} "
                             "(divide a uint8 threshold by 255)")
        self.params = {"threshold": float(threshold)}

    def reference(self, img: "torch.Tensor", **params: float) -> "torch.Tensor":
        import torchvision.transforms.v2.functional as F

        return F.solarize(img, params["threshold"])


# ------------------------------------------------------- neighbourhood ops
class GaussianBlur(Transform):
    """torchvision GaussianBlur (reflect border), sigma drawn per image.
    Separable: a horizontal pre-pass, and the vertical half fused into the
    next pass. Wrap it in RandomApply to blur only some images."""

    _kind = "neighbourhood"

    def __init__(self, kernel_size: Any, sigma: Any = (0.1, 2.0), same_on_batch: bool = False):
        kx, ky = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        if any(k <= 0 or k % 2 == 0 for k in (kx, ky)):
            raise ValueError(f"GaussianBlur: kernel_size must be positive and odd, got {kernel_size}")
        self.kernel_size = (int(kx), int(ky))
        lo, hi = (sigma, sigma) if isinstance(sigma, (int, float)) else sigma
        if not 0 < float(lo) <= float(hi):
            raise ValueError(f"GaussianBlur: sigma must satisfy 0 < min <= max, got {sigma}")
        self.sigma = Uniform(lo, hi)
        self.same_on_batch = bool(same_on_batch)

    def _slot_count(self) -> int:
        return 1

    def _sample(self, b: Binding, rng: str) -> list[str]:
        return [f"p[uint2({b.k(self)}, b)] = {self.sigma._draw(rng)};"]

    def _taps(self, b: Binding, radius: int, coord: str, size: str, load: str) -> list[str]:
        return [f"float sig = {P(b.k(self))}; float4 acc = float4(0.0); float ws = 0.0;",
                f"for (int i = -{radius}; i <= {radius}; ++i) {{",
                "    float w = gauss(i, sig); ws += w;",
                f"    int q = reflect({coord} + i, {size}); acc += w * {load}; }}",
                "return acc / ws;"]

    def _pre(self, b: Binding) -> list[str] | None:
        return self._taps(b, self.kernel_size[0] // 2, "x", "OW", "load_mid(src, q, y, b)")

    def _input(self, b: Binding) -> list[str]:
        return self._taps(b, self.kernel_size[1] // 2, "y", "OH", "load_mid(src, x, q, b)")

    def _reference(self, b: Binding, img: "torch.Tensor", row: list[float]) -> "torch.Tensor":
        import torchvision.transforms.v2.functional as F

        s = row[b.k(self)]
        return F.gaussian_blur(img, list(self.kernel_size), [s, s])

    def __repr__(self) -> str:
        return f"GaussianBlur(kernel_size={self.kernel_size}, sigma={self.sigma})"


# --------------------------------------------------------------- others
class RandomApply(Transform):
    """torchvision RandomApply: apply ``transforms`` to an image with
    probability ``p``. Holds pointwise ops, or exactly one op that reads
    neighbouring pixels (a blur)."""

    def __init__(self, transforms: Sequence[Transform], p: float = 0.5,
                 same_on_batch: bool = False):
        self.transforms = list(transforms)
        self.p = _probability(p)
        self.same_on_batch = bool(same_on_batch)
        if not self.transforms:
            raise ValueError("RandomApply needs at least one transform")
        for t in self.transforms:
            if not isinstance(t, Transform):
                raise TypeError(f"RandomApply takes slangimage.augment transforms, got {t!r}")
        kinds = {t._kind for t in self.transforms}
        if "neighbourhood" in kinds and len(self.transforms) != 1:
            raise ValueError("RandomApply holds either pointwise ops or exactly one op that "
                             "reads neighbouring pixels (such as GaussianBlur)")
        if kinds - {"pointwise", "neighbourhood"}:
            raise ValueError("RandomApply cannot hold geometric transforms or Normalize")
        self._kind = "neighbourhood" if "neighbourhood" in kinds else "pointwise"

    def _children(self) -> list[Transform]:
        return self.transforms

    def _slot_count(self) -> int:
        return 1

    def _sample(self, b: Binding, rng: str) -> list[str]:
        return [f"p[uint2({b.k(self)}, b)] = rnd({rng}) < {lit(self.p)} ? 1.0 : 0.0;"]

    def _apply(self, b: Binding) -> list[str]:
        body = [ln for t in self.transforms for ln in t._apply(b)]
        return [f"if ({P(b.k(self))} > 0.5) {{", *[f"  {ln}" for ln in body], "}"]

    def _reference(self, b: Binding, img: "torch.Tensor", row: list[float]) -> "torch.Tensor":
        if row[b.k(self)] > 0.5:
            for t in self.transforms:
                img = t._reference(b, img, row)
        return img

    def __repr__(self) -> str:
        return f"RandomApply({self.transforms}, p={self.p})"


class Normalize(Transform):
    """torchvision Normalize; must be the last transform."""

    _kind = "normalize"

    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        self.mean = tuple(float(m) for m in mean)
        self.std = tuple(float(s) for s in std)
        if len(self.mean) != 3 or len(self.std) != 3:
            raise ValueError("Normalize: mean and std need three values (RGB)")
        if any(s == 0 for s in self.std):
            raise ValueError("Normalize: std must be non-zero")

    def _reference(self, b: Binding, img: "torch.Tensor", row: list[float]) -> "torch.Tensor":
        import torchvision.transforms.v2.functional as F

        return F.normalize(img, list(self.mean), list(self.std))

    def __repr__(self) -> str:
        return f"Normalize(mean={self.mean}, std={self.std})"
