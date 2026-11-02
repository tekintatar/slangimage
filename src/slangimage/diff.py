# slangimage/diff.py — differentiable image ops on torch tensors.
#
#     import torch, slangimage.diff as sid
#
#     img = torch.rand(512, 512, 3, device="cuda", requires_grad=True)
#     out = sid.gaussian_blur(img, radius=9, sigma=3.0)
#     out = sid.sepia(out, intensity=0.8)
#     loss = (out - target).pow(2).mean()
#     loss.backward()          # gradients flow through the Slang kernels
#
# Every op is a fused Slang kernel wired into torch.autograd — most via
# slangpy's automatic path (compiler-generated backward); bilateral uses
# a hand-written gather backward through a manual torch.autograd.Function
# bridge (the generated scatter backward was ~400x slower). CUDA only
# (torch interop);
# images are (H, W, 3) — or (H, W) / (H, W, C) where noted — float32,
# on the CUDA device. Values are UNCLAMPED so gradients stay alive at
# saturation; clamp for display yourself.

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from .kernels.lut import validate_lut_size

_SHADER = Path(__file__).resolve().parent / "shaders" / "diff" / "diff_kernels.slang"


def _create_torch_cuda_device(spy: Any) -> Any:
    """SlangPy's CUDA device for torch interop, with setup guidance on failure.

    Like the texture path, the new device must compile and run a one-pixel test
    kernel: device creation can succeed while NVRTC cannot be loaded, which
    would otherwise surface only as a raw E52002 error on the first operation.
    """
    from .runtime import device as runtime

    hint = runtime._NVRTC_HINT + " and 'PyTorch extra'."
    try:
        device = spy.create_torch_device(type=spy.DeviceType.cuda)
    except RuntimeError as e:
        raise RuntimeError(
            f"slangimage.diff could not create SlangPy's CUDA device ({runtime._summarize(e)}). "
            + hint
        ) from e
    try:
        runtime._verify_runtime_compiler(device)
    except Exception as e:
        runtime._close_quietly(device)
        raise RuntimeError(
            "slangimage.diff created SlangPy's CUDA device but could not compile a "
            f"test kernel on it ({runtime._summarize(e)}). " + hint
        ) from e
    return device


@functools.lru_cache(maxsize=1)
def _ctx() -> tuple:
    try:
        import torch  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "slangimage.diff requires PyTorch and SlangPy's torch bridge (Linux + "
            "NVIDIA). A plain pip install cannot build the bridge; follow the "
            "three steps in the README section 'PyTorch extra'."
        ) from e
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "slangimage.diff requires a CUDA device (torch interop is CUDA-only)"
        )
    import slangpy as spy

    device = _create_torch_cuda_device(spy)
    # Two module instances = two separate call-data caches. slangpy caches
    # the generated dispatch per call signature WITHOUT the inputs' grad-ness:
    # a no-grad call would poison the cache and silently drop the autograd
    # hook from all later grad-requiring calls (slangpy#1052; drop the twin
    # modules once fixed upstream).
    src = _SHADER.read_text()
    module_grad = spy.Module.load_from_source(device, "slangimage_diff_grad", src)
    module_nograd = spy.Module.load_from_source(device, "slangimage_diff_nograd", src)
    return device, module_grad, module_nograd, spy


def _module(*tensors: torch.Tensor) -> tuple[Any, Any]:
    """Pick the module by whether any input participates in autograd."""
    _, module_grad, module_nograd, spy = _ctx()
    needs_grad = any(getattr(t, "requires_grad", False) for t in tensors)
    return (module_grad if needs_grad else module_nograd), spy


def available() -> bool:
    """True if the differentiable path can actually run: torch + CUDA present
    AND the slangpy torch bridge works.

    Verified with a real 1-pixel forward pass — building the device/modules
    (`_ctx()`) succeeds even when the `slangpy-torch` bridge is missing, so the
    bridge only actually fails at dispatch. Checking it here keeps `available()`
    honest instead of false-positive.
    """
    try:
        import torch

        _ctx()
        exposure(torch.zeros(1, 1, 3, device="cuda"), 0.0)
        return True
    except Exception:
        return False


def _hwc(image: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Normalize to (H, W, C); remember if input was 2D."""
    if image.ndim == 2:
        return image.unsqueeze(-1), True
    if image.ndim == 3:
        return image, False
    raise ValueError(f"expected (H, W) or (H, W, C) tensor, got shape {tuple(image.shape)}")


def _f32c(t: torch.Tensor) -> torch.Tensor:
    """Mixed-precision friendliness: the Slang kernels run fp32. Upcast
    half/bfloat16 inputs (autograd routes gradients back through the cast,
    so fp16 leaves get fp16 grads) and make contiguous for the bridge.
    Outputs are fp32 — cast back yourself if you need storage dtype."""
    import torch

    if t.dtype != torch.float32:
        t = t.float()
    return t.contiguous()


# ------------------------------------------------------------------- blurs
def gaussian_blur(image: torch.Tensor, radius: int = 9,
                  sigma: float = 3.0) -> torch.Tensor:
    """Separable gaussian blur, differentiable. (H, W) or (H, W, C)."""
    radius = int(radius)
    sigma = float(sigma)
    if radius < 1:
        raise ValueError("radius must be >= 1")
    if radius > 64:
        raise ValueError("radius > 64 not supported (kernel MaxIters bound)")
    if not sigma > 0:
        raise ValueError("sigma must be > 0 (0 makes the kernel weights NaN)")
    module, spy = _module(image)
    x, squeeze = _hwc(_f32c(image))
    H, W, C = x.shape
    tmp = module.gaussian_h(spy.grid((H, W, C)), x, radius, sigma, W)
    out = module.gaussian_v(spy.grid((H, W, C)), tmp, radius, sigma, H)
    return out.squeeze(-1) if squeeze else out


# --------------------------------------------------------------- color ops
def exposure(image: torch.Tensor, ev: float = 0.0) -> torch.Tensor:
    """rgb * 2^ev, differentiable, unclamped. (H, W, 3)."""
    module, _ = _module(image)
    return module.exposure(_f32c(image), float(ev))


def sepia(image: torch.Tensor, intensity: float = 1.0) -> torch.Tensor:
    """Classic sepia matrix, lerped by intensity. (H, W, 3)."""
    module, _ = _module(image)
    return module.sepia(_f32c(image), float(intensity))


def hue_rotate(image: torch.Tensor, angle: float = 0.0) -> torch.Tensor:
    """YIQ chroma rotation (radians); luminance-preserving, fully linear. (H, W, 3)."""
    module, _ = _module(image)
    return module.hue_rotate(_f32c(image), float(angle))


@functools.lru_cache(maxsize=1)
def _bilateral_fn():
    """Manual torch.autograd.Function bridge: plain forward kernel + plain
    hand-written GATHER backward kernel (no atomics, deterministic). Slang's
    generated backward for bilateral is an atomic scatter measured at ~400x
    the forward cost (benchmarks/compare_kornia_diff.py). Built lazily so
    importing slangimage.diff never requires torch."""
    import torch

    class _BilateralFn(torch.autograd.Function):
        @staticmethod
        @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
        def forward(ctx, src, radius, sigma_spatial, sigma_range):
            # plain (non-differentiable) dispatches: nograd module twin
            _, _, module, spy = _ctx()
            H, W, _ = src.shape
            outw = module.bilateral_fwd(
                spy.grid((H, W)), src.detach(), int(radius),
                float(sigma_spatial), float(sigma_range), W, H,
            )
            ctx.save_for_backward(src, outw)
            ctx.meta = (radius, sigma_spatial, sigma_range, H, W)
            return outw[..., :3].contiguous()

        @staticmethod
        @torch.amp.custom_bwd(device_type="cuda")
        @torch.autograd.function.once_differentiable
        def backward(ctx, gout):
            src, outw = ctx.saved_tensors
            radius, sigma_spatial, sigma_range, H, W = ctx.meta
            _, _, module, spy = _ctx()
            grad = module.bilateral_bwd(
                spy.grid((H, W)), src.detach(), outw,
                gout.contiguous().detach(), int(radius),
                float(sigma_spatial), float(sigma_range), W, H,
            )
            return grad, None, None, None

    return _BilateralFn


def bilateral(image: torch.Tensor, radius: int = 5, sigma_spatial: float = 2.5,
              sigma_range: float = 0.15) -> torch.Tensor:
    """Differentiable bilateral filter, (H, W, 3). Gradients flow through the
    range weights too — the edge-preservation itself is trainable. Gaussian
    spatial only (the box/cutoff modes of si.ops are not gradient-friendly).
    Border: out-of-bounds taps are skipped (truncated window), not clamped."""
    radius = int(radius)
    sigma_spatial = float(sigma_spatial)
    sigma_range = float(sigma_range)
    if radius < 1:
        raise ValueError("radius must be >= 1")
    if radius > 15:
        raise ValueError("radius > 15 not supported")
    if not sigma_spatial > 0:
        raise ValueError("sigma_spatial must be > 0 in the differentiable path")
    if not sigma_range > 0:
        raise ValueError("sigma_range must be > 0 in the differentiable path")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) tensor, got shape {tuple(image.shape)}")
    _ctx()  # fail early with the torch/CUDA message, not inside apply()
    return _bilateral_fn().apply(
        _f32c(image), radius, sigma_spatial, sigma_range
    )


def unsharp_mask(image: torch.Tensor, radius: int = 2, sigma: float = 1.5,
                 amount: float = 1.0) -> torch.Tensor:
    """Unsharp mask: src + amount * (src - gaussian(src)), differentiable.
    (H, W) or (H, W, C). Sharpen = this with a small radius. Composed from
    the differentiable gaussian + torch arithmetic — gradients are exact
    by construction."""
    blurred = gaussian_blur(image, radius=radius, sigma=sigma)
    return image + float(amount) * (image - blurred)


# ----------------------------------------------------------------- resize
@functools.lru_cache(maxsize=1)
def _resize_fn():
    """Manual bridge like bilateral's, but lighter: resample weights are
    geometry-only, so resize is LINEAR — backward is just the transposed
    resample of the upstream gradient, no saved activations."""
    import torch

    class _ResizeFn(torch.autograd.Function):
        @staticmethod
        @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
        def forward(ctx, src, out_h, out_w, mode):
            _, _, module, spy = _ctx()
            H, W, C = src.shape
            x = src.detach()
            tmp = module.resample_x(
                spy.grid((H, out_w, C)), x, W / out_w, W, mode)
            out = module.resample_y(
                spy.grid((out_h, out_w, C)), tmp, H / out_h, H, mode)
            ctx.meta = (H, W, C, out_h, out_w, mode)
            return out

        @staticmethod
        @torch.amp.custom_bwd(device_type="cuda")
        @torch.autograd.function.once_differentiable
        def backward(ctx, gout):
            H, W, C, out_h, out_w, mode = ctx.meta
            _, _, module, spy = _ctx()
            g = gout.contiguous().detach()
            # forward = Y ∘ X, so adjoint = Xᵀ ∘ Yᵀ
            gtmp = module.resample_y_adj(
                spy.grid((H, out_w, C)), g, H / out_h, H, out_h, mode)
            grad = module.resample_x_adj(
                spy.grid((H, W, C)), gtmp, W / out_w, W, out_w, mode)
            return grad, None, None, None

    return _ResizeFn


_RESIZE_MODES = {"bilinear": 0, "bicubic": 1}


def resize(image: torch.Tensor, width: int | None = None,
           height: int | None = None, scale: float | None = None,
           mode: str = "bilinear") -> torch.Tensor:
    """Differentiable resize (bilinear or bicubic), (H, W) or (H, W, C).
    Give scale=, or width/height — a single dimension preserves aspect
    ratio. Matches torch F.interpolate(align_corners=False,
    antialias=False) exactly, including the a=-0.75 bicubic and edge
    clamping; bicubic overshoot is left unclamped by design."""
    mode_int = _RESIZE_MODES.get(mode)
    if mode_int is None:
        raise ValueError(f"mode must be one of {sorted(_RESIZE_MODES)}")
    for name, value in (("width", width), ("height", height), ("scale", scale)):
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be > 0")
    x, squeeze = _hwc(image)
    H, W, _ = x.shape
    if scale is not None:
        width, height = round(W * scale), round(H * scale)
    elif width is not None and height is None:
        height = round(H * width / W)
    elif height is not None and width is None:
        width = round(W * height / H)
    if width is None or height is None:
        raise ValueError("give scale=, or width= and/or height=")
    width, height = max(1, int(width)), max(1, int(height))
    _ctx()  # fail early with the torch/CUDA message, not inside apply()
    out = _resize_fn().apply(_f32c(x), height, width, mode_int)
    return out.squeeze(-1) if squeeze else out


def _align_grads(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Multi-tensor differentiable calls crash (illegal memory access) when
    only SOME inputs require grad — the generated backward scatters into the
    absent grad buffer (slangpy 0.42). Give no-grad inputs a throwaway
    grad-requiring alias; torch discards the unused leaf gradients."""
    if any(t.requires_grad for t in tensors):
        return tuple(
            t if t.requires_grad else t.detach().requires_grad_(True) for t in tensors
        )
    return tensors


_BLEND_KERNELS = {"multiply": "blend_multiply", "screen": "blend_screen", "overlay": "blend_overlay"}


def blend(base: torch.Tensor, blend_image: torch.Tensor,
          mode: str = "multiply", opacity: float = 1.0) -> torch.Tensor:
    """Differentiable blend modes (multiply / screen / overlay), (H, W, 3)."""
    kernel = _BLEND_KERNELS.get(mode)
    if kernel is None:
        raise ValueError(f"mode must be one of {sorted(_BLEND_KERNELS)}")
    module, _ = _module(base, blend_image)
    base, blend_image = _align_grads(_f32c(base), _f32c(blend_image))
    return getattr(module, kernel)(base, blend_image, float(opacity))


def lut(image: torch.Tensor, lut_tensor: torch.Tensor) -> torch.Tensor:
    """Square 8x8-grid CLUT, differentiable w.r.t. BOTH image and LUT —
    make the LUT a torch.nn.Parameter and it becomes learnable color
    grading. Layout matches slangimage.ops.identity_lut_np()."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) image, got {tuple(image.shape)}")
    if lut_tensor.ndim != 3 or lut_tensor.shape[2] != 3:
        raise ValueError(f"expected an (N, N, 3) LUT, got {tuple(lut_tensor.shape)}")
    validate_lut_size(lut_tensor.shape[1], lut_tensor.shape[0])
    module, spy = _module(image, lut_tensor)
    H, W, _ = image.shape
    image, lut_tensor = _align_grads(_f32c(image), _f32c(lut_tensor))
    return module.lut_apply(spy.grid((H, W)), image, lut_tensor, int(lut_tensor.shape[0]))
