# playground/app.py
# Interactive SlangImage playground: upload a photo, explore every operator
# with live parameter sliders, and stack ops into pipelines.
#
#   pip install ".[playground]"
#   slangimage-playground
#
# Runs on whatever backend the machine has (Metal/CUDA/Vulkan). The same app
# works on a Hugging Face Space with a CUDA GPU.

from __future__ import annotations

import concurrent.futures
import time

import gradio as gr
import numpy as np

import slangimage as si

MAX_SIDE = 2048  # keep interaction snappy on big uploads

# ---------------------------------------------------------------- op registry
# name -> (fn, [(param, min, max, step, default)], choices: {param: [..]})
OPS: dict[str, dict] = {
    "gaussian_blur": dict(
        fn=lambda img, radius, sigma: si.ops.gaussian_blur(img, radius=int(radius), sigma=sigma),
        params=[("radius", 1, 40, 1, 9), ("sigma", 0.1, 15.0, 0.1, 3.0)],
    ),
    "surface_blur": dict(
        fn=lambda img, radius, bsigma: si.ops.surface_blur(img, radius=int(radius), bsigma=bsigma),
        params=[("radius", 1, 25, 1, 9), ("bsigma", 0.01, 0.5, 0.01, 0.1)],
    ),
    "bilateral": dict(
        fn=lambda img, radius, sigma_spatial, sigma_range: si.ops.bilateral(
            img, radius=int(radius), sigma_spatial=sigma_spatial, sigma_range=sigma_range
        ),
        params=[("radius", 1, 15, 1, 5), ("sigma_spatial", 0.0, 8.0, 0.1, 2.5),
                ("sigma_range", 0.01, 0.6, 0.01, 0.15)],
    ),
    "nl_means": dict(
        fn=lambda img, search_radius, h: si.ops.nl_means(img, search_radius=int(search_radius), h=h),
        params=[("search_radius", 1, 8, 1, 4), ("h", 0.01, 0.4, 0.01, 0.12)],
    ),
    "unsharp_mask": dict(
        fn=lambda img, radius, sigma, amount: si.ops.unsharp_mask(img, radius=radius, sigma=sigma, amount=amount),
        params=[("radius", 1, 30, 1, 6), ("sigma", 0.5, 10.0, 0.1, 2.5), ("amount", 0.0, 4.0, 0.05, 1.2)],
    ),
    "sharpen_luma": dict(
        fn=lambda img, radius, sharpness: si.ops.sharpen_luma(img, radius=radius, sharpness=sharpness),
        params=[("radius", 1, 10, 1, 4), ("sharpness", 0.0, 3.0, 0.05, 1.0)],
    ),
    "vibrance": dict(
        fn=lambda img, amount: si.ops.vibrance(img, amount=amount),
        params=[("amount", -1.0, 2.0, 0.05, 0.6)],
    ),
    "exposure": dict(
        fn=lambda img, ev: si.ops.exposure(img, ev=ev),
        params=[("ev", -3.0, 3.0, 0.05, 0.0)],
    ),
    "gamma": dict(
        fn=lambda img, power: si.ops.gamma(img, power=power),
        params=[("power", 0.2, 3.0, 0.02, 1.0)],
    ),
    "sepia": dict(
        fn=lambda img, intensity: si.ops.sepia(img, intensity=intensity),
        params=[("intensity", 0.0, 1.0, 0.02, 0.8)],
    ),
    "hue_rotate": dict(
        fn=lambda img, angle: si.ops.hue_rotate(img, angle=angle),
        params=[("angle", -3.14, 3.14, 0.02, 0.5)],
    ),
    "highlight_shadow": dict(
        fn=lambda img, shadow_amount, highlight_amount: si.ops.highlight_shadow(
            img, shadow_amount=shadow_amount, highlight_amount=highlight_amount
        ),
        params=[("shadow_amount", -1.0, 1.0, 0.02, 0.3), ("highlight_amount", 0.0, 1.0, 0.02, 0.9)],
    ),
    "clahe": dict(
        fn=lambda img, clip_limit, tiles, strength: si.ops.clahe(
            img, clip_limit=clip_limit, tiles=(int(tiles), int(tiles)), strength=strength
        ),
        params=[("clip_limit", 1.0, 10.0, 0.1, 3.0), ("tiles", 2, 16, 1, 8), ("strength", 0.0, 1.0, 0.02, 1.0)],
    ),
    "equalize": dict(
        fn=lambda img: si.ops.equalize(img),
        params=[],
    ),
    "autocontrast": dict(
        fn=lambda img: si.ops.autocontrast(img),
        params=[],
    ),
    "rotate": dict(
        fn=lambda img, angle: si.ops.rotate(img, angle=angle),
        params=[("angle", -180, 180, 1, 15)],
    ),
    "flip": dict(
        fn=lambda img, axis: si.ops.flip(
            img, horizontal=axis in ("horizontal", "both"),
            vertical=axis in ("vertical", "both")),
        params=[],
        choices={"axis": ["horizontal", "vertical", "both"]},
    ),
    "white_balance": dict(
        fn=lambda img, temperature, tint: si.ops.white_balance(
            img, temperature=temperature, tint=tint),
        params=[("temperature", 2000, 25000, 50, 6500), ("tint", -150, 150, 1, 0)],
    ),
    "solarize": dict(
        fn=lambda img, threshold: si.ops.solarize(img, threshold=threshold),
        params=[("threshold", 0.0, 1.0, 0.01, 0.5)],
    ),
    "posterize": dict(
        fn=lambda img, bits: si.ops.posterize(img, bits=int(bits)),
        params=[("bits", 1, 8, 1, 4)],
    ),
    "grayscale": dict(
        fn=lambda img: si.ops.grayscale(img),
        params=[],
    ),
    "kuwahara": dict(
        fn=lambda img, radius: si.ops.kuwahara(img, radius=int(radius)),
        params=[("radius", 2, 12, 1, 6)],
    ),
    "pointillize": dict(
        fn=lambda img, radius, jitter, seed: si.ops.pointillize(img, radius=radius, jitter=jitter, seed=seed),
        params=[("radius", 3.0, 40.0, 0.5, 12.0), ("jitter", 0.0, 1.0, 0.05, 1.0), ("seed", 0, 10, 1, 0)],
    ),
    "cmyk_halftone": dict(
        fn=lambda img, width, angle_deg: si.ops.cmyk_halftone(img, width=width, angle_deg=angle_deg),
        params=[("width", 2.0, 24.0, 0.5, 7.0), ("angle_deg", 0.0, 90.0, 1.0, 0.0)],
    ),
    "smooth_threshold": dict(
        fn=lambda img, threshold, smooth_amount: si.ops.smooth_threshold(
            img, threshold=threshold, smooth_amount=smooth_amount
        ),
        params=[("threshold", 0.0, 1.0, 0.01, 0.5), ("smooth_amount", 0.0, 0.5, 0.01, 0.15)],
    ),
    "invert": dict(fn=lambda img: si.ops.invert(img), params=[]),
    "extract_channel": dict(
        fn=lambda img, channel: si.ops.extract_channel(img, channel=channel),
        params=[],
        choices={"channel": ["r", "g", "b"]},
    ),
    "false_color (duotone)": dict(
        fn=lambda img: si.ops.false_color(img, color0=(0.05, 0.0, 0.2), color1=(1.0, 0.95, 0.8)),
        params=[],
    ),
    "blend with inverted self": dict(
        fn=lambda img, mode, opacity: si.ops.blend(img, si.ops.invert(img), mode=mode, opacity=opacity),
        params=[("opacity", 0.0, 1.0, 0.02, 0.6)],
        choices={"mode": sorted(si.ops.BLEND_MODES)},
    ),
}

MAX_SLIDERS = 3

# SlangImage devices are thread-affine: a device and everything derived from it
# stay on the thread that created it. Gradio runs callbacks on a pool of worker
# threads, so every GPU call goes through this one dedicated thread, which also
# creates the process-wide default device.
_gpu = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="slangimage-gpu")


def _on_gpu(fn, *args):
    """Run ``fn(*args)`` on the playground's GPU thread and return the result."""
    return _gpu.submit(fn, *args).result()


def _device_label() -> str:
    dev = si.device()
    return f"{str(dev.info.type).split('.')[-1]} · {dev.info.adapter_name}"


DEVICE_LABEL = _on_gpu(_device_label)


# ------------------------------------------------------------------ helpers
def _to_image(arr_u8: np.ndarray, precision: str) -> si.Image:
    arr = (arr_u8.astype(np.float32) / 255.0)
    if precision == "fp16":
        arr = arr.astype(np.float16)
    img = si.from_numpy(arr)
    if max(img.width, img.height) > MAX_SIDE:
        img = si.ops.resize(img, scale=MAX_SIDE / max(img.width, img.height))
    return img


def _to_u8(img: si.Image) -> np.ndarray:
    return (img.to_numpy(clip=True)[..., :3] * 255.0).round().astype(np.uint8)


def _run_stack(base_u8: np.ndarray, precision: str, stack: list) -> tuple[si.Image, float]:
    img = _to_image(base_u8, precision)
    t0 = time.perf_counter()
    for op_name, kwargs in stack:
        img = OPS[op_name]["fn"](img, **kwargs)
    img.device.wait_for_idle()
    return img, (time.perf_counter() - t0) * 1000.0


def _render_stack(base_u8: np.ndarray, precision: str, stack: list) -> tuple[np.ndarray, float]:
    """GPU thread: run the pipeline and read the result back as uint8 RGB."""
    img, ms = _run_stack(base_u8, precision, stack)
    return _to_u8(img), ms


def _render_preview(base_u8, precision, stack, op_name, kwargs) -> tuple[np.ndarray, str]:
    """GPU thread: the pipeline plus the current operator, with timings."""
    img, stack_ms = _run_stack(base_u8, precision, stack)
    t0 = time.perf_counter()
    out = OPS[op_name]["fn"](img, **kwargs)
    out.device.wait_for_idle()
    op_ms = (time.perf_counter() - t0) * 1000.0
    label = f"{out.width}×{out.height} {precision} · stack {stack_ms:.1f} ms · {op_name} {op_ms:.1f} ms · {DEVICE_LABEL}"
    return _to_u8(out), label


def _collect_kwargs(op_name: str, slider_vals: list, choice_val: str) -> dict:
    spec = OPS[op_name]
    kwargs = {p[0]: v for p, v in zip(spec["params"], slider_vals)}
    if spec.get("choices"):
        kwargs[next(iter(spec["choices"]))] = choice_val
    return kwargs


# ------------------------------------------------------------------ callbacks
def on_op_change(op_name: str):
    spec = OPS[op_name]
    slider_updates = []
    for i in range(MAX_SLIDERS):
        if i < len(spec["params"]):
            name, lo, hi, step, default = spec["params"][i]
            slider_updates.append(gr.update(visible=True, label=name, minimum=lo,
                                            maximum=hi, step=step, value=default))
        else:
            slider_updates.append(gr.update(visible=False))
    if spec.get("choices"):
        cname, cvals = next(iter(spec["choices"].items()))
        choice_update = gr.update(visible=True, label=cname, choices=cvals, value=cvals[0])
    else:
        choice_update = gr.update(visible=False)
    return *slider_updates, choice_update


def preview(image_u8, precision, stack, op_name, s1, s2, s3, choice):
    if image_u8 is None:
        return None, "upload an image first"
    kwargs = _collect_kwargs(op_name, [s1, s2, s3], choice)
    return _on_gpu(_render_preview, image_u8, precision, stack, op_name, kwargs)


def add_to_stack(image_u8, precision, stack, op_name, s1, s2, s3, choice):
    if image_u8 is None:
        return stack, "upload an image first", None, ""
    kwargs = _collect_kwargs(op_name, [s1, s2, s3], choice)
    stack = stack + [(op_name, kwargs)]
    out, ms = _on_gpu(_render_stack, image_u8, precision, stack)
    return stack, _stack_text(stack), out, f"pipeline: {ms:.1f} ms on {DEVICE_LABEL}"


def undo_stack(image_u8, precision, stack):
    stack = stack[:-1]
    if image_u8 is None:
        return stack, _stack_text(stack), None, ""
    out, ms = _on_gpu(_render_stack, image_u8, precision, stack)
    return stack, _stack_text(stack), out, f"pipeline: {ms:.1f} ms on {DEVICE_LABEL}"


def clear_stack(image_u8, precision):
    if image_u8 is None:
        return [], "(empty)", None, ""
    out, _ = _on_gpu(_render_stack, image_u8, precision, [])
    return [], "(empty)", out, ""


def _stack_text(stack: list) -> str:
    if not stack:
        return "(empty)"
    lines = []
    for i, (name, kwargs) in enumerate(stack, 1):
        args = ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in kwargs.items())
        lines.append(f"{i}. {name}({args})")
    return "\n".join(lines)


# ------------------------------------------------------------------------ UI
first_op = next(iter(OPS))


def _sample_image() -> np.ndarray:
    """Generate the bundled default in memory so installed wheels need no asset."""
    y, x = np.mgrid[0:512, 0:512].astype(np.float32) / 511.0
    r = 0.5 + 0.45 * np.sin(8.0 * x) * np.cos(5.0 * y)
    g = np.clip(0.15 + 0.85 * y, 0.0, 1.0)
    b = np.clip(0.2 + 0.8 * x, 0.0, 1.0)
    rgb = np.stack((r, g, b), axis=-1)
    return (np.clip(rgb, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def _initial_sliders():
    spec = OPS[first_op]
    out = []
    for i in range(MAX_SLIDERS):
        if i < len(spec["params"]):
            name, lo, hi, step, default = spec["params"][i]
            out.append(gr.Slider(lo, hi, value=default, step=step, label=name, visible=True))
        else:
            out.append(gr.Slider(0, 1, value=0, visible=False))
    return out


with gr.Blocks(title="SlangImage playground") as demo:
    gr.Markdown(f"# SlangImage playground\nrunning on **{DEVICE_LABEL}** — every op is a live GPU kernel")
    stack_state = gr.State([])

    with gr.Row():
        with gr.Column(scale=1):
            # format="png": gradio re-encodes displayed images as lossy webp
            # by default, which adds block artifacts on smooth gradients
            image_in = gr.Image(label="input (sample preloaded — drop your own)",
                                type="numpy", height=320, value=_sample_image(),
                                format="png")
            precision = gr.Radio(["fp16", "fp32"], value="fp16", label="precision")
            op = gr.Dropdown(sorted(OPS), value=first_op, label="operator")
            sliders = _initial_sliders()
            choice = gr.Dropdown(visible=False)
            with gr.Row():
                add_btn = gr.Button("add to pipeline", variant="primary")
                undo_btn = gr.Button("undo")
                clear_btn = gr.Button("clear")
            stack_view = gr.Textbox(label="pipeline", value="(empty)", lines=6, interactive=False)
        with gr.Column(scale=2):
            image_out = gr.Image(label="result (pipeline + current op preview)",
                                 type="numpy", format="png")
            status = gr.Markdown("")

    live_inputs = [image_in, precision, stack_state, op, *sliders, choice]
    op.change(on_op_change, inputs=[op], outputs=[*sliders, choice]).then(
        preview, inputs=live_inputs, outputs=[image_out, status]
    )
    for comp in (*sliders, choice, image_in, precision):
        comp.change(preview, inputs=live_inputs, outputs=[image_out, status])
    add_btn.click(add_to_stack, inputs=live_inputs[:len(live_inputs)],
                  outputs=[stack_state, stack_view, image_out, status])
    undo_btn.click(undo_stack, inputs=[image_in, precision, stack_state],
                   outputs=[stack_state, stack_view, image_out, status])
    clear_btn.click(clear_stack, inputs=[image_in, precision],
                    outputs=[stack_state, stack_view, image_out, status])
    demo.load(preview, inputs=live_inputs, outputs=[image_out, status])

def main() -> None:
    """Launch the packaged interactive playground."""
    _on_gpu(si.prewarm)  # compile the default fp16 kernels before the first preview
    demo.launch()


if __name__ == "__main__":
    main()
