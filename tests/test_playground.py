"""The packaged playground renders from Gradio-style worker threads.

Gradio calls event handlers on its own worker threads, while SlangImage devices
are thread-affine. The check runs in a subprocess: importing the playground
creates the process-wide default device on the playground's GPU thread, which
must not become this test process's default.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.auto_device

SCRIPT = textwrap.dedent(
    """
    import importlib.util
    import sys
    import threading

    try:
        import slangimage.playground as app          # installed wheel
    except ModuleNotFoundError:                      # source checkout
        spec = importlib.util.spec_from_file_location("playground_app", sys.argv[1])
        app = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(app)

    image = app._sample_image()
    results = []

    def handler(fn, *args):
        # a fresh thread per call, like Gradio's worker pool
        thread = threading.Thread(target=lambda: results.append(fn(*args)))
        thread.start()
        thread.join()

    handler(app.preview, image, "fp16", [], "gaussian_blur", 9, 3.0, 0, None)
    handler(app.add_to_stack, image, "fp32", [], "vibrance", 0.6, 0, 0, None)
    handler(app.clear_stack, image, "fp16")
    assert len(results) == 3, results

    out, label = results[0]
    assert out.shape == (512, 512, 3) and out.dtype.name == "uint8", (out.shape, out.dtype)
    assert "gaussian_blur" in label, label
    stack, _, arr, _ = results[1]
    assert stack == [("vibrance", {"amount": 0.6})], stack
    assert arr.shape == (512, 512, 3), arr.shape
    assert results[2][0] == [], results[2]
    print("ok")
    """
)


def test_playground_callbacks_run_from_worker_threads():
    pytest.importorskip("gradio")
    app = Path(__file__).resolve().parents[1] / "playground" / "app.py"
    proc = subprocess.run(
        [sys.executable, "-c", SCRIPT, str(app)],
        capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-4000:]
    assert proc.stdout.strip().endswith("ok")
