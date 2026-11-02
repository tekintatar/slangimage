# tests/test_doctor.py
# `python -m slangimage` setup report: a working backend must pass its test
# kernel, and failures (including crashes in a backend's child process) must be
# reported, not raised, with exit status 1 when nothing works.

from __future__ import annotations

import json
import os
import sys

import pytest

from slangimage import __main__ as doctor


def _backend_name(device):
    return str(device.info.type).split(".")[-1]


def test_selected_backend_passes_the_test_kernel(ctx):
    device, _ = ctx
    result = doctor.check_backend(_backend_name(device))
    assert result["ok"] and result["created"]
    assert result["adapter"] == device.info.adapter_name


def test_isolated_check_in_a_child_process(ctx):
    device, _ = ctx
    result = doctor.run_check(_backend_name(device))
    assert result["ok"] and result["adapter"] == device.info.adapter_name


def _fake_child(code):
    return [sys.executable, "-c", code]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_isolated_check_reports_a_crash():
    result = doctor.run_check("vulkan", _fake_child(
        "import os, signal; os.kill(os.getpid(), signal.SIGSEGV)"))
    assert not result["ok"] and not result["created"]
    assert "crashed (SIGSEGV)" in result["error"]


def test_isolated_check_reads_the_result_between_native_noise():
    result = doctor.run_check("cuda", _fake_child(
        "print('native noise'); print('{\"ok\": true, \"created\": true, \"adapter\": \"X\"}');"
        "print('rhi: more noise')"))
    assert result == {"ok": True, "created": True, "adapter": "X"}


def test_isolated_check_reports_an_error_exit():
    result = doctor.run_check("metal", _fake_child("import sys; sys.exit('broken install')"))
    assert not result["ok"] and "status 1: broken install" in result["error"]


def _no_gpu(monkeypatch):
    monkeypatch.setattr(doctor, "run_check", lambda backend: {
        "ok": False, "created": False, "error": f"could not create a {backend} device (test)"})
    monkeypatch.setattr(doctor, "_nvidia_driver", lambda: None)
    monkeypatch.setattr(doctor, "_torch_status", lambda: {"installed": False})


def test_report_without_any_gpu(monkeypatch, capsys):
    _no_gpu(monkeypatch)
    assert doctor.main([]) == 1
    text = capsys.readouterr().out
    for backend in doctor.BACKENDS:
        assert f"could not create a {backend} device (test)" in text
    assert "none — no working GPU backend" in text

    assert doctor.main(["--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["auto_backend"] is None
    assert not any(result["ok"] for result in report["backends"].values())
    assert set(report["environment"].values()) <= {"set", "not set"}


def test_native_stdout_noise_goes_to_stderr(monkeypatch, capfd):
    # Without an NVIDIA driver, SlangPy's native code writes to fd 1 directly.
    _no_gpu(monkeypatch)

    def noisy_torch_status():
        os.write(1, b"native noise\n")
        return {"installed": False}

    monkeypatch.setattr(doctor, "_torch_status", noisy_torch_status)
    assert doctor.main(["--json"]) == 1
    out, err = capfd.readouterr()
    assert json.loads(out)["auto_backend"] is None
    assert "native noise" in err and "native noise" not in out


def test_software_renderer_is_flagged(monkeypatch, capsys):
    _no_gpu(monkeypatch)
    monkeypatch.setattr(doctor, "run_check", lambda backend: (
        {"ok": True, "created": True, "adapter": "llvmpipe (LLVM 20.1.2, 256 bits)",
         "software": True} if backend == "vulkan" else
        {"ok": False, "created": False, "error": "unavailable (test)"}))
    doctor.main([])
    assert "software renderer on the CPU" in capsys.readouterr().out
    assert doctor._SOFTWARE_ADAPTER.search("SwiftShader Device (Subzero)")
    assert not doctor._SOFTWARE_ADAPTER.search("NVIDIA RTX PRO 4500 Blackwell")


@pytest.mark.parametrize("darwin,results,expected", [
    (True, {"metal": (True, True), "vulkan": (True, True)}, "metal"),
    (True, {"metal": (False, False), "vulkan": (True, True)}, "vulkan"),
    # AUTO keeps a Metal/Vulkan device even if its kernel fails...
    (True, {"metal": (True, False), "vulkan": (True, True)}, "metal"),
    # ...but a CUDA device must pass the test kernel (the library's probe).
    (False, {"cuda": (True, False), "vulkan": (True, True)}, "vulkan"),
    (False, {"cuda": (True, True), "vulkan": (True, True)}, "cuda"),
    (False, {"metal": (True, True)}, None),
])
def test_auto_choice_matches_the_library(monkeypatch, darwin, results, expected):
    monkeypatch.setattr(doctor.platform, "system", lambda: "Darwin" if darwin else "Linux")
    full = {b: {"created": False, "ok": False} for b in doctor.BACKENDS}
    for backend, (created, ok) in results.items():
        full[backend] = {"created": created, "ok": ok}
    assert doctor._auto_choice(full) == expected
