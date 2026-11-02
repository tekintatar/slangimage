# tests/test_doctor.py
# `python -m slangimage` setup report: a working backend must pass its test
# kernel, and failures must be reported (not raised) with exit status 1.

from __future__ import annotations

import json

import pytest

import slangimage as si
from slangimage import __main__ as doctor


def test_selected_backend_passes_the_test_kernel(ctx):
    device, _ = ctx
    backend = str(device.info.type).split(".")[-1]
    result = doctor.check_backend(backend)
    assert result == {"ok": True, "adapter": device.info.adapter_name}


def test_report_without_any_gpu(monkeypatch, capsys):
    def no_device(backend):
        raise RuntimeError(f"could not create a {backend} device (test)")

    monkeypatch.setattr(si, "DeviceManager", no_device)
    monkeypatch.setattr(doctor, "_nvidia_driver", lambda: None)
    monkeypatch.setattr(doctor, "_torch_status", lambda: {"installed": False})

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


@pytest.mark.parametrize("darwin,working,expected", [
    (True, {"metal", "vulkan"}, "metal"),
    (True, {"vulkan"}, "vulkan"),
    (False, {"cuda", "vulkan"}, "cuda"),
    (False, {"metal"}, None),
])
def test_auto_choice_follows_platform_order(monkeypatch, darwin, working, expected):
    monkeypatch.setattr(doctor.platform, "system", lambda: "Darwin" if darwin else "Linux")
    results = {b: {"ok": b in working} for b in doctor.BACKENDS}
    assert doctor._auto_choice(results) == expected
