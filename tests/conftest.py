# tests/conftest.py
# Shared texture-test device + module cache, selected by SLANGIMAGE_BACKEND.
# No-plumbing API/lifecycle tests also exercise AUTO or explicit extra devices.
# The torch suites use their own CUDA device, never the texture fixture.

from __future__ import annotations

import os
from collections import Counter

import pytest
import slangpy as spy

import slangimage as si
from slangimage.runtime.modules import get_module_cache


CUDA_AUTOGRAD_TEST_FILES = {
    "test_diff.py",
    "test_diff_core_contracts.py",
    "test_nn.py",
}
CUDA_AUGMENT_TEST_FILES = {"test_augment.py"}


def pytest_collection_modifyitems(config, items):
    """Mark every test that needs a GPU, so `pytest -m "not gpu"` runs the
    GPU-free remainder (lint-level CI, machines without a supported GPU)."""
    for item in items:
        if ("ctx" in item.fixturenames or item.get_closest_marker("auto_device")
                or item.path.name in CUDA_AUTOGRAD_TEST_FILES | CUDA_AUGMENT_TEST_FILES):
            item.add_marker(pytest.mark.gpu)


@pytest.fixture(scope="session")
def ctx(request):
    backend = os.environ.get("SLANGIMAGE_BACKEND", "auto")
    device = si.get_default_device(backend)
    if backend != "auto":
        assert device.info.type == getattr(spy.DeviceType, backend), (
            f"Requested {backend}, got {device.info.type}"
        )
    request.config._slangimage_texture_device = (
        f"{device.info.type} ({device.info.adapter_name})"
    )
    print(f"\n[slangimage] device: {device.info.type} ({device.info.adapter_name})")
    return device, get_module_cache(device)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Separate CUDA test evidence from texture/host checks, including skips."""
    if config.option.collectonly:
        return
    groups = {
        "Selected-backend texture + host tests": Counter(),
        "AUTO-device API/lifecycle tests": Counter(),
        "CUDA autograd tests": Counter(),
        "CUDA augment tests": Counter(),
    }
    for outcome in ("passed", "failed", "error", "skipped", "xfailed", "xpassed"):
        for report in terminalreporter.stats.get(outcome, ()):
            path = report.nodeid.split("::", 1)[0].replace("\\", "/")
            if path.rsplit("/", 1)[-1] in CUDA_AUTOGRAD_TEST_FILES:
                group = "CUDA autograd tests"
            elif path.rsplit("/", 1)[-1] in CUDA_AUGMENT_TEST_FILES:
                group = "CUDA augment tests"
            elif "auto_device" in getattr(report, "keywords", {}):
                group = "AUTO-device API/lifecycle tests"
            else:
                group = "Selected-backend texture + host tests"
            label = outcome
            if getattr(report, "when", None) == "collect":
                label = f"{outcome} collection nodes (not test cases)"
            groups[group][label] += 1

    terminalreporter.section("coverage groups (not operator/precision coverage)")
    selected = os.environ.get("SLANGIMAGE_BACKEND", "auto")
    actual = getattr(config, "_slangimage_texture_device", "not initialized")
    terminalreporter.write_line(f"Requested texture backend: {selected}; ctx device: {actual}")
    for group, counts in groups.items():
        summary = ", ".join(f"{count} {label}" for label, count in counts.items())
        terminalreporter.write_line(f"{group}: {summary or 'no test results'}")
    terminalreporter.write_line(
        "Selected-backend texture + host includes CPU checks; AUTO-device tests are separate. "
        "CUDA autograd and augment skips are not coverage. See docs/TEST_COVERAGE.md."
    )
