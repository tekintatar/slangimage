"""CPU-only checks for honest grouping of pytest outcomes."""

from types import SimpleNamespace

import pytest

from conftest import pytest_terminal_summary


def render(stats, *, collectonly=False):
    lines = []
    reporter = SimpleNamespace(stats=stats, section=lines.append, write_line=lines.append)
    config = SimpleNamespace(option=SimpleNamespace(collectonly=collectonly))
    pytest_terminal_summary(reporter, 0, config)
    return "\n".join(lines)


def report(path, when="call", *, auto_device=False):
    return SimpleNamespace(
        nodeid=path, when=when, keywords={"auto_device": True} if auto_device else {},
    )


def test_separates_cuda_from_texture_and_host(monkeypatch):
    monkeypatch.setenv("SLANGIMAGE_BACKEND", "cuda")
    output = render({"passed": [
        report("tests/test_geometry.py::test_tile[nearest]"),
        report("tests/test_artifact_checks.py::test_valid_archives"),
        report("tests/test_diff.py::test_gradient"),
        report("tests/test_diff_core_contracts.py::test_vjp"),
        report("tests/test_nn.py::test_module"),
    ]})
    assert "Requested texture backend: cuda; ctx device: not initialized" in output
    assert "Selected-backend texture + host tests: 2 passed" in output
    assert "AUTO-device API/lifecycle tests: no test results" in output
    assert "CUDA autograd tests: 3 passed" in output


def test_module_skips_are_not_skipped_test_cases():
    output = render({"skipped": [
        report("tests/test_diff.py", "collect"),
        report("tests/test_diff_core_contracts.py", "collect"),
        report("tests/test_nn.py", "collect"),
    ]})
    assert "CUDA autograd tests: 3 skipped collection nodes (not test cases)" in output
    assert "Selected-backend texture + host tests: no test results" in output


@pytest.mark.parametrize("outcome", ["failed", "error", "skipped", "xfailed", "xpassed"])
def test_nonpassing_test_outcomes_remain_visible(outcome):
    output = render({outcome: [report("tests/test_diff.py::test_gradient")]})
    assert f"CUDA autograd tests: 1 {outcome}" in output


def test_collect_only_is_not_execution_evidence():
    assert render({}, collectonly=True) == ""


@pytest.mark.parametrize("outcome", ["passed", "failed", "error", "skipped", "xfailed", "xpassed"])
def test_auto_device_results_are_separate(outcome):
    output = render({outcome: [
        report("tests/test_public_api.py::test_pipe_chaining", auto_device=True),
        report("tests/test_regressions.py::test_release_device_clears_manager_and_pool_state",
               auto_device=True),
        report("tests/test_prewarm.py::test_prewarm_covers_all_ops[fp32]"),
    ]})
    assert f"AUTO-device API/lifecycle tests: 2 {outcome}" in output
    assert f"Selected-backend texture + host tests: 1 {outcome}" in output
