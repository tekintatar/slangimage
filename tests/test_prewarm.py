"""Prewarm dispatch coverage on the selected texture backend, not a math oracle."""

from functools import wraps

import pytest

import slangimage as si


# Everything else in ops.__all__ must actually execute during prewarm.
_NON_OPERATORS = {"prewarm", "BLEND_MODES", "identity_lut_np"}


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
def test_prewarm_covers_all_ops(ctx, monkeypatch, precision):
    device, _ = ctx
    expected = set(si.ops.__all__) - _NON_OPERATORS
    completed = set()

    def trace(name, operation):
        @wraps(operation)
        def run(*args, **kwargs):
            result = operation(*args, **kwargs)
            assert isinstance(result, si.Image), name
            assert result.device is device, f"{name} escaped the selected device"
            assert result.precision == precision, f"{name} used the wrong storage precision"
            completed.add(name)
            return result
        return run

    for name in expected:
        monkeypatch.setattr(si.ops, name, trace(name, getattr(si.ops, name)))

    # Never let an omitted device argument silently turn this into AUTO coverage.
    def unexpected_default_device(*args, **kwargs):
        pytest.fail("explicit-device prewarm requested a default device")

    monkeypatch.setattr(si.ops, "get_default_device", unexpected_default_device)
    si.prewarm(device=device, precision=precision, size=48)
    # Includes nested public calls (affine through rotate).
    # A newly exported op or removed prewarm path fails without a count update.
    assert completed == expected, f"prewarm missing operators: {sorted(expected - completed)}"


@pytest.mark.parametrize("kwargs, message", [
    ({"size": size}, "size must be >= 20") for size in (1, 8, 16, 19)
] + [({"precision": "float16"}, "precision must be 'fp16' or 'fp32'")])
def test_prewarm_rejects_invalid_arguments_before_device_init(monkeypatch, kwargs, message):
    def unexpected_default_device(*args, **kwargs):
        pytest.fail("invalid prewarm arguments initialized a device")

    monkeypatch.setattr(si.ops, "get_default_device", unexpected_default_device)
    with pytest.raises(ValueError, match=message):
        si.prewarm(**kwargs)
