"""Host contracts for the reusable cross-backend capture/compare harness."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

import slangimage as si


SCRIPT = Path(__file__).parents[1]/"scripts"/"cross_backend_numeric.py"
SPEC = importlib.util.spec_from_file_location("cross_backend_numeric", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
COVERED_EXPORTS = MODULE.COVERED_EXPORTS
NON_OPERATORS = MODULE.NON_OPERATORS
comparison_bound = MODULE.comparison_bound


def test_cross_backend_manifest_covers_every_texture_export():
    assert COVERED_EXPORTS == set(si.ops.__all__)-NON_OPERATORS


def test_cross_backend_policy_is_exact_only_for_discrete_remaps():
    assert comparison_bound("fp16__flip") == (0, 0)
    assert comparison_bound("fp32__crop") == (0, 0)
    assert comparison_bound("fp16__gamma")[0] > 0
    assert comparison_bound("fp32__kuwahara")[0] > comparison_bound("fp32__gamma")[0]


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
def test_cross_backend_bounds_reject_material_changes(precision):
    atol, rtol = comparison_bound(f"{precision}__gamma")
    base = np.array([0, .25, .5, 1], dtype=np.float64)
    harmless = base+atol*.5
    material = base+max(.05, atol*10)
    assert np.all(np.abs(harmless-base) <= atol+rtol*np.maximum(abs(harmless), abs(base)))
    assert not np.all(np.abs(material-base) <= atol+rtol*np.maximum(abs(material), abs(base)))
