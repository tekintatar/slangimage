# Contributing to SlangImage

Thanks for your interest! SlangImage is pre-alpha and moving quickly, so please
open an issue to discuss anything non-trivial before sending a large PR.

## Development setup

You need a working GPU backend — **Metal** (macOS), **CUDA**, or **Vulkan** —
and Python 3.10–3.13. There is no CPU fallback for the kernels; the test suite
requires a GPU. Name a supported interpreter explicitly when creating the
environment (a bare `python3` may be Apple's 3.9 or Homebrew's 3.14); on
Debian/Ubuntu install `python3-venv` first. Platform requirements and known
issues are in the README and [docs/PLATFORM_NOTES.md](docs/PLATFORM_NOTES.md).

```bash
python3.12 -m venv .venv && . .venv/bin/activate   # or 3.10, 3.11, 3.13
pip install -e ".[dev]"      # texture ops on Metal, CUDA or Vulkan
python -m slangimage          # setup report: which GPU backends work, and why not
pytest                        # runs on the auto-selected backend
```

If a backend fails or the tests cannot find a GPU, start with
`python -m slangimage`: it runs a test kernel on each backend and reports the
reason for every failure (paste its output into bug reports).

For the differentiable ops (Linux + NVIDIA), build SlangPy's torch bridge
before the editable install; the prerequisites are in the README's *PyTorch
extra* section and [docs/INSTALL_TORCH.md](docs/INSTALL_TORCH.md). Passing
`--no-build-isolation` to the editable install itself fails with
`No module named 'hatchling'`.

```bash
pip install "torch>=2.13,<2.14"
pip install --no-build-isolation --no-cache-dir "slangpy-torch==0.7.0"
pip install -e ".[dev,torch]"
```

On Python 3.10, `tests/test_artifact_checks.py` is skipped because the artifact
checker needs `tomllib` (Python 3.11+); use 3.11 or later for release checks.

Select a backend for the test suite explicitly with the `SLANGIMAGE_BACKEND`
env var (the library itself ignores it; application code passes a device, see
the README's *Choosing a backend*):

```bash
SLANGIMAGE_BACKEND=metal  pytest          # macOS
SLANGIMAGE_BACKEND=vulkan pytest          # Linux/Windows (Vulkan driver)
SLANGIMAGE_BACKEND=cuda   pytest -ra      # NVIDIA + NVRTC; also runs CUDA forward/gradient tests if available
```

The differentiable tests (`test_diff.py`, `test_diff_core_contracts.py`, and
`test_nn.py`) require `auto`/`cuda` selection and available `torch` + CUDA.
Explicit Metal/Vulkan selection skips all three modules before probing CUDA,
even on a CUDA-capable machine.

The summary separates **Selected-backend texture + host tests**, intentional
**AUTO-device API/lifecycle tests**, and **CUDA autograd tests**. The first group
includes CPU-only checks, so its total is not a count of numerical tests.
Use `-m 'not auto_device'` to exclude intentional automatic-device checks, or
`-m auto_device` to select them separately. Three skipped CUDA collection nodes
mean three entire modules were excluded, not three test cases.
A CUDA release check with skipped autograd modules has not validated gradients.
Use `-ra` to inspect skip reasons and record the actual device/dependency stack.
See the [operator evidence map and reporting rules](docs/TEST_COVERAGE.md).

## Checking a change

One command runs everything a change needs locally — whitespace, ruff, mypy
(the package and the typing contract in `typing_tests/`) and the test suite on
your GPU:

```bash
scripts/check.sh                       # SLANGIMAGE_BACKEND=metal|cuda|vulkan to pick a backend
scripts/check.sh --no-gpu              # only the tests that need no GPU
scripts/check.sh -k bilateral          # extra arguments go to pytest
```

Tests that need a GPU carry the `gpu` marker (applied automatically to tests
that use the `ctx` fixture, AUTO-device tests and the CUDA autograd modules);
mark any other test that creates a device yourself.

GitHub Actions runs `scripts/check.sh --no-gpu` on Python 3.10 and 3.13,
shellcheck, and the reproducible wheel/sdist build with its archive checks on
every pull request. Hosted runners have no supported GPU, so run the GPU tests
on the backend(s) you have before opening a PR, and state in the PR which
backend, device and dependency versions you tested. Maintainers additionally run artifact validation,
fresh-install checks and the other GPU suites: Metal on Apple Silicon, and
CUDA/Vulkan on a Linux NVIDIA host set up with the helper scripts in
`scripts/cuda_box/`. Record the actual device and dependency stack before
claiming a check passed.

## Building and checking artifacts

The wheel and sdist are reproducible for a fixed source date and the build
backend pinned in `build-constraints.txt`. Use Python 3.11 or newer for these
checks (`scripts/check_artifacts.py` needs `tomllib`). From a clean checkout, in
an environment with the `dev` extra:

```bash
pip install build==1.4.0 twine==7.0.0
SOURCE_DATE_EPOCH=$(git log -1 --format=%ct) \
  PIP_CONSTRAINT=build-constraints.txt python -m build
python scripts/check_artifacts.py dist      # the dist directory, not dist/*
twine check dist/*
```

Then install the wheel into a fresh environment and, from a directory outside
the checkout, run
`python <checkout>/scripts/validate_installed_artifact.py --source-root <checkout> --gpu-smoke --backend metal`
(or `cuda`/`vulkan`). `twine check` does not render Markdown, so keep README
links absolute (`https://github.com/tekintatar/slangimage/blob/main/...`):
relative links break on PyPI.

## Adding an operator

An op is five small pieces:

1. **Shader** — `src/slangimage/shaders/ops/<name>.slang`. Every `RWTexture2D`
   parameter must carry `[format(SLANGIMAGE_FMT)]` (CUDA does no format
   conversion on surface writes). Do manual loads + clamp — no hardware
   samplers. Dispatch grids are row-major `spy.grid((h, w))`. Math runs in fp32
   registers regardless of storage.
2. **Kernel class** — `src/slangimage/kernels/<name>.py`, one class per shader,
   allocating outputs via `alloc_output` (never `device.create_texture`).
3. **Functional wrapper** — a function in `src/slangimage/ops.py` with a
   docstring.
4. **Prewarm** — add a case to the `prewarm()` registry so the op is
   compile-checked at startup.
5. **Tests** — a property test (a numpy reference or a mathematical invariant,
   not just "no NaN") in `tests/`.

Update [docs/TEST_COVERAGE.md](docs/TEST_COVERAGE.md) with the operator's exact
test names, storage precisions, and direct/indirect prewarm path. Smoke tests
and same-backend fp16/fp32 comparisons are not independent numerical references.
`tests/test_prewarm.py` executes every exported operator at fp16/fp32 on the
selected backend and verifies the output device/precision. An export without
an executed prewarm path fails this test. Mark only deliberately automatic-device
tests with `@pytest.mark.auto_device`; use `ctx` for selected-backend tests.

Downstream packages can ship their own private `.slang` kernels without forking:
register a directory with `ModuleCache.add_ops_dir()` (see
`examples/05_custom_kernels.py`).

## Conventions

- Images are RGBA on the GPU; 3-channel input auto-expands to opaque RGBA.
- `to_numpy()` readback is **lossless** — never clip there (sharpening
  overshoot and unclamped outputs such as `normalize` depend on it). Clipping
  belongs only in `to_pil()` / `save()`.
- fp16 is the default storage format; keep kernel arithmetic in fp32.

## Pull requests

- Keep PRs focused; one operator or one fix per PR where possible.
- Run `scripts/check.sh` (on at least one GPU backend) before pushing.
- Update `CHANGELOG.md` under the `Unreleased` heading.
- By contributing you agree your work is licensed under the project's
  [Apache-2.0](LICENSE) license.
