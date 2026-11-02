# runtime/modules.py

from __future__ import annotations

import hashlib
import re
import weakref
from pathlib import Path
from typing import Any

import slangpy as spy


# Storage-format annotation injected into every op shader. CUDA performs no
# automatic format conversion on surface writes (slangpy #648), so every
# RWTexture2D parameter carries [format(SLANGIMAGE_FMT)] and we compile one
# module variant per precision. Metal/Vulkan accept the annotation too, so
# the same source stays fully portable.
_FMT_BY_PRECISION = {
    "fp16": "rgba16f",
    "fp32": "rgba32f",
}

_FP16_FORMATS = {spy.Format.rgba16_float}

_QUOTED_IMPORT_RE = re.compile(r'^\s*import\s+"([^"\n]+)"\s*;', re.MULTILINE)


def _infer_precision(args: tuple, kwargs: dict) -> str:
    """Pick the module variant from the first texture argument's format."""
    for a in list(args) + list(kwargs.values()):
        if isinstance(a, spy.Texture):
            return "fp16" if a.format in _FP16_FORMATS else "fp32"
    return "fp32"


class _OpDispatcher:
    """
    Callable proxy for one op module.

    Attribute access returns a dispatcher that selects the fp16 or fp32
    compiled variant based on the formats of the texture arguments, so op
    wrappers can keep calling `self.module.kernel_name(...)` unchanged.
    """

    def __init__(self, cache: "ModuleCache", op_name: str):
        self._cache = cache
        self._op_name = op_name

    def __getattr__(self, fn_name: str):
        def dispatch(*args: Any, **kwargs: Any):
            precision = _infer_precision(args, kwargs)
            mod = self._cache.load_variant(self._op_name, precision)
            return getattr(mod, fn_name)(*args, **kwargs)

        return dispatch


# One shared cache per device for the functional API (like texture_pool's
# get_pool). The cache owns its device, so release_device() is required to
# remove the package root. Downstream packages register custom kernel dirs on
# this shared instance: get_module_cache(device).add_ops_dir(...)
_caches: "weakref.WeakKeyDictionary[spy.Device, ModuleCache]" = weakref.WeakKeyDictionary()


def get_module_cache(device: spy.Device) -> "ModuleCache":
    from .device import _claim_device_thread

    _claim_device_thread(device)
    cache = _caches.get(device)
    if cache is None:
        cache = ModuleCache(device)
        _caches[device] = cache
    return cache


class ModuleCache:
    """Per-device cache of compiled operator shader modules (fp16/fp32 variants).

    Effects take ``(device, modules)``; the functional API keeps one shared
    cache per device internally. ``extra_ops_dirs`` and :meth:`add_ops_dir`
    register directories of custom ``.slang`` op files that are searched
    before the built-in ops (see examples/05_custom_kernels.py). Like its
    device, a cache belongs to the thread that created the device.
    """

    def __init__(self, device: spy.Device, extra_ops_dirs: list[str | Path] | None = None):
        from .device import _claim_device_thread

        _claim_device_thread(device)
        self.device = device
        self._modules: dict[tuple[str, str], spy.Module] = {}
        self._dispatchers: dict[str, _OpDispatcher] = {}

        repo_root = Path(__file__).resolve().parents[1]
        self.shaders_root = repo_root / "shaders"
        self.ops_root = self.shaders_root / "ops"
        # searched before the built-in ops dir — downstream packages
        # (e.g. third-party effect packages) register their own kernel dirs
        # here and may shadow built-in op names
        self.extra_ops_dirs: list[Path] = [Path(p) for p in (extra_ops_dirs or [])]

    def _check_thread(self) -> None:
        from .device import _claim_device_thread

        _claim_device_thread(self.device)

    def add_ops_dir(self, path: str | Path) -> None:
        """Register a directory of .slang op files (searched before built-ins).
        A custom op in such a directory may `import "sibling.slang"` from its own
        directory, as well as the built-in shared shaders."""
        self._check_thread()
        self.extra_ops_dirs.insert(0, Path(path))
        # A shadow dir registered after some ops were compiled must invalidate
        # the cache, or the already-loaded originals keep winning.
        self._modules.clear()
        self._dispatchers.clear()

    def _find_op_source(self, op_name: str) -> Path:
        for d in self.extra_ops_dirs:
            candidate = d / f"{op_name}.slang"
            if candidate.exists():
                return candidate
        candidate = self.ops_root / f"{op_name}.slang"
        if candidate.exists():
            return candidate
        searched = [str(d) for d in self.extra_ops_dirs] + [str(self.ops_root)]
        raise FileNotFoundError(f"op '{op_name}.slang' not found in: {searched}")

    def _resolve_import(self, importer: Path, import_name: str) -> Path | None:
        """Resolve quoted imports using the same roots as the device session."""
        candidates = (
            importer.parent / import_name,
            self.shaders_root / import_name,
            self.shaders_root / "common" / import_name,
            self.ops_root / import_name,
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        return None

    def _source_graph_hash(self, source_path: Path, root_source: str) -> str:
        """Hash a module's canonical path and transitive quoted imports.

        Slang module names are session-global. Including the complete resolved
        source graph keeps two custom roots, or a helper-only edit, from
        aliasing an already-loaded module with different behavior.
        """
        digest = hashlib.sha1()
        visited: set[Path] = set()

        def visit(path: Path, source: str | None = None) -> None:
            path = path.resolve()
            if path in visited:
                return
            visited.add(path)
            if source is None:
                source = path.read_text()
            digest.update(b"\0path\0")
            digest.update(str(path).encode())
            digest.update(b"\0source\0")
            digest.update(source.encode())
            for import_name in _QUOTED_IMPORT_RE.findall(source):
                imported = self._resolve_import(path, import_name)
                if imported is None:
                    # Compilation will report the missing import. Keep it in
                    # the identity so unresolved graphs cannot alias either.
                    digest.update(b"\0missing-import\0")
                    digest.update(import_name.encode())
                else:
                    visit(imported)

        visit(source_path, root_source)
        return digest.hexdigest()[:12]

    def _inline_local_imports(
        self,
        source_path: Path,
        source: str,
        inlined: set[Path] | None = None,
        stack: tuple[Path, ...] = (),
    ) -> str:
        """Inline quoted imports that resolve beside a custom source file.

        Slang's session caches imported modules by their import name, not their
        resolved file path. Two extension roots that both import
        ``"helper.slang"`` would otherwise share whichever helper compiled
        first even when their top-level module names differ. Local helpers are
        source dependencies rather than public modules, so embedding them in
        the uniquely named top-level module avoids that session-global alias.
        Imports resolved through SlangImage's built-in include roots stay as
        imports and continue to share the canonical built-in module.
        """
        source_path = source_path.resolve()
        if inlined is None:
            inlined = set()

        def replace(match: re.Match) -> str:
            import_name = match.group(1)
            local_path = source_path.parent / import_name
            if not local_path.is_file():
                return match.group(0)

            imported = local_path.resolve()
            if imported in stack or imported == source_path:
                chain = " -> ".join(str(p) for p in (*stack, source_path, imported))
                raise RuntimeError(f"cyclic local Slang import: {chain}")
            if imported in inlined:
                return f"// local import already inlined: {import_name}"

            inlined.add(imported)
            imported_source = self._inline_local_imports(
                imported,
                imported.read_text(),
                inlined,
                (*stack, source_path),
            )
            resume_line = source.count("\n", 0, match.end()) + 2
            return (
                f"// begin inlined local import: {import_name}\n"
                f"#line 1\n{imported_source}\n"
                f"// end inlined local import: {import_name}\n"
                f"#line {resume_line}"
            )

        return _QUOTED_IMPORT_RE.sub(replace, source)

    def load_variant(self, op_name: str, precision: str) -> spy.Module:
        self._check_thread()
        key = (op_name, precision)
        mod = self._modules.get(key)
        if mod is None:
            fmt = _FMT_BY_PRECISION[precision]
            source_path = self._find_op_source(op_name)
            raw_source = source_path.read_text()
            identity_source = f'#define SLANGIMAGE_FMT "{fmt}"\n#line 1\n{raw_source}'
            # module names are session-global in slangpy and re-loading a
            # name with different source is a hard error — include a source
            # hash so shadowed/updated ops coexist with already-loaded ones
            src_hash = self._source_graph_hash(source_path, identity_source)
            name = f"{op_name}__{precision}__{src_hash}"
            if source_path.parent != self.ops_root:
                source = self._inline_local_imports(source_path, raw_source)
                source = f'#define SLANGIMAGE_FMT "{fmt}"\n#line 1\n{source}'
                # The session also keys source modules by path. Use a virtual
                # path containing the graph hash so a helper-only edit can be
                # loaded alongside the prior version. Local quoted imports are
                # already inlined; canonical built-in imports resolve through
                # the device's normal include paths.
                virtual_path = source_path.with_name(f"{name}.slang")
                sm = self.device.slang_session.load_module_from_source(
                    name, source, path=str(virtual_path)
                )
                mod = spy.Module.load_from_module(self.device, sm)
            else:
                source = identity_source
                mod = spy.Module.load_from_source(self.device, name, source)
            self._modules[key] = mod
        return mod

    def get_op(self, op_name: str) -> _OpDispatcher:
        self._check_thread()
        disp = self._dispatchers.get(op_name)
        if disp is None:
            disp = _OpDispatcher(self, op_name)
            self._dispatchers[op_name] = disp
        return disp
