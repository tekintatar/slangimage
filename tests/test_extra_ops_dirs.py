# tests/test_extra_ops_dirs.py
# ModuleCache can load .slang ops from registered external directories —
# the mechanism downstream packages use to ship custom kernels from their
# own directories.

from __future__ import annotations


import numpy as np
import pytest

from slangimage.runtime.textures import np_to_texture, texture_to_np
from slangimage.runtime.texture_pool import alloc_output

CUSTOM_OP = """
// a custom kernel: scales rgb by a factor
void custom_scale(
    uint2               coord,
    Texture2D<float4>   src,
    [format(SLANGIMAGE_FMT)] RWTexture2D<float4> dst,
    float               factor
)
{
    uint width, height;
    src.GetDimensions(width, height);
    if (coord.x >= width || coord.y >= height)
        return;
    float4 c = src[coord];
    dst[coord] = float4(clamp(c.rgb * factor, 0.0, 1.0), c.a);
}
"""




def test_external_ops_dir(ctx, tmp_path):
    import slangpy as spy

    device, modules = ctx
    (tmp_path / "custom_scale.slang").write_text(CUSTOM_OP)
    modules.add_ops_dir(tmp_path)

    img = np.full((16, 16, 4), 0.25, dtype=np.float32)
    img[..., 3] = 1.0
    src = np_to_texture(device, img, usage="read", precision="fp32")
    dst = alloc_output(device, 16, 16, src.format)

    mod = modules.get_op("custom_scale")
    mod.custom_scale(spy.grid((16, 16)), src, dst, 2.0)

    out = texture_to_np(dst)
    np.testing.assert_allclose(out[..., :3], 0.5, atol=1e-3)


def test_missing_op_reports_search_paths(ctx):
    _, modules = ctx
    with pytest.raises(FileNotFoundError, match="no_such_op"):
        modules.load_variant("no_such_op", "fp32")


HELPER = "float3 tint(float3 c){ return c * float3(1.0, 0.5, 0.25); }\n"
SIBLING_OP = (
    'import "helper.slang";\n'
    "void tint_op(uint2 coord, Texture2D<float4> src,\n"
    "             [format(SLANGIMAGE_FMT)] RWTexture2D<float4> dst){\n"
    "  uint w,h; src.GetDimensions(w,h);\n"
    "  if(coord.x>=w||coord.y>=h) return;\n"
    "  dst[coord] = float4(tint(src[coord].rgb), src[coord].a);\n"
    "}\n"
)


def test_custom_op_imports_sibling(ctx, tmp_path):
    """A custom op can `import "sibling.slang"` from its own directory."""
    import slangpy as spy

    device, _ = ctx
    (tmp_path / "helper.slang").write_text(HELPER)
    (tmp_path / "tint_op.slang").write_text(SIBLING_OP)
    cache = __import__("slangimage").ModuleCache(device, extra_ops_dirs=[tmp_path])

    img = np.full((8, 8, 4), 0.6, dtype=np.float32)
    img[..., 3] = 1.0
    src = np_to_texture(device, img, usage="read", precision="fp32")
    dst = alloc_output(device, 8, 8, src.format)
    cache.get_op("tint_op").tint_op(spy.grid((8, 8)), src, dst)

    out = texture_to_np(dst)
    np.testing.assert_allclose(out[0, 0, :3], [0.6, 0.3, 0.15], atol=2e-3)


def test_shadow_after_load_invalidates_cache(ctx, tmp_path):
    """Registering a shadow dir AFTER an op was compiled must take effect."""
    import slangpy as spy

    device, _ = ctx
    cache = __import__("slangimage").ModuleCache(device)

    img = np.full((8, 8, 4), 0.25, dtype=np.float32)
    img[..., 3] = 1.0
    src = np_to_texture(device, img, usage="read", precision="fp32")

    dst = alloc_output(device, 8, 8, src.format)
    cache.get_op("point_ops").invert(spy.grid((8, 8)), src, dst)  # loads the built-in

    (tmp_path / "point_ops.slang").write_text(
        "void invert(uint2 c, Texture2D<float4> src,"
        " [format(SLANGIMAGE_FMT)] RWTexture2D<float4> dst){"
        " dst[c]=float4(0.123,0.123,0.123,1.0); }"
    )
    cache.add_ops_dir(tmp_path)  # must invalidate the already-compiled invert

    dst2 = alloc_output(device, 8, 8, src.format)
    cache.get_op("point_ops").invert(spy.grid((8, 8)), src, dst2)
    out = texture_to_np(dst2)
    np.testing.assert_allclose(out[..., :3], 0.123, atol=2e-3)


def test_shadowing_already_loaded_builtin(ctx, tmp_path):
    """A custom dir may shadow a built-in op that was ALREADY compiled in
    this session — the source-hash in module names prevents slangpy's
    'already loaded with different source' error."""
    import slangpy as spy

    device, _ = ctx

    # fresh cache: load the built-in first
    cache = __import__("slangimage").ModuleCache(device)
    cache.load_variant("point_ops", "fp32")

    # now shadow it with different source in a second cache on the same device
    shadow = 'void invert(uint2 coord, Texture2D<float4> src, [format(SLANGIMAGE_FMT)] RWTexture2D<float4> dst) { dst[coord] = src[coord]; }'
    (tmp_path / "point_ops.slang").write_text(shadow)
    cache2 = __import__("slangimage").ModuleCache(device, extra_ops_dirs=[tmp_path])

    img = np.full((8, 8, 4), 0.25, dtype=np.float32)
    src = np_to_texture(device, img, usage="read", precision="fp32")
    dst = alloc_output(device, 8, 8, src.format)
    cache2.get_op("point_ops").invert(spy.grid((8, 8)), src, dst)

    out = texture_to_np(dst)
    np.testing.assert_allclose(out, img, atol=1e-6)  # shadowed no-op, not builtin invert


VARIANT_OP = (
    'import "helper.slang";\n'
    "void variant(uint2 coord, Texture2D<float4> src,\n"
    "             [format(SLANGIMAGE_FMT)] RWTexture2D<float4> dst){\n"
    "  uint w,h; src.GetDimensions(w,h);\n"
    "  if(coord.x>=w||coord.y>=h) return;\n"
    "  float4 c = src[coord];\n"
    "  dst[coord] = float4(c.rgb * helper_gain(), c.a);\n"
    "}\n"
)


def test_module_identity_includes_path_and_transitive_imports(ctx, tmp_path):
    """Identical roots with different helpers, plus helper-only edits, must
    compile as distinct session-global modules."""
    import slangpy as spy

    device, _ = ctx
    root_a, root_b = tmp_path / "a", tmp_path / "b"
    root_a.mkdir()
    root_b.mkdir()
    for root, gain in ((root_a, 0.5), (root_b, 0.8)):
        (root / "variant.slang").write_text(VARIANT_OP)
        (root / "helper.slang").write_text(
            f"float helper_gain() {{ return {gain}; }}\n"
        )

    img = np.ones((8, 8, 4), dtype=np.float32)
    src = np_to_texture(device, img, usage="read", precision="fp32")

    def run(cache):
        dst = alloc_output(device, 8, 8, src.format)
        cache.get_op("variant").variant(spy.grid((8, 8)), src, dst)
        return texture_to_np(dst)[0, 0, 0]

    cache_a = __import__("slangimage").ModuleCache(device, extra_ops_dirs=[root_a])
    cache_b = __import__("slangimage").ModuleCache(device, extra_ops_dirs=[root_b])
    assert run(cache_a) == pytest.approx(0.5, abs=2e-3)
    assert run(cache_b) == pytest.approx(0.8, abs=2e-3)

    (root_a / "helper.slang").write_text(
        "float helper_gain() { return 0.2; }\n"
    )
    cache_a.add_ops_dir(root_a)  # invalidate after the helper-only edit
    assert run(cache_a) == pytest.approx(0.2, abs=2e-3)
