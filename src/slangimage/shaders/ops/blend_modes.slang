// shaders/ops/blend_modes.slang
// Photoshop/PDF-style blend modes matching Core Image's CI*BlendMode set.
// Formulas follow the W3C compositing spec (which CI implements); inputs are
// straight (unpremultiplied) RGBA. Full source-over alpha compositing, which
// reduces to the plain blend formula for opaque images.
//
// Mode ids (keep in sync with ops/blend_modes.py):
//   0 multiply, 1 screen, 2 overlay, 3 darken, 4 lighten, 5 color_dodge,
//   6 color_burn, 7 hard_light, 8 soft_light, 9 linear_light, 10 vivid_light,
//   11 color

static const float BLEND_EPS = 1e-6;

float3 bm_multiply(float3 b, float3 s) { return b * s; }
float3 bm_screen(float3 b, float3 s)   { return b + s - b * s; }

float bm_overlay1(float b, float s)
{
    return (b <= 0.5) ? 2.0 * b * s : 1.0 - 2.0 * (1.0 - b) * (1.0 - s);
}
float3 bm_overlay(float3 b, float3 s)
{
    return float3(bm_overlay1(b.r, s.r), bm_overlay1(b.g, s.g), bm_overlay1(b.b, s.b));
}

float3 bm_darken(float3 b, float3 s)  { return min(b, s); }
float3 bm_lighten(float3 b, float3 s) { return max(b, s); }

float bm_dodge1(float b, float s)
{
    if (b <= 0.0) return 0.0;
    if (s >= 1.0) return 1.0;
    return min(1.0, b / (1.0 - s));
}
float3 bm_color_dodge(float3 b, float3 s)
{
    return float3(bm_dodge1(b.r, s.r), bm_dodge1(b.g, s.g), bm_dodge1(b.b, s.b));
}

float bm_burn1(float b, float s)
{
    if (b >= 1.0) return 1.0;
    if (s <= 0.0) return 0.0;
    return 1.0 - min(1.0, (1.0 - b) / s);
}
float3 bm_color_burn(float3 b, float3 s)
{
    return float3(bm_burn1(b.r, s.r), bm_burn1(b.g, s.g), bm_burn1(b.b, s.b));
}

float3 bm_hard_light(float3 b, float3 s) { return bm_overlay(s, b); }

float bm_soft_light1(float b, float s)
{
    // W3C soft-light
    if (s <= 0.5)
        return b - (1.0 - 2.0 * s) * b * (1.0 - b);
    float d = (b <= 0.25) ? ((16.0 * b - 12.0) * b + 4.0) * b : sqrt(b);
    return b + (2.0 * s - 1.0) * (d - b);
}
float3 bm_soft_light(float3 b, float3 s)
{
    return float3(bm_soft_light1(b.r, s.r), bm_soft_light1(b.g, s.g), bm_soft_light1(b.b, s.b));
}

float3 bm_linear_light(float3 b, float3 s)
{
    return clamp(b + 2.0 * s - 1.0, 0.0, 1.0);
}

float bm_vivid_light1(float b, float s)
{
    return (s <= 0.5) ? bm_burn1(b, 2.0 * s) : bm_dodge1(b, 2.0 * (s - 0.5));
}
float3 bm_vivid_light(float3 b, float3 s)
{
    return float3(bm_vivid_light1(b.r, s.r), bm_vivid_light1(b.g, s.g), bm_vivid_light1(b.b, s.b));
}

// ---- Nonseparable "color" mode: weighted luminosity transfer + gamut
// compression toward neutral gray, not a literal HSL conversion. ----
float bm_lum(float3 c) { return dot(c, float3(0.3, 0.59, 0.11)); }

float3 bm_clip_color(float3 c)
{
    float l = bm_lum(c);
    float n = min(c.r, min(c.g, c.b));
    float x = max(c.r, max(c.g, c.b));
    if (n < 0.0)
        c = l + (c - l) * l / max(l - n, BLEND_EPS);
    if (x > 1.0)
        c = l + (c - l) * (1.0 - l) / max(x - l, BLEND_EPS);
    return c;
}

float3 bm_set_lum(float3 c, float l)
{
    return bm_clip_color(c + (l - bm_lum(c)));
}

float3 bm_color(float3 b, float3 s)
{
    return bm_set_lum(s, bm_lum(b));
}

float3 apply_blend_mode(int mode, float3 b, float3 s)
{
    switch (mode)
    {
    case 0:  return bm_multiply(b, s);
    case 1:  return bm_screen(b, s);
    case 2:  return bm_overlay(b, s);
    case 3:  return bm_darken(b, s);
    case 4:  return bm_lighten(b, s);
    case 5:  return bm_color_dodge(b, s);
    case 6:  return bm_color_burn(b, s);
    case 7:  return bm_hard_light(b, s);
    case 8:  return bm_soft_light(b, s);
    case 9:  return bm_linear_light(b, s);
    case 10: return bm_vivid_light(b, s);
    case 11: return bm_color(b, s);
    default: return s;
    }
}

void blend_mode(
    uint2               coord,
    Texture2D<float4>   base,      // background
    Texture2D<float4>   blend,     // foreground
    [format(SLANGIMAGE_FMT)] RWTexture2D<float4> dst,
    int                 mode,
    float               opacity    // scales foreground alpha
)
{
    uint width, height;
    base.GetDimensions(width, height);
    if (coord.x >= width || coord.y >= height)
        return;

    uint bw, bh;
    blend.GetDimensions(bw, bh);
    uint2 bc = uint2(min(coord.x, bw - 1), min(coord.y, bh - 1));

    float4 cb = base[coord];
    float4 cs = blend[bc];

    float ab = cb.a;
    float as = cs.a * clamp(opacity, 0.0, 1.0);

    float3 blended = apply_blend_mode(mode, cb.rgb, cs.rgb);

    // source-over with blended color (W3C compositing)
    float ao = as + ab * (1.0 - as);
    float3 co = (1.0 - ab) * as * cs.rgb
              + (1.0 - as) * ab * cb.rgb
              + as * ab * blended;
    co = (ao > BLEND_EPS) ? co / ao : float3(0.0, 0.0, 0.0);

    dst[coord] = float4(co, ao);
}
