// shaders/ops/resize.slang
// Resize with:
//   - Bicubic (Catmull Rom) filter
//   - Lanczos3 filter
// No color space conversions

static const float PI        = 3.14159265;
static const float RESIZE_EPS = 1e-6;


// ============================================================
// Lanczos3 support
// ============================================================

float sinc_pi(float x)
{
    x = abs(x);
    if (x < 1e-5) return 1.0;
    float px = PI * x;
    return sin(px) / px;
}

float lanczos3_kernel(float x)
{
    x = abs(x);
    if (x >= 3.0) return 0.0;
    return sinc_pi(x) * sinc_pi(x / 3.0);
}

// ------------------------------------------------------------
// Separable Lanczos3 (two 1D passes). Lanczos is separable, so H-then-V gives
// the same mathematical filter as a 2D footprint, but with O(ratio) taps/pixel and
// full GPU occupancy instead of O(ratio^2) taps on very few threads (the 2D
// path starves the GPU at large downscales). Per-pass normalization by the tap
// weight sum matches the 2D joint normalization because the per-column x-weight
// sum is constant across rows. Edge handling is the same truncated window
// (out-of-bounds taps dropped, not clamp-replicated). The intermediate texture
// rounds to the input storage precision; the result is not bitwise identical
// to a single 2D dispatch with only one output store.
// ------------------------------------------------------------

// Horizontal pass: src (Ws x Hs) -> dst (Wd x Hs); rows pass through.
void resize_lanczos3_h(
    uint2               coord,
    Texture2D<float4>   src,
    [format(SLANGIMAGE_FMT)] RWTexture2D<float4> dst
)
{
    uint dst_w, dst_h;
    dst.GetDimensions(dst_w, dst_h);
    if (coord.x >= dst_w || coord.y >= dst_h)
        return;

    uint src_w, src_h;
    src.GetDimensions(src_w, src_h);

    float scale     = float(src_w) / float(dst_w);
    float inv_scale = scale > 1.0 ? scale : 1.0;
    float center    = (float(coord.x) + 0.5) * scale - 0.5;
    float radius    = 3.0 * inv_scale;
    float rcp_is    = 1.0 / inv_scale;

    int x0 = clamp(int(floor(center - radius)), 0, int(src_w) - 1);
    int x1 = clamp(int(ceil (center + radius)), 0, int(src_w) - 1);
    int sy = int(coord.y);

    float4 acc   = float4(0.0);
    float  acc_w = 0.0;
    for (int sx = x0; sx <= x1; ++sx)
    {
        float wx = lanczos3_kernel((float(sx) - center) * rcp_is);
        if (wx == 0.0) continue;
        acc   += wx * src[uint2(sx, sy)];
        acc_w += wx;
    }

    if (acc_w < RESIZE_EPS)
    {
        uint px = clamp(int(center + 0.5), 0, int(src_w) - 1);
        dst[coord] = src[uint2(px, coord.y)];
        return;
    }
    dst[coord] = acc / acc_w;
}

// Vertical pass: src (Wd x Hs) -> dst (Wd x Hd); columns pass through.
void resize_lanczos3_v(
    uint2               coord,
    Texture2D<float4>   src,
    [format(SLANGIMAGE_FMT)] RWTexture2D<float4> dst
)
{
    uint dst_w, dst_h;
    dst.GetDimensions(dst_w, dst_h);
    if (coord.x >= dst_w || coord.y >= dst_h)
        return;

    uint src_w, src_h;
    src.GetDimensions(src_w, src_h);

    float scale     = float(src_h) / float(dst_h);
    float inv_scale = scale > 1.0 ? scale : 1.0;
    float center    = (float(coord.y) + 0.5) * scale - 0.5;
    float radius    = 3.0 * inv_scale;
    float rcp_is    = 1.0 / inv_scale;

    int y0 = clamp(int(floor(center - radius)), 0, int(src_h) - 1);
    int y1 = clamp(int(ceil (center + radius)), 0, int(src_h) - 1);
    int sx = int(coord.x);

    float4 acc   = float4(0.0);
    float  acc_w = 0.0;
    for (int sy = y0; sy <= y1; ++sy)
    {
        float wy = lanczos3_kernel((float(sy) - center) * rcp_is);
        if (wy == 0.0) continue;
        acc   += wy * src[uint2(sx, sy)];
        acc_w += wy;
    }

    if (acc_w < RESIZE_EPS)
    {
        uint py = clamp(int(center + 0.5), 0, int(src_h) - 1);
        dst[coord] = src[uint2(coord.x, py)];
        return;
    }
    dst[coord] = acc / acc_w;
}


// ============================================================
// Bicubic support
// ============================================================

// Catmull Rom style cubic kernel (a = -0.5)
float cubic_kernel(float x)
{
    x = abs(x);
    float a = -0.5;

    if (x <= 1.0)
    {
        return (a + 2.0) * x * x * x
             - (a + 3.0) * x * x
             + 1.0;
    }
    else if (x < 2.0)
    {
        return a * x * x * x
             - 5.0 * a * x * x
             + 8.0 * a * x
             - 4.0 * a;
    }
    return 0.0;
}

// Bicubic sample in texture value space, 4x4 footprint
float4 sample_bicubic(Texture2D<float4> src, float2 uv)
{
    uint src_w, src_h;
    src.GetDimensions(src_w, src_h);

    // Map uv to source pixel coordinates
    float2 pos  = uv * float2(src_w, src_h) - 0.5;
    float2 base = floor(pos);
    float2 frac = pos - base;

    int2 i0 = int2(base);

    float4 acc  = float4(0, 0, 0, 0);
    float  wsum = 0.0;

    [unroll]
    for (int j = -1; j <= 2; ++j)
    {
        int sy = clamp(i0.y + j, 0, int(src_h) - 1);
        float wy = cubic_kernel(float(j) - frac.y);

        [unroll]
        for (int i = -1; i <= 2; ++i)
        {
            int sx = clamp(i0.x + i, 0, int(src_w) - 1);
            float wx = cubic_kernel(float(i) - frac.x);

            float  w = wx * wy;
            float4 s = src[uint2(sx, sy)];

            acc  += w * s;
            wsum += w;
        }
    }

    return acc / max(wsum, 1e-6);
}

void resize_bicubic(
    uint2               coord,
    Texture2D<float4>   src,
    [format(SLANGIMAGE_FMT)] RWTexture2D<float4> dst
)
{
    uint dst_w, dst_h;
    dst.GetDimensions(dst_w, dst_h);
    if (coord.x >= dst_w || coord.y >= dst_h)
        return;

    float2 uv = (float2(coord) + 0.5) / float2(dst_w, dst_h);
    dst[coord] = sample_bicubic(src, uv);
}

// ============================================================
// Bilinear support
// ============================================================

// Bilinear sample in texture value space
float4 sample_bilinear(Texture2D<float4> src, float2 uv)
{
    uint src_w, src_h;
    src.GetDimensions(src_w, src_h);

    // Map uv to source pixel coordinates
    // integer coordinates are pixel centers
    float2 pos  = uv * float2(src_w, src_h) - 0.5;
    float2 base = floor(pos);
    float2 frac = pos - base;

    int2 p00 = int2(base);
    int2 p10 = p00 + int2(1, 0);
    int2 p01 = p00 + int2(0, 1);
    int2 p11 = p00 + int2(1, 1);

    // Clamp to edges
    p00.x = clamp(p00.x, 0, int(src_w) - 1);
    p10.x = clamp(p10.x, 0, int(src_w) - 1);
    p01.x = clamp(p01.x, 0, int(src_w) - 1);
    p11.x = clamp(p11.x, 0, int(src_w) - 1);

    p00.y = clamp(p00.y, 0, int(src_h) - 1);
    p10.y = clamp(p10.y, 0, int(src_h) - 1);
    p01.y = clamp(p01.y, 0, int(src_h) - 1);
    p11.y = clamp(p11.y, 0, int(src_h) - 1);

    float4 s00 = src[uint2(p00)];
    float4 s10 = src[uint2(p10)];
    float4 s01 = src[uint2(p01)];
    float4 s11 = src[uint2(p11)];

    float4 x0 = lerp(s00, s10, frac.x);
    float4 x1 = lerp(s01, s11, frac.x);
    return lerp(x0, x1, frac.y);
}

void resize_bilinear(
    uint2               coord,
    Texture2D<float4>   src,
    [format(SLANGIMAGE_FMT)] RWTexture2D<float4> dst
)
{
    uint dst_w, dst_h;
    dst.GetDimensions(dst_w, dst_h);
    if (coord.x >= dst_w || coord.y >= dst_h)
        return;

    // Same uv convention as bicubic path
    float2 uv = (float2(coord) + 0.5) / float2(dst_w, dst_h);
    dst[coord] = sample_bilinear(src, uv);
}
