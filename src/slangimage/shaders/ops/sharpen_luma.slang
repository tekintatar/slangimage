import "color_conversions.slang";

// ----------------------------------------------------------------------
// Bicubic / sharpening code
// ----------------------------------------------------------------------

float cubic(float x)
{
    x = abs(x);
    if (x <= 1.0)
    {
        return 1.0 - 2.0 * x * x + x * x * x;
    }
    else if (x < 2.0)
    {
        return 4.0 - 8.0 * x + 5.0 * x * x - x * x * x;
    }
    return 0.0;
}

// Bicubic on HSL lightness (for luma sharpen)
float sample_bicubic_luma_wide(
    Texture2D<float4> src,
    uint2             tid,
    float             radius_pixels
)
{
    uint w, h;
    src.GetDimensions(w, h);

    float r = max(radius_pixels, 0.5); // avoid division by zero

    float r_clamped = clamp(radius_pixels, 1.0, 4.0);
    int   N         = int(ceil(r_clamped));

    float Y     = 0.0;
    float wsum  = 0.0;
    float inv_r = 1.0 / r;   // hoisted: per-tap fdiv is slow on NVRTC

    // no [unroll]: the trip count is runtime (N from the radius param) —
    // forcing it makes NVRTC emit ~3x slower code
    for (int j = -N; j <= N; ++j)
    {
        int sy = clamp(int(tid.y) + j, 0, int(h) - 1);
        float wy = cubic(float(j) * inv_r);

        for (int i = -N; i <= N; ++i)
        {
            int sx = clamp(int(tid.x) + i, 0, int(w) - 1);
            float wx = cubic(float(i) * inv_r);

            float  w   = wx * wy;
            float4 s   = src[uint2(sx, sy)];

            // HSL lightness = (max+min)/2 on the saturated rgb — computed
            // with scalars (rgb_to_hsl per tap is float3-heavy; float3 loop
            // math is ~3x on CUDA, experiments/cuda_hotloop)
            float r = saturate(s.r);
            float g = saturate(s.g);
            float b = saturate(s.b);
            float L = 0.5 * (max(r, max(g, b)) + min(r, min(g, b)));

            Y    += w * L;
            wsum += w;
        }
    }

    return Y / max(wsum, 1e-6);
}

// Bicubic blur in RGB space (for RGB sharpen). Returns float4 — even the
// once-per-pixel float3 return/math poisoned NVRTC codegen for the whole
// kernel (measured 2.9 vs 1.1 ms with an identical float4-only clone).
float4 sample_bicubic_rgb_wide(
    Texture2D<float4> src,
    uint2             tid,
    float             radius_pixels
)
{
    uint w, h;
    src.GetDimensions(w, h);

    float r = max(radius_pixels, 0.5); // avoid division by zero

    float r_clamped = clamp(radius_pixels, 1.0, 4.0);
    int   N         = int(ceil(r_clamped));

    // float4 accumulator (float3 loop math is ~3x on CUDA;
    // experiments/cuda_hotloop) — the alpha lane is discarded by callers
    float4 acc  = float4(0.0);
    float  wsum = 0.0;
    float inv_r = 1.0 / r;   // hoisted: per-tap fdiv is slow on NVRTC

    // no [unroll]: the trip count is runtime (N from the radius param) —
    // forcing it makes NVRTC emit ~3x slower code
    for (int j = -N; j <= N; ++j)
    {
        int sy = clamp(int(tid.y) + j, 0, int(h) - 1);
        float wy = cubic(float(j) * inv_r);

        for (int i = -N; i <= N; ++i)
        {
            int sx = clamp(int(tid.x) + i, 0, int(w) - 1);
            float wx = cubic(float(i) * inv_r);

            float  w = wx * wy;
            float4 s = saturate(src[uint2(sx, sy)]);

            acc  += w * s;
            wsum += w;
        }
    }

    return acc / max(wsum, 1e-6);
}

// Sharpen luma, keep chroma as is (HSL space)
void sharpen_luma(
    uint2               tid,        // from spy.grid((h, w))
    Texture2D<float4>   src,
    [format(SLANGIMAGE_FMT)] RWTexture2D<float4> dst,
    float               radius,     // blur radius in pixels for Y_smooth
    float               sharpness   // effect strength
)
{
    uint w, h;
    src.GetDimensions(w, h);
    if (tid.x >= w || tid.y >= h)
        return;

    float4 s = src[tid];

    s.rgb = saturate(s.rgb);
    float3 hsl = rgb_to_hsl(s.rgb);
    float L = hsl.z;
    float H = hsl.x;
    float S = hsl.y;

    float radius_px   = clamp(radius, 0.5, 64.0);
    float sharp_local = clamp(sharpness, -2.0, 4.0);

    float L_smooth = sample_bicubic_luma_wide(src, tid, radius_px);

    float detail  = L - L_smooth;
    float L_sharp = L + sharp_local * detail;

    L_sharp = clamp(L_sharp, 0.0, 1.0);

    float3 out_rgb = hsl_to_rgb(float3(H, S, L_sharp));
    out_rgb = saturate(out_rgb);

    dst[tid] = float4(out_rgb, s.a);
}

// Sharpen full RGB with bicubic blur (classic unsharp mask on RGB)
void sharpen_rgb(
    uint2               tid,        // from spy.grid((h, w))
    Texture2D<float4>   src,
    [format(SLANGIMAGE_FMT)] RWTexture2D<float4> dst,
    float               radius,     // blur radius in pixels
    float               sharpness   // effect strength
)
{
    uint w, h;
    src.GetDimensions(w, h);
    if (tid.x >= w || tid.y >= h)
        return;

    // float4/scalar math only — see note on sample_bicubic_rgb_wide
    float4 s = src[tid];
    float4 rgba = saturate(s);

    float radius_px   = clamp(radius, 0.5, 64.0);
    float sharp_local = clamp(sharpness, -2.0, 4.0);

    float4 smooth4 = sample_bicubic_rgb_wide(src, tid, radius_px);

    float4 sharp4 = saturate(rgba + sharp_local * (rgba - smooth4));

    // write via lane assignment: float4(sharp4.rgb, s.a) materializes a
    // float3 and poisons NVRTC codegen (measured 2.6x on this kernel)
    sharp4.a = s.a;
    dst[tid] = sharp4;
}