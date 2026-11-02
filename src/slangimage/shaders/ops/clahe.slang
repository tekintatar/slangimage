// shaders/ops/clahe.slang
// Contrast-Limited Adaptive Histogram Equalization on luminance.
// Three passes (see ops/clahe.py):
//   1. clahe_hist  — per-tile 256-bin luminance histogram (atomics)
//   2. clahe_cdf   — one thread per tile: clip, redistribute, prefix-sum
//                    to a normalized mapping
//   3. clahe_apply — per pixel: bilinear blend of the 4 neighboring tile
//                    mappings; RGB scaled by newLuma/oldLuma
//
// Weighted RGB luminance, not Lab. RGB ratios are preserved only before the
// final gamut clamp. All pixels contribute regardless of alpha.

static const uint CLAHE_BINS = 256;

float clahe_luma(float3 rgb)
{
    return dot(rgb, float3(0.299, 0.587, 0.114));
}

void clahe_hist(
    uint2                    coord,
    Texture2D<float4>        src,
    RWStructuredBuffer<uint> hist,     // [tiles_x*tiles_y*256]
    int                      tiles_x,
    int                      tiles_y
)
{
    uint width, height;
    src.GetDimensions(width, height);
    if (coord.x >= width || coord.y >= height)
        return;

    uint tw = (width + tiles_x - 1) / tiles_x;
    uint th = (height + tiles_y - 1) / tiles_y;
    uint tx = min(coord.x / tw, uint(tiles_x) - 1);
    uint ty = min(coord.y / th, uint(tiles_y) - 1);

    float luma = clamp(clahe_luma(src[coord].rgb), 0.0, 1.0);
    uint bin = min(uint(luma * 255.0 + 0.5), CLAHE_BINS - 1);

    InterlockedAdd(hist[(ty * uint(tiles_x) + tx) * CLAHE_BINS + bin], 1u);
}

// one thread per tile
void clahe_cdf(
    uint2                     tile,
    RWStructuredBuffer<uint>  hist,
    RWStructuredBuffer<float> mapping,  // [tiles_x*tiles_y*256], normalized
    int                       tiles_x,
    int                       tiles_y,
    float                     clip_limit // relative; effective bin cap >= 1 count
)
{
    if (tile.x >= uint(tiles_x) || tile.y >= uint(tiles_y))
        return;

    uint base = (tile.y * uint(tiles_x) + tile.x) * CLAHE_BINS;

    // tile pixel count
    float total = 0.0;
    for (uint b = 0; b < CLAHE_BINS; ++b)
        total += float(hist[base + b]);
    if (total <= 0.0)
    {
        for (uint b = 0; b < CLAHE_BINS; ++b)
            mapping[base + b] = float(b) / 255.0;
        return;
    }

    // clip + collect excess
    float clip = max(clip_limit * total / float(CLAHE_BINS), 1.0);
    float excess = 0.0;
    for (uint b = 0; b < CLAHE_BINS; ++b)
    {
        float v = float(hist[base + b]);
        if (v > clip)
        {
            excess += v - clip;
            v = clip;
        }
        mapping[base + b] = v; // reuse as scratch
    }

    // redistribute excess uniformly
    float add = excess / float(CLAHE_BINS);

    // prefix sum -> normalized cdf
    float cum = 0.0;
    for (uint b = 0; b < CLAHE_BINS; ++b)
    {
        cum += mapping[base + b] + add;
        mapping[base + b] = clamp(cum / total, 0.0, 1.0);
    }
}

void clahe_apply(
    uint2                     coord,
    Texture2D<float4>         src,
    [format(SLANGIMAGE_FMT)] RWTexture2D<float4> dst,
    RWStructuredBuffer<float> mapping,
    int                       tiles_x,
    int                       tiles_y,
    float                     strength   // clamped [0,1]; 0 still clips HDR RGB
)
{
    uint width, height;
    src.GetDimensions(width, height);
    if (coord.x >= width || coord.y >= height)
        return;

    float4 c = src[coord];
    float luma = clamp(clahe_luma(c.rgb), 0.0, 1.0);
    uint bin = min(uint(luma * 255.0 + 0.5), CLAHE_BINS - 1);

    // Match the histogram's ceil-sized footprint, including partial edge tiles.
    uint tw = (width + tiles_x - 1) / tiles_x;
    uint th = (height + tiles_y - 1) / tiles_y;
    // Allocated grids can contain empty trailing tiles, even when tiles <= size.
    // Keep buffer strides unchanged but extend only the last occupied mapping.
    int last_x = int((width - 1) / tw);
    int last_y = int((height - 1) / th);

    // fractional tile-space position, centered on tile centers
    float fx = (float(coord.x) + 0.5) / float(tw) - 0.5;
    float fy = (float(coord.y) + 0.5) / float(th) - 0.5;

    int x0 = int(floor(fx));
    int y0 = int(floor(fy));
    float ax = fx - float(x0);
    float ay = fy - float(y0);

    int x1 = clamp(x0 + 1, 0, last_x);
    int y1 = clamp(y0 + 1, 0, last_y);
    x0 = clamp(x0, 0, last_x);
    y0 = clamp(y0, 0, last_y);

    float m00 = mapping[(uint(y0) * uint(tiles_x) + uint(x0)) * CLAHE_BINS + bin];
    float m10 = mapping[(uint(y0) * uint(tiles_x) + uint(x1)) * CLAHE_BINS + bin];
    float m01 = mapping[(uint(y1) * uint(tiles_x) + uint(x0)) * CLAHE_BINS + bin];
    float m11 = mapping[(uint(y1) * uint(tiles_x) + uint(x1)) * CLAHE_BINS + bin];

    float new_luma = lerp(lerp(m00, m10, ax), lerp(m01, m11, ax), ay);
    new_luma = lerp(luma, new_luma, clamp(strength, 0.0, 1.0));

    float ratio = (luma > 1e-4) ? new_luma / luma : 1.0;
    float3 out_rgb = clamp(c.rgb * ratio, 0.0, 1.0);

    dst[coord] = float4(out_rgb, c.a);
}
