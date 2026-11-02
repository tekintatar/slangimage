// shaders/ops/hist_color.slang
// Augmentation color-op parity set, histogram half (see ops/hist_color.py):
//
// equalize — global PER-CHANNEL histogram equalization (normalized
// 256-bin CDF; CLAHE in clahe.slang
// is the tiled, luma-only relative):
//   1. equalize_hist  — 3x256-bin histogram over all pixels (atomics)
//   2. equalize_cdf   — one thread per channel: prefix sum to a LUT,
//                       (cdf - cdf_min) / (total - cdf_min)
//   3. equalize_apply — per pixel LUT lookup per channel
//
// autocontrast — per-channel (x - min) / (max - min) stretch with EXACT
// float min/max (order-preserving uint encoding + InterlockedMin/Max):
//   1. autocontrast_minmax — reduce min/max per channel
//   2. autocontrast_apply  — remap; constant channels pass through

static const uint HC_BINS = 256;

void equalize_hist(
    uint2                    coord,
    Texture2D<float4>        src,
    RWStructuredBuffer<uint> hist      // [3 * 256]
)
{
    uint width, height;
    src.GetDimensions(width, height);
    if (coord.x >= width || coord.y >= height)
        return;

    float3 c = clamp(src[coord].rgb, 0.0, 1.0);
    [ForceUnroll]
    for (uint ch = 0; ch < 3; ++ch)
    {
        uint bin = min(uint(c[ch] * 255.0 + 0.5), HC_BINS - 1);
        InterlockedAdd(hist[ch * HC_BINS + bin], 1u);
    }
}

// one thread per channel: coord.x = channel (dispatch grid (1, 3))
void equalize_cdf(
    uint2                     coord,
    RWStructuredBuffer<uint>  hist,
    RWStructuredBuffer<float> mapping   // [3 * 256], normalized LUT
)
{
    if (coord.x >= 3)
        return;
    uint base = coord.x * HC_BINS;

    float total = 0.0;
    float cdf_min = 0.0;
    for (uint b = 0; b < HC_BINS; ++b)
    {
        float v = float(hist[base + b]);
        if (total == 0.0 && v > 0.0)
            cdf_min = v;              // cdf at the first occupied bin
        total += v;
    }

    float denom = total - cdf_min;
    if (denom <= 0.0)
    {
        // Empty or single-bin channel: quantized identity mapping (bin/255),
        // not an exact input-value passthrough.
        for (uint b = 0; b < HC_BINS; ++b)
            mapping[base + b] = float(b) / 255.0;
        return;
    }

    float cum = 0.0;
    for (uint b = 0; b < HC_BINS; ++b)
    {
        cum += float(hist[base + b]);
        mapping[base + b] = clamp((cum - cdf_min) / denom, 0.0, 1.0);
    }
}

void equalize_apply(
    uint2                     coord,
    Texture2D<float4>         src,
    [format(SLANGIMAGE_FMT)] RWTexture2D<float4> dst,
    RWStructuredBuffer<float> mapping
)
{
    uint width, height;
    src.GetDimensions(width, height);
    if (coord.x >= width || coord.y >= height)
        return;

    float4 c = src[coord];
    float3 v = clamp(c.rgb, 0.0, 1.0);
    float3 o;
    [ForceUnroll]
    for (uint ch = 0; ch < 3; ++ch)
    {
        uint bin = min(uint(v[ch] * 255.0 + 0.5), HC_BINS - 1);
        o[ch] = mapping[ch * HC_BINS + bin];
    }
    dst[coord] = float4(o, c.a);
}

// ------------------------------------------------------- autocontrast
// order-preserving float <-> uint so InterlockedMin/Max sort correctly
// (works for negative values too — inputs are unclamped by convention)
uint _mm_encode(float f)
{
    uint u = asuint(f);
    return (u & 0x80000000u) != 0 ? ~u : (u | 0x80000000u);
}

float _mm_decode(uint k)
{
    return (k & 0x80000000u) != 0 ? asfloat(k & 0x7FFFFFFFu) : asfloat(~k);
}

void autocontrast_minmax(
    uint2                    coord,
    Texture2D<float4>        src,
    RWStructuredBuffer<uint> mm        // [6]: min r,g,b then max r,g,b
)
{
    uint width, height;
    src.GetDimensions(width, height);
    if (coord.x >= width || coord.y >= height)
        return;

    float3 c = src[coord].rgb;
    [ForceUnroll]
    for (uint ch = 0; ch < 3; ++ch)
    {
        uint key = _mm_encode(c[ch]);
        InterlockedMin(mm[ch], key);
        InterlockedMax(mm[3 + ch], key);
    }
}

void autocontrast_apply(
    uint2                    coord,
    Texture2D<float4>        src,
    [format(SLANGIMAGE_FMT)] RWTexture2D<float4> dst,
    RWStructuredBuffer<uint> mm
)
{
    uint width, height;
    src.GetDimensions(width, height);
    if (coord.x >= width || coord.y >= height)
        return;

    float4 c = src[coord];
    float3 o;
    [ForceUnroll]
    for (uint ch = 0; ch < 3; ++ch)
    {
        float lo = _mm_decode(mm[ch]);
        float hi = _mm_decode(mm[3 + ch]);
        // Degenerate range (<= fp32 1e-12) passes through, even outside [0,1].
        o[ch] = (hi - lo) > 1e-12 ? (c[ch] - lo) / (hi - lo) : c[ch];
    }
    dst[coord] = float4(o, c.a);
}
