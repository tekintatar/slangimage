// shaders/ops/cmyk_halftone.slang
// Classic CMYK halftone (rotated screens per ink, like CICMYKHalftone):
// naive RGB->CMYK separation, each ink screened on its own rotated dot
// grid (C 15°, M 75°, Y 0°, K 45°), dot area proportional to ink value,
// smoothstep-antialiased, recombined multiplicatively on white paper.

static const float HT_DEG = 3.14159265 / 180.0;

float ht_channel(float2 p, float angle, float width, float value, float sharpness)
{
    // rotate into screen space
    float s = sin(angle);
    float c = cos(angle);
    float2 q = float2(c * p.x - s * p.y, s * p.x + c * p.y) / max(width, 1.0);

    float2 cellCenter = floor(q) + 0.5;
    float dist = length(q - cellCenter);

    // dot radius: value 1 -> covers the cell diagonal (solid ink)
    float r = sqrt(clamp(value, 0.0, 1.0)) * 0.7071;
    float aa = max(sharpness, 1e-3);
    return 1.0 - smoothstep(r - aa, r + aa, dist);
}

void cmyk_halftone(
    uint2               coord,
    Texture2D<float4>   src,
    [format(SLANGIMAGE_FMT)] RWTexture2D<float4> dst,
    float               width,      // screen cell size in pixels
    float               angle_deg,  // extra rotation added to all screens
    float               sharpness   // dot edge AA in cell units (~0.08)
)
{
    uint w, h;
    src.GetDimensions(w, h);
    if (coord.x >= w || coord.y >= h)
        return;

    float4 px = src[coord];
    float3 rgb = clamp(px.rgb, 0.0, 1.0);

    // Normalized RGB -> CMYK separation. Keep this as the fixed public
    // contract until a continuous, independently validated alternative is
    // ready to expose.
    float k = 1.0 - max(rgb.r, max(rgb.g, rgb.b));
    float denom = max(1.0 - k, 1e-4);
    float cy = (1.0 - rgb.r - k) / denom;
    float mg = (1.0 - rgb.g - k) / denom;
    float ye = (1.0 - rgb.b - k) / denom;

    float2 p = float2(coord) + 0.5;
    float base = angle_deg * HT_DEG;

    float C = ht_channel(p, base + 15.0 * HT_DEG, width, cy, sharpness);
    float M = ht_channel(p, base + 75.0 * HT_DEG, width, mg, sharpness);
    float Y = ht_channel(p, base + 0.0 * HT_DEG, width, ye, sharpness);
    float K = ht_channel(p, base + 45.0 * HT_DEG, width, k, sharpness);

    // ink over white paper, multiplicative
    float3 out_rgb;
    out_rgb.r = (1.0 - C) * (1.0 - K);
    out_rgb.g = (1.0 - M) * (1.0 - K);
    out_rgb.b = (1.0 - Y) * (1.0 - K);

    dst[coord] = float4(out_rgb, px.a);
}
