// shaders/ops/lut.slang
// Square 8x8-grid color LUT (64 slices, each size/8 by size/8). Blue
// channel picks two neighboring slices, red/green index within a slice,
// results are blended by frac(blue). LUT texels are fetched with manual
// bilinear filtering (no hardware samplers, matching the rest of the
// library). LUT images are stored top-left origin (as loaded from PNG).
// Source RGBA and intensity are clamped; LUT alpha is ignored and mapped
// RGB is not clipped. This is an 8x8 tiled atlas, not an arbitrary HALD layout.

float4 lut_bilinear(Texture2D<float4> lut, float2 pos, uint lw, uint lh)
{
    float2 p = pos - 0.5;
    int x0 = int(floor(p.x));
    int y0 = int(floor(p.y));
    float fx = p.x - float(x0);
    float fy = p.y - float(y0);

    int x1 = min(x0 + 1, int(lw) - 1);
    int y1 = min(y0 + 1, int(lh) - 1);
    x0 = clamp(x0, 0, int(lw) - 1);
    y0 = clamp(y0, 0, int(lh) - 1);

    float4 c00 = lut[uint2(x0, y0)];
    float4 c10 = lut[uint2(x1, y0)];
    float4 c01 = lut[uint2(x0, y1)];
    float4 c11 = lut[uint2(x1, y1)];

    return lerp(lerp(c00, c10, fx), lerp(c01, c11, fx), fy);
}

void lut_apply(
    uint2               coord,
    Texture2D<float4>   src,
    Texture2D<float4>   lut,       // square 8x8 grid, size >=16 and divisible by 8
    [format(SLANGIMAGE_FMT)] RWTexture2D<float4> dst,
    float               intensity  // 1.0 = full LUT, 0.0 = clamped source
)
{
    uint width, height;
    src.GetDimensions(width, height);
    if (coord.x >= width || coord.y >= height)
        return;

    uint lw, lh;
    lut.GetDimensions(lw, lh);

    float4 c = clamp(src[coord], 0.0, 1.0);

    float b = c.b * 63.0;

    float slice0 = floor(b);
    float slice1 = ceil(b);

    float2 cell0 = float2(fmod(slice0, 8.0), floor(slice0 / 8.0));
    float2 cell1 = float2(fmod(slice1, 8.0), floor(slice1 / 8.0));

    // Normalized position in a 512px 8x8-tile LUT is
    // cell * 0.125 + 0.5/512 + (0.125 - 1/512) * rg (texel-center inset);
    // generalized to the actual LUT width and scaled to pixels.
    float scale = float(lw); // 512 for standard LUTs
    float2 pos0 = (cell0 * 0.125 + 0.5 / scale + (0.125 - 1.0 / scale) * c.rg) * scale;
    float2 pos1 = (cell1 * 0.125 + 0.5 / scale + (0.125 - 1.0 / scale) * c.rg) * scale;

    float4 n0 = lut_bilinear(lut, pos0, lw, lh);
    float4 n1 = lut_bilinear(lut, pos1, lw, lh);
    float3 mapped = lerp(n0, n1, frac(b)).rgb;

    float3 out_rgb = lerp(c.rgb, mapped, clamp(intensity, 0.0, 1.0));
    dst[coord] = float4(out_rgb, c.a);
}
