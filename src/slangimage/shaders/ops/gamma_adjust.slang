// shaders/ops/gamma_adjust.slang
// Simple gamma adjustment in working (sRGB) space.
//
// power < 1.0  → brightens midtones
// power > 1.0  → darkens midtones
//
// out = rgb^power, alpha passed through.

static const float GAMMA_EPS = 1e-6;

void gamma_adjust(
    uint2               coord,
    Texture2D<float4>   src,
    [format(SLANGIMAGE_FMT)] RWTexture2D<float4> dst,
    float               power
)
{
    uint w, h;
    src.GetDimensions(w, h);

    if (coord.x >= w || coord.y >= h)
        return;

    // Clamp power to a sane range to avoid NaNs
    float p = clamp(power, 0.01, 10.0);

    float4 c  = src[coord];
    float3 rgb = saturate(c.rgb);

    // Apply per channel gamma
    float3 out_rgb = pow(rgb, float3(p, p, p));

    dst[coord] = float4(out_rgb, c.a);
}