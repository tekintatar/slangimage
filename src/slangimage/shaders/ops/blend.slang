// shaders/ops/blend.slang
// Simple blend kernels for effects.

float3 screen_blend(float3 base, float3 blend)
{
    return 1.0 - (1.0 - base) * (1.0 - blend);
}

// Soft glow: mix original with screen(original, blurred)
void soft_glow(
    uint2               coord,
    Texture2D<float4>   src,
    Texture2D<float4>   blur,
    [format(SLANGIMAGE_FMT)] RWTexture2D<float4> dst,
    float               strength
)
{
    uint width, height;
    src.GetDimensions(width, height);

    if (coord.x >= width || coord.y >= height)
        return;

    float4 c_src  = src[coord];
    float4 c_blur = blur[coord];

    float3 base   = c_src.rgb;
    float3 glow   = screen_blend(base, c_blur.rgb);

    float k = clamp(strength, 0.0, 1.0);
    float3 out_rgb = lerp(base, glow, k);

    float  out_a = c_src.a; // keep original alpha

    dst[coord] = float4(out_rgb, out_a);
}