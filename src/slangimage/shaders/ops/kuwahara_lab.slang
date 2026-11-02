// shaders/ops/kuwahara_lab.slang
// Blend Kuwahara luminance in Lab space to avoid color distortion.

import "color_conversions.slang";
import "lab.slang";

void kuwahara_lab_blend(
    uint2               coord,
    Texture2D<float4>   src,        // original image, sRGB in [0,1]
    Texture2D<float4>   kuw,        // Kuwahara-smoothed RGB result
    [format(SLANGIMAGE_FMT)] RWTexture2D<float4> dst,        // output
    float               strength    // 0..1, how much to push toward Kuwahara L
)
{
    uint width, height;
    src.GetDimensions(width, height);

    if (coord.x >= width || coord.y >= height)
        return;

    float4 c_src  = src[coord];
    float4 c_kuw  = kuw[coord];

    float3 lab     = rgb_to_lab(saturate(c_src.rgb));

    // Push only the LUMINANCE toward the Kuwahara result's actual luminance
    // (Lab L of the smoothed RGB — NOT its red channel), keeping the original
    // a*/b* so a constant color stays put and only luminance is stylized.
    float kuw_L = rgb_to_lab(saturate(c_kuw.rgb)).x;

    // Blend L channel
    float k = clamp(strength, 0.0, 1.0);
    lab.x = lerp(lab.x, kuw_L, k);

    float3 rgb_out = lab_to_rgb(lab);

    dst[coord] = float4(rgb_out, c_src.a);
}