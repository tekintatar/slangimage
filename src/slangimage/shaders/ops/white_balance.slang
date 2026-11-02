// shaders/ops/white_balance.slang
// Photographic white balance: applies a 3x3 chromatic-adaptation matrix
// in LINEAR RGB. The matrix (Bradford CAT from the assumed illuminant —
// Kelvin temperature + green/magenta tint — to the 6500K daylight
// neutral) is computed on the CPU per call: see kernels/white_balance.py
// for the color science.

import "color_conversions.slang";

void white_balance(
    uint2               coord,
    Texture2D<float4>   src,
    [format(SLANGIMAGE_FMT)] RWTexture2D<float4> dst,
    float3              row0,     // rows of the adaptation matrix
    float3              row1,     //   (linear-RGB to linear-RGB)
    float3              row2
)
{
    uint width, height;
    src.GetDimensions(width, height);
    if (coord.x >= width || coord.y >= height)
        return;

    float4 c = src[coord];
    float3 lin = srgb_to_linear(saturate(c.rgb));

    float3 adapted = float3(dot(row0, lin), dot(row1, lin), dot(row2, lin));

    // adaptation can push channels slightly out of gamut; clamp before
    // the sRGB encode (display-referred op — documented, unlike readback)
    float3 outp = linear_to_srgb(saturate(adapted));
    dst[coord] = float4(outp, c.a);
}
