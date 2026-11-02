// shaders/ops/lab_luma_extract.slang

import "lab.slang";

// Input:  sRGB texture
// Output: R = Lab L in [0,1], GBA unused but we can keep A for convenience

void lab_luma_extract(
    uint2               coord,
    Texture2D<float4>   src,
    [format(SLANGIMAGE_FMT)] RWTexture2D<float4> dst
)
{
    uint w, h;
    src.GetDimensions(w, h);
    if (coord.x >= w || coord.y >= h)
        return;

    float4 input = src[coord];
    float3 lab = rgb_to_lab_packed(input.rgb);      // L_norm in lab.x
    dst[coord] = float4(lab.xxx, input.a); // L replicated to RGB
}