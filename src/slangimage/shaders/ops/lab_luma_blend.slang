// shaders/ops/lab_luma_blend.slang

import "lab.slang";

void lab_luma_blend(
    uint2               coord,
    Texture2D<float4>   luma_tex,   // R = L_norm [0,1]
    Texture2D<float4>   src_rgb,    // original sRGB image
    [format(SLANGIMAGE_FMT)] RWTexture2D<float4> dst_rgb
)
{
    uint w, h;
    src_rgb.GetDimensions(w, h);
    if (coord.x >= w || coord.y >= h)
        return;

    float4 src_c4 = src_rgb[coord];
    float  L_norm = luma_tex[coord].r;

    float3 rgb = src_c4.rgb;
    float3 lab = rgb_to_lab_packed(rgb);           // L in [0,100]

    lab.x = L_norm;

    float3 rgb_out = lab_packed_to_rgb(lab);

    dst_rgb[coord] = float4(rgb_out, src_c4.a);
}