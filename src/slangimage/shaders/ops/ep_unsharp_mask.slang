// shaders/ops/ep_unsharp_mask.slang
// Edge Preserving Unsharp Mask (EP-USM)

import "color_conversions.slang";

void ep_unsharp_mask(
    uint2               tid,
    Texture2D<float4>   src,
    Texture2D<float4>   smooth,
    [format(SLANGIMAGE_FMT)] RWTexture2D<float4> dst,
    float               amount
)
{
    uint w, h;
    src.GetDimensions(w, h);
    if (tid.x >= w || tid.y >= h)
        return;

    float4 s    = src[tid];
    float4 sm   = smooth[tid];

    // convert to YCbCr
    float3 srcYCC = srgb_to_ycbcr(s.rgb);
    float3 smoothYCC = srgb_to_ycbcr(sm.rgb);

    float Y   = srcYCC.x;
    float Y_s = smoothYCC.x;

    float Cb = srcYCC.y;
    float Cr = srcYCC.z;

    // Edge-preserving detail
    float detail = Y - Y_s;

    float Y_sharp = Y + amount * detail;
    Y_sharp = clamp(Y_sharp, 0.0, 1.0);

    float3 outRGB = ycbcr_to_srgb(float3(Y_sharp, Cb, Cr));
    dst[tid] = float4(outRGB, s.a);
}