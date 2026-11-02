import "color_conversions.slang";

void unsharp_mask(
    uint2 tid,
    Texture2D<float4> src,
    Texture2D<float4> blur,
    [format(SLANGIMAGE_FMT)] RWTexture2D<float4> dst,
    float amount
){
    uint w, h;
    src.GetDimensions(w, h);
    if (tid.x >= w || tid.y >= h) return;

    float4 s = src[tid];
    float4 b = blur[tid];

    // Classic RGB-space unsharp mask: out = s + amount * (s - blur)
    float3 delta = s.rgb - b.rgb;

    float3 sharp = s.rgb + amount * delta;

    // clamp only at the end
    sharp = clamp(sharp, 0.0, 1.0);

    dst[tid] = float4(sharp, s.a);
}

void unsharp_mask_luma(
    uint2 tid,
    Texture2D<float4> src,
    Texture2D<float4> blurred,
    [format(SLANGIMAGE_FMT)] RWTexture2D<float4> dst,
    float amount, // Y' = lerp(Y_blur, Y, 1 + amount)
)
{
    uint w, h;
    src.GetDimensions(w, h);
    if (tid.x >= w || tid.y >= h)
        return;

    float4 src_rgba = src[tid];
    float4 blur_rgba = blurred[tid];

    // Convert both to YCbCr (Y behaves like LAB L because converted from linear)
    float3 src_ycc = srgb_to_ycbcr(src_rgba.rgb);
    float3 blur_ycc = srgb_to_ycbcr(blur_rgba.rgb);

    float Y = src_ycc.x;
    float Y_b = blur_ycc.x;

    float Cb = src_ycc.y;
    float Cr = src_ycc.z;

    // Sharpen luma only: Y' = lerp(Y_blur, Y, 1 + amount)
    //                       = Y + amount * (Y - Y_blur), clamped to [0, 1].
    float t = 1.0 + amount;
    float Y_sharp = lerp(Y_b, Y, t);
    Y_sharp = clamp(Y_sharp, 0.0, 1.0);

    float3 out_rgb = ycbcr_to_srgb(float3(Y_sharp, Cb, Cr));
    out_rgb = saturate(out_rgb);

    dst[tid] = float4(out_rgb, src_rgba.a);
}