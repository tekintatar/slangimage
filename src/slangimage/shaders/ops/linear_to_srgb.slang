import "color_conversions.slang";

void linear_to_srgb_pass(
    uint2 tid,
    Texture2D<float4> src,
    [format(SLANGIMAGE_FMT)] RWTexture2D<float4> dst
)
{
    uint w, h;
    src.GetDimensions(w, h);
    if (tid.x >= w || tid.y >= h) return;

    float4 c = src[tid];
    float3 srgb = linear_to_srgb(c.rgb);
    dst[tid] = float4(srgb, c.a);
}