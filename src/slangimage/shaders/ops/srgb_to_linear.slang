import "color_conversions.slang";

void srgb_to_linear_pass(
    uint2 tid,
    Texture2D<float4> src,
    [format(SLANGIMAGE_FMT)] RWTexture2D<float4> dst
)
{
    uint w, h;
    src.GetDimensions(w, h);
    if (tid.x >= w || tid.y >= h) return;

    float4 c = src[tid];
    float3 lin = srgb_to_linear(c.rgb);
    dst[tid] = float4(lin, c.a);
}