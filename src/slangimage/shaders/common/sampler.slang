inline float4 sample_bilinear_tex(Texture2D<float4> src, float2 pos_pix)
{
    uint w, h;
    src.GetDimensions(w, h);

    // Integer coordinates are centers
    float2 base = floor(pos_pix);
    float2 frac = pos_pix - base;

    int2 p00 = int2(base);
    int2 p10 = p00 + int2(1, 0);
    int2 p01 = p00 + int2(0, 1);
    int2 p11 = p00 + int2(1, 1);

    p00.x = clamp(p00.x, 0, int(w) - 1);
    p10.x = clamp(p10.x, 0, int(w) - 1);
    p01.x = clamp(p01.x, 0, int(w) - 1);
    p11.x = clamp(p11.x, 0, int(w) - 1);

    p00.y = clamp(p00.y, 0, int(h) - 1);
    p10.y = clamp(p10.y, 0, int(h) - 1);
    p01.y = clamp(p01.y, 0, int(h) - 1);
    p11.y = clamp(p11.y, 0, int(h) - 1);

    float4 s00 = src[uint2(p00)];
    float4 s10 = src[uint2(p10)];
    float4 s01 = src[uint2(p01)];
    float4 s11 = src[uint2(p11)];

    float4 x0 = lerp(s00, s10, frac.x);
    float4 x1 = lerp(s01, s11, frac.x);
    return lerp(x0, x1, frac.y);
}