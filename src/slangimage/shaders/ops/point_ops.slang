// shaders/ops/point_ops.slang
// Trivial per-pixel kernels:
// invert, extract_channel, clamp_multiply, high_pass_contrast, random_noise.

void invert(
    uint2               coord,
    Texture2D<float4>   src,
    [format(SLANGIMAGE_FMT)] RWTexture2D<float4> dst
)
{
    uint width, height;
    src.GetDimensions(width, height);
    if (coord.x >= width || coord.y >= height)
        return;

    float4 c = src[coord];
    // invert RGB, preserve alpha (consistent with the other per-pixel ops)
    dst[coord] = float4(1.0 - c.rgb, c.a);
}

void extract_channel(
    uint2               coord,
    Texture2D<float4>   src,
    [format(SLANGIMAGE_FMT)] RWTexture2D<float4> dst,
    int                 channel   // 0=r, 1=g, 2=b
)
{
    uint width, height;
    src.GetDimensions(width, height);
    if (coord.x >= width || coord.y >= height)
        return;

    float4 c = src[coord];
    float v = (channel == 0) ? c.r : (channel == 1) ? c.g : c.b;
    dst[coord] = float4(v, v, v, c.a);
}

void clamp_multiply(
    uint2               coord,
    Texture2D<float4>   src,
    Texture2D<float4>   src2,
    [format(SLANGIMAGE_FMT)] RWTexture2D<float4> dst
)
{
    uint width, height;
    src.GetDimensions(width, height);
    if (coord.x >= width || coord.y >= height)
        return;

    uint w2, h2;
    src2.GetDimensions(w2, h2);
    uint2 c2 = uint2(min(coord.x, w2 - 1), min(coord.y, h2 - 1));

    float3 a = clamp(src[coord].rgb, 0.0, 1.0);
    float3 b = clamp(src2[c2].rgb, 0.0, 1.0);
    dst[coord] = float4(a * b, src[coord].a);   // preserve first input's alpha
}

void high_pass_contrast(
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

    float3 a = src[coord].rgb;
    float3 b = blur[coord].rgb;
    // strength*(a/2 - b/2) + 0.5, preserving the source alpha
    dst[coord] = float4(strength * (a * 0.5 - b * 0.5) + 0.5, src[coord].a);
}

void random_noise(
    uint2               coord,
    [format(SLANGIMAGE_FMT)] RWTexture2D<float4> dst,
    float               random_num
)
{
    uint width, height;
    dst.GetDimensions(width, height);
    if (coord.x >= width || coord.y >= height)
        return;

    // classic fract(sin(dot)) hash; keep math in fp32 regardless of storage
    float2 uv = float2(coord);
    float a = 12.9898 + random_num;
    float c = 151.7182;
    float d = 43758.5453;
    float dt = dot(uv, float2(a, c));
    float sn = fmod(dt, 3.14);
    float noise = frac(sin(sn) * d);
    noise = lerp(0.5, noise, 0.9);

    dst[coord] = float4(noise, noise, noise, 1.0);
}
