// shaders/ops/color_aug.slang
// Augmentation color-op parity set, pointwise half: solarize, posterize,
// grayscale, normalize (see ops/color_aug.py; histogram-based equalize /
// autocontrast live in hist_color.slang).

void solarize(
    uint2               coord,
    Texture2D<float4>   src,
    [format(SLANGIMAGE_FMT)] RWTexture2D<float4> dst,
    float               threshold
)
{
    uint width, height;
    src.GetDimensions(width, height);
    if (coord.x >= width || coord.y >= height)
        return;

    float4 c = src[coord];
    float3 o = float3(
        c.r >= threshold ? 1.0 - c.r : c.r,
        c.g >= threshold ? 1.0 - c.g : c.g,
        c.b >= threshold ? 1.0 - c.b : c.b);
    dst[coord] = float4(o, c.a);
}

void posterize(
    uint2               coord,
    Texture2D<float4>   src,
    [format(SLANGIMAGE_FMT)] RWTexture2D<float4> dst,
    int                 bits      // 1..8: keep the top `bits` of uint8
)
{
    uint width, height;
    src.GetDimensions(width, height);
    if (coord.x >= width || coord.y >= height)
        return;

    float4 c = src[coord];
    // PIL semantics on the 8-bit scale: round to uint8, mask low bits
    float step = float(1u << uint(8 - bits));
    float3 v = floor(clamp(c.rgb, 0.0, 1.0) * 255.0 + 0.5);
    v = v - fmod(v, step);
    dst[coord] = float4(v / 255.0, c.a);
}

void grayscale(
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
    // ITU-R 601 luma — same weights as kornia / torchvision rgb_to_grayscale
    float y = dot(c.rgb, float3(0.299, 0.587, 0.114));
    dst[coord] = float4(y, y, y, c.a);
}

void normalize(
    uint2               coord,
    Texture2D<float4>   src,
    [format(SLANGIMAGE_FMT)] RWTexture2D<float4> dst,
    float3              mean,
    float3              std
)
{
    uint width, height;
    src.GetDimensions(width, height);
    if (coord.x >= width || coord.y >= height)
        return;

    float4 c = src[coord];
    // output is intentionally unclamped (training-tensor convention);
    // readback is lossless, so negative values survive
    dst[coord] = float4((c.rgb - mean) / std, c.a);
}
