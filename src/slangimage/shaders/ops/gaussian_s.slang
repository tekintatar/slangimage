// shaders/ops/gaussian_s.slang
// Separable gaussian blur (horizontal + vertical passes, clamp-to-edge).

void gaussian_blur_h(
    uint2               coord,     // from spy.grid
    Texture2D<float4>   src,       // read only, sRGB encoded
    [format(SLANGIMAGE_FMT)] RWTexture2D<float4> dst,       // write, sRGB encoded
    int                 radius,
    float               sigma
)
{
    uint width, height;
    src.GetDimensions(width, height);

    if (coord.x >= width || coord.y >= height)
        return;

    float twoSigma2 = 2.0 * sigma * sigma;

    // single float4 accumulator (float3 loop math is ~3x on CUDA;
    // experiments/cuda_hotloop) — same weight hits all four lanes
    float4 acc  = float4(0.0);
    float  wsum = 0.0;

    // Blur along X only, Y fixed
    for (int dx = -radius; dx <= radius; ++dx)
    {
        int qx = int(coord.x) + dx;
        qx = clamp(qx, 0, int(width) - 1);

        float w = exp(-(dx * dx) / twoSigma2);

        float4 inputImage = src[uint2(qx, coord.y)];


        acc  += w * inputImage;
        wsum += w;
    }

    if (wsum > 0.0)
        acc /= wsum;

    dst[coord] = acc;
}

void gaussian_blur_v(
    uint2               coord,     // from spy.grid
    Texture2D<float4>   src,       // read only, sRGB encoded
    [format(SLANGIMAGE_FMT)] RWTexture2D<float4> dst,       // write, sRGB encoded
    int                 radius,
    float               sigma
)
{
    uint width, height;
    src.GetDimensions(width, height);

    if (coord.x >= width || coord.y >= height)
        return;

    float twoSigma2 = 2.0 * sigma * sigma;

    // single float4 accumulator (float3 loop math is ~3x on CUDA;
    // experiments/cuda_hotloop) — same weight hits all four lanes
    float4 acc  = float4(0.0);
    float  wsum = 0.0;

    // Blur along Y only, X fixed
    for (int dy = -radius; dy <= radius; ++dy)
    {
        int qy = int(coord.y) + dy;
        qy = clamp(qy, 0, int(height) - 1);

        float w = exp(-(dy * dy) / twoSigma2);

        float4 inputImage = src[uint2(coord.x, qy)];

        acc  += w * inputImage;
        wsum += w;
    }

    if (wsum > 0.0)
        acc /= wsum;

    dst[coord] = acc;
}