// shaders/ops/smooth_threshold.slang
// Luminance threshold kernels:
//   smooth_threshold    — smoothstep around threshold
//   smooth_threshold_bw — linear ramp black..white

void smooth_threshold(
    uint2               coord,
    Texture2D<float4>   src,
    [format(SLANGIMAGE_FMT)] RWTexture2D<float4> dst,
    float               threshold,
    float               smooth_amount
)
{
    uint width, height;
    src.GetDimensions(width, height);
    if (coord.x >= width || coord.y >= height)
        return;

    float4 c = src[coord];
    // luminance weights (0.2125, 0.7154, 0.0721)
    float luminance = dot(c.rgb, float3(0.2125, 0.7154, 0.0721));
    float lo = max(threshold - smooth_amount, 0.0);
    float hi = min(threshold + smooth_amount, 1.0);
    float t = smoothstep(lo, hi, luminance);

    dst[coord] = float4(t, t, t, c.a);
}

void smooth_threshold_bw(
    uint2               coord,
    Texture2D<float4>   src,
    [format(SLANGIMAGE_FMT)] RWTexture2D<float4> dst,
    float               black_threshold,
    float               white_threshold
)
{
    uint width, height;
    src.GetDimensions(width, height);
    if (coord.x >= width || coord.y >= height)
        return;

    float4 c = src[coord];
    // Rec. 601 luma: brightness = 0.299 r + 0.587 g + 0.114 b
    float brightness = dot(c.rgb, float3(0.299, 0.587, 0.114));

    float v;
    if (brightness < black_threshold)
        v = 0.0;
    else if (brightness > white_threshold)
        v = 1.0;
    else
        // Host wrappers require a positive, finite, normal fp32 gap.
        v = (brightness - black_threshold)
          / (white_threshold - black_threshold);

    dst[coord] = float4(v, v, v, c.a);
}
