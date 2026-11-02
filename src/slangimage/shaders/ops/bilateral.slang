// shaders/ops/bilateral.slang
// Classic bilateral filter, full 2D and separable 1D variants.
//
// Spatial weighting: gaussian(sigma_spatial), or box when sigma_spatial <= 0.
// Range weighting (range_type): 0 = gaussian(sigma_range) on RGB distance,
//                               1 = hard cutoff (weight 1 if |dc| <= sigma_range)
// The cutoff mode matches the common "epsilon filter" / fast-bilateral
// formulation (box spatial + threshold range) when sigma_spatial <= 0.
// (cutoff compares squared distance against sigma_range^2 — identical to
// the |dc| <= sigma_range formulation, without the sqrt.)
//
// Hot loops follow the CUDA-path rules (experiments/cuda_hotloop):
// float4/scalar math only (no float3 — ~3x on CUDA), loop-invariant
// divisions hoisted.

void bilateral(
    uint2               coord,
    Texture2D<float4>   src,
    [format(SLANGIMAGE_FMT)] RWTexture2D<float4> dst,
    int                 radius,
    float               sigma_spatial,
    float               sigma_range,
    int                 range_type
)
{
    uint width, height;
    src.GetDimensions(width, height);
    if (coord.x >= width || coord.y >= height)
        return;

    float4 center = src[coord];

    bool  box = sigma_spatial <= 0.0;
    float neg_half_inv_ss2 = box ? 0.0 : -0.5 / (sigma_spatial * sigma_spatial);
    float neg_half_inv_sr2 = -0.5 / (sigma_range * sigma_range);
    float sr2 = sigma_range * sigma_range;

    float4 accum = float4(0.0);
    float  n = 0.0;

    for (int j = -radius; j <= radius; ++j)
    {
        for (int i = -radius; i <= radius; ++i)
        {
            int sx = clamp(int(coord.x) + i, 0, int(width) - 1);
            int sy = clamp(int(coord.y) + j, 0, int(height) - 1);
            float4 s = src[uint2(sx, sy)];

            float dr = s.r - center.r;
            float dg = s.g - center.g;
            float db = s.b - center.b;
            float d2 = dr * dr + dg * dg + db * db;

            float w_spatial = box ? 1.0
                : exp(float(i * i + j * j) * neg_half_inv_ss2);
            float w_range = (range_type == 1)
                ? ((d2 <= sr2) ? 1.0 : 0.0)
                : exp(d2 * neg_half_inv_sr2);

            float w = w_spatial * w_range;
            accum += s * w;
            n += w;
        }
    }

    // epilogue without float3 temporaries or float4(x.rgb, y) swizzle-
    // constructors — either poisons NVRTC codegen for the whole kernel
    // (measured 2.6x on sharpen_rgb; experiments/cuda_hotloop)
    float4 outp = (n > 1e-6) ? accum * (1.0 / n) : center;
    outp.a = center.a;
    dst[coord] = outp;
}

// One directional pass; run with (dir_x, dir_y) = (1,0) then (0,1) for the
// separable approximation of the bilateral filter.
void bilateral_1d(
    uint2               coord,
    Texture2D<float4>   src,
    [format(SLANGIMAGE_FMT)] RWTexture2D<float4> dst,
    int                 radius,
    int                 dir_x,
    int                 dir_y,
    float               sigma_spatial,
    float               sigma_range,
    int                 range_type
)
{
    uint width, height;
    src.GetDimensions(width, height);
    if (coord.x >= width || coord.y >= height)
        return;

    float4 center = src[coord];

    bool  box = sigma_spatial <= 0.0;
    float neg_half_inv_ss2 = box ? 0.0 : -0.5 / (sigma_spatial * sigma_spatial);
    float neg_half_inv_sr2 = -0.5 / (sigma_range * sigma_range);
    float sr2 = sigma_range * sigma_range;

    float4 accum = float4(0.0);
    float  n = 0.0;

    for (int k = -radius; k <= radius; ++k)
    {
        int sx = clamp(int(coord.x) + k * dir_x, 0, int(width) - 1);
        int sy = clamp(int(coord.y) + k * dir_y, 0, int(height) - 1);
        float4 s = src[uint2(sx, sy)];

        float dr = s.r - center.r;
        float dg = s.g - center.g;
        float db = s.b - center.b;
        float d2 = dr * dr + dg * dg + db * db;

        float w_spatial = box ? 1.0 : exp(float(k * k) * neg_half_inv_ss2);
        float w_range = (range_type == 1)
            ? ((d2 <= sr2) ? 1.0 : 0.0)
            : exp(d2 * neg_half_inv_sr2);

        float w = w_spatial * w_range;
        accum += s * w;
        n += w;
    }

    // epilogue without float3 temporaries or float4(x.rgb, y) swizzle-
    // constructors — either poisons NVRTC codegen for the whole kernel
    // (measured 2.6x on sharpen_rgb; experiments/cuda_hotloop)
    float4 outp = (n > 1e-6) ? accum * (1.0 / n) : center;
    outp.a = center.a;
    dst[coord] = outp;
}
