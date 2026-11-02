// shaders/ops/kuwahara.slang
// Generalized Kuwahara filter, 8 sectors around each pixel.
// Operates fully in RGB space: sector means are RGB, only variance is scalar.
//
// Input:  src  -> RGB image (any working space, typically sRGB or linear)
// Output: dst  -> RGB Kuwahara smoothed result
//
// Two accumulation strategies, selected per target (measured on L40S +
// M3 Max at 3456^2 fp16, r=6 — see ROADMAP Track P, CUDA-vs-Vulkan gap):
//  - named float4 registers + divergence-free branch: CUDA 4.6 ms,
//    Vulkan 3.9 ms (dynamically-indexed local arrays go to local memory
//    on these targets: 18.4 / 8.1 ms)
//  - fixed arrays + dynamic index: Metal 62 ms (the register variant
//    tanks M3 occupancy: 84 ms)

import "fastmath.slang";

static const float KUWAHARA_EPS = 1e-5;
static const float PI = 3.14159265;

// Sector index for an integer offset — exactly equivalent to
// floor((atan2(dy, dx) + PI) / (PI/4)) & 7, verified exhaustively for all
// offsets up to radius 64, but pure comparisons: the atan2-per-tap was a
// large term in the CUDA-vs-Vulkan kuwahara gap (NVRTC compiles atan2 as
// a big precise software routine; Vulkan uses a fast approx).
int kuwahara_sector(int dx, int dy)
{
    int a = abs(dx);
    int b = abs(dy);
    if (dy > 0)
        return dx > 0 ? (a > b ? 4 : 5) : (b > a ? 6 : 7);
    if (dy < 0)
        return dx < 0 ? (a > b ? 0 : 1) : (b > a ? 2 : 3);
    return dx < 0 ? 0 : 4;
}

// Combine sector statistics: mean RGB weighted by inverse variance.
float4 kuwahara_combine(
    float4 sums[8], float lums[8], float lum2[8],
    float varPower, float4 center)
{
    float variances[8];
    float minVariance = 3.402823466e+38;

    [ForceUnroll]
    for (int i = 0; i < 8; ++i)
    {
        float wsum = sums[i].w;
        variances[i] = KUWAHARA_EPS;
        if (wsum <= KUWAHARA_EPS)
            continue;

        float invW = 1.0 / wsum;

        float mean_l  = lums[i] * invW;
        float mean_l2 = lum2[i] * invW;

        float var_l = mean_l2 - mean_l * mean_l;
        var_l = (var_l > KUWAHARA_EPS) ? var_l : KUWAHARA_EPS;

        variances[i] = var_l;
        minVariance = min(minVariance, var_l);
    }

    float4 de_rgb = float4(0.0);
    float  nu     = 0.0;
    [ForceUnroll]
    for (int i = 0; i < 8; ++i)
    {
        float wsum = sums[i].w;
        if (wsum <= KUWAHARA_EPS)
            continue;

        // Scale every inverse-variance weight by minVariance^varPower.
        // Ratios are unchanged, but weights are <=1 and at least one is 1;
        // no overflowing raw weights or subnormal normalization reciprocal.
        float w = fast_pow(minVariance / variances[i], varPower);
        float invW = 1.0 / wsum;

        de_rgb += sums[i] * (invW * w);
        nu     += w;
    }

    // no float3 temporaries / float4(x.rgb, y) constructors — either
    // poisons NVRTC codegen for the whole kernel (experiments/cuda_hotloop)
    float4 outp = (nu > KUWAHARA_EPS) ? de_rgb * (1.0 / nu) : center;
    outp.a = center.a;
    return outp;
}

// CUDA/Vulkan variant: per-sector accumulators as NAMED float4 registers
// (rgb + weight in .w) + scalar luma sums. Two hot-loop rules meet here:
//  - no float3 arithmetic (slang emits it through a struct NVRTC won't
//    keep in registers; float4 stays vectorized — measured 3x)
//  - no dynamically-indexed local arrays (forced to local memory). The
//    sector index is uniform per loop iteration (it depends only on the
//    offset), so the branch below is divergence-free.
float4 kuwahara_regs(
    uint2 coord, Texture2D<float4> src,
    int radius, float sigmaSpatial, float varPower,
    uint width, uint height)
{
    float4 s0 = float4(0.0), s1 = float4(0.0), s2 = float4(0.0), s3 = float4(0.0);
    float4 s4 = float4(0.0), s5 = float4(0.0), s6 = float4(0.0), s7 = float4(0.0);
    float l0 = 0.0, l1 = 0.0, l2 = 0.0, l3 = 0.0, l4 = 0.0, l5 = 0.0, l6 = 0.0, l7 = 0.0;
    float m0 = 0.0, m1 = 0.0, m2 = 0.0, m3 = 0.0, m4 = 0.0, m5 = 0.0, m6 = 0.0, m7 = 0.0;

    // spatial gaussian: exp(-dist / (sigma * size)) with the divisor hoisted
    float neg_inv_ss = -1.0 / (sigmaSpatial * float(radius));

    for (int dy = -radius; dy <= radius; ++dy)
    {
        for (int dx = -radius; dx <= radius; ++dx)
        {
            int sx = clamp(int(coord.x) + dx, 0, int(width)  - 1);
            int sy = clamp(int(coord.y) + dy, 0, int(height) - 1);

            float g = fast_exp(float(dx * dx + dy * dy) * neg_inv_ss);
            int sector = kuwahara_sector(dx, dy);

            float4 texel = src[uint2(sx, sy)];

            // scalar signal for variance (was length(rgb) on a float3)
            float luma2 = texel.r * texel.r + texel.g * texel.g + texel.b * texel.b;
            float luma  = sqrt(luma2);

            float4 c4 = g * float4(texel.rgb, 1.0);
            float gl = g * luma;
            float gm = g * luma2;
            if      (sector == 0) { s0 += c4; l0 += gl; m0 += gm; }
            else if (sector == 1) { s1 += c4; l1 += gl; m1 += gm; }
            else if (sector == 2) { s2 += c4; l2 += gl; m2 += gm; }
            else if (sector == 3) { s3 += c4; l3 += gl; m3 += gm; }
            else if (sector == 4) { s4 += c4; l4 += gl; m4 += gm; }
            else if (sector == 5) { s5 += c4; l5 += gl; m5 += gm; }
            else if (sector == 6) { s6 += c4; l6 += gl; m6 += gm; }
            else                  { s7 += c4; l7 += gl; m7 += gm; }
        }
    }

    float4 sums[8] = { s0, s1, s2, s3, s4, s5, s6, s7 };
    float  lums[8] = { l0, l1, l2, l3, l4, l5, l6, l7 };
    float  lum2[8] = { m0, m1, m2, m3, m4, m5, m6, m7 };
    return kuwahara_combine(sums, lums, lum2, varPower, src[coord]);
}

// Metal variant: fixed arrays with dynamic sector index — M3 handles the
// private arrays fine, while 40 live named registers tank its occupancy.
float4 kuwahara_array(
    uint2 coord, Texture2D<float4> src,
    int radius, float sigmaSpatial, float varPower,
    uint width, uint height)
{
    float4 sums[8];
    float  lums[8];
    float  lum2[8];
    [ForceUnroll]
    for (int k = 0; k < 8; ++k)
    {
        sums[k] = float4(0.0);
        lums[k] = 0.0;
        lum2[k] = 0.0;
    }

    float neg_inv_ss = -1.0 / (sigmaSpatial * float(radius));

    for (int dy = -radius; dy <= radius; ++dy)
    {
        for (int dx = -radius; dx <= radius; ++dx)
        {
            int sx = clamp(int(coord.x) + dx, 0, int(width)  - 1);
            int sy = clamp(int(coord.y) + dy, 0, int(height) - 1);

            float g = fast_exp(float(dx * dx + dy * dy) * neg_inv_ss);
            int sector = kuwahara_sector(dx, dy);

            float4 texel = src[uint2(sx, sy)];
            float luma2 = texel.r * texel.r + texel.g * texel.g + texel.b * texel.b;
            float luma  = sqrt(luma2);

            sums[sector] += g * float4(texel.rgb, 1.0);
            lums[sector] += g * luma;
            lum2[sector] += g * luma2;
        }
    }

    return kuwahara_combine(sums, lums, lum2, varPower, src[coord]);
}

// 8 sector generalized Kuwahara in RGB
void generalized_kuwahara(
    uint2               coord,         // from spy.grid
    Texture2D<float4>   src,           // read only
    [format(SLANGIMAGE_FMT)] RWTexture2D<float4> dst,           // write
    int                 radius,        // kernel radius (kSize)
    float               sigmaSpatial,  // spatial sigma used in gauss
    float               varPower       // exponent on variance (eg 7.0)
)
{
    uint width, height;
    src.GetDimensions(width, height);

    if (coord.x >= width || coord.y >= height)
        return;

    __target_switch
    {
    case metal:
        dst[coord] = kuwahara_array(coord, src, radius, sigmaSpatial, varPower, width, height);
        break;
    default:
        dst[coord] = kuwahara_regs(coord, src, radius, sigmaSpatial, varPower, width, height);
        break;
    }
}
