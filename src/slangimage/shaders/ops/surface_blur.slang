// shaders/ops/surface_blur.slang
// Bilateral surface blur. `radius` is a sample count, not a pixel
// half-width: the window spans (radius - 1) / 2 pixels on each side, with a
// spatial gaussian of sigma = half-width and an RGB range gaussian of sigma =
// bsigma. Operates in whatever color space the texture is already in (no
// srgb/linear conversions).

import "fastmath.slang";

static const float PI = 3.14159265;
static const float SURFACE_EPS = 1e-5;

// Per tap, the weight is the product of a range gaussian on the RGB
// difference d = cc - c and separable spatial gaussians in i and j. The
// gaussians' normalization constants are the same for every tap, so they
// cancel in the finalColor / Z ratio. What remains per tap is a single
// exponential: exp(kR * dot(d, d) + kS * (i^2 + j^2)) — one fast_exp
// instead of three precise exps, divisions and sqrts (NVRTC compiles
// those as slow precise routines).

void surface_blur(
    uint2               coord,      // spy.grid((h, w))
    Texture2D<float4>   src,        // read (fp16/fp32)
    [format(SLANGIMAGE_FMT)] RWTexture2D<float4> dst,        // write
    float               radius,     // sample count, not pixel half-width
    float               bsigma      // RGB range sigma
)
{
    uint width, height;
    src.GetDimensions(width, height);

    if (coord.x >= width || coord.y >= height)
        return;

    // NOTE: the loop below uses float4/scalar math ONLY. float3 arithmetic
    // in a CUDA-path hot loop is ~3x slower (slang emits it through a
    // struct that NVRTC won't keep in registers; float4 stays vectorized) —
    // this was the bulk of the CUDA-vs-Vulkan gap on this op.
    float4 center4 = src[coord];

    // Half-width of the window: (sample count - 1) / 2.
    int SAMPLES_COUNT = int(radius);
    int kSize         = (SAMPLES_COUNT - 1) / 2;

    // A one-pixel/empty footprint is identity. Avoid 0 * infinity in the
    // spatial weights and do not depend on a NaN comparison for this path.
    if (kSize <= 0)
    {
        dst[coord] = center4;
        return;
    }

    float4 finalColor = float4(0.0);
    float  Z          = 0.0;

    float kR = -0.5 / (bsigma * bsigma);
    float kS = -0.5 / (float(kSize) * float(kSize));

    for (int j = -kSize; j <= kSize; ++j)
    {
        for (int i = -kSize; i <= kSize; ++i)
        {
            // clamp-to-edge tap
            int sx = clamp(int(coord.x) + i, 0, int(width)  - 1);
            int sy = clamp(int(coord.y) + j, 0, int(height) - 1);

            float4 cc = src[uint2(sx, sy)];
            float dr = cc.r - center4.r;
            float dg = cc.g - center4.g;
            float db = cc.b - center4.b;

            float factor = fast_exp(kR * (dr * dr + dg * dg + db * db)
                                    + kS * float(i * i + j * j));

            Z          += factor;
            finalColor += factor * cc;   // alpha lane rides along, discarded
        }
    }

    // epilogue without float3 temporaries or float4(x.rgb, y) swizzle-
    // constructors — either poisons NVRTC codegen for the whole kernel
    // (measured 2.6x on sharpen_rgb; experiments/cuda_hotloop)
    float4 outp = (Z > SURFACE_EPS) ? finalColor * (1.0 / Z) : center4;
    // keep alpha from the source pixel
    outp.a = center4.a;
    dst[coord] = outp;
}
