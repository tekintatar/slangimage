// microbench.slang — isolate the CUDA-vs-Vulkan gap: reads vs math.

void reads49(uint2 coord, Texture2D<float4> src,
             [format(SLANGIMAGE_FMT)] RWTexture2D<float4> dst)
{
    uint w, h; src.GetDimensions(w, h);
    if (coord.x >= w || coord.y >= h) return;
    float4 acc = float4(0.0);
    for (int dy = -3; dy <= 3; ++dy)
        for (int dx = -3; dx <= 3; ++dx)
        {
            uint2 q = uint2(clamp(int(coord.x) + dx, 0, int(w) - 1),
                            clamp(int(coord.y) + dy, 0, int(h) - 1));
            acc += src[q];
        }
    dst[coord] = acc / 49.0;
}

void exp_loop(uint2 coord, Texture2D<float4> src,
              [format(SLANGIMAGE_FMT)] RWTexture2D<float4> dst, int iters)
{
    uint w, h; src.GetDimensions(w, h);
    if (coord.x >= w || coord.y >= h) return;
    float3 c = src[coord].rgb;
    float acc = c.x;
    for (int i = 0; i < iters; ++i)
        acc = exp(-acc - float(i) * 1e-4) + c.y * 1e-3;
    dst[coord] = float4(acc, 0.0, 0.0, 1.0);
}

void pow_loop(uint2 coord, Texture2D<float4> src,
              [format(SLANGIMAGE_FMT)] RWTexture2D<float4> dst, int iters)
{
    uint w, h; src.GetDimensions(w, h);
    if (coord.x >= w || coord.y >= h) return;
    float3 c = src[coord].rgb;
    float acc = c.x + 0.5;
    for (int i = 0; i < iters; ++i)
        acc = pow(abs(acc) + 0.5, 1.7) * 0.31 + c.y * 1e-3;
    dst[coord] = float4(acc, 0.0, 0.0, 1.0);
}

void fma_loop(uint2 coord, Texture2D<float4> src,
              [format(SLANGIMAGE_FMT)] RWTexture2D<float4> dst, int iters)
{
    uint w, h; src.GetDimensions(w, h);
    if (coord.x >= w || coord.y >= h) return;
    float3 c = src[coord].rgb;
    float acc = c.x;
    for (int i = 0; i < iters; ++i)
        acc = acc * 1.000001 + c.y * 1e-6;
    dst[coord] = float4(acc, 0.0, 0.0, 1.0);
}

// same as reads49 but with RUNTIME loop bounds (the real kernels' shape)
void reads_rt(uint2 coord, Texture2D<float4> src,
              [format(SLANGIMAGE_FMT)] RWTexture2D<float4> dst, int radius)
{
    uint w, h; src.GetDimensions(w, h);
    if (coord.x >= w || coord.y >= h) return;
    float4 acc = float4(0.0);
    for (int dy = -radius; dy <= radius; ++dy)
        for (int dx = -radius; dx <= radius; ++dx)
        {
            uint2 q = uint2(clamp(int(coord.x) + dx, 0, int(w) - 1),
                            clamp(int(coord.y) + dy, 0, int(h) - 1));
            acc += src[q];
        }
    dst[coord] = acc / float((2 * radius + 1) * (2 * radius + 1));
}

// runtime bounds + one exp per tap (surface_blur's exact shape)
void surfb_sim(uint2 coord, Texture2D<float4> src,
               [format(SLANGIMAGE_FMT)] RWTexture2D<float4> dst, int radius)
{
    uint w, h; src.GetDimensions(w, h);
    if (coord.x >= w || coord.y >= h) return;
    float3 c = src[coord].rgb;
    float3 acc = float3(0.0);
    float z = 0.0;
    for (int dy = -radius; dy <= radius; ++dy)
        for (int dx = -radius; dx <= radius; ++dx)
        {
            uint2 q = uint2(clamp(int(coord.x) + dx, 0, int(w) - 1),
                            clamp(int(coord.y) + dy, 0, int(h) - 1));
            float3 cc = src[q].rgb;
            float3 d = cc - c;
            float f = exp2((-8.0 * dot(d, d) - 0.03 * float(dx * dx + dy * dy)) * 1.4427);
            z += f;
            acc += f * cc;
        }
    dst[coord] = float4(acc / max(z, 1e-5), 1.0);
}

// CUDA hardware ex2.approx via intrinsic; exp2 elsewhere
float hw_exp2(float x)
{
    __target_switch
    {
    case cuda: __intrinsic_asm "__expf($0)";
    default: return exp(x);
    }
}

void surfb_hw(uint2 coord, Texture2D<float4> src,
              [format(SLANGIMAGE_FMT)] RWTexture2D<float4> dst, int radius)
{
    uint w, h; src.GetDimensions(w, h);
    if (coord.x >= w || coord.y >= h) return;
    float3 c = src[coord].rgb;
    float3 acc = float3(0.0);
    float z = 0.0;
    for (int dy = -radius; dy <= radius; ++dy)
        for (int dx = -radius; dx <= radius; ++dx)
        {
            uint2 q = uint2(clamp(int(coord.x) + dx, 0, int(w) - 1),
                            clamp(int(coord.y) + dy, 0, int(h) - 1));
            float3 cc = src[q].rgb;
            float3 d = cc - c;
            float f = hw_exp2(-8.0 * dot(d, d) - 0.03 * float(dx * dx + dy * dy));
            z += f;
            acc += f * cc;
        }
    dst[coord] = float4(acc / max(z, 1e-5), 1.0);
}

// bisect surfb_sim: same body, NO exp — f is the raw polynomial
void surfb_noexp(uint2 coord, Texture2D<float4> src,
                 [format(SLANGIMAGE_FMT)] RWTexture2D<float4> dst, int radius)
{
    uint w, h; src.GetDimensions(w, h);
    if (coord.x >= w || coord.y >= h) return;
    float3 c = src[coord].rgb;
    float3 acc = float3(0.0);
    float z = 0.0;
    for (int dy = -radius; dy <= radius; ++dy)
        for (int dx = -radius; dx <= radius; ++dx)
        {
            uint2 q = uint2(clamp(int(coord.x) + dx, 0, int(w) - 1),
                            clamp(int(coord.y) + dy, 0, int(h) - 1));
            float3 cc = src[q].rgb;
            float3 d = cc - c;
            float f = -8.0 * dot(d, d) - 0.03 * float(dx * dx + dy * dy);
            z += f;
            acc += f * cc;
        }
    dst[coord] = float4(acc / max(z, 1e-5), 1.0);
}

// bisect: f constant — reads + weighted accumulation only
void surfb_constf(uint2 coord, Texture2D<float4> src,
                  [format(SLANGIMAGE_FMT)] RWTexture2D<float4> dst, int radius)
{
    uint w, h; src.GetDimensions(w, h);
    if (coord.x >= w || coord.y >= h) return;
    float3 acc = float3(0.0);
    float z = 0.0;
    for (int dy = -radius; dy <= radius; ++dy)
        for (int dx = -radius; dx <= radius; ++dx)
        {
            uint2 q = uint2(clamp(int(coord.x) + dx, 0, int(w) - 1),
                            clamp(int(coord.y) + dy, 0, int(h) - 1));
            float3 cc = src[q].rgb;
            float f = 0.5;
            z += f;
            acc += f * cc;
        }
    dst[coord] = float4(acc / max(z, 1e-5), 1.0);
}

// like surfb_constf but load the FULL float4 into a local, swizzle after
void surfb_constf4(uint2 coord, Texture2D<float4> src,
                   [format(SLANGIMAGE_FMT)] RWTexture2D<float4> dst, int radius)
{
    uint w, h; src.GetDimensions(w, h);
    if (coord.x >= w || coord.y >= h) return;
    float3 acc = float3(0.0);
    float z = 0.0;
    for (int dy = -radius; dy <= radius; ++dy)
        for (int dx = -radius; dx <= radius; ++dx)
        {
            uint2 q = uint2(clamp(int(coord.x) + dx, 0, int(w) - 1),
                            clamp(int(coord.y) + dy, 0, int(h) - 1));
            float4 cc4 = src[q];
            float3 cc = cc4.rgb;
            float f = 0.5;
            z += f;
            acc += f * cc;
        }
    dst[coord] = float4(acc / max(z, 1e-5), 1.0);
}

// float4 accumulator, same weighted accumulation
void surfb_f4acc(uint2 coord, Texture2D<float4> src,
                 [format(SLANGIMAGE_FMT)] RWTexture2D<float4> dst, int radius)
{
    uint w, h; src.GetDimensions(w, h);
    if (coord.x >= w || coord.y >= h) return;
    float4 acc = float4(0.0);
    float z = 0.0;
    for (int dy = -radius; dy <= radius; ++dy)
        for (int dx = -radius; dx <= radius; ++dx)
        {
            uint2 q = uint2(clamp(int(coord.x) + dx, 0, int(w) - 1),
                            clamp(int(coord.y) + dy, 0, int(h) - 1));
            float4 cc4 = src[q];
            float f = 0.5;
            z += f;
            acc += f * cc4;
        }
    dst[coord] = float4(acc.rgb / max(z, 1e-5), 1.0);
}

// three scalar accumulators
void surfb_scalar3(uint2 coord, Texture2D<float4> src,
                   [format(SLANGIMAGE_FMT)] RWTexture2D<float4> dst, int radius)
{
    uint w, h; src.GetDimensions(w, h);
    if (coord.x >= w || coord.y >= h) return;
    float ar = 0.0, ag = 0.0, ab = 0.0, z = 0.0;
    for (int dy = -radius; dy <= radius; ++dy)
        for (int dx = -radius; dx <= radius; ++dx)
        {
            uint2 q = uint2(clamp(int(coord.x) + dx, 0, int(w) - 1),
                            clamp(int(coord.y) + dy, 0, int(h) - 1));
            float4 cc4 = src[q];
            float f = 0.5;
            z += f;
            ar += f * cc4.r;
            ag += f * cc4.g;
            ab += f * cc4.b;
        }
    float inv = 1.0 / max(z, 1e-5);
    dst[coord] = float4(ar * inv, ag * inv, ab * inv, 1.0);
}
