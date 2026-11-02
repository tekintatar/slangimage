// shaders/ops/pointillize.slang
// Pointillist stylization: the image is partitioned into jittered cells
// (one paint dab per cell); every pixel takes the color sampled at its
// nearest jittered cell center. Deterministic per (cell, seed).

float2 point_hash2(float2 p, float seed)
{
    // deterministic 2d hash -> [0,1)^2, fp32 math regardless of storage
    float3 a = frac(float3(p.x, p.y, p.x) * float3(0.1031, 0.1030, 0.0973) + seed * 0.113);
    a += dot(a, float3(a.y, a.z, a.x) + 33.33);
    return frac(float2((a.x + a.y) * a.z, (a.x + a.z) * a.y));
}

void pointillize(
    uint2               coord,
    Texture2D<float4>   src,
    [format(SLANGIMAGE_FMT)] RWTexture2D<float4> dst,
    float               radius,   // cell size in pixels (dab radius ~ radius)
    float               jitter,   // 0 = regular grid, 1 = full jitter
    float               seed
)
{
    uint width, height;
    src.GetDimensions(width, height);
    if (coord.x >= width || coord.y >= height)
        return;

    float cell = max(radius, 2.0);
    float2 p = float2(coord) + 0.5;
    float2 pc = p / cell;
    int2 base = int2(floor(pc));

    float best_d2 = 1e30;
    float2 best_center = p;

    for (int j = -1; j <= 1; ++j)
    {
        for (int i = -1; i <= 1; ++i)
        {
            float2 cid = float2(base + int2(i, j));
            float2 off = (point_hash2(cid, seed) - 0.5) * clamp(jitter, 0.0, 1.0);
            float2 center = (cid + 0.5 + off) * cell;
            float2 d = center - p;
            float d2 = dot(d, d);
            if (d2 < best_d2)
            {
                best_d2 = d2;
                best_center = center;
            }
        }
    }

    int sx = clamp(int(best_center.x), 0, int(width) - 1);
    int sy = clamp(int(best_center.y), 0, int(height) - 1);
    float4 c = src[uint2(sx, sy)];

    dst[coord] = float4(c.rgb, src[coord].a);
}
