// shaders/ops/nl_means.slang
// Non-local means denoising (Buades et al.), patch-based.
//
// weight(p, q) = exp(-max(patchDist2 - 2*sigma^2, 0) / h^2)
// patchDist2 = mean squared RGB difference over the patch.
// The center pixel uses the max neighbor weight (standard trick to avoid
// self over-weighting). Cost is O((2S+1)^2 * (2P+1)^2) per pixel — this is
// the expensive, quality denoiser; keep search/patch radii modest.

void nl_means(
    uint2               coord,
    Texture2D<float4>   src,
    [format(SLANGIMAGE_FMT)] RWTexture2D<float4> dst,
    int                 search_radius,  // S, e.g. 5
    int                 patch_radius,   // P, e.g. 1
    float               h,              // filtering strength, e.g. 0.1
    float               sigma           // noise std estimate (0 = off)
)
{
    uint width, height;
    src.GetDimensions(width, height);
    if (coord.x >= width || coord.y >= height)
        return;

    int W = int(width);
    int H = int(height);
    int cx = int(coord.x);
    int cy = int(coord.y);

    float patch_n = float((2 * patch_radius + 1) * (2 * patch_radius + 1)) * 3.0;
    float inv_h2 = 1.0 / max(h * h, 1e-8);
    float two_sigma2 = 2.0 * sigma * sigma;

    // float4/scalar math only in the hot loops (experiments/cuda_hotloop:
    // float3 arithmetic is ~3x slower on the CUDA path)
    float4 accum = float4(0.0);
    float  n = 0.0;
    float  wmax = 0.0;

    for (int sy = -search_radius; sy <= search_radius; ++sy)
    {
        for (int sx = -search_radius; sx <= search_radius; ++sx)
        {
            if (sx == 0 && sy == 0)
                continue;

            // mean squared patch distance
            float d2 = 0.0;
            for (int py = -patch_radius; py <= patch_radius; ++py)
            {
                for (int px = -patch_radius; px <= patch_radius; ++px)
                {
                    int ax = clamp(cx + px, 0, W - 1);
                    int ay = clamp(cy + py, 0, H - 1);
                    int bx = clamp(cx + sx + px, 0, W - 1);
                    int by = clamp(cy + sy + py, 0, H - 1);
                    float4 a = src[uint2(ax, ay)];
                    float4 b = src[uint2(bx, by)];
                    float dr = a.r - b.r;
                    float dg = a.g - b.g;
                    float db = a.b - b.b;
                    d2 += dr * dr + dg * dg + db * db;
                }
            }
            d2 /= patch_n;

            float w = exp(-max(d2 - two_sigma2, 0.0) * inv_h2);
            wmax = max(wmax, w);

            int qx = clamp(cx + sx, 0, W - 1);
            int qy = clamp(cy + sy, 0, H - 1);
            accum += src[uint2(qx, qy)] * w;
            n += w;
        }
    }

    // center pixel with max neighbor weight
    float4 center = src[coord];
    float wc = max(wmax, 1e-6);
    accum += center * wc;
    n += wc;

    // lane assignment, not float4(accum.rgb/n, a) — the .rgb swizzle
    // materializes a float3 and poisons NVRTC codegen
    float4 outp = accum * (1.0 / n);
    outp.a = center.a;
    dst[coord] = outp;
}
