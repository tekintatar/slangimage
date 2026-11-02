// experiments/d0_diff_gaussian/diff_gaussian.slang
// D0 spike: differentiable ops via Slang autodiff.

import slangpy;   // ITensor & friends

// stage 1: plumbing sanity — per-element polynomial (slangpy docs pattern)
[Differentiable]
float polynomial(float a, float b, float c, float x)
{
    return a * x * x + b * x + c;
}

// stage 2: neighborhood op — 1D horizontal gaussian over a 2D tensor.
// Reads are differentiable; Slang generates the backward (scatter into
// src gradients). Weights are data-independent (no_diff math on dx/sigma).
[Differentiable]
float gaussian_h(
    no_diff int2 pix,             // pix.x = col, pix.y = row (spy.grid convention)
    IDiffTensor<float, 2> src,    // H x W
    no_diff int radius,
    no_diff float sigma,
    no_diff int width
)
{
    float two_sigma2 = 2.0 * sigma * sigma;
    float acc = 0.0;
    float wsum = 0.0;

    [MaxIters(81)]   // autodiff needs a loop bound: radius <= 40
    for (int dx = -radius; dx <= radius; ++dx)
    {
        int q = clamp(pix.x + dx, 0, width - 1);
        float w = exp(-float(dx * dx) / two_sigma2);
        acc += w * src[uint2(q, pix.y)];
        wsum += w;
    }
    return acc / wsum;
}

// variant: index via call_id generator instead of explicit grid
[Differentiable]
float gaussian_h_cid(
    no_diff uint2 pix,
    IDiffTensor<float, 2> src,
    no_diff int radius,
    no_diff float sigma,
    no_diff int width
)
{
    float two_sigma2 = 2.0 * sigma * sigma;
    float acc = 0.0;
    float wsum = 0.0;
    [MaxIters(81)]
    for (int dx = -radius; dx <= radius; ++dx)
    {
        int q = clamp(int(pix.x) + dx, 0, width - 1);
        float w = exp(-float(dx * dx) / two_sigma2);
        acc += w * src[uint2(q, pix.y)];
        wsum += w;
    }
    return acc / wsum;
}

// -------- workaround path: hand-written backward, normal dispatch --------
// grad_src[i,j] = sum over output pixels p in [j-r, j+r] of og[i,p] *
// w_eff(p, j) / wsum, where w_eff folds clamp-padding (taps that clamped
// onto column j). Dispatched as a regular kernel — no .bwds() plumbing.
float gaussian_h_bwd(
    uint2 pix,                    // pix.x = col, pix.y = row
    Tensor<float, 2> out_grad,    // upstream gradient, H x W
    int radius,
    float sigma,
    int width
)
{
    float two_sigma2 = 2.0 * sigma * sigma;

    float wsum = 0.0;
    for (int dx = -radius; dx <= radius; ++dx)
        wsum += exp(-float(dx * dx) / two_sigma2);

    int j = int(pix.x);
    float acc = 0.0;
    for (int p = j - radius; p <= j + radius; ++p)
    {
        if (p < 0 || p >= width)
            continue;
        float w_eff = 0.0;
        for (int dx = -radius; dx <= radius; ++dx)
        {
            int target = clamp(p + dx, 0, width - 1);
            if (target == j)
                w_eff += exp(-float(dx * dx) / two_sigma2);
        }
        acc += w_eff * out_grad[uint2(uint(p), pix.y)];
    }
    return acc / wsum;
}
