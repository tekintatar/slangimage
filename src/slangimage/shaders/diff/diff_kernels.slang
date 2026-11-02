// shaders/diff/diff_kernels.slang
// Differentiable kernels for slangimage.diff (torch autograd, CUDA).
//
// Conventions (probed, see experiments/d0_diff_gaussian/FINDINGS.md):
// - images are (H, W, C) tensors; for grid((H,W,C)): pix.x = channel
//   (fastest dim), pix.y = column, pix.z = row; subscripts pass through
//   as t[uint3(pix.x, pix.y, pix.z)].
// - (H, W, 3) tensors auto-map to per-element float3 functions.
// - In [Differentiable] functions, ALL loops zero-based with an offset —
//   negative runtime loop starts crash backward codegen (slangpy#1051).
//   Plain kernels (the manual-bridge bilateral pair) are exempt.
// - Values are UNCLAMPED: clamps zero gradients at saturation; display
//   clamping is the caller's concern (differs from si.ops semantics).

import slangpy;

// ------------------------------------------------ separable gaussian blur
[Differentiable]
float gaussian_h(
    no_diff int3 pix,                 // x=channel, y=col, z=row
    IDiffTensor<float, 3> src,        // H x W x C
    no_diff int radius,
    no_diff float sigma,
    no_diff int width
)
{
    float two_sigma2 = 2.0 * sigma * sigma;
    float acc = 0.0;
    float wsum = 0.0;
    [MaxIters(129)]                   // radius <= 64
    for (int i = 0; i < 2 * radius + 1; ++i)
    {
        int d = i - radius;
        int q = clamp(pix.y + d, 0, width - 1);
        float w = exp(-float(d * d) / two_sigma2);
        acc += w * src[uint3(pix.x, q, pix.z)];
        wsum += w;
    }
    return acc / wsum;
}

[Differentiable]
float gaussian_v(
    no_diff int3 pix,
    IDiffTensor<float, 3> src,
    no_diff int radius,
    no_diff float sigma,
    no_diff int height
)
{
    float two_sigma2 = 2.0 * sigma * sigma;
    float acc = 0.0;
    float wsum = 0.0;
    [MaxIters(129)]
    for (int i = 0; i < 2 * radius + 1; ++i)
    {
        int d = i - radius;
        int q = clamp(pix.z + d, 0, height - 1);
        float w = exp(-float(d * d) / two_sigma2);
        acc += w * src[uint3(pix.x, pix.y, q)];
        wsum += w;
    }
    return acc / wsum;
}

// ------------------------------------------- per-element color ops (float3)
[Differentiable]
float3 exposure(float3 rgb, no_diff float ev)
{
    return rgb * exp2(ev);
}

[Differentiable]
float3 sepia(float3 rgb, no_diff float intensity)
{
    float3 sep = float3(
        dot(rgb, float3(0.393, 0.769, 0.189)),
        dot(rgb, float3(0.349, 0.686, 0.168)),
        dot(rgb, float3(0.272, 0.534, 0.131)));
    return lerp(rgb, sep, intensity);
}

[Differentiable]
float3 hue_rotate(float3 rgb, no_diff float angle)
{
    // chroma rotation in YIQ; luminance-exact, fully linear -> smooth grads
    float y = dot(rgb, float3(0.299, 0.587, 0.114));
    float i = dot(rgb, float3(0.596, -0.274, -0.322));
    float q = dot(rgb, float3(0.211, -0.523, 0.312));

    float s = sin(angle);
    float c = cos(angle);
    float i2 = i * c - q * s;
    float q2 = i * s + q * c;

    return float3(
        y + 0.956 * i2 + 0.621 * q2,
        y - 0.272 * i2 - 0.647 * q2,
        y - 1.106 * i2 + 1.703 * q2);
}

// ------------------------------- bilateral (manual bridge, PLAIN kernels)
// Differentiable in BOTH the sampled values and the range weights — the
// edge-preservation itself carries gradients. Gaussian spatial only.
//
// Forward + hand-written gather backward, dispatched as plain (non-
// [Differentiable]) kernels and wired into torch.autograd by diff.py.
// Why not the generated backward: Slang autodiff emits an atomic-scatter
// backward that measured ~400x the forward cost (see
// benchmarks/compare_kornia_diff.py), and float3 returns in loops are
// mis-differentiated anyway (slangpy#1055). The gather adjoint below is
// exact, deterministic (no atomics), and runs at ~forward speed. Being
// plain kernels, natural loops and vector math are fine here.
//
// Border: out-of-bounds taps are SKIPPED (truncated window + wsum
// renormalization) rather than clamped like si.ops.bilateral — the
// truncated window keeps the weight matrix symmetric, w(p,q) == w(q,p),
// which is what makes the one-pass gather adjoint possible.

float4 bilateral_fwd(
    int2 pix,                         // x = col, y = row
    Tensor<float, 3> src,             // H x W x 3
    int radius,
    float sigma_spatial,
    float sigma_range,
    int width,
    int height
)
{
    float3 c = float3(src[uint3(0, pix.x, pix.y)],
                      src[uint3(1, pix.x, pix.y)],
                      src[uint3(2, pix.x, pix.y)]);
    float inv2ss = 1.0 / (2.0 * sigma_spatial * sigma_spatial);
    float inv2sr = 1.0 / (2.0 * sigma_range * sigma_range);

    float3 acc = float3(0.0);
    float wsum = 0.0;
    for (int dy = -radius; dy <= radius; ++dy)
    {
        int qy = pix.y + dy;
        if (qy < 0 || qy >= height) continue;
        for (int dx = -radius; dx <= radius; ++dx)
        {
            int qx = pix.x + dx;
            if (qx < 0 || qx >= width) continue;
            float3 s = float3(src[uint3(0, qx, qy)],
                              src[uint3(1, qx, qy)],
                              src[uint3(2, qx, qy)]);
            float3 d = s - c;
            float w = exp(-float(dx * dx + dy * dy) * inv2ss
                          - dot(d, d) * inv2sr);
            acc += w * s;
            wsum += w;
        }
    }
    // wsum rides along in .w — the backward needs it per pixel
    return float4(acc / wsum, wsum);
}

// Gather adjoint. Each thread m sums, over every window partner n, the
// three roles its own value s[m] plays in the forward:
//   value in out[n]:    G[n][k] * w / W[n]
//   center of w(m,n):   A(m,n) * dw/ds[m], A(p,q) = dot(G[p], s[q]-out[p])/W[p]
//   neighbor in w(n,m): A(n,m) * dw/ds[m]
// Both weight derivatives reduce to +w * (s[n]-s[m]) / sigma_range^2, so
// the tap contribution is (w/W[n])*G[n] + (A(m,n)+A(n,m))*(w/sr^2)*d.
float3 bilateral_bwd(
    int2 pix,                         // x = col, y = row (pixel m)
    Tensor<float, 3> src,             // H x W x 3
    Tensor<float, 3> outw,            // H x W x 4: forward rgb + wsum
    Tensor<float, 3> gout,            // H x W x 3: upstream dL/dout
    int radius,
    float sigma_spatial,
    float sigma_range,
    int width,
    int height
)
{
    float3 sm = float3(src[uint3(0, pix.x, pix.y)],
                       src[uint3(1, pix.x, pix.y)],
                       src[uint3(2, pix.x, pix.y)]);
    float3 gm = float3(gout[uint3(0, pix.x, pix.y)],
                       gout[uint3(1, pix.x, pix.y)],
                       gout[uint3(2, pix.x, pix.y)]);
    float3 om = float3(outw[uint3(0, pix.x, pix.y)],
                       outw[uint3(1, pix.x, pix.y)],
                       outw[uint3(2, pix.x, pix.y)]);
    float wm = outw[uint3(3, pix.x, pix.y)];

    float inv2ss = 1.0 / (2.0 * sigma_spatial * sigma_spatial);
    float inv2sr = 1.0 / (2.0 * sigma_range * sigma_range);
    float invsr2 = 1.0 / (sigma_range * sigma_range);

    float3 grad = float3(0.0);
    for (int dy = -radius; dy <= radius; ++dy)
    {
        int qy = pix.y + dy;
        if (qy < 0 || qy >= height) continue;
        for (int dx = -radius; dx <= radius; ++dx)
        {
            int qx = pix.x + dx;
            if (qx < 0 || qx >= width) continue;
            float3 sn = float3(src[uint3(0, qx, qy)],
                               src[uint3(1, qx, qy)],
                               src[uint3(2, qx, qy)]);
            float3 gn = float3(gout[uint3(0, qx, qy)],
                               gout[uint3(1, qx, qy)],
                               gout[uint3(2, qx, qy)]);
            float3 on = float3(outw[uint3(0, qx, qy)],
                               outw[uint3(1, qx, qy)],
                               outw[uint3(2, qx, qy)]);
            float wn = outw[uint3(3, qx, qy)];

            float3 d = sn - sm;
            float w = exp(-float(dx * dx + dy * dy) * inv2ss
                          - dot(d, d) * inv2sr);
            float a_mn = dot(gm, sn - om) / wm;
            float a_nm = dot(gn, sm - on) / wn;
            grad += (w / wn) * gn + (a_mn + a_nm) * (w * invsr2) * d;
        }
    }
    return grad;
}

// ------------------------------- resize (manual bridge, PLAIN kernels)
// Separable resampling, bilinear (mode 0) or bicubic Keys a=-0.75
// (mode 1), align_corners=False with edge-clamped taps — matches torch
// F.interpolate (antialias=False) exactly. Weights are geometry-only, so
// resize is LINEAR: the adjoint kernels need only the upstream gradient
// (no saved activations). Adjoints use the recompute-and-match trick
// (like d0's gaussian_h_bwd): scan the influenced output range, rebuild
// each output pixel's taps, and accumulate the weights whose clamped
// target lands on this source pixel — exact at the edges.

float _resample_w(float d, int mode)
{
    float ad = abs(d);
    if (mode == 0)
        return max(0.0, 1.0 - ad);
    const float a = -0.75;               // torch's bicubic constant
    if (ad <= 1.0)
        return ((a + 2.0) * ad - (a + 3.0)) * ad * ad + 1.0;
    if (ad < 2.0)
        return (((ad - 5.0) * ad + 8.0) * ad - 4.0) * a;
    return 0.0;
}

float resample_x(
    int3 pix,                         // x = ch, y = col (out), z = row
    Tensor<float, 3> src,             // H x W_src x C
    float scale,                      // W_src / W_out
    int src_width,
    int mode
)
{
    float sx = (float(pix.y) + 0.5) * scale - 0.5;
    int x0 = int(floor(sx));
    int off = mode == 0 ? 0 : 1;
    float acc = 0.0;
    for (int x = x0 - off; x <= x0 + off + 1; ++x)
    {
        float w = _resample_w(sx - float(x), mode);
        int q = clamp(x, 0, src_width - 1);
        acc += w * src[uint3(pix.x, q, pix.z)];
    }
    return acc;
}

float resample_y(
    int3 pix,                         // x = ch, y = col, z = row (out)
    Tensor<float, 3> src,             // H_src x W x C
    float scale,                      // H_src / H_out
    int src_height,
    int mode
)
{
    float sy = (float(pix.z) + 0.5) * scale - 0.5;
    int y0 = int(floor(sy));
    int off = mode == 0 ? 0 : 1;
    float acc = 0.0;
    for (int y = y0 - off; y <= y0 + off + 1; ++y)
    {
        float w = _resample_w(sy - float(y), mode);
        int q = clamp(y, 0, src_height - 1);
        acc += w * src[uint3(pix.x, pix.y, q)];
    }
    return acc;
}

float resample_x_adj(
    int3 pix,                         // x = ch, y = col (SRC), z = row
    Tensor<float, 3> gout,            // H x W_out x C
    float scale,                      // W_src / W_out (forward's scale)
    int src_width,
    int out_width,
    int mode
)
{
    int xs = pix.y;
    float support = mode == 0 ? 1.0 : 2.0;
    // output pixels whose (unclamped) taps can reach xs; widened by 1 for
    // float safety — stray candidates contribute w == 0
    int lo = int(ceil((float(xs) - support + 0.5) / scale - 0.5)) - 1;
    int hi = int(floor((float(xs) + support + 0.5) / scale - 0.5)) + 1;
    if (xs == 0) lo = 0;                       // edge clamp pulls far taps in
    if (xs == src_width - 1) hi = out_width - 1;
    lo = max(lo, 0);
    hi = min(hi, out_width - 1);

    int off = mode == 0 ? 0 : 1;
    float acc = 0.0;
    for (int xo = lo; xo <= hi; ++xo)
    {
        float sx = (float(xo) + 0.5) * scale - 0.5;
        int x0 = int(floor(sx));
        float w_eff = 0.0;
        for (int x = x0 - off; x <= x0 + off + 1; ++x)
        {
            if (clamp(x, 0, src_width - 1) == xs)
                w_eff += _resample_w(sx - float(x), mode);
        }
        acc += w_eff * gout[uint3(pix.x, xo, pix.z)];
    }
    return acc;
}

float resample_y_adj(
    int3 pix,                         // x = ch, y = col, z = row (SRC)
    Tensor<float, 3> gout,            // H_out x W x C
    float scale,                      // H_src / H_out (forward's scale)
    int src_height,
    int out_height,
    int mode
)
{
    int ys = pix.z;
    float support = mode == 0 ? 1.0 : 2.0;
    int lo = int(ceil((float(ys) - support + 0.5) / scale - 0.5)) - 1;
    int hi = int(floor((float(ys) + support + 0.5) / scale - 0.5)) + 1;
    if (ys == 0) lo = 0;
    if (ys == src_height - 1) hi = out_height - 1;
    lo = max(lo, 0);
    hi = min(hi, out_height - 1);

    int off = mode == 0 ? 0 : 1;
    float acc = 0.0;
    for (int yo = lo; yo <= hi; ++yo)
    {
        float sy = (float(yo) + 0.5) * scale - 0.5;
        int y0 = int(floor(sy));
        float w_eff = 0.0;
        for (int y = y0 - off; y <= y0 + off + 1; ++y)
        {
            if (clamp(y, 0, src_height - 1) == ys)
                w_eff += _resample_w(sy - float(y), mode);
        }
        acc += w_eff * gout[uint3(pix.x, pix.y, yo)];
    }
    return acc;
}

// -------------------------------------------- blend modes (element float3)
[Differentiable]
float3 blend_multiply(float3 base, float3 blend, no_diff float opacity)
{
    return lerp(base, base * blend, opacity);
}

[Differentiable]
float3 blend_screen(float3 base, float3 blend, no_diff float opacity)
{
    return lerp(base, base + blend - base * blend, opacity);
}

[Differentiable]
float3 blend_overlay(float3 base, float3 blend, no_diff float opacity)
{
    // piecewise-smooth per channel (same class as ReLU for autodiff)
    float3 lo = 2.0 * base * blend;
    float3 hi = 1.0 - 2.0 * (1.0 - base) * (1.0 - blend);
    float3 o = float3(
        base.x <= 0.5 ? lo.x : hi.x,
        base.y <= 0.5 ? lo.y : hi.y,
        base.z <= 0.5 ? lo.z : hi.z);
    return lerp(base, o, opacity);
}

// --------------------------------------- learnable LUT (512x512 8x8 CLUT)
// Differentiable w.r.t. BOTH the image and the LUT: image gradients flow
// through the bilinear weights (grid_sample-style, floor() treated as
// piecewise-constant); LUT gradients scatter into the tapped texels.
// Layout matches slangimage.ops.lut / identity_lut_np: 64 slices of
// 64x64 in an 8x8 grid, blue selects the slice, red/green index within.
// plain helper (called via no_diff): staircase slice coords for blue value
float4 _lut_cells(float b)
{
    float s0 = floor(b);
    float s1 = min(s0 + 1.0, 63.0);
    return float4(fmod(s0, 8.0), floor(s0 / 8.0), fmod(s1, 8.0), floor(s1 / 8.0));
}

[Differentiable]
float3 _lut_tap(IDiffTensor<float, 3> lut, no_diff int x, no_diff int y, no_diff int size)
{
    int cx = clamp(x, 0, size - 1);
    int cy = clamp(y, 0, size - 1);
    return float3(
        lut[uint3(0, cx, cy)],
        lut[uint3(1, cx, cy)],
        lut[uint3(2, cx, cy)]);
}

[Differentiable]
float3 _lut_bilinear(IDiffTensor<float, 3> lut, float2 pos, no_diff int size)
{
    float2 p = pos - 0.5;
    no_diff int x0 = int(floor(p.x));
    no_diff int y0 = int(floor(p.y));
    float fx = p.x - no_diff floor(p.x);   // detach: d(fx)/d(p.x) = 1
    float fy = p.y - no_diff floor(p.y);

    float3 c00 = _lut_tap(lut, x0, y0, size);
    float3 c10 = _lut_tap(lut, x0 + 1, y0, size);
    float3 c01 = _lut_tap(lut, x0, y0 + 1, size);
    float3 c11 = _lut_tap(lut, x0 + 1, y0 + 1, size);

    return lerp(lerp(c00, c10, fx), lerp(c01, c11, fx), fy);
}

[Differentiable]
float3 lut_apply(
    no_diff int2 pix,
    IDiffTensor<float, 3> img,        // H x W x 3
    IDiffTensor<float, 3> lut,        // 512 x 512 x 3
    no_diff int lut_size              // 512
)
{
    float3 c = float3(
        img[uint3(0, pix.x, pix.y)],
        img[uint3(1, pix.x, pix.y)],
        img[uint3(2, pix.x, pix.y)]);
    c = clamp(c, 0.0, 1.0);

    float b = c.z * 63.0;
    no_diff float4 cells = no_diff _lut_cells(b);
    float fb = b - no_diff floor(b);          // detach: d(fb)/d(b) = 1

    float scale = float(lut_size);
    float2 cell0 = cells.xy;
    float2 cell1 = cells.zw;
    float2 rg = float2(c.x, c.y);
    float2 pos0 = (cell0 * 0.125 + 0.5 / scale + (0.125 - 1.0 / scale) * rg) * scale;
    float2 pos1 = (cell1 * 0.125 + 0.5 / scale + (0.125 - 1.0 / scale) * rg) * scale;

    float3 n0 = _lut_bilinear(lut, pos0, lut_size);
    float3 n1 = _lut_bilinear(lut, pos1, lut_size);
    return lerp(n0, n1, fb);
}
