// shaders/augment/augment.slang — op library for slangimage.augment.
//
// slangimage.augment generates one entry point per pipeline pass from the
// transforms the pipeline uses and appends it to this file. Every op here is
// a small function on one pixel: float4 with rgb in xyz, values in [0, 1],
// fp32 math. Semantics follow torchvision.transforms.v2 on float images,
// including the clamp after each colour op.
//
// Tensor vector subscripts are reversed (x = last dim): a (B, P) parameter
// table is read as p[uint2(k, b)]. No float3 in this file (NVRTC spills it).

import slangpy;

// ---------------------------------------------------------------- helpers
float prm(Tensor<float, 2> p, int b, int k) { return p[uint2(k, b)]; }

float gray(float4 c) { return 0.2989 * c.x + 0.587 * c.y + 0.114 * c.z; }

float4 clamp01(float4 c) { return clamp(c, 0.0, 1.0); }

// ------------------------------------------------------------- colour ops
float4 brightness(float4 c, float f) { return clamp01(c * f); }

float4 contrast(float4 c, float f, float mean) { return clamp01(f * c + (1.0 - f) * mean); }

float4 saturation(float4 c, float f) { return clamp01(f * c + (1.0 - f) * gray(c)); }

// torchvision adjust_hue: RGB -> HSV, shift the hue, HSV -> RGB
float4 hue(float4 c, float shift)
{
    float r = c.x, g = c.y, b = c.z;
    float maxc = max(r, max(g, b));
    float minc = min(r, min(g, b));
    bool eqc = maxc == minc;
    float cr = maxc - minc;
    float s = cr / (eqc ? 1.0 : maxc);
    float div = eqc ? 1.0 : cr;
    float rc = (maxc - r) / div;
    float gc = (maxc - g) / div;
    float bc = (maxc - b) / div;
    float h;
    if (maxc == r)      h = bc - gc;
    else if (maxc == g) h = 2.0 + rc - bc;
    else                h = 4.0 + gc - rc;
    h = fmod(h * (1.0 / 6.0) + 1.0, 1.0) + shift;
    h = h - floor(h);
    float v = maxc;
    float h6 = h * 6.0;
    float fi = floor(h6);
    float f = h6 - fi;
    int i = int(fi) % 6;
    float sxf = s * f;
    float q = clamp((1.0 - sxf) * v, 0.0, 1.0);
    float t = clamp((sxf + 1.0 - s) * v, 0.0, 1.0);
    float pp = clamp((1.0 - s) * v, 0.0, 1.0);
    switch (i)
    {
    case 0: return float4(v, t, pp, 0.0);
    case 1: return float4(q, v, pp, 0.0);
    case 2: return float4(pp, v, t, 0.0);
    case 3: return float4(pp, q, v, 0.0);
    case 4: return float4(t, pp, v, 0.0);
    default: return float4(v, pp, q, 0.0);
    }
}

// ------------------------------------------------------- image sources
// Where one uint8 RGB image lives. Kernels are generic over the source, and
// Slang specializes each one, so the indirection costs nothing.
interface IImageSource
{
    float4 load(int x, int y);   // 0-255 values
}

// one (3, H, W) image read in place through its device address
struct SourceCHW : IImageSource
{
    uint8_t* p;
    int w;
    int plane;
    float4 load(int x, int y)
    {
        int i = y * w + x;
        return float4(float(p[i]), float(p[i + plane]), float(p[i + 2 * plane]), 0.0);
    }
}

// one (H, W, 3) image, as DALI and nvImageCodec decode
struct SourceHWC : IImageSource
{
    uint8_t* p;
    int w;
    float4 load(int x, int y)
    {
        int i = (y * w + x) * 3;
        return float4(float(p[i]), float(p[i + 1]), float(p[i + 2]), 0.0);
    }
}

// The image table: ptrs holds device addresses, meta rows are (H, W).
SourceCHW source_chw(Tensor<int64_t, 1> ptrs, Tensor<int, 2> meta, int b)
{
    int h = meta[uint2(0, b)];
    int w = meta[uint2(1, b)];
    SourceCHW s = { (uint8_t*)(uint64_t(ptrs[uint1(b)])), w, h * w };
    return s;
}

SourceHWC source_hwc(Tensor<int64_t, 1> ptrs, Tensor<int, 2> meta, int b)
{
    SourceHWC s = { (uint8_t*)(uint64_t(ptrs[uint1(b)])), meta[uint2(1, b)] };
    return s;
}

// One output pixel of torchvision's resize of the box (left, top, cw, ch)
// to (OW, OH): bilinear, align_corners=False; with aa != 0 the antialias
// triangle filter, whose support widens with the downscale factor. Taps stay
// inside the box. Returns [0, 1].
float4 resample<S : IImageSource>(S src, int left, int top, int cw, int ch,
                                  int ox, int oy, int OW, int OH, int aa)
{
    float sx = float(cw) / float(OW);
    float sy = float(ch) / float(OH);
    if (aa == 0)
    {
        float fx = max((float(ox) + 0.5) * sx - 0.5, 0.0);
        float fy = max((float(oy) + 0.5) * sy - 0.5, 0.0);
        int x0 = min(int(fx), cw - 1);
        int y0 = min(int(fy), ch - 1);
        int x1 = min(x0 + 1, cw - 1);
        int y1 = min(y0 + 1, ch - 1);
        float ax = fx - float(x0);
        float ay = fy - float(y0);
        float4 r0 = lerp(src.load(left + x0, top + y0), src.load(left + x1, top + y0), ax);
        float4 r1 = lerp(src.load(left + x0, top + y1), src.load(left + x1, top + y1), ax);
        return lerp(r0, r1, ay) * (1.0 / 255.0);
    }
    float supx = sx >= 1.0 ? sx : 1.0;
    float supy = sy >= 1.0 ? sy : 1.0;
    float invx = sx >= 1.0 ? 1.0 / sx : 1.0;
    float invy = sy >= 1.0 ? 1.0 / sy : 1.0;
    float cx = (float(ox) + 0.5) * sx;
    float cy = (float(oy) + 0.5) * sy;
    int xmin = max(int(cx - supx + 0.5), 0);
    int xmax = min(int(cx + supx + 0.5), cw);
    int ymin = max(int(cy - supy + 0.5), 0);
    int ymax = min(int(cy + supy + 0.5), ch);
    float wxsum = 0.0;
    for (int x = xmin; x < xmax; ++x)
        wxsum += max(0.0, 1.0 - abs((float(x) - cx + 0.5) * invx));
    float4 acc = float4(0.0);
    float wysum = 0.0;
    for (int y = ymin; y < ymax; ++y)
    {
        float wy = max(0.0, 1.0 - abs((float(y) - cy + 0.5) * invy));
        if (wy == 0.0) continue;
        float4 row = float4(0.0);
        for (int x = xmin; x < xmax; ++x)
            row += max(0.0, 1.0 - abs((float(x) - cx + 0.5) * invx)) * src.load(left + x, top + y);
        acc += wy * row;
        wysum += wy;
    }
    return acc * (1.0 / (255.0 * wxsum * wysum));
}

// --------------------------------------------------------------- geometry
// torchvision's centre-crop offset: int(round((size - crop) / 2.0)), where
// Python's round sends halves to the even neighbour. A crop larger than the
// image pads (crop - size) // 2 zeros before it: a negative offset.
int center_offset(int size, int crop)
{
    int d = size - crop;
    if (d < 0)
        return -((-d) / 2);
    int q = d / 2;
    if ((d & 1) != 0 && (q & 1) != 0)   // exactly x.5 with an odd floor
        q += 1;
    return q;
}

// torchvision Resize(int): the shorter side becomes `size`, the longer side
// int(size * long / short). Writes the new (w, h).
void resize_shorter(int W, int H, int size, out int nw, out int nh)
{
    if (W <= H) { nw = size; nh = (size * H) / W; }
    else        { nh = size; nw = (size * W) / H; }
}

// ------------------------------------------------- neighbourhood helpers
// Neighbourhood ops read the previous pass's output through this accessor;
// load applies the op's border mode.
interface IImage
{
    float4 load(int x, int y);
    int width();
    int height();
}

// torch "reflect" padding (edge pixel not repeated), then clamped so radii
// larger than the image stay in bounds.
int reflect(int i, int n)
{
    if (i < 0) i = -i;
    if (i >= n) i = 2 * n - 2 - i;
    return clamp(i, 0, n - 1);
}

// torchvision's gaussian kernel weight (unnormalized) at offset d
float gauss(int d, float sigma) { float x = float(d) / sigma; return exp(-0.5 * x * x); }

// ---------------------------------------------------------------- random
// Counter-based: every value is a function of (call seed, image, draw).
uint pcg(uint v)
{
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float rnd(inout uint s)
{
    s = pcg(s);
    return float(s >> 8) * (1.0 / 16777216.0);
}

float urand(inout uint s, float lo, float hi) { return lo + (hi - lo) * rnd(s); }

int irand(inout uint s, int n) { return min(int(rnd(s) * float(n)), n - 1); }

// torchvision RandomResizedCrop box for an image of W x H: 10 tries, then a
// centre crop clamped to the ratio range.
void sample_rrc(inout uint s, int W, int H, float smin, float smax, float rmin, float rmax,
                out int left, out int top, out int w, out int h)
{
    float area = float(H * W);
    float lo = log(rmin);
    float hi = log(rmax);
    w = 0;
    h = 0;
    bool found = false;
    for (int t = 0; t < 10; ++t)
    {
        float ta = area * urand(s, smin, smax);
        float ar = exp(urand(s, lo, hi));
        int cw = int(round(sqrt(ta * ar)));
        int ch = int(round(sqrt(ta / ar)));
        if (!found && cw > 0 && cw <= W && ch > 0 && ch <= H)
        {
            w = cw;
            h = ch;
            found = true;
        }
    }
    float u0 = rnd(s);
    float u1 = rnd(s);
    if (found)
    {
        top = min(int(u0 * float(H - h + 1)), H - h);
        left = min(int(u1 * float(W - w + 1)), W - w);
        return;
    }
    float ratio = float(W) / float(H);
    if (ratio < rmin)      { w = W; h = int(round(float(W) / rmin)); }
    else if (ratio > rmax) { h = H; w = int(round(float(H) * rmax)); }
    else                   { w = W; h = H; }
    top = (H - h) / 2;
    left = (W - w) / 2;
}

// A random order of 0..3 packed in 2-bit fields (Fisher-Yates).
int random_order(inout uint s)
{
    int code = 0 | (1 << 2) | (2 << 4) | (3 << 6);
    for (int i = 3; i > 0; --i)
    {
        int j = min(int(rnd(s) * float(i + 1)), i);
        int a = (code >> (2 * i)) & 3;
        int c = (code >> (2 * j)) & 3;
        code &= ~((3 << (2 * i)) | (3 << (2 * j)));
        code |= (c << (2 * i)) | (a << (2 * j));
    }
    return code;
}
