// sRGB to linear
float3 srgb_to_linear(float3 c)
{   
    float3 lo = c / 12.92;
    float3 hi = pow((c + 0.055) / 1.055, float3(2.4, 2.4, 2.4));
    float3 res;
    res.r = (c.r <= 0.04045) ? lo.r : hi.r;
    res.g = (c.g <= 0.04045) ? lo.g : hi.g;
    res.b = (c.b <= 0.04045) ? lo.b : hi.b;
    return res;
}

// linear to sRGB
float3 linear_to_srgb(float3 c)
{
    float3 lo = c * 12.92;
    float3 hi = 1.055 * pow(c, float3(1.0 / 2.4, 1.0 / 2.4, 1.0 / 2.4)) - 0.055;
    float3 res;
    res.r = (c.r <= 0.0031308) ? lo.r : hi.r;
    res.g = (c.g <= 0.0031308) ? lo.g : hi.g;
    res.b = (c.b <= 0.0031308) ? lo.b : hi.b;
    return res;
}

// RGB (sRGB) -> XYZ (D65)
float3 rgb_to_xyz(float3 c)
{
    float3 cl = srgb_to_linear(c);

    // Standard D65 matrix
    float x = dot(cl, float3(0.4124564, 0.3575761, 0.1804375));
    float y = dot(cl, float3(0.2126729, 0.7151522, 0.0721750));
    float z = dot(cl, float3(0.0193339, 0.1191920, 0.9503041));

    return float3(x, y, z);
}

// XYZ -> RGB (sRGB)
float3 xyz_to_rgb(float3 xyz)
{
    float x = xyz.x;
    float y = xyz.y;
    float z = xyz.z;

    float3 cl;
    cl.r =  3.2404542 * x + -1.5371385 * y + -0.4985314 * z;
    cl.g = -0.9692660 * x +  1.8760108 * y +  0.0415560 * z;
    cl.b =  0.0556434 * x + -0.2040259 * y +  1.0572252 * z;

    return linear_to_srgb(cl);
}

float f_lab(float t)
{
    const float delta = 6.0 / 29.0;
    float delta3 = delta * delta * delta;
    if (t > delta3)
        return pow(t, 1.0 / 3.0);
    return t / (3.0 * delta * delta) + 4.0 / 29.0;
}

float3 xyz_to_lab(float3 xyz)
{
    // D65 white
    float3 white = float3(0.95047, 1.00000, 1.08883);
    float3 r = float3(
        f_lab(xyz.x / white.x),
        f_lab(xyz.y / white.y),
        f_lab(xyz.z / white.z)
    );

    float L = 116.0 * r.y - 16.0;
    float a = 500.0 * (r.x - r.y);
    float b = 200.0 * (r.y - r.z);

    return float3(L, a, b);
}

float finv_lab(float t)
{
    const float delta = 6.0 / 29.0;
    float delta2 = delta * delta;
    if (t > delta)
        return t * t * t;
    return 3.0 * delta2 * (t - 4.0 / 29.0);
}

float3 lab_to_xyz(float3 lab)
{
    float L = lab.x;
    float a = lab.y;
    float b = lab.z;

    float fy = (L + 16.0) / 116.0;
    float fx = fy + (a / 500.0);
    float fz = fy - (b / 200.0);

    // D65 white
    float3 white = float3(0.95047, 1.00000, 1.08883);

    float x = white.x * finv_lab(fx);
    float y = white.y * finv_lab(fy);
    float z = white.z * finv_lab(fz);

    return float3(x, y, z);
}

// sRGB 0..1 <-> Lab (L 0..100, a b approx -128..128)
float3 rgb_to_lab(float3 rgb)
{
    float3 xyz = rgb_to_xyz(rgb);
    return xyz_to_lab(xyz);
}

float3 lab_to_rgb(float3 lab)
{
    float3 xyz = lab_to_xyz(lab);
    float3 rgb = xyz_to_rgb(xyz);
    return saturate(rgb);
}

// Assumes input is *linear* RGB in [0,1]
float3 linear_rgb_to_ycbcr(float3 lin)
{
    // BT.601 style luma in linear space
    float y = dot(lin, float3(0.299, 0.587, 0.114));
    float cb = (lin.b - y) * 0.564;
    float cr = (lin.r - y) * 0.713;

    return float3(y, cb, cr);
}

// Assumes output should be *linear* RGB in [0,1]
float3 ycbcr_to_linear_rgb(float3 ycc)
{
    float y = ycc.x;
    float cb = ycc.y;
    float cr = ycc.z;

    float3 lin;
    lin.r = y + 1.403 * cr;
    lin.g = y - 0.714 * cr - 0.344 * cb;
    lin.b = y + 1.770 * cb;

    return lin;
}

float3 srgb_to_ycbcr(float3 srgb)
{
    float3 lin = srgb_to_linear(srgb);
    return linear_rgb_to_ycbcr(lin);
}

float3 ycbcr_to_srgb(float3 ycc)
{
    float3 lin = ycbcr_to_linear_rgb(ycc);
    return linear_to_srgb(lin);
}

float3 rgb_to_hsv(float3 c)
{
    float4 K = float4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    float4 p = lerp(float4(c.bg, K.wz), float4(c.gb, K.xy), step(c.b, c.g));
    float4 q = lerp(float4(p.xyw, c.r), float4(c.r, p.yzx), step(p.x, c.r));

    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return float3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

float3 hsv_to_rgb(float3 c)
{
    float4 K = float4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    float3 p = abs(frac(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * lerp(K.xxx, saturate(p - K.xxx), c.y);
}

float3 rgb_to_hsl(float3 rgb)
{
    // Clamp to [0,1] first; out-of-range input has no meaningful HSL.
    rgb = saturate(rgb);

    float r = rgb.r;
    float g = rgb.g;
    float b = rgb.b;

    float maxc = max(r, max(g, b));
    float minc = min(r, min(g, b));
    float delta = maxc - minc;

    float H = 0.0;
    float S = 0.0;
    float L = 0.5 * (maxc + minc);

    if (delta > 1e-6)
    {
        float den = 1.0 - abs(2.0 * L - 1.0);
        den = max(den, 1e-6); // avoid division by zero

        S = delta / den;
        S = saturate(S); // keep in [0,1] for sanity

        if (maxc == r)
        {
            H = (g - b) / delta;
        }
        else if (maxc == g)
        {
            H = 2.0 + (b - r) / delta;
        }
        else
        {
            H = 4.0 + (r - g) / delta;
        }

        H /= 6.0;
        H = frac(H);
    }

    return float3(H, S, L);
}

float hue_to_rgb(float p, float q, float t)
{
    t = frac(t);

    if (t < 1.0 / 6.0) return p + (q - p) * 6.0 * t;
    if (t < 1.0 / 2.0) return q;
    if (t < 2.0 / 3.0) return p + (q - p) * (2.0 / 3.0 - t) * 6.0;

    return p;
}

float3 hsl_to_rgb(float3 hsl)
{
    float H = hsl.x;
    float S = hsl.y;
    float L = hsl.z;

    float3 rgb;

    if (S < 1e-6)
    {
        // Gray
        rgb = float3(L, L, L);
    }
    else
    {
        float q = (L < 0.5)
                      ? (L * (1.0 + S))
                      : (L + S - L * S);

        float p = 2.0 * L - q;

        rgb.r = hue_to_rgb(p, q, H + 1.0 / 3.0);
        rgb.g = hue_to_rgb(p, q, H);
        rgb.b = hue_to_rgb(p, q, H - 1.0 / 3.0);
    }

    return rgb;
}

float max_rgb(float3 input) {

    return rgb_to_hsv(input).z;
}