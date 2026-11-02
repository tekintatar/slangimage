// shaders/ops/color_controls.slang
// Combined color controls in YCbCr space.
//
// Low-level parameters (the functional API maps signed contrast to a factor):
//   hue        in [-1, 1]   → hue shift in [-PI, PI] radians
//   saturation in [-1, 1]   → scale = max(0, 1 + saturation)
//   contrast   in [0, 5]    → encoded RGB factor around 0.5; <=1e-4 bypasses
//   brightness in [-1, 1]   → Y' += brightness * 0.5
//
// Decodes sRGB before the rounded YCbCr-style linear-RGB transform; encodes
// before contrast. Luma and chroma are limited, but final RGB is not clipped.

import "color_conversions.slang";

static const float PI  = 3.14159265359;
static const float EPS = 1e-6;

// ------------------------------------------------------------
// Contrast adjustment using a 4x5 color matrix, Core Image style
// contrast = 1.0 -> identity
// contrast = 0.0 -> flat gray (0.5)
// ------------------------------------------------------------

float3 apply_contrast(float3 rgb, float contrast)
{
    contrast = max(contrast, 0.0);

    float t = (1.0 - contrast) * 0.5;

    // Matrix multiply expanded inline:
    // result = rgb * contrast + t
    float3 out_rgb;
    out_rgb.r = rgb.r * contrast + t;
    out_rgb.g = rgb.g * contrast + t;
    out_rgb.b = rgb.b * contrast + t;

    return out_rgb;
}

void color_controls(
    uint2 coord,
    Texture2D<float4> src,
    [format(SLANGIMAGE_FMT)] RWTexture2D<float4> dst,
    float hue_amount,        // [-1, 1]
    float saturation_amount, // [-1, 1]
    float contrast_amount,   // factor, NOT the public signed control
    float brightness_amount  // [-1, 1]
)
{
    uint w, h;
    src.GetDimensions(w, h);
    if (coord.x >= w || coord.y >= h)
        return;

    float4 c4 = src[coord];
    float3 rgb = c4.rgb;

    // sRGB → YCbCr
    float3 ycc = srgb_to_ycbcr(rgb);
    float Y = ycc.x;
    float Cb = ycc.y;
    float Cr = ycc.z;

    // Clamp controls
    float hue_t = clamp(hue_amount, -1.0, 1.0);
    float sat_t = clamp(saturation_amount, -1.0, 1.0);
    float contrast_t = clamp(contrast_amount, 0.0, 5.0);
    float bright_t = clamp(brightness_amount, -1.0, 1.0);

    // Legacy negative-luma branch is unreachable after the factor clamp.
    if (contrast_t < 0.0)
    {
        // avoid slope ≤ 0
        float slope = 1.0 + max(contrast_t, -0.99);
        Y = (Y - 0.5) * slope + 0.5;
    }

    // 2) Brightness as additive offset in Y
    Y += bright_t * 0.5;
    Y = clamp(Y, 0.0, 1.0);

    // 3) Hue rotation in CbCr plane
    float angle = hue_t * PI;
    float cosA = cos(angle);
    float sinA = sin(angle);

    float newCb = Cb * cosA - Cr * sinA;
    float newCr = Cb * sinA + Cr * cosA;

    // 4) Saturation scaling
    float sat_scale = max(0.0, 1.0 + sat_t);
    newCb *= sat_scale;
    newCr *= sat_scale;

    float max_chroma = 0.6;
    float sat_mag = length(float2(newCb, newCr));
    if (sat_mag > max_chroma && sat_mag > EPS)
    {
        float scale = max_chroma / sat_mag;
        newCb *= scale;
        newCr *= scale;
    }

    float3 out_ycc = float3(Y, newCb, newCr);
    float3 out_rgb = ycbcr_to_srgb(out_ycc);

    // 5) Factor-style affine contrast on encoded RGB around 0.5.
    // The legacy <=1e-4 bypass is not reachable from the public signed API.
    float pos_contrast = max(contrast_t, 0.0); // ignore negative here
    if (pos_contrast > 1e-4)
    {
        out_rgb = apply_contrast(out_rgb,pos_contrast);
    }

    dst[coord] = float4(out_rgb, c4.a);
}
