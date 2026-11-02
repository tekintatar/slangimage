// shaders/ops/color_adjust.slang
// Standard color utilities matching these CI built-ins:
// exposure (CIExposureAdjust), sepia (CISepiaTone), hue rotation
// (CIHueAdjust, YIQ-space), false color (CIFalseColor), and global
// highlight/shadow adjust (pointwise power-law formulation; exact Core Image
// parity is not established).

static const float3 CA_LUM = float3(0.299, 0.587, 0.114);

void exposure_adjust(
    uint2               coord,
    Texture2D<float4>   src,
    [format(SLANGIMAGE_FMT)] RWTexture2D<float4> dst,
    float               ev
)
{
    uint w, h;
    src.GetDimensions(w, h);
    if (coord.x >= w || coord.y >= h)
        return;
    float4 c = src[coord];
    dst[coord] = float4(clamp(c.rgb * exp2(ev), 0.0, 1.0), c.a);
}

void sepia_tone(
    uint2               coord,
    Texture2D<float4>   src,
    [format(SLANGIMAGE_FMT)] RWTexture2D<float4> dst,
    float               intensity
)
{
    uint w, h;
    src.GetDimensions(w, h);
    if (coord.x >= w || coord.y >= h)
        return;
    float4 c = src[coord];

    // classic sepia matrix
    float3 sep;
    sep.r = dot(c.rgb, float3(0.393, 0.769, 0.189));
    sep.g = dot(c.rgb, float3(0.349, 0.686, 0.168));
    sep.b = dot(c.rgb, float3(0.272, 0.534, 0.131));

    float3 out_rgb = lerp(c.rgb, clamp(sep, 0.0, 1.0), clamp(intensity, 0.0, 1.0));
    dst[coord] = float4(out_rgb, c.a);
}

void hue_adjust(
    uint2               coord,
    Texture2D<float4>   src,
    [format(SLANGIMAGE_FMT)] RWTexture2D<float4> dst,
    float               angle    // radians
)
{
    uint w, h;
    src.GetDimensions(w, h);
    if (coord.x >= w || coord.y >= h)
        return;
    float4 c = src[coord];

    // Rotate encoded-RGB chroma in YIQ; internal Y is untouched. Rounded
    // reconstruction coefficients and final clipping can change output luma.
    float y = dot(c.rgb, CA_LUM);
    float i = dot(c.rgb, float3(0.596, -0.274, -0.322));
    float q = dot(c.rgb, float3(0.211, -0.523, 0.312));

    float s = sin(angle);
    float co = cos(angle);
    float i2 = i * co - q * s;
    float q2 = i * s + q * co;

    float3 out_rgb;
    out_rgb.r = y + 0.956 * i2 + 0.621 * q2;
    out_rgb.g = y - 0.272 * i2 - 0.647 * q2;
    out_rgb.b = y - 1.106 * i2 + 1.703 * q2;

    dst[coord] = float4(clamp(out_rgb, 0.0, 1.0), c.a);
}

void false_color(
    uint2               coord,
    Texture2D<float4>   src,
    [format(SLANGIMAGE_FMT)] RWTexture2D<float4> dst,
    float3              color0,  // mapped at luminance 0
    float3              color1   // mapped at luminance 1
)
{
    uint w, h;
    src.GetDimensions(w, h);
    if (coord.x >= w || coord.y >= h)
        return;
    float4 c = src[coord];
    float luma = clamp(dot(c.rgb, CA_LUM), 0.0, 1.0);
    dst[coord] = float4(lerp(color0, color1, luma), c.a);
}

void highlight_shadow(
    uint2               coord,
    Texture2D<float4>   src,
    [format(SLANGIMAGE_FMT)] RWTexture2D<float4> dst,
    float               shadow_amount,    // -1..1, 0 = no change, >0 lifts
    float               highlight_amount  //  0..1, 1 = no change, <1 compresses
)
{
    uint w, h;
    src.GetDimensions(w, h);
    if (coord.x >= w || coord.y >= h)
        return;
    float4 c = src[coord];

    float l = clamp(dot(c.rgb, CA_LUM), 0.0, 1.0);

    // shadows: canonical GPUImage lift for s >= 0 (identity at 0 via clamp);
    // gamma-style deepening for s < 0
    float shadow;
    if (shadow_amount >= 0.0)
    {
        float e = 1.0 / (shadow_amount + 1.0);
        shadow = clamp(pow(l, e) - 0.76 * pow(l, 2.0 * e) - l, 0.0, 1.0);
    }
    else
    {
        shadow = pow(l, 1.0 - shadow_amount) - l; // negative delta, hits darks hardest
    }

    // highlights: canonical GPUImage compression (identity at 1.0 via clamp)
    float he = 1.0 / (2.0 - clamp(highlight_amount, 0.0, 1.0));
    float inv = 1.0 - l;
    float highlight = clamp(
        (1.0 - (pow(inv, he) - 0.8 * pow(inv, 2.0 * he))) - l,
        -1.0, 0.0);

    float new_l = clamp(l + shadow + highlight, 0.0, 1.0);
    // Every positive luma follows the curve, including near-black pixels.
    // Only zero/clipped-negative luma bypasses division; exact black stays black.
    float ratio = (l > 0.0) ? new_l / l : 1.0;

    dst[coord] = float4(clamp(c.rgb * ratio, 0.0, 1.0), c.a);
}
