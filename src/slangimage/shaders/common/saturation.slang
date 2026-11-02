// ============================================================
// sRGB <-> Linear
// ============================================================

// sRGB -> Linear
float3 rgb2linear(float3 c)
{
    // Make sure we never feed nonsense into the transfer
    c = saturate(c);

    float3 low  = c / 12.92;
    float3 high = pow(c * (1.0 / 1.055) + (0.055 / 1.055), 2.4);

    float3 res;
    res.r = (c.r <= 0.04045) ? low.r : high.r;
    res.g = (c.g <= 0.04045) ? low.g : high.g;
    res.b = (c.b <= 0.04045) ? low.b : high.b;
    return res;
}

// Linear -> sRGB
float3 linear2rgb(float3 c)
{
    // Avoid pow(negative, 1/2.4) -> NaN
    c = max(c, float3(0.0, 0.0, 0.0));

    float3 low  = 12.92 * c;
    float3 high = 1.055 * pow(c, 1.0 / 2.4) - 0.055;

    float3 res;
    res.r = (c.r <= 0.0031308) ? low.r : high.r;
    res.g = (c.g <= 0.0031308) ? low.g : high.g;
    res.b = (c.b <= 0.0031308) ? low.b : high.b;

    // Make sure this is safe to feed back into later stages
    return saturate(res);
}
// ============================================================
// RGB -> HSL helpers (no hue needed for saturation-only)
// ============================================================

void rgb2hsl_sat_lum(
    float3 rgb_in,
    out float saturation,
    out float lightness,
    out float delta
)
{
    float3 rgb = saturate(rgb_in);

    float cMin = min(rgb.r, min(rgb.g, rgb.b));
    float cMax = max(rgb.r, max(rgb.g, rgb.b));
    delta      = cMax - cMin;

    lightness  = 0.5 * (cMax + cMin);
    saturation = 0.0;

    if (delta > 1e-6)
    {
        float denom;
        if (lightness < 0.5)
            denom = cMax + cMin;
        else
            denom = 2.0 - cMax - cMin;

        if (denom > 1e-6)
            saturation = delta / denom;
    }

    lightness  = saturate(lightness);
    saturation = saturate(saturation);
}

// ============================================================
// Pure HSL saturation adjustment (Photoshop-style) with chroma cap
// ============================================================
//
// satDelta  in [-1, 1]  > 0 increases saturation, < 0 decreases
// maxChroma in [0, 1]   maximum allowed chroma (cMax - cMin) in sRGB
float3 hslAdjustSaturation(
    float3 col,
    float satDelta,
    bool useLinearInput,
    float maxChroma
)
{
    // Clamp control params to sane ranges
    satDelta = clamp(satDelta, -1.0, 1.0);
    maxChroma = saturate(maxChroma);

    // Convert to sRGB space if our input is linear
    float3 rgb = useLinearInput ? linear2rgb(col) : col;
    rgb = saturate(rgb);

    float saturation;
    float lightness;
    float deltaOrig;
    rgb2hsl_sat_lum(rgb, saturation, lightness, deltaOrig);

    // No chroma, nothing to do
    if (deltaOrig <= 1e-6)
    {
        // Preserve original space
        return useLinearInput ? col : saturate(col);
    }

    float a = 0.0;

    if (satDelta >= 0.0)
    {
        // Increase saturation
        float satPlus = saturation + satDelta;

        if (satPlus >= 1.0)
        {
            a = saturation; // Photoshop style
        }
        else
        {
            // Avoid a == 0 accidentally
            a = max(1.0 - satDelta, 1e-6);
        }

        a = 1.0 / a - 1.0;
        rgb = rgb + (rgb - lightness) * a;
    }
    else
    {
        // Decrease saturation
        a = satDelta;
        rgb = lightness + (rgb - lightness) * (1.0 + a);
    }

    // Optional chroma cap
    if (maxChroma < 1.0)
    {
        float cMin2 = min(rgb.r, min(rgb.g, rgb.b));
        float cMax2 = max(rgb.r, max(rgb.g, rgb.b));
        float deltaNew = cMax2 - cMin2;

        if (deltaNew > maxChroma)
        {
            float scale = maxChroma / max(deltaNew, 1e-6);
            // Pull back towards same HSL lightness pivot
            rgb = lightness + (rgb - lightness) * scale;
        }
    }

    // Stay in displayable range before going back to linear
    rgb = saturate(rgb);

    return useLinearInput ? rgb2linear(rgb) : rgb;
}