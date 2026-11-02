import "common/color_conversions.slang";
import "common/saturation.slang";

static const float EPS = 1e-6;

float skinMask(float3 input) {
    float3 pixel = saturate(input);
    float gray = max_rgb(pixel);
    float gi = 1.0 / (gray + EPS);
    float gii = 1.0 / (1.0 - gray + EPS);
    float3 rgbsat = max((pixel.rgb - gray) * gii, (gray - pixel.rgb) * gi);
    float skin = min(pixel.r - pixel.g, pixel.g * 2.0 - pixel.b) * 4.0 * (1.0 - rgbsat.r) * gi;

    skin = 0.15 + clamp(skin, 0.0, 1.0) * 0.7;
    return 1.0 - skin;
}


void vibrance(
    uint2 tid,
    Texture2D<float4> src,
    [format(SLANGIMAGE_FMT)] RWTexture2D<float4> dst,
    float amount,      // expected -1..1
)
{
    uint w, h;
    src.GetDimensions(w, h);
    if (tid.x >= w || tid.y >= h)
        return;

    float4 s = src[tid];


    // Clamp overall range to be safe
    float A = clamp(amount, -1.0, 1.0);

    // Make sure RGB is in a sane range
    float3 c = saturate(s.rgb);
    float3 saturated_image = hslAdjustSaturation(c, A, false, 1.0);

    float skinM = skinMask(c);

    float t = smoothstep(0.5, 1.0, skinM);
    float localM = skinM * (1.0 - 0.5 * t);

    float3 outColor = lerp(c, saturated_image, localM);
    outColor = saturate(outColor);

    dst[tid] = float4(float3(outColor), s.a);
}
