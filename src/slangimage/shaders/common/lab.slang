// common/lab.slang
// sRGB <-> CIELAB helpers (D65 white point).
//
// Conventions:
// - RGB is sRGB in [0,1].
// - rgb_to_lab_packed / lab_packed_to_rgb use a normalized Lab packing so
//   every channel fits in [0,1]:
//      L_norm = L / 100
//      a_norm = 0.5 + a / 254
//      b_norm = 0.5 + b / 254
//   (a, b in roughly [-127, 127] map to [0, 1]).
// - The 4-digit sRGB<->XYZ matrices and the 7.787 / 0.008856 piecewise
//   CIELAB constants are used as-is; changing them changes output values.

// ---------------------------------------------------------------------
// sRGB -> XYZ (D65), XYZ scaled so Y is in [0, 100]
// ---------------------------------------------------------------------
inline float3 rgb_to_xyz_d65(float3 c)
{
    float3 tmp;

    tmp.x = (c.r > 0.04045)
        ? pow((c.r + 0.055) / 1.055, 2.4)
        : c.r / 12.92;

    tmp.y = (c.g > 0.04045)
        ? pow((c.g + 0.055) / 1.055, 2.4)
        : c.g / 12.92;

    tmp.z = (c.b > 0.04045)
        ? pow((c.b + 0.055) / 1.055, 2.4)
        : c.b / 12.92;

    // Each colN is one ROW of the linear-sRGB -> XYZ matrix, so
    // XYZ = 100 * (dot(rgb, row0), dot(rgb, row1), dot(rgb, row2)).
    float3 col0 = float3(0.4124, 0.3576, 0.1805);
    float3 col1 = float3(0.2126, 0.7152, 0.0722);
    float3 col2 = float3(0.0193, 0.1192, 0.9505);

    float x = dot(tmp, col0);
    float y = dot(tmp, col1);
    float z = dot(tmp, col2);

    return 100.0 * float3(x, y, z);
}

// ---------------------------------------------------------------------
// XYZ -> CIELAB (D65 white point 95.047, 100, 108.883)
// c is XYZ with X,Y,Z scaled as in rgb_to_xyz_d65 (X ≈ 0..95, Y ≈ 0..100, Z ≈ 0..108)
// ---------------------------------------------------------------------
inline float3 xyz_to_lab_packed(float3 c)
{
    float3 n = c / float3(95.047, 100.0, 108.883);

    float3 v;
    v.x = (n.x > 0.008856) ? pow(n.x, 1.0 / 3.0) : (7.787 * n.x) + (16.0 / 116.0);
    v.y = (n.y > 0.008856) ? pow(n.y, 1.0 / 3.0) : (7.787 * n.y) + (16.0 / 116.0);
    v.z = (n.z > 0.008856) ? pow(n.z, 1.0 / 3.0) : (7.787 * n.z) + (16.0 / 116.0);

    float L = (116.0 * v.y) - 16.0;
    float a = 500.0 * (v.x - v.y);
    float b = 200.0 * (v.y - v.z);

    return float3(L, a, b);
}

// ---------------------------------------------------------------------
// sRGB (0..1) -> normalized Lab packing:
// (L/100, 0.5 + a/254, 0.5 + b/254)
// ---------------------------------------------------------------------
inline float3 rgb_to_lab_packed(float3 c)
{
    float3 lab = xyz_to_lab_packed(rgb_to_xyz_d65(c));

    float L_norm = lab.x / 100.0;
    float a_norm = 0.5 + 0.5 * (lab.y / 127.0);
    float b_norm = 0.5 + 0.5 * (lab.z / 127.0);

    return float3(L_norm, a_norm, b_norm);
}

// ---------------------------------------------------------------------
// Lab (unpacked, L in [0..100], a,b approx -127..127) -> XYZ
// This is the inverse of xyz_to_lab_packed.
// ---------------------------------------------------------------------
inline float3 lab_packed_to_xyz(float3 c)
{
    float fy = (c.x + 16.0) / 116.0;
    float fx = c.y / 500.0 + fy;
    float fz = fy - c.z / 200.0;

    float3 res;
    res.x = 95.047 * ((fx > 0.206897) ? fx * fx * fx : (fx - 16.0 / 116.0) / 7.787);
    res.y = 100.000 * ((fy > 0.206897) ? fy * fy * fy : (fy - 16.0 / 116.0) / 7.787);
    res.z = 108.883 * ((fz > 0.206897) ? fz * fz * fz : (fz - 16.0 / 116.0) / 7.787);

    return res;
}

// ---------------------------------------------------------------------
// XYZ -> sRGB (0..1), unclamped
// c is XYZ with same scaling as rgb_to_xyz_d65 output.
// ---------------------------------------------------------------------
inline float3 xyz_to_rgb_d65(float3 c)
{
    float3 n = c / 100.0;

    // Each colN is one ROW of the XYZ -> linear-sRGB matrix.
    float3 col0 = float3( 3.2406, -1.5372, -0.4986);
    float3 col1 = float3(-0.9689,  1.8758,  0.0415);
    float3 col2 = float3( 0.0557, -0.2040,  1.0570);

    float r_lin = dot(n, col0);
    float g_lin = dot(n, col1);
    float b_lin = dot(n, col2);

    float3 v = float3(r_lin, g_lin, b_lin);

    float3 r;
    r.x = (v.r > 0.0031308) ? (1.055 * pow(v.r, 1.0 / 2.4) - 0.055) : 12.92 * v.r;
    r.y = (v.g > 0.0031308) ? (1.055 * pow(v.g, 1.0 / 2.4) - 0.055) : 12.92 * v.g;
    r.z = (v.b > 0.0031308) ? (1.055 * pow(v.b, 1.0 / 2.4) - 0.055) : 12.92 * v.b;

    return r;
}

// ---------------------------------------------------------------------
// Normalized Lab packing -> sRGB (0..1); inverse of rgb_to_lab_packed.
// Input c is (L_norm, a_norm, b_norm) as produced by rgb_to_lab_packed.
// ---------------------------------------------------------------------
inline float3 lab_packed_to_rgb(float3 c)
{
    float L = 100.0 * c.x;
    float a = 2.0 * 127.0 * (c.y - 0.5);
    float b = 2.0 * 127.0 * (c.z - 0.5);

    float3 lab_unpacked = float3(L, a, b);
    float3 xyz          = lab_packed_to_xyz(lab_unpacked);
    float3 rgb          = xyz_to_rgb_d65(xyz);

    return rgb;
}
