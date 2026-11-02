// shaders/ops/geometry.slang
// Geometric transforms: flip, 90-degree rotations, crop (exact index
// remaps), wrapped tiling, and the general engine — inverse-warp affine with
// bilinear or nearest sampling and zeros outside the source
// (torchvision/kornia fill convention). The forward 2x3 matrix is inverted
// on the CPU: see kernels/geometry.py.
//
// Hot code follows the CUDA-path rules (experiments/cuda_hotloop):
// float4/scalar math only.

void flip(
    uint2               coord,
    Texture2D<float4>   src,
    [format(SLANGIMAGE_FMT)] RWTexture2D<float4> dst,
    int                 horizontal,
    int                 vertical
)
{
    uint width, height;
    src.GetDimensions(width, height);
    if (coord.x >= width || coord.y >= height)
        return;

    uint sx = horizontal != 0 ? (width - 1 - coord.x) : coord.x;
    uint sy = vertical != 0 ? (height - 1 - coord.y) : coord.y;
    dst[coord] = src[uint2(sx, sy)];
}

// k counter-clockwise quarter turns, matching numpy.rot90 exactly
// (verified in tests). dst dims are swapped for odd k.
void rot90(
    uint2               coord,
    Texture2D<float4>   src,
    [format(SLANGIMAGE_FMT)] RWTexture2D<float4> dst,
    int                 k
)
{
    uint sw, sh, dw, dh;
    src.GetDimensions(sw, sh);
    dst.GetDimensions(dw, dh);
    if (coord.x >= dw || coord.y >= dh)
        return;

    uint sx, sy;
    if (k == 1)      { sx = sw - 1 - coord.y; sy = coord.x; }
    else if (k == 2) { sx = sw - 1 - coord.x; sy = sh - 1 - coord.y; }
    else             { sx = coord.y; sy = sh - 1 - coord.x; }  // k == 3
    dst[coord] = src[uint2(sx, sy)];
}

void crop(
    uint2               coord,
    Texture2D<float4>   src,
    [format(SLANGIMAGE_FMT)] RWTexture2D<float4> dst,
    int                 x0,
    int                 y0
)
{
    uint dw, dh;
    dst.GetDimensions(dw, dh);
    if (coord.x >= dw || coord.y >= dh)
        return;
    // the rect is validated in Python to lie inside the source
    dst[coord] = src[uint2(coord.x + uint(x0), coord.y + uint(y0))];
}

int wrap_tile_index(int value, int size)
{
    int wrapped = value % size;
    return wrapped < 0 ? wrapped + size : wrapped;
}

// Repeat a scaled source over an explicit output extent. offset_x/y are
// source-coordinate phase offsets (in source pixels), applied after inverse
// scaling. Every bilinear tap wraps independently so interpolation is seamless
// across tile boundaries.
void tile_repeat(
    uint2               coord,
    Texture2D<float4>   src,
    [format(SLANGIMAGE_FMT)] RWTexture2D<float4> dst,
    float               scale_x,
    float               scale_y,
    float               offset_x,
    float               offset_y,
    int                 mode        // 0 = bilinear, 1 = nearest
)
{
    uint sw, sh, dw, dh;
    src.GetDimensions(sw, sh);
    dst.GetDimensions(dw, dh);
    if (coord.x >= dw || coord.y >= dh)
        return;

    // Python normalizes offsets to one non-negative period. Add one more
    // period so px/py below also stay positive: Vulkan truncates negative
    // float-to-int conversions in this shader instead of preserving floor().
    // The added period cannot change a wrapped sample.
    float sx = (float(coord.x) + 0.5) / scale_x + offset_x + float(sw);
    float sy = (float(coord.y) + 0.5) / scale_y + offset_y + float(sh);

    if (mode == 1)
    {
        int xi = wrap_tile_index(int(floor(sx)), int(sw));
        int yi = wrap_tile_index(int(floor(sy)), int(sh));
        dst[coord] = src[uint2(uint(xi), uint(yi))];
        return;
    }

    float px = sx - 0.5;
    float py = sy - 0.5;
    int x0 = int(floor(px));
    int y0 = int(floor(py));
    float fx = px - float(x0);
    float fy = py - float(y0);

    int x1 = wrap_tile_index(x0 + 1, int(sw));
    int y1 = wrap_tile_index(y0 + 1, int(sh));
    x0 = wrap_tile_index(x0, int(sw));
    y0 = wrap_tile_index(y0, int(sh));

    float4 top = (1.0 - fx) * src[uint2(uint(x0), uint(y0))]
               + fx * src[uint2(uint(x1), uint(y0))];
    float4 bottom = (1.0 - fx) * src[uint2(uint(x0), uint(y1))]
                  + fx * src[uint2(uint(x1), uint(y1))];
    dst[coord] = (1.0 - fy) * top + fy * bottom;
}

// Inverse-warp affine. inv_row0/inv_row1 map OUTPUT pixel-center coords
// to SOURCE pixel-center coords; taps outside the source contribute
// zeros (transparent black), like grid_sample / warp_affine fill=0.
void affine(
    uint2               coord,
    Texture2D<float4>   src,
    [format(SLANGIMAGE_FMT)] RWTexture2D<float4> dst,
    float3              inv_row0,   // (a, b, tx)
    float3              inv_row1,   // (c, d, ty)
    int                 mode        // 0 = bilinear, 1 = nearest
)
{
    uint sw, sh, dw, dh;
    src.GetDimensions(sw, sh);
    dst.GetDimensions(dw, dh);
    if (coord.x >= dw || coord.y >= dh)
        return;

    // pixel-center coordinates
    float xo = float(coord.x) + 0.5;
    float yo = float(coord.y) + 0.5;
    float sx = inv_row0.x * xo + inv_row0.y * yo + inv_row0.z;
    float sy = inv_row1.x * xo + inv_row1.y * yo + inv_row1.z;

    if (mode == 1)
    {
        int xi = int(floor(sx));
        int yi = int(floor(sy));
        bool inside = xi >= 0 && xi < int(sw) && yi >= 0 && yi < int(sh);
        dst[coord] = inside ? src[uint2(uint(xi), uint(yi))] : float4(0.0);
        return;
    }

    // bilinear over pixel centers; out-of-bounds taps are zero
    float px = sx - 0.5;
    float py = sy - 0.5;
    int x0 = int(floor(px));
    int y0 = int(floor(py));
    float fx = px - float(x0);
    float fy = py - float(y0);

    float4 acc = float4(0.0);
    float w00 = (1.0 - fx) * (1.0 - fy);
    float w10 = fx * (1.0 - fy);
    float w01 = (1.0 - fx) * fy;
    float w11 = fx * fy;

    if (x0 >= 0 && x0 < int(sw) && y0 >= 0 && y0 < int(sh))
        acc += w00 * src[uint2(uint(x0), uint(y0))];
    if (x0 + 1 >= 0 && x0 + 1 < int(sw) && y0 >= 0 && y0 < int(sh))
        acc += w10 * src[uint2(uint(x0 + 1), uint(y0))];
    if (x0 >= 0 && x0 < int(sw) && y0 + 1 >= 0 && y0 + 1 < int(sh))
        acc += w01 * src[uint2(uint(x0), uint(y0 + 1))];
    if (x0 + 1 >= 0 && x0 + 1 < int(sw) && y0 + 1 >= 0 && y0 + 1 < int(sh))
        acc += w11 * src[uint2(uint(x0 + 1), uint(y0 + 1))];

    dst[coord] = acc;
}
