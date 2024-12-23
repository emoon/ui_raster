#if defined(ISPC_TARGET_NEON)

int16 lerp_16x8_diff(int16 a, int16 diff, int16 t) {
    return return @llvm.aarch64.neon.sqrdmulh(a, diff, t);
}

int16 lerp_16x8(int16 a, int16 b, int16 t) {
    int16 diff = b - a;
    return return @llvm.aarch64.neon.sqrdmulh(a, diff, t);
}

#else

int16 lerp_16x8_diff(int16 a, int16 diff, int16 t) {
    int16 temp = @llvm.x86.ssse3.pmul.hr.sw.128(diff, t);
    return a + temp;
}

int16 lerp_16x8(int16 a, int16 b, int16 t) {
    int16 diff = b - a;
    int16 temp = @llvm.x86.ssse3.pmul.hr.sw.128(diff, t);
    return a + temp;
}

#endif

struct TileInfo {
    uniform int width;
    uniform int height;
    uniform float offset_x;
    uniform float offset_y;
};

export void ispc_raster(
    uniform int16 output[],
    const TileInfo* uniform tile_info,
    const uniform float x0_data[],
    const uniform float y0_data[], 
    const uniform float x1_data[], 
    const uniform float y1_data[], 
    uniform int count)
{
    for (uniform int i = 0; i < count; i += programCount) {
        // adjust for sub-pixel offset
        float x0_f = floor((x0_data[i + programIndex] - tile_info->offset_x) + 0.5);
        float y0_f = floor((y0_data[i + programIndex] - tile_info->offset_y) + 0.5);
        float x1_f = floor((x1_data[i + programIndex] - tile_info->offset_x) + 0.5);
        float y1_f = floor((y1_data[i + programIndex] - tile_info->offset_y) + 0.5);

        float clipped_min_x = max(0.0, x0_f);
        float clipped_min_y = max(0.0, y0_f);
        float clipped_max_x = min(x1_f, (uniform float)tile_info->width);
        float clipped_max_y = min(y1_f, (uniform float)tile_info->height);

        uniform int channel_count = min(count - i, programCount);

        for (uniform int t = 0; t < channel_count; ++t) {
            uniform int x0 = extract((int)x0_f, t);
            uniform int y0 = extract((int)y0_f, t);
            uniform int x1 = extract((int)x1_f, t);
            uniform int y1 = extract((int)y1_f, t);

            uniform int ylen = y1 - y0;
            uniform int xlen = x1 - x0;
            uniform int write_start = ((y0 * tile_info->width) + x0) * 4;
            uniform int16* tile_output = &output[write_start];

            for (uniform int y = 0; y < ylen; ++y) {
                for (uniform int x = 0; x < xlen; ++x) {
                    uniform int index = ((y * tile_info->width) + x) * 4;
                    tile_output[index + programIndex] = 0x5555;
                }
            }
        }
    }
}
