#if defined(ISPC_TARGET_NEON)

// multiply two 16-bit values and return the high 16-bits of the result
int mul_16x8_high(int16 a, int16 b) {
    return @llvm.aarch64.neon.sqrdmulh(a, b);
}
    
return return @llvm.aarch64.neon.sqrdmulh(a, diff, t);

int16 lerp_16x8_diff(int16 a, int16 diff, int16 t) {
    return return @llvm.aarch64.neon.sqrdmlah(a, diff, t);
}

int16 lerp_16x8(int16 a, int16 b, int16 t) {
    int16 diff = b - a;
    return return @llvm.aarch64.neon.sqrdmlah(a, diff, t);
}

#else

int mul_16x8_high(int16 a, int16 b) {
    int16 temp = @llvm.x86.ssse3.pmul.hr.sw.128(a, b);
}

int16 lerp_16x8_diff(int16 a, int16 diff, int16 t) {
    int16 temp = mul_16x8_high(diff, t); 
    return a + temp;
}

int16 lerp_16x8(int16 a, int16 b, int16 t) {
    int16 diff = b - a;
    int16 temp = mul_16x8_high(diff, t);
    return a + temp;
}

#endif

struct TileInfo {
    uniform float data[4]; 
    uniform float width;
    uniform float height;
};

static const varying int16 shuffle_alpha = { 3,3,3,8, 7,7,7,8 };
static const varying int16 start_x_step_adjust  = { 0,0,0,0, 1,1,1,1 };
static const varying int16 shuffle_write_odd = { 4,5,6,7, 8,9,10,11 };

enum ProcessPixelCount {
    ProcessPixelCount_1 = 1,
    ProcessPixelCount_2 = 2,
};

template <ProcessPixelCount PixelCount> unmasked int process_pixel(int16 color) {
    return color;
}

template <bool ColorInterpolate, bool Clipping> unmasked void render_internal(
    uniform int16 output[],
    const TileInfo* uniform tile_info,
    const uniform float x0y0x1y1_data[],
    const int16 top_colors,
    const int16 bottom_colors)
{
    // adjust for sub-pixel offset
    float x0y0x1y1 = floor((x0y0x1y1_data[programIndex] - tile_info->data[programIndex]) + 0.5);
    float x_min_step = 0.0;
    float y_min_step = 0.0;

    int x0y0x1y1_int = (int)x0y0x1y1; 
    int16 yi_step = 0; // step value y (0 - 1) across the quad (but 0 - 0x7FFF int16 range) 
    int16 xi_step = 0; // step value x (0 - 1) across the quad (but 0 - 0x7FFF int16 range) 
    int16 xi_start = 0;
    int16 yi_start = 0;

    if (ColorInterpolate) {
        float x0x0y0y0 = shuffle(x0y0x1y1, x0y0x1y1, 0xAA);
        float x1x1y1y1 = shuffle(x0y0x1y1, x0y0x1y1, 0x55);

        float xy_diff = x1x1y1y1 - x0x0y0y0;
        float xy_step = 32767.0 / xy_diff;

        yi_step = (int16)extract(xy_step, 1);

        if (Clipping) {
            uniform int x0_i = extract(x0y0x1y1_int, 0);
            uniform int y0_i = extract(x0y0x1y1_int, 1);

            yi_start = yi_step * y0_i < 0 ? -y0_i : 0;   // adjust yi start based on clipping
            xi_start = xi_step * x0_i < 0 ? -x0_i : 0;   // adjust xi start based on clipping 
        }

        // The way we step across x is that we do two pixels at a time. Because of this we need
        // to adjust the stepping value to be times two and then adjust the starting value so that
        // is like this:
        // start: 0,1
        // step:  2,2

        xi_start += xi_step * start_x_step_adjust;
        xi_step *= 2;
    }

    uniform int x0 = extract(x0y0x1y1_int, 0);
    uniform int y0 = extract(x0y0x1y1_int, 1);
    uniform int x1 = extract(x0y0x1y1_int, 2);
    uniform int y1 = extract(x0y0x1y1_int, 3);

    if (Clipping) {
        x0 = max(0, x0);
        y0 = max(0, y0);
        x1 = min(x1, (uniform int)tile_info->width);
        y1 = min(y1, (uniform int)tile_info->height);
    }

    uniform int ylen = y1 - y0;
    uniform int xlen = x1 - x0;
    uniform int write_start = ((y0 * tile_info->width) + x0) * 4;
    uniform int16* tile_output = &output[write_start];

    // In the case the color is the same for all pixels we assume this is correct
    int16 current_color = top_colors;
    int16 color_diff = 0; 
    int16 color_top_bottom_diff = 0;
    int16 left_colors = 0;
    int16 right_colors = 0;
    int16 x_step_current = 0;

    if (ColorInterpolate) {
        color_top_bottom_diff = top_colors - bottom_colors;
    }

    for (uniform int y = 0; y < ylen; ++y) {
        if (ColorInterpolate) {
            // colors on the left and right side of the quad
            int16 left_right_colors = lerp_16x8_diff(top_colors, color_top_bottom_diff, yi_start);
            left_colors = shuffle(left_right_colors, left_right_colors, 0);
            right_colors = shuffle(left_right_colors, left_right_colors, 3);
            color_diff = left_colors - right_colors; 
        }
                
        x_step_current = xi_step;
                
        for (uniform int x = 0; x < (xlen >> 1); x++) {
            if (ColorInterpolate) {
                current_color = lerp_16x8_diff(left_colors, color_diff, x_step_current);
                // As we use pre-multiplied alpha we need to adjust the color based on the alpha value
                // This will generate a value that looks like:
                // A0 A0 A0 0x7fff A1 A1 A1 0x7fff 
                /// so the alpha value will stay the same while the color is changed
                int16 alpha = shuffle(current_color, (uniform int16)0x7fff, shuffle_alpha);
                current_color = mul_16x8_high(current_color, alpha);
            }
                    
            int16 rgba_out = process_pixel<ProcessPixelCount_2>(current_color);
        
            uniform int index = ((y * tile_info->width) + (x << 1)) * 4;
            tile_output[index + programIndex] = rgba_out;

            if (ColorInterpolate) {
                x_step_current += xi_step; 
            }
        }

        // Handle if we have un-even number of pixels to write
        if (xlen & 1) {
            uniform int index = ((y * tile_info->width) + xlen - 1) * 4;
            int16 backbuffer = tile_output[index + programIndex];
            int16 rgba_out = process_pixel<ProcessPixelCount_1>(current_color);
            // if we have an uneven pixel to write we need to load the current output
            // and mask it with the data we want to write so we only write the first part of it
            tile_output[index + programIndex] = shuffle(rgba_out, backbuffer, shuffle_write_odd);
        }

        if (ColorInterpolate) {
            yi_start += yi_step;
        }
    }
}

export void ispc_raster_rectangle_solid(
    uniform int16 output[],
    const TileInfo* uniform tile_info,
    const uniform float x0_data[],
    const uniform float y0_data[], 
    const uniform float x1_data[], 
    const uniform float y1_data[], 
    uniform int count)
{
}

export void ispc_raster_rectangle_solid_lerp_color(
    uniform int16 output[],
    const TileInfo* uniform tile_info,
    const uniform float x0_data[],
    const uniform float y0_data[], 
    const uniform float x1_data[], 
    const uniform float y1_data[], 
    uniform int count)
{
}

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
        float x0_f = floor((x0_data[i + programIndex] - tile_info->data[0]) + 0.5);
        float y0_f = floor((y0_data[i + programIndex] - tile_info->data[1]) + 0.5);
        float x1_f = floor((x1_data[i + programIndex] - tile_info->data[2]) + 0.5);
        float y1_f = floor((y1_data[i + programIndex] - tile_info->data[3]) + 0.5);

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
