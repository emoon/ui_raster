#if defined(ISPC_TARGET_NEON)

// multiply two 16-bit values and return the high 16-bits of the result
int mul_16x8_high(int16 a, int16 b) {
    return @llvm.aarch64.neon.sqrdmulh(a, b);
}
    
int16 lerp_16x8_diff(int16 a, int16 diff, int16 t) {
    return return @llvm.aarch64.neon.sqrdmlah(a, diff, t);
}

int16 lerp_16x8(int16 a, int16 b, int16 t) {
    int16 diff = b - a;
    return return @llvm.aarch64.neon.sqrdmlah(a, diff, t);
}

#else

int mul_16x8_high(int16 a, int16 b) {
    return @llvm.x86.ssse3.pmul.hr.sw.128(a, b);
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
    uniform float data[8]; 
    uniform float width;
    uniform float height;
};

static const varying int16 shuffle_alpha = { 3,3,3,8, 7,7,7,8 };
static const varying int16 start_x_step_adjust  = { 0,0,0,0, 1,1,1,1 };
static const varying int16 shuffle_write_odd = { 4,5,6,7, 8,9,10,11 };
static const varying int16 shuffle_colors_left = { 0,1,2,3, 0,1,2,3 };
static const varying int16 shuffle_colors_right = { 4,5,6,7, 4,5,6,7 };
static const varying int shuffle_x0x0y0y0 = { 0,0,1,1, 0,0,1,1 };
static const varying int shuffle_x1x1y1y1 = { 2,2,3,3, 2,2,3,3 };
static const varying int shuffle_mix = { 0, 1, 2, 3, 8, 9, 10, 11 };
static const varying int shuffle_splat_alpha = { 3, 3, 3, 3, 7, 7, 7, 7 };

enum ProcessPixelCount {
    ProcessPixelCount_1 = 1,
    ProcessPixelCount_2 = 2,
};

template <ProcessPixelCount PixelCount, bool TextureAlignedUVs> 
    unmasked int process_pixel(int16 color, const uniform int16* texture_ptr, int16 fixed_texture_fraction, int texture_width) 
{
    varying int16 tex_rgba_rgba_0 = 0;
    varying int16 tex_rgba_rgba_1 = 0;
    varying int16 texture_fraction = fixed_texture_fraction;
    varying int16 tex_rgba0 = 0;

    if (TextureAlignedUVs) {
        // samples two texels from the texture
        tex_rgba_rgba_0 = texture_ptr[programIndex];
        tex_rgba_rgba_1 = texture_ptr[programIndex + texture_width];
    }

    if (TextureAlignedUVs) {
        varying int16 t0_t1 = lerp_16x8(tex_rgba_rgba_0, tex_rgba_rgba_1, extract(fixed_texture_fraction, 1)); // lerp in y direction
        varying int16 t = rotate(t0_t1, 4); // rotate so we can lerp in x direction
        tex_rgba0 = lerp_16x8(t, t, extract(fixed_texture_fraction, 0)); // lerp in x direction
    } 

    if (PixelCount == ProcessPixelCount_2) {
        if (TextureAlignedUVs) {
            // samples two texels from the texture
            tex_rgba_rgba_0 = texture_ptr[programIndex + 4];
            tex_rgba_rgba_1 = texture_ptr[programIndex + texture_width + 4];
        }

        if (TextureAlignedUVs) {
            varying int16 t0_t1 = lerp_16x8(tex_rgba_rgba_0, tex_rgba_rgba_1, extract(fixed_texture_fraction, 1)); // lerp in y direction
            varying int16 t = rotate(t0_t1, 4); // rotate so we can lerp in x direction
            varying int16 tex_rgba1 = lerp_16x8(t, t, extract(fixed_texture_fraction, 0)); // lerp in x direction
            tex_rgba0 = shuffle(tex_rgba0, tex_rgba1, shuffle_mix);
        } 
    }

    if (TextureAlignedUVs) {
        varying int16 one_minus_alpha = 0x7fff - shuffle(color, color, shuffle_splat_alpha); 
        color = lerp_16x8(color, tex_rgba0, one_minus_alpha);
    }

    return color;
}

template <bool ColorInterpolate, bool Clipping, bool Texture, bool TextureAlignedUVs> unmasked void render_internal(
    uniform int16 output[],
    const TileInfo* uniform tile_info,
    const uniform int16 texture_data[],
    const uniform float uv_data[],
    const uniform int texture_size[],
    const uniform float x0y0x1y1_data[],
    const uniform int16 top_colors_a[],
    const uniform int16 bottom_colors_a[])
{
    int16 top_colors = top_colors_a[programIndex];
    int16 bottom_colors = bottom_colors_a[programIndex];
    int16 fixed_texture_fraction = 0;

    // adjust for sub-pixel offset
    float x0y0x1y1_adjust = (x0y0x1y1_data[programIndex] - tile_info->data[programIndex]) + 0.5;
    float x0y0x1y1 = floor(x0y0x1y1_adjust);
    float uv = 0.0;

    if (Texture || TextureAlignedUVs) {
        uv = uv_data[programIndex];
        uv *= texture_size[programIndex];
        float uv_fraction = (x0y0x1y1_adjust - x0y0x1y1);

        if (Texture) {
            // Adjust UVs for sub-pixel offset based on the fractional part of the quads position
            uv += uv_fraction; 
        } else {
            // If we have aligned UVs (1:1) with the screen pixels we can use the fractional part
            // part when we sample the texture
            fixed_texture_fraction = (int16)(uv_fraction * 0x7FFF);
        }
    }

    float x_min_step = 0.0;
    float y_min_step = 0.0;

    int x0y0x1y1_int = (int)x0y0x1y1; 
    uniform int16 yi_step = 0; // step value y (0 - 1) across the quad (but 0 - 0x7FFF int16 range) 
    uniform int16 xi_step = 0; // step value x (0 - 1) across the quad (but 0 - 0x7FFF int16 range) 
    int16 xi_start = 0;
    int16 yi_start = 0;

    if (ColorInterpolate || Texture) {
        float x0x0y0y0 = shuffle(x0y0x1y1, shuffle_x0x0y0y0);
        float x1x1y1y1 = shuffle(x0y0x1y1, shuffle_x1x1y1y1);

        float xy_diff = x1x1y1y1 - x0x0y0y0;
        float xy_step = 32767.0 / xy_diff;

        uniform int x0_i = 0; 
        uniform int y0_i = 0; 

        if (Clipping) {
            x0_i = extract(x0y0x1y1_int, 0);
            y0_i = extract(x0y0x1y1_int, 1);
            y0_i = y0_i < 0 ? -y0_i : 0;
            x0_i = x0_i < 0 ? -x0_i : 0;
        }

        if (ColorInterpolate && Clipping) {
            xi_step = (uniform int16)extract(xy_step, 0);
            yi_step = (uniform int16)extract(xy_step, 1);

            yi_start = yi_step * y0_i;   // adjust yi start based on clipping
            xi_start = xi_step * x0_i;   // adjust xi start based on clipping 
        }

        if (Texture && Clipping) {

        }

        // The way we step across x is that we do two pixels at a time. Because of this we need
        // to adjust the stepping value to be times two and then adjust the starting value so that
        // is like this:
        // start: 0,1
        // step:  2,2

        xi_start += xi_step * start_x_step_adjust;
        xi_step *= 2;
    }

    const uniform int16* texture_ptr = 0;
    uniform int texture_width = 0;

    if (TextureAlignedUVs || Texture) {
        texture_width = extract(texture_size[programIndex], 0); 
        texture_ptr = &texture_data[0];

        if (TextureAlignedUVs) {
            uniform int u0 = (uniform int)extract(uv, 0);
            uniform int v0 = (uniform int)extract(uv, 1);
            uniform int texture_start = ((v0 * texture_width) + u0);
            texture_ptr += texture_start; 
        }
    } 
 
    uniform int x0 = extract(x0y0x1y1_int, 0);
    uniform int y0 = extract(x0y0x1y1_int, 1);
    uniform int x1 = extract(x0y0x1y1_int, 2);
    uniform int y1 = extract(x0y0x1y1_int, 3);

    if (Clipping) {
        x0 = max((uniform int)0, x0);
        y0 = max((uniform int)0, y0);
        x1 = min(x1, (uniform int)tile_info->width);
        y1 = min(y1, (uniform int)tile_info->height);
    }

    uniform int ylen = y1 - y0;
    uniform int xlen = x1 - x0;

    assume(ylen > 0);
    assume(xlen > 0);

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
        color_top_bottom_diff = bottom_colors - top_colors;
    }

    for (uniform int y = 0; y < ylen; ++y) {
        if (ColorInterpolate) {
            // colors on the left and right side of the quad
            int16 left_right_colors = lerp_16x8_diff(top_colors, color_top_bottom_diff, yi_start);
            left_colors = shuffle(left_right_colors, shuffle_colors_left);
            right_colors = shuffle(left_right_colors, shuffle_colors_right);
            color_diff = right_colors - left_colors;
        }

        x_step_current = xi_start;
                
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
            
            int16 rgba_out = process_pixel<ProcessPixelCount_2, TextureAlignedUVs>(
                current_color, texture_ptr, fixed_texture_fraction, texture_width);
        
            uniform int index = ((y * tile_info->width) + (x << 1)) * 4;
            tile_output[index + programIndex] = rgba_out;

            if (TextureAlignedUVs) {
                texture_ptr += 2;
            } 

            if (ColorInterpolate) {
                x_step_current += xi_step; 
            }
        }

        // Handle if we have un-even number of pixels to write
        if (xlen & 1) {
            uniform int index = ((y * tile_info->width) + xlen - 1) * 4;
            int16 backbuffer = tile_output[index + programIndex];
            int16 rgba_out = process_pixel<ProcessPixelCount_1, TextureAlignedUVs>(
                current_color, texture_ptr, fixed_texture_fraction, texture_width);
            // if we have an uneven pixel to write we need to load the current output
            // and mask it with the data we want to write so we only write the first part of it
            tile_output[index + programIndex] = shuffle(rgba_out, backbuffer, shuffle_write_odd);
        }

        if (ColorInterpolate) {
            yi_start += yi_step;
        }

        if (TextureAlignedUVs) {
            texture_ptr += texture_width - xlen;
        }
    }
}

export void ispc_raster_fixed_texture_solid(
    uniform int16 output[],
    const TileInfo* uniform tile_info,
    const uniform int16 texture_data[],
    const uniform float uv_data[],
    const uniform int texture_size[],
    const uniform float x0y0x1y1_data[],
    const uniform int16 top_colors_a[],
    const uniform int16 bottom_colors_a[])
{
    render_internal<false, false, false, true>(output, tile_info, texture_data, uv_data, texture_size, x0y0x1y1_data, top_colors_a, bottom_colors_a);
}

export void ispc_raster_rectangle_solid(
    uniform int16 output[],
    const TileInfo* uniform tile_info,
    const uniform float x0y0x1y1_data[],
    const uniform int16 top_colors_a[],
    const uniform int16 bottom_colors_a[])
{
    render_internal<false, false, false, false>(output, tile_info, 0, 0, 0, x0y0x1y1_data, top_colors_a, bottom_colors_a);
}

export void ispc_raster_rectangle_solid_lerp_color(
    uniform int16 output[],
    const TileInfo* uniform tile_info,
    const uniform float x0y0x1y1_data[],
    const uniform int16 top_colors[],
    const uniform int16 bottom_colors[])
{
    render_internal<true, false, false, false>(output, tile_info, 0, 0, 0, x0y0x1y1_data, top_colors, bottom_colors);
}

/*
export void ispc_raster_rectangle_texture(
    uniform int16 output[],
    const TileInfo* uniform tile_info,
    const uniform float x0y0x1y1[],
    const uniform int16 top_colors[],
    const uniform int16 bottom_colors[])
{

    render_internal<false, false>(output, tile_info, x0y0x1y1, top_colors, bottom_colors);
}
*/

