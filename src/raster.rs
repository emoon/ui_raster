use crate::simd::*;

pub struct Raster;

pub struct TileInfo {
    pub offsets: f32x4,
    pub width: i32,
    pub height: i32,
}

const TEXTURE_MODE_NONE: usize = 0;
const TEXTURE_MODE_ALIGNED: usize = 1;
const PIXEL_COUNT_1: usize = 1;
const PIXEL_COUNT_2: usize = 2;
const PIXEL_COUNT_3: usize = 3;
const PIXEL_COUNT_4: usize = 4;

const COLOR_MODE_NONE: usize = 0;
const COLOR_MODE_SOLID: usize = 1;
const COLOR_MODE_LERP: usize = 2;

impl Raster {
    #[inline(always)]
    fn sample_aligned_texture(
        texture: *const i16,
        texture_width: usize,
        u_fraction: i16x8,
        v_fraction: i16x8,
        offset: usize) -> i16x8
    {
        let rgba_rgba_0 = i16x8::load_unaligned_ptr(unsafe { texture.add(offset) });
        let rgba_rgba_1 = i16x8::load_unaligned_ptr(unsafe { texture.add((texture_width * 4) + offset) });
        let t0_t1 = i16x8::lerp(rgba_rgba_0, rgba_rgba_1, v_fraction);
        let t = t0_t1.rotate_4();
        i16x8::lerp(t0_t1, t, u_fraction)
    }

    #[inline(always)]
    fn interpolate_color(
        left_colors: i16x8,
        color_diff: i16x8,
        step: i16x8) -> i16x8
    {
        let color = i16x8::lerp_diff(left_colors, color_diff, step);
        // As we use pre-multiplied alpha we need to adjust the color based on the alpha value
        // This will generate a value that looks like:
        // A0 A0 A0 0x7fff A1 A1 A1 0x7fff 
        // so the alpha value will stay the same while the color is changed
        let alpha = color.shuffle_333_0x7fff_777_0x7fff();
        i16x8::mul_high(color, alpha)
    }

    #[inline(always)]
    fn process_pixels<const COUNT: usize, const COLOR_MODE: usize, const TEXTURE_MODE: usize>(
        mut color: i16x8,
        texture: *const i16,
        texture_width: usize,
        fixed_u_fraction: i16x8, 
        fixed_v_fraction: i16x8,
        color_diff: i16x8,
        left_colors: i16x8,
        x_step_current: i16x8,
        xi_step: i16x8) -> (i16x8, i16x8)
    {
        let mut color_0 = i16x8::new_splat(0);
        let mut color_1 = i16x8::new_splat(0);

        if COUNT == PIXEL_COUNT_4 {
            if TEXTURE_MODE == TEXTURE_MODE_ALIGNED {
                let t0 = Self::sample_aligned_texture(texture, texture_width, fixed_u_fraction, fixed_v_fraction, 0);
                let t1 = Self::sample_aligned_texture(texture, texture_width, fixed_u_fraction, fixed_v_fraction, 4);
                let t2 = Self::sample_aligned_texture(texture, texture_width, fixed_u_fraction, fixed_v_fraction, 8);
                let t3 = Self::sample_aligned_texture(texture, texture_width, fixed_u_fraction, fixed_v_fraction, 12);

                color_0 = i16x8::merge(t0, t1);
                color_1 = i16x8::merge(t2, t3);
            }

            if COLOR_MODE == COLOR_MODE_LERP {
                color_0 = Self::interpolate_color(left_colors, color_diff, x_step_current);
                color_1 = Self::interpolate_color(left_colors, color_diff, x_step_current + xi_step);
            }

        } else if COUNT == PIXEL_COUNT_3 {
            if TEXTURE_MODE == TEXTURE_MODE_ALIGNED {
                let t0 = Self::sample_aligned_texture(texture, texture_width, fixed_u_fraction, fixed_v_fraction, 0);
                let t1 = Self::sample_aligned_texture(texture, texture_width, fixed_u_fraction, fixed_v_fraction, 4);
                let t2 = Self::sample_aligned_texture(texture, texture_width, fixed_u_fraction, fixed_v_fraction, 8);

                color_0 = i16x8::merge(t0, t1);
                color_1 = t2;
            }

            if COLOR_MODE == COLOR_MODE_LERP {
                color_0 = Self::interpolate_color(left_colors, color_diff, x_step_current);
                color_1 = Self::interpolate_color(left_colors, color_diff, x_step_current + xi_step);
            }

        } else if COUNT == PIXEL_COUNT_2 {
            if TEXTURE_MODE == TEXTURE_MODE_ALIGNED {
                let t0 = Self::sample_aligned_texture(texture, texture_width, fixed_u_fraction, fixed_v_fraction, 0);
                let t1 = Self::sample_aligned_texture(texture, texture_width, fixed_u_fraction, fixed_v_fraction, 4);
                color_0 = i16x8::merge(t0, t1);
            }

            if COLOR_MODE == COLOR_MODE_LERP {
                color_1 = Self::interpolate_color(left_colors, color_diff, x_step_current + xi_step);
            }
        }

        (color_0, color_1) 
    }


    fn render_internal<const COLOR_MODE: usize, const TEXTURE_MODE: usize>(
        output: &mut [i16],
        texture_data: *const i16,
        tile_info: &TileInfo,
        uv_data: &[f32],
        texture_sizes: &[i32],
        coords: &[f32],
        top_colors: i16x8,
        bottom_colors: i16x8)
    {
        let x0y0x1y1_adjust = (f32x4::load_unaligned(coords) - tile_info.offsets) + f32x4::new_splat(0.5);
        let x0y0x1y1 = x0y0x1y1_adjust.floor();
        let x0y0x1y1_int = x0y0x1y1.as_i32x4(); 

        let mut xi_start = i16x8::new_splat(0);
        let mut yi_start = i16x8::new_splat(0);

        let mut fixed_u_fraction = i16x8::new_splat(0);
        let mut fixed_v_fraction = i16x8::new_splat(0);

        let mut xi_step = i16x8::new_splat(0);
        let mut yi_step = i16x8::new_splat(0);

        let mut texture_ptr = texture_data;//.as_ptr();
        let mut texture_width = 0;

        if TEXTURE_MODE == TEXTURE_MODE_ALIGNED {
            let texture_sizes = i32x4::load_unaligned(texture_sizes);
            let uv = f32x4::load_unaligned(uv_data) * texture_sizes.as_f32x4();
            let uv_i = uv.as_i32x4();

            let uv_fraction = (x0y0x1y1_adjust - x0y0x1y1) * f32x4::new_splat(0x7fff as f32);
            let uv_fraction = i16x8::new_splat(0x7fff) - uv_fraction.as_i32x4().as_i16x8();

            fixed_u_fraction = uv_fraction.splat_0000_0000();
            fixed_v_fraction = uv_fraction.splat_2222_2222();

            texture_width = texture_sizes.extract::<0>() as usize;

            let u = uv_i.extract::<0>() as usize; 
            let v = uv_i.extract::<1>() as usize;

            texture_ptr = unsafe { texture_ptr.add((v * texture_width + u) * 4) };
        }

        let x0 = x0y0x1y1_int.extract::<0>();
        let y0 = x0y0x1y1_int.extract::<1>();
        let x1 = x0y0x1y1_int.extract::<2>();
        let y1 = x0y0x1y1_int.extract::<3>();

        if COLOR_MODE == COLOR_MODE_LERP {
            let x0y0x0y0 = x0y0x1y1.shuffle_0101();
            let x1y1x1y1 = x0y0x1y1.shuffle_2323();

            let xy_diff = x1y1x1y1 - x0y0x0y0;
            let xy_step = f32x4::new_splat(32767.0) / xy_diff;

            let x0_i = if x0 < 0 { -x0 } else { 1 };
            let y0_i = if y0 < 0 { -y0 } else { 1 };

            xi_step = xy_step.as_i32x4().as_i16x8().splat_0000_0000() * i16x8::new_splat(x0_i as _);
            yi_step = xy_step.as_i32x4().as_i16x8().splat_2222_2222() * i16x8::new_splat(y0_i as _);

            // The way we step across x is that we do two pixels at a time. Because of this we need
            // to adjust the stepping value to be times two and then adjust the starting value so that
            // is like this:
            // start: 0,1
            // step:  2,2
            xi_start += xi_step * i16x8::new(0,0,0,0,1,1,1,1);
            xi_step = xi_step * i16x8::new_splat(2);
        }
        
        let x0 = x0.max(0);
        let y0 = y0.max(0);
        let x1 = x1.min(tile_info.width);
        let y1 = y1.min(tile_info.height);

        let ylen = y1 - y0;
        let xlen = x1 - x0;
        
        let tile_width = tile_info.width as usize;
        let current_color = i16x8::new_splat(0);
        let output = &mut output[((y0 as usize * tile_width + x0 as usize) * 4)..];
        let mut output_ptr = output.as_mut_ptr();

        let mut current_color = top_colors;
        let mut color_diff = i16x8::new_splat(0); 
        let mut color_top_bottom_diff = i16x8::new_splat(0);
        let mut left_colors = i16x8::new_splat(0);
        let mut right_colors = i16x8::new_splat(0);
        let mut x_step_current = i16x8::new_splat(0);

        if COLOR_MODE == COLOR_MODE_LERP {
            color_top_bottom_diff = bottom_colors - top_colors;
        }

        let mut tile_line_ptr = output_ptr;
        let mut texture_line_ptr = texture_ptr;

        for _y in 0..ylen {
            if COLOR_MODE == COLOR_MODE_LERP {
                let left_right_colors = i16x8::lerp_diff(top_colors, color_top_bottom_diff, yi_start);
                left_colors = left_right_colors.shuffle_0123_0123();
                right_colors = left_right_colors.shuffle_4567_4567();
                color_diff = right_colors - left_colors;
            }

            x_step_current = xi_start;

            for _x in 0..(xlen >> 2) {
                let (c0, c1) = Self::process_pixels::<PIXEL_COUNT_4, COLOR_MODE, TEXTURE_MODE>(
                    current_color,
                    texture_ptr,
                    texture_width,
                    fixed_u_fraction,
                    fixed_v_fraction,
                    color_diff,
                    left_colors,
                    x_step_current,
                    xi_step
                );

                c0.store_unaligned_ptr(output_ptr);
                c1.store_unaligned_ptr(unsafe { output_ptr.add(8) });

                output_ptr = unsafe { output_ptr.add(16) };

                if TEXTURE_MODE == TEXTURE_MODE_ALIGNED {
                    texture_ptr = unsafe { texture_ptr.add(16) };
                }

                if COLOR_MODE == COLOR_MODE_LERP {
                    x_step_current += xi_step * i16x8::new_splat(2);
                }
            }

            let xlen_rest = xlen & 3;

            if xlen_rest == 1 {
                let (c0, c1) = Self::process_pixels::<PIXEL_COUNT_3, COLOR_MODE, TEXTURE_MODE>(
                    current_color,
                    texture_ptr,
                    texture_width,
                    fixed_u_fraction,
                    fixed_v_fraction,
                    color_diff,
                    left_colors,
                    x_step_current,
                    xi_step
                );

                // TODO: Merge with output
                c0.store_unaligned_ptr(output_ptr);
                c1.store_unaligned_ptr(unsafe { output_ptr.add(8) });
            } else if xlen_rest == 2 {
                let (c0, _) = Self::process_pixels::<PIXEL_COUNT_2, COLOR_MODE, TEXTURE_MODE>(
                    current_color,
                    texture_ptr,
                    texture_width,
                    fixed_u_fraction,
                    fixed_v_fraction,
                    color_diff,
                    left_colors,
                    x_step_current,
                    xi_step
                );

                c0.store_unaligned_ptr(output_ptr);
            } else if xlen & 3 == 3 {
                let (c0, _) = Self::process_pixels::<PIXEL_COUNT_1, COLOR_MODE, TEXTURE_MODE>(
                    current_color,
                    texture_ptr,
                    texture_width,
                    fixed_u_fraction,
                    fixed_v_fraction,
                    color_diff,
                    left_colors,
                    x_step_current,
                    xi_step
                );
                    
                c0.store_unaligned_ptr(output_ptr);
            }

            tile_line_ptr = unsafe { tile_line_ptr.add(tile_width * 4) };
            output_ptr = tile_line_ptr;

            if TEXTURE_MODE == TEXTURE_MODE_ALIGNED {
                texture_line_ptr = unsafe { texture_line_ptr.add(texture_width * 4) };
                texture_ptr = texture_line_ptr;
            }

            if COLOR_MODE == COLOR_MODE_LERP {
                yi_start += yi_step;
            }
        }
    }

    #[inline(never)]
    pub fn render_aligned_texture(
        output: &mut [i16],
        tile_info: &TileInfo,
        coords: &[f32],
        texture_data: *const i16,
        uv_data: &[f32],
        texture_sizes: &[i32])
    {
        Self::render_internal::<COLOR_MODE_NONE, TEXTURE_MODE_ALIGNED>(
            output,
            texture_data,
            tile_info,
            uv_data,
            texture_sizes,
            coords,
            i16x8::new_splat(0),
            i16x8::new_splat(0),
        );
    }

    #[inline(never)]
    pub fn render_solid_lerp(
        output: &mut [i16],
        tile_info: &TileInfo,
        coords: &[f32],
        top_colors: i16x8,
        bottom_colors: i16x8)
    {
        let uv_data = [0.0];
        let texture_sizes = [0];

        Self::render_internal::<COLOR_MODE_LERP, TEXTURE_MODE_NONE>(
            output,
            std::ptr::null(),
            tile_info,
            &uv_data,
            &texture_sizes,
            coords,
            top_colors,
            bottom_colors
        );
    }
}
