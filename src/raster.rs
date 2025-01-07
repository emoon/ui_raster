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

const BLEND_MODE_NONE: usize = 0;
const BLEND_MODE_COLOR_BG: usize = 1;
const BLEND_MODE_TEXTURE_COLOR_BG: usize = 2;

const ROUND_MODE_NONE: usize = 0;
const ROUND_MODE_ENABLED: usize = 1;

#[derive(Copy, Clone)]
enum Corner {
    TopLeft,
    TopRight,
    BottomRight,
    BottomLeft,
}

const CORNER_OFFSETS: [(f32, f32); 4] = [
    (1.0, 1.0), // TopLeft: No shift
    (0.0, 1.0), // TopRight: Shift down
    (1.0, 0.0), // BottemLeft: Shift right
    (0.0, 0.0), // BottomRight: Shift right and down
];

#[derive(Copy, Clone)]
pub enum BlendMode {
    None,
    WithBackground,
    WithTexture,
    WithTextureAndBackground,
    Enabled,
}

/// Calculates the blending factor for rounded corners in vectorized form.
///
/// # Parameters
/// - `center_y2`: Squared y-coordinates from circle centers.
/// - `current_x`: X-coordinates of current points.
/// - `circle_center_x`: X-coordinates of circle centers.
/// - `border_radius_v`: Vertical border radii of circles.
///
/// # Returns
/// A vector of 15-bit integers representing blending factors for anti-aliasing,
/// scaled to fit within 0 to 32767.
#[inline(always)]
fn calculate_rounding_blend(
    circle_y2: f32x4,
    current_x: f32x4,
    circle_center_x: f32x4,
    border_radius_v: f32x4,
) -> i16x8 {
    let t0 = current_x - circle_center_x;
    let circle_x2 = t0 * t0;
    let dist = (circle_x2 + circle_y2).sqrt();
    let dist_to_edge = dist - border_radius_v;

    let dist_to_edge =
        f32x4::new_splat(1.0) - dist_to_edge.clamp(f32x4::new_splat(0.0), f32x4::new_splat(1.0));

    (dist_to_edge * f32x4::new_splat(32767.0))
        .as_i32x4()
        .as_i16x8()
}

/// Samples aligned texture data with bilinear interpolation in vectorized form.
///
/// # Parameters
/// - `texture`: Pointer to the texture data.
/// - `texture_width`: Width of the texture in pixels.
/// - `u_fraction`, `v_fraction`: Fractions for bilinear interpolation in U and V directions.
/// - `offset`: Starting offset in the texture data.
///
/// # Returns
/// An `i16x8` vector with sampled and interpolated texture values.
#[inline(always)]
fn sample_aligned_texture(
    texture: *const i16,
    texture_width: usize,
    u_fraction: i16x8,
    v_fraction: i16x8,
    offset: usize,
) -> i16x8 {
    let rgba_rgba_0 = i16x8::load_unaligned_ptr(unsafe { texture.add(offset) });
    let rgba_rgba_1 =
        i16x8::load_unaligned_ptr(unsafe { texture.add((texture_width * 4) + offset) });
    let t0_t1 = i16x8::lerp(rgba_rgba_0, rgba_rgba_1, v_fraction);
    let t = t0_t1.rotate_4();
    i16x8::lerp(t0_t1, t, u_fraction)
}

#[inline(always)]
fn premultiply_alpha(color: i16x8) -> i16x8 {
    // As we use pre-multiplied alpha we need to adjust the color based on the alpha value
    // This will generate a value that looks like:
    // A0 A0 A0 0x7fff A1 A1 A1 0x7fff
    // so the alpha value will stay the same while the color is changed
    let alpha = color.shuffle_333_0x7fff_777_0x7fff();
    i16x8::mul_high(color, alpha)
}

#[inline(always)]
fn interpolate_color(left_colors: i16x8, color_diff: i16x8, step: i16x8) -> i16x8 {
    let color = i16x8::lerp_diff(left_colors, color_diff, step);
    premultiply_alpha(color)
}

#[inline(always)]
fn multi_sample_aligned_texture<const COUNT: usize>(
    texture: *const i16,
    width: usize,
    u: i16x8,
    v: i16x8,
) -> (i16x8, i16x8) {
    let zero = i16x8::new_splat(0);

    if COUNT == PIXEL_COUNT_1 {
        let t0 = sample_aligned_texture(texture, width, u, v, 0);
        (t0, zero)
    } else if COUNT == PIXEL_COUNT_2 {
        let t0 = sample_aligned_texture(texture, width, u, v, 0);
        let t1 = sample_aligned_texture(texture, width, u, v, 4);
        (i16x8::merge(t0, t1), zero)
    } else if COUNT == PIXEL_COUNT_3 {
        let t0 = sample_aligned_texture(texture, width, u, v, 0);
        let t1 = sample_aligned_texture(texture, width, u, v, 4);
        let t2 = sample_aligned_texture(texture, width, u, v, 8);
        (i16x8::merge(t0, t1), t2)
    } else if COUNT == PIXEL_COUNT_3 {
        let t0 = sample_aligned_texture(texture, width, u, v, 0);
        let t1 = sample_aligned_texture(texture, width, u, v, 4);
        let t2 = sample_aligned_texture(texture, width, u, v, 8);
        let t3 = sample_aligned_texture(texture, width, u, v, 12);
        (i16x8::merge(t0, t1), i16x8::merge(t2, t3))
    } else {
        unimplemented!()
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn process_pixels<
    const COUNT: usize,
    const COLOR_MODE: usize,
    const TEXTURE_MODE: usize,
    const BLEND_MODE: usize,
    const ROUND_MODE: usize,
>(
    output: *mut i16,
    fixed_color: i16x8,
    texture: *const i16,
    texture_width: usize,
    fixed_u_fraction: i16x8,
    fixed_v_fraction: i16x8,
    color_diff: i16x8,
    left_colors: i16x8,
    x_step_current: i16x8,
    xi_step: i16x8,
    c_blend: i16x8,
) {
    let mut color_0 = fixed_color;
    let mut color_1 = fixed_color;

    let mut tex_0 = i16x8::new_splat(0);
    let mut tex_1 = i16x8::new_splat(0);

    if TEXTURE_MODE == TEXTURE_MODE_ALIGNED {
        (tex_0, tex_1) = multi_sample_aligned_texture::<COUNT>(
            texture,
            texture_width,
            fixed_u_fraction,
            fixed_v_fraction,
        );
    }

    if COLOR_MODE == COLOR_MODE_LERP {
        color_0 = interpolate_color(left_colors, color_diff, x_step_current);
        color_1 = interpolate_color(left_colors, color_diff, x_step_current + xi_step);
    } else if TEXTURE_MODE == TEXTURE_MODE_ALIGNED {
        color_0 = tex_0;
        color_1 = tex_1;
    }

    if BLEND_MODE == BLEND_MODE_TEXTURE_COLOR_BG {
        // TODO: Blend between texture and color
    }

    // If we have rounded we need to adjust the color based on the distance to the circle center
    if ROUND_MODE == ROUND_MODE_ENABLED {
        if COUNT >= PIXEL_COUNT_3 {
            // At his point c0,c1 contains 4 colors that we need to blend based on the
            // distance to the circle center. So we need to splat distance for each radius
            // calculated to get the correct blending value.
            color_0 = i16x8::mul_high(color_0, c_blend.shuffle_0000_2222());
            color_1 = i16x8::mul_high(color_1, c_blend.shuffle_4444_6666());
        } else {
            // Only one or two pixels so we only need one shuffle/mul
            color_0 = i16x8::mul_high(color_0, c_blend.shuffle_0000_2222());
        }
    }

    match COUNT {
        PIXEL_COUNT_1 => color_0.store_unaligned_ptr_lower(output),
        PIXEL_COUNT_2 => color_0.store_unaligned_ptr(output),
        PIXEL_COUNT_3 => {
            color_0.store_unaligned_ptr(output);
            color_1.store_unaligned_ptr_lower(unsafe { output.add(8) });
        }
        PIXEL_COUNT_4 => {
            color_0.store_unaligned_ptr(output);
            color_1.store_unaligned_ptr(unsafe { output.add(8) });
        }
        _ => unimplemented!(),
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn render_internal<
    const COLOR_MODE: usize,
    const TEXTURE_MODE: usize,
    const ROUND_MODE: usize,
    const BLEND_MODE: usize,
>(
    output: &mut [i16],
    texture_data: *const i16,
    tile_info: &TileInfo,
    uv_data: &[f32],
    texture_sizes: &[i32],
    coords: &[f32],
    border_radius: f32,
    radius_direction: usize,
    top_colors: i16x8,
    bottom_colors: i16x8,
) {
    let x0y0x1y1_adjust =
        (f32x4::load_unaligned(coords) - tile_info.offsets) + f32x4::new_splat(0.5);
    let x0y0x1y1 = x0y0x1y1_adjust.floor();
    let x0y0x1y1_int = x0y0x1y1.as_i32x4();

    // Used for stepping for edges with radius
    let mut rounding_y_step = f32x4::new_splat(0.0);
    let mut rounding_x_step = f32x4::new_splat(0.0);
    let mut rounding_y_current = f32x4::new_splat(0.0);
    let mut rounding_x_current = f32x4::new_splat(0.0);
    let mut border_radius_v = f32x4::new_splat(0.0);
    let mut circle_center_x = f32x4::new_splat(0.0);
    let mut circle_center_y = f32x4::new_splat(0.0);

    let mut xi_start = i16x8::new_splat(0);
    let mut yi_start = i16x8::new_splat(0);

    let mut fixed_u_fraction = i16x8::new_splat(0);
    let mut fixed_v_fraction = i16x8::new_splat(0);

    let mut xi_step = i16x8::new_splat(0);
    let mut yi_step = i16x8::new_splat(0);

    let mut texture_ptr = texture_data; //.as_ptr();
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
        xi_start += xi_step * i16x8::new(0, 0, 0, 0, 1, 1, 1, 1);
        xi_step = xi_step * i16x8::new_splat(2);
    }

    // If we have rounded edges we need to adjust the start and end values
    if ROUND_MODE == ROUND_MODE_ENABLED {
        let x0f = x0y0x1y1_adjust.extract::<0>();
        let y0f = x0y0x1y1_adjust.extract::<1>();

        let center_adjust = CORNER_OFFSETS[radius_direction & 3];

        // TODO: Get the corret corner direction
        rounding_y_step = f32x4::new_splat(1.0);
        rounding_x_step = f32x4::new(4.0, 4.0, 4.0, 4.0);
        rounding_y_current = f32x4::new_splat(x0y0x1y1.extract::<1>());
        border_radius_v = f32x4::new_splat(border_radius);
        // TODO: Fixme
        circle_center_x = f32x4::new_splat(x0f + border_radius * center_adjust.0);
        circle_center_y = f32x4::new_splat(y0f + border_radius * center_adjust.1);
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
    let mut circle_y2 = f32x4::new_splat(0.0);
    let mut circle_distance = i16x8::new_splat(0);

    for _y in 0..ylen {
        // as y2 for the circle is constant in the inner loop we can calculate it here
        if ROUND_MODE == ROUND_MODE_ENABLED {
            let x0f = x0y0x1y1.extract::<0>();

            let t0 = rounding_y_current - circle_center_y;
            circle_y2 = t0 * t0;
            rounding_x_current = f32x4::new(x0f, x0f + 1.0, x0f + 2.0, x0f + 3.0);
        }

        if COLOR_MODE == COLOR_MODE_LERP {
            let left_right_colors = i16x8::lerp_diff(top_colors, color_top_bottom_diff, yi_start);
            left_colors = left_right_colors.shuffle_0123_0123();
            right_colors = left_right_colors.shuffle_4567_4567();
            color_diff = right_colors - left_colors;
        }

        x_step_current = xi_start;

        for _x in 0..(xlen >> 2) {
            // Calculate the distance to the circle center
            if ROUND_MODE == ROUND_MODE_ENABLED {
                circle_distance = calculate_rounding_blend(
                    circle_y2,
                    rounding_x_current,
                    circle_center_x,
                    border_radius_v,
                );
            }

            process_pixels::<PIXEL_COUNT_4, COLOR_MODE, TEXTURE_MODE, BLEND_MODE, ROUND_MODE>(
                output_ptr,
                current_color,
                texture_ptr,
                texture_width,
                fixed_u_fraction,
                fixed_v_fraction,
                color_diff,
                left_colors,
                x_step_current,
                xi_step,
                circle_distance,
            );

            output_ptr = unsafe { output_ptr.add(16) };

            if TEXTURE_MODE == TEXTURE_MODE_ALIGNED {
                texture_ptr = unsafe { texture_ptr.add(16) };
            }

            if COLOR_MODE == COLOR_MODE_LERP {
                x_step_current += xi_step * i16x8::new_splat(2);
            }

            if ROUND_MODE == ROUND_MODE_ENABLED {
                rounding_x_current += rounding_x_step;
            }
        }

        // Calculate the distance to the circle center
        if ROUND_MODE == ROUND_MODE_ENABLED {
            circle_distance = calculate_rounding_blend(
                circle_y2,
                rounding_x_current,
                circle_center_x,
                border_radius_v,
            );
        }

        // Process the remaining pixels
        match xlen & 3 {
            1 => {
                process_pixels::<PIXEL_COUNT_1, COLOR_MODE, TEXTURE_MODE, BLEND_MODE, ROUND_MODE>(
                    output_ptr,
                    current_color,
                    texture_ptr,
                    texture_width,
                    fixed_u_fraction,
                    fixed_v_fraction,
                    color_diff,
                    left_colors,
                    x_step_current,
                    xi_step,
                    circle_distance,
                );
            }
            2 => {
                process_pixels::<PIXEL_COUNT_2, COLOR_MODE, TEXTURE_MODE, BLEND_MODE, ROUND_MODE>(
                    output_ptr,
                    current_color,
                    texture_ptr,
                    texture_width,
                    fixed_u_fraction,
                    fixed_v_fraction,
                    color_diff,
                    left_colors,
                    x_step_current,
                    xi_step,
                    circle_distance,
                );
            }
            3 => {
                process_pixels::<PIXEL_COUNT_3, COLOR_MODE, TEXTURE_MODE, BLEND_MODE, ROUND_MODE>(
                    output_ptr,
                    current_color,
                    texture_ptr,
                    texture_width,
                    fixed_u_fraction,
                    fixed_v_fraction,
                    color_diff,
                    left_colors,
                    x_step_current,
                    xi_step,
                    circle_distance,
                );
            }
            _ => {}
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

        if ROUND_MODE == ROUND_MODE_ENABLED {
            rounding_y_current += rounding_y_step;
        }
    }
}

impl Raster {
    #[inline(never)]
    pub fn render_aligned_texture(
        output: &mut [i16],
        tile_info: &TileInfo,
        coords: &[f32],
        texture_data: *const i16,
        uv_data: &[f32],
        texture_sizes: &[i32],
    ) {
        render_internal::<COLOR_MODE_NONE, TEXTURE_MODE_ALIGNED, ROUND_MODE_NONE, BLEND_MODE_NONE>(
            output,
            texture_data,
            tile_info,
            uv_data,
            texture_sizes,
            coords,
            0.0,
            0,
            i16x8::new_splat(0),
            i16x8::new_splat(0),
        );
    }

    #[inline(never)]
    pub fn render_solid_quad(
        output: &mut [i16],
        tile_info: &TileInfo,
        coords: &[f32],
        color: i16x8,
        blend_mode: BlendMode,
    ) {
        let uv_data = [0.0];
        let texture_sizes = [0];

        render_internal::<COLOR_MODE_NONE, TEXTURE_MODE_NONE, ROUND_MODE_NONE, BLEND_MODE_NONE>(
            output,
            std::ptr::null(),
            tile_info,
            &uv_data,
            &texture_sizes,
            coords,
            0.0,
            0,
            color,
            color,
        );
    }

    #[inline(never)]
    fn render_soild_rounded_corner(
        output: &mut [i16],
        tile_info: &TileInfo,
        coords: &[f32],
        color: i16x8,
        radius: f32,
        blend_mode: BlendMode,
        corner: Corner,
    ) {
        let uv_data = [0.0];
        let texture_sizes = [0];

        render_internal::<COLOR_MODE_NONE, TEXTURE_MODE_NONE, ROUND_MODE_ENABLED, BLEND_MODE_NONE>(
            output,
            std::ptr::null(),
            tile_info,
            &uv_data,
            &texture_sizes,
            coords,
            radius,
            corner as usize,
            color,
            color,
        );
    }

    #[inline(never)]
    pub fn render_solid_quad_rounded(
        output: &mut [i16],
        tile_info: &TileInfo,
        coords: &[f32],
        color: i16x8,
        radius: f32,
        blend_mode: BlendMode,
    ) {
        let uv_data = [0.0];
        let texture_sizes = [0];

        // As we use pre-multiplied alpha we need to adjust the color based on the alpha value
        let color = premultiply_alpha(color);

        // First caluclate how large the corners has to be
        let corner_size = radius.ceil();

        let upper_left_coords = [
            coords[0],
            coords[1],
            coords[0] + corner_size,
            coords[1] + corner_size,
        ];

        // We first render all the corners as the icache has is warm with this code and the
        // branches we have should be predicted the same for all the corners.
        Self::render_soild_rounded_corner(
            output,
            tile_info,
            &upper_left_coords,
            color,
            radius,
            blend_mode,
            Corner::TopLeft,
        );

        let upper_right_coords = [
            coords[2] - corner_size,
            coords[1],
            coords[2],
            coords[1] + corner_size,
        ];

        Self::render_soild_rounded_corner(
            output,
            tile_info,
            &upper_right_coords,
            color,
            radius,
            blend_mode,
            Corner::TopRight,
        );

        let lower_right_coords = [
            coords[2] - corner_size,
            coords[3] - corner_size,
            coords[2],
            coords[3],
        ];

        Self::render_soild_rounded_corner(
            output,
            tile_info,
            &lower_right_coords,
            color,
            radius,
            blend_mode,
            Corner::BottomLeft,
        );

        let lower_left_coords = [
            coords[0],
            coords[3] - corner_size,
            coords[0] + corner_size,
            coords[3],
        ];

        Self::render_soild_rounded_corner(
            output,
            tile_info,
            &lower_left_coords,
            color,
            radius,
            blend_mode,
            Corner::BottomRight,
        );

        // Now we render the sides
        let top_coords = [
            coords[0] + corner_size,
            coords[1],
            coords[2] - corner_size,
            coords[1] + corner_size,
        ];

        Self::render_solid_quad(output, tile_info, &top_coords, color, blend_mode);

        let bottom_coords = [
            coords[0] + corner_size,
            coords[3] - corner_size,
            coords[2] - corner_size,
            coords[3],
        ];

        Self::render_solid_quad(output, tile_info, &bottom_coords, color, blend_mode);

        let left_coords = [
            coords[0],
            coords[1] + corner_size,
            coords[2],
            coords[3] - corner_size,
        ];

        Self::render_solid_quad(output, tile_info, &left_coords, color, blend_mode);
    }

    #[inline(never)]
    pub fn render_solid_lerp_radius(
        output: &mut [i16],
        tile_info: &TileInfo,
        coords: &[f32],
        radius: f32,
        top_colors: i16x8,
        bottom_colors: i16x8,
    ) {
        let uv_data = [0.0];
        let texture_sizes = [0];

        render_internal::<COLOR_MODE_LERP, TEXTURE_MODE_NONE, ROUND_MODE_ENABLED, BLEND_MODE_NONE>(
            output,
            std::ptr::null(),
            tile_info,
            &uv_data,
            &texture_sizes,
            coords,
            radius,
            2,
            top_colors,
            bottom_colors,
        );
    }
}
