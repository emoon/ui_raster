use crate::simd::*;

pub struct Raster;

pub struct TileInfo {
    pub offsets: f32x4,
    pub width: i32,
    pub height: i32,
}

const TEXTURE_MODE_NONE: usize = 0;
const TEXTURE_MODE_ALIGNED: usize = 1;
const PIXEL_COUNT_2: usize = 2;

impl Raster {
    fn process_pixel<const COUNT: usize, const TEXTURE_MODE: usize>(
        mut color: i16x8,
        texture: *const i16,
        texture_width: usize,
        fixed_u_fraction: i16x8, 
        fixed_v_fraction: i16x8) -> i16x8
    {
        let mut tex_rgba0 = i16x8::new_splat(0);

        if TEXTURE_MODE == TEXTURE_MODE_ALIGNED {
            let rgba_rgba_0 = i16x8::load_unaligned_ptr(texture);
            let rgba_rgba_1 = i16x8::load_unaligned_ptr(unsafe { texture.add(texture_width * 4) });
            let t0_t1 = i16x8::lerp(rgba_rgba_0, rgba_rgba_1, fixed_v_fraction);
            //let t = t0_t1.rotate_4();
            tex_rgba0 = i16x8::lerp(t0_t1, t0_t1, fixed_u_fraction);
        }

        if COUNT == 2 {
            if TEXTURE_MODE == TEXTURE_MODE_ALIGNED {
                let rgba_rgba_0 = i16x8::load_unaligned_ptr(unsafe { texture.add(4) });
                let rgba_rgba_1 = i16x8::load_unaligned_ptr(unsafe { texture.add((texture_width * 4) + 4)});
                let t0_t1 = i16x8::lerp(rgba_rgba_0, rgba_rgba_1, fixed_v_fraction);
                //let t = t0_t1.rotate_4();
                let rgba = i16x8::lerp(t0_t1, t0_t1, fixed_u_fraction);
                color = i16x8::merge(tex_rgba0, rgba);
            }
        } else {
            color = tex_rgba0;
        }

        color
    }


    fn render_internal<const TEXTURE_MODE: usize>(
        output: &mut [i16],
        texture_data: *const i16,
        tile_info: &TileInfo,
        uv_data: &[f32],
        texture_sizes: &[i32],
        coords: &[f32])
    {
        let x0y0x1y1_adjust = (f32x4::load_unaligned(coords) - tile_info.offsets) + f32x4::new_splat(0.5);
        let x0y0x1y1 = x0y0x1y1_adjust.floor();
        let x0y0x1y1_int = x0y0x1y1.as_i32x4(); 
        let mut fixed_u_fraction = i16x8::new_splat(0);
        let mut fixed_v_fraction = i16x8::new_splat(0);
        let mut texture_ptr = texture_data;//.as_ptr();
        let mut texture_width = 0;
        //

        if TEXTURE_MODE == TEXTURE_MODE_ALIGNED {
            let texture_sizes = i32x4::load_unaligned(texture_sizes);
            let uv = f32x4::load_unaligned(uv_data) * texture_sizes.as_f32x4();
            let uv_i = uv.as_i32x4();

            let uv_fraction = (x0y0x1y1_adjust - x0y0x1y1) * f32x4::new_splat(0x7fff as f32);
            let uv_fraction = i16x8::new_splat(0x7fff) - uv_fraction.as_i32x4().as_i16x8();

            fixed_u_fraction = uv_fraction.splat_1111_1111();
            fixed_v_fraction = uv_fraction.splat_3333_3333();
            
            texture_width = texture_sizes.extract::<0>() as usize;

            let u = uv_i.extract::<0>() as usize; 
            let v = uv_i.extract::<1>() as usize;

            unsafe { texture_ptr = texture_ptr.add((v * texture_width + u) * 4) };
        }

        let x0 = x0y0x1y1_int.extract::<0>();
        let y0 = x0y0x1y1_int.extract::<1>();
        let x1 = x0y0x1y1_int.extract::<2>();
        let y1 = x0y0x1y1_int.extract::<3>();

        let ylen = y1 - y0;
        let xlen = x1 - x0;

        let current_color = i16x8::new_splat(0);
        let mut output_ptr = output.as_mut_ptr();

        for _y in 0..ylen {
            for _x in (0..xlen).step_by(2) {
                let color = Self::process_pixel::<PIXEL_COUNT_2, TEXTURE_MODE>(
                    current_color,
                    texture_ptr,
                    texture_width,
                    fixed_u_fraction,
                    fixed_v_fraction,
                );

                color.store_unaligned_ptr(output_ptr);
                texture_ptr = unsafe { texture_ptr.add(8) };
                output_ptr = unsafe { output_ptr.add(8) };
            }

            if TEXTURE_MODE == TEXTURE_MODE_ALIGNED {
                texture_ptr = unsafe { texture_ptr.add((texture_width - xlen as usize) * 8) };
                output_ptr = unsafe { output_ptr.add(512) };
            }
        }
    }

    #[inline(never)]
    pub fn render_aligned_texture(
        output: &mut [i16],
        texture_data: *const i16,
        tile_info: &TileInfo,
        uv_data: &[f32],
        texture_sizes: &[i32],
        coords: &[f32])
    {
        Self::render_internal::<TEXTURE_MODE_ALIGNED>(
            output,
            texture_data,
            tile_info,
            uv_data,
            texture_sizes,
            coords,
        );
    }
}
