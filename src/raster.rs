use crate::simd::*;

struct Raster {


}

enum ColorInterpolation {
    None,
    Solid,
    Linear,
}

enum TextureMode {
    None,
    Aligned,
    Unaligned,
}

struct TileInfo {
    offsets: f32x4,
    width: i32,
    height: i32,
}

impl Raster {
    fn render_internal<const C: usize, const T: usize>(
        output: &mut [i16],
        tile_info: &TileInfo,
        uv_data: &[f32],
        texture_sizes: &[i32],
        coords: &[f32])
    {
        let x0y0x1y1_adjust = (f32x4::load_unaligned(coords) - tile_info.offsets) + f32x4::new_splat(0.5);
        let x0y0x1y1 = x0y0x1y1_adjust.floor();
        let mut fixed_u_fraction = i16x8::new_splat(0);
        let mut fixed_v_fraction = i16x8::new_splat(0);

        if T == 0 {
            let texture_sizes = i32x4::load_unaligned(texture_sizes);
            let uv = f32x4::load_unaligned(uv_data) * texture_sizes.as_f32x4();
            let uv_fraction = (x0y0x1y1_adjust - x0y0x1y1) * f32x4::new_splat(0x7fff as f32);
            let uv_fraction = i16x8::new_splat(0x7fff) - uv_fraction.as_i32x4().as_i16x8();
            fixed_u_fraction = uv_fraction.splat_1111_1111();
            fixed_v_fraction = uv_fraction.splat_3333_3333();
        }


        //let mut fixed_texture_fraction = i16x8::splat(0);
    }


    pub fn render_aligned_texture(output: &mut [i16]) {



    }
}
