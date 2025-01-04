use minifb::{Key, Scale, Window, WindowOptions};
use png::{Decoder, Transformations};
use std::fs::File;

//use ispc_rt::ispc_module;
//ispc_module!(ui_raster);

mod simd;
pub mod raster;

const WIDTH: usize = 1280;
const HEIGHT: usize = 720;

const SRGB_BIT_COUNT: u32 = 12;
const LINEAR_BIT_COUNT: u32 = 15;
const LINEAR_TO_SRGB_SHIFT: u32 = LINEAR_BIT_COUNT - SRGB_BIT_COUNT;

fn linear_to_srgb(x: f32) -> f32 {
    if x <= 0.0031308 {
        x * 12.92
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    }
}

fn srgb_to_linear(x: f32) -> f32 {
    if x <= 0.04045 {
        x / 12.92
    } else {
        ((x + 0.055) / 1.055).powf(2.4)
    }
}

// TODO: Verify that we are building the range correctly here
fn build_srgb_to_linear_table() -> [u16; 1 << 8] {
    let mut table = [0; 1 << 8];

    for (i, entry) in table.iter_mut().enumerate().take((1 << 8)) {
        let srgb = i as f32 / 255.0;
        let linear = srgb_to_linear(srgb);
        *entry = (linear * ((1 << LINEAR_BIT_COUNT) - 1) as f32) as u16;
    }

    table
}

// TODO: Verify that we are building the range correctly here
fn build_linear_to_srgb_table() -> [u8; 1 << SRGB_BIT_COUNT] {
    let mut table = [0; 1 << SRGB_BIT_COUNT];

    for (i, entry) in table.iter_mut().enumerate().take((1 << SRGB_BIT_COUNT)) {
        let linear = i as f32 / ((1 << SRGB_BIT_COUNT) - 1) as f32;
        let srgb = linear_to_srgb(linear);
        *entry = (srgb * (1 << 8) as f32) as u8;
    }

    table
}

// Reference implementation. This will run in hw on the device.
fn copy_tile_to_output_buffer(
    output: &mut [u32],
    tile: &[i16],
    linear_to_srgb: &[u8; 4096],
    tile_width: usize,
    tile_height: usize,
    width: usize,
) {
    let and_mask = (1 << SRGB_BIT_COUNT) - 1;

    for y in 0..tile_height {
        for x in 0..tile_width {
            let tile_index = (y * tile_width) + x;
            let output_index = (y * width) + x;
            let r = tile[tile_index * 4];
            let g = tile[(tile_index * 4) + 1];
            let b = tile[(tile_index * 4) + 2];

            let r = (r >> LINEAR_TO_SRGB_SHIFT) & and_mask;
            let g = (g >> LINEAR_TO_SRGB_SHIFT) & and_mask;
            let b = (b >> LINEAR_TO_SRGB_SHIFT) & and_mask;

            let r = linear_to_srgb[r as usize];
            let g = linear_to_srgb[g as usize];
            let b = linear_to_srgb[b as usize];

            let color = (r as u32) << 16 | (g as u32) << 8 | b as u32;
            output[output_index] = color;
        }
    }
}

struct Texture {
    data: Vec<u16>,
    width: usize,
    height: usize,
}

fn read_texture(path: &str, srgb_to_linear: &[u16; 256]) -> Texture {
   let mut decoder = Decoder::new(File::open(path).unwrap());

    // Reading the image in RGBA format.
    decoder.set_transformations(Transformations::ALPHA);

    let mut reader = decoder.read_info().unwrap();
    let size = reader.output_buffer_size();

    let mut u8_buffer = vec![0u8; size]; 
    let mut data = Vec::with_capacity(size * 4);

    // Read the next frame. Currently this function should only be called once.
    reader.next_frame(&mut u8_buffer).unwrap();

    for t in &u8_buffer {
        let temp = srgb_to_linear[*t as usize];
        data.push(temp);
    }

    for entry in data.iter_mut().take(512 * 4) {
        *entry = 0;
    }

    let width = reader.info().width as usize;
    let height = reader.info().height as usize;

    Texture {
        data,
        width,
        height,
    }
}


fn color_from_u8(srgb_to_linear: &[u16; 256], r: u8, g: u8, b: u8, a: u8) -> (i16, i16, i16, i16) {
    let r = srgb_to_linear[r as usize] as i16;
    let g = srgb_to_linear[g as usize] as i16;
    let b = srgb_to_linear[b as usize] as i16;
    (r, g, b, (a as i16) << 7)
} 

/*
fn generate_test_texture(srgb_to_linear: &[u16; 256]) -> Texture { 
    let mut texture = vec![0u16; 512 * 512 * 4];

    for y in 0..512 {
        let color = if y % 2 == 1 {
            srgb_to_linear[255]
        } else {
            srgb_to_linear[0]
        };

        for x in 0..256 {
            let index = (y * 512 + x) * 4;
            texture[index + 0] = color;
            texture[index + 1] = color;
            texture[index + 2] = color;
            texture[index + 3] = 0;
        }
    }

    Texture {
        data: texture,
        width: 512,
        height: 512,
    }
}
*/

fn main() {
    let srgb_to_linear = build_srgb_to_linear_table();
    let linear_to_srgb = build_linear_to_srgb_table();
    let texture = read_texture("assets/uv.png", &srgb_to_linear);
    //let texture = generate_test_texture(&srgb_to_linear); 

    let tile_width = 1280;
    let tile_height = 512;
    let mut output = vec![0i16; tile_width * tile_height * 4];

    let mut window = Window::new(
        "Test - ESC to exit",
        WIDTH,
        HEIGHT,
        WindowOptions {
            scale: Scale::X1,
            ..WindowOptions::default()
        },
    )
    .unwrap_or_else(|e| {
        panic!("{}", e);
    });

    let mut buffer: Vec<u32> = vec![0; WIDTH * HEIGHT];

    // Limit to max ~60 fps update rate
    window.set_target_fps(60);

    let mut y_pos = 0.0;

    while window.is_open() && !window.is_key_down(Key::Escape) {
        for i in output.iter_mut() {
            *i = 0;
        }

        /*
        let tile_info = ui_raster::TileInfo {
            data: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            width: tile_width as _,
            height: tile_height as _,
        };
        */

        let tile_info_2 = raster::TileInfo {
            offsets: crate::simd::f32x4::new_splat(0.0),
            width: tile_width as _,
            height: tile_height as _,
        };

        let t0 = color_from_u8(&srgb_to_linear, 0, 0, 0, 255);
        let t1 = color_from_u8(&srgb_to_linear, 128, 255, 0, 255);
        let top_colors = [t0.0, t0.1, t0.2, t0.3, t1.0, t1.1, t1.2, t1.3];

        let t0 = color_from_u8(&srgb_to_linear, 128, 0, 0, 255);
        let t1 = color_from_u8(&srgb_to_linear, 255, 255, 255, 255);
        let bottom_colors = [t0.0, t0.1, t0.2, t0.3, t1.0, t1.1, t1.2, t1.3];

        let coords = [
            10.0, 10.0 + y_pos, 100.0 * 4.0, (100.0 * 4.0) + y_pos,
            1.0, 1.0, 2.0, 2.0,
        ];

        /*
        unsafe {
            ui_raster::ispc_raster_rectangle_solid_lerp_color(
                output.as_mut_ptr(),
                &tile_info,
                coords.as_ptr(),
                top_colors.as_ptr(),
                bottom_colors.as_ptr(),
            );
        }
        */

        let uv_coords = [
            0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];

        let texture_sizes = [
            texture.width as i32, texture.height as i32, 
            texture.width as i32, texture.height as i32, 
            texture.width as i32, texture.height as i32, 
            texture.width as i32, texture.height as i32, 
        ];

        /*
        unsafe {
            ui_raster::ispc_raster_texture_aligned(
                output.as_mut_ptr(),
                &tile_info,
                texture.data.as_ptr() as *const i16,
                uv_coords.as_ptr(),
                texture_sizes.as_ptr(),
                coords.as_ptr(),
                top_colors.as_ptr(),
                bottom_colors.as_ptr(),
            );
        }
        */

        raster::Raster::render_aligned_texture(
            &mut output,
            texture.data.as_ptr() as *const i16,
            &tile_info_2,
            &uv_coords,
            &texture_sizes,
            &coords,
        );

        copy_tile_to_output_buffer(
            &mut buffer,
            &output,
            &linear_to_srgb,
            tile_width,
            tile_height,
            WIDTH,
        );

        window.update_with_buffer(&buffer, WIDTH, HEIGHT).unwrap();

        y_pos += 0.21;
    }
}
