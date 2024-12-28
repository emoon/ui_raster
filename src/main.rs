use ispc_rt::ispc_module;
use minifb::{Key, Window, WindowOptions, Scale};

ispc_module!(ui_raster);

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

    for i in 0..(1 << 8) {
        let srgb = i as f32 / 255.0;
        let linear = srgb_to_linear(srgb);
        table[i] = (linear * ((1 << LINEAR_BIT_COUNT) - 1) as f32) as u16;
    }

    table
}

// TODO: Verify that we are building the range correctly here
fn build_linear_to_srgb_table() -> [u8; 1 << SRGB_BIT_COUNT] {
    let mut table = [0; 1 << SRGB_BIT_COUNT];

    for i in 0..(1 << SRGB_BIT_COUNT) {
        let linear = i as f32 / ((1 << SRGB_BIT_COUNT) - 1) as f32;
        let srgb = linear_to_srgb(linear);
        table[i] = (srgb * (1 << 8) as f32) as u8;
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
            let r = tile[(tile_index * 4) + 0];
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

fn main() {
    let srgb_to_linear = build_srgb_to_linear_table();
    let linear_to_srgb = build_linear_to_srgb_table();

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

    while window.is_open() && !window.is_key_down(Key::Escape) {
        for i in output.iter_mut() {
            *i = 0;
        }

        let x0_data = [10.0f32, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0];
        let y0_data = [10.0f32, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0];
        let x1_data = [1101.0f32, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0];
        let y1_data = [500.0f32, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0];

        let tile_info = ui_raster::TileInfo {
            data: [0.0, 0.0, 0.0, 0.0],
            width: tile_width as _,
            height: tile_height as _,
        };

        unsafe {
            ui_raster::ispc_raster(
                output.as_mut_ptr(),
                &tile_info,
                x0_data.as_ptr(),
                y0_data.as_ptr(),
                x1_data.as_ptr(),
                y1_data.as_ptr(),
                1
            );
        }

        copy_tile_to_output_buffer(&mut buffer, &output, &linear_to_srgb, tile_width, tile_height, WIDTH);

        window
            .update_with_buffer(&buffer, WIDTH, HEIGHT)
            .unwrap();
    }
}
