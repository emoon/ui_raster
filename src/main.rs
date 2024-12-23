use ispc_rt::ispc_module;

ispc_module!(ui_raster);

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
        let srgb = i as f32 / (1 << 8) as f32;
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
    let _srgb_to_linear = build_srgb_to_linear_table();
    let _linear_to_srgb = build_linear_to_srgb_table();

    unsafe {
        ui_raster::ispc_raster();
    }
}
