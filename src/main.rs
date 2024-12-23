use ispc_rt::ispc_module;

ispc_module!(ui_raster);

fn main() {
    unsafe {
        ui_raster::ispc_raster();
    }
}
