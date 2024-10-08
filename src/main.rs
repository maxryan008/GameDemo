mod wgpulib;
mod texture;
mod camera;
mod vertex_types;
mod world_generation;
mod voxels;
mod logger;

use wgpulib::run;

fn main() {
    pollster::block_on(run());
}
