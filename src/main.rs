mod wgpulib;
mod texture;
mod camera;
mod vertex_types;
mod voxels;
mod logger;
mod texture_packer;
mod world_handler;

use std::time::{Duration, SystemTime};
use wgpulib::run;

fn main() {
    pollster::block_on(run());
}

pub fn timeit<F: FnMut() -> T, T>(mut f: F) -> (T, Duration) {
    let start = SystemTime::now();
    let result = f();
    let end = SystemTime::now();
    let duration = end.duration_since(start).unwrap();
    (result, duration)
}
