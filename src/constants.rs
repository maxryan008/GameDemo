use cgmath::{Vector2, Vector3};

pub const TEXTURE_OUTPUT_SIZE: u32 = 4096;

pub const CHUNK_SIZE: usize = 64;
pub const CHUNK_SIZE_I32: i32 = CHUNK_SIZE as i32;
pub const CHUNK_SIZE_P: usize = CHUNK_SIZE + 2;
pub const CHUNK_SIZE_P2: usize = CHUNK_SIZE_P * CHUNK_SIZE_P;
pub const CHUNK_SIZE_P3: usize = CHUNK_SIZE_P * CHUNK_SIZE_P * CHUNK_SIZE_P;
pub const CHUNK_SIZE2: usize = CHUNK_SIZE * CHUNK_SIZE;
pub const CHUNK_SIZE2_I32: i32 = CHUNK_SIZE2 as i32;
pub const CHUNK_SIZE3: usize = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;