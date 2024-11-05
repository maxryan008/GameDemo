use cgmath::{Vector2, Vector3};

pub const CHUNK_SIZE: usize = 2;
pub const CHUNK_SIZE_I32: i32 = CHUNK_SIZE as i32;
pub const CHUNK_SIZE_P: usize = CHUNK_SIZE + 2;
pub const CHUNK_SIZE_P2: usize = CHUNK_SIZE_P * CHUNK_SIZE_P;
pub const CHUNK_SIZE_P3: usize = CHUNK_SIZE_P * CHUNK_SIZE_P * CHUNK_SIZE_P;
pub const CHUNK_SIZE2: usize = CHUNK_SIZE * CHUNK_SIZE;
pub const CHUNK_SIZE2_I32: i32 = CHUNK_SIZE2 as i32;
pub const CHUNK_SIZE3: usize = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;

pub const ADJACENT_CHUNK_DIRECTIONS: [Vector3<i32>; 27] = [
    Vector3 { x: 0, y: 0, z: 0 },
    // moore neighbours in the negative direction
    Vector3 { x: 0, y: -1, z: -1 },
    Vector3 { x: -1, y: 0, z: -1 },
    Vector3 { x: -1, y: 0, z: 1 },
    Vector3 { x: -1, y: -1, z: 0 },
    Vector3 {
        x: -1,
        y: -1,
        z: -1,
    },
    Vector3 { x: -1, y: 1, z: -1 },
    Vector3 { x: -1, y: -1, z: 1 },
    Vector3 { x: -1, y: 1, z: 1 },
    Vector3 { x: 1, y: 0, z: -1 },
    Vector3 { x: 1, y: -1, z: -1 },
    Vector3 { x: 0, y: 1, z: -1 },
    Vector3 { x: 1, y: 1, z: 1 },
    Vector3 { x: 1, y: -1, z: 1 },
    Vector3 { x: 1, y: 1, z: -1 },
    Vector3 { x: 1, y: 1, z: 0 },
    Vector3 { x: 0, y: 1, z: 1 },
    Vector3 { x: 1, y: -1, z: 0 },
    Vector3 { x: 0, y: -1, z: 1 },
    Vector3 { x: 1, y: 0, z: 1 },
    Vector3 { x: -1, y: 1, z: 0 },
    // von neumann neighbour
    Vector3 { x: -1, y: 0, z: 0 },
    Vector3 { x: 1, y: 0, z: 0 },
    Vector3 { x: 0, y: -1, z: 0 },
    Vector3 { x: 0, y: 1, z: 0 },
    Vector3 { x: 0, y: 0, z: -1 },
    Vector3 { x: 0, y: 0, z: 1 },
];

pub const ADJACENT_AO_DIRS: [Vector2<i32>; 9] = [
    Vector2::new(-1, -1),
    Vector2::new(-1, 0),
    Vector2::new(-1, 1),
    Vector2::new(0, -1),
    Vector2::new(0, 0),
    Vector2::new(0, 1),
    Vector2::new(1, -1),
    Vector2::new(1, 0),
    Vector2::new(1, 1),
];