use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use cgmath::{Array, Vector3};
use crate::constants::CHUNK_SIZE;
use crate::quad::Direction;
use crate::world_handler::ChunkData;

#[derive(Clone)]
pub struct ChunksRefs {
    pub chunks: Vec<Arc<ChunkData>>,
}

impl ChunksRefs {
    pub fn try_new(
        world_data: Arc<Mutex<HashMap<Vector3<i32>, Arc<ChunkData>>>>,
        middle_chunk: Vector3<i32>,
    ) -> Option<Self> {
        let mut chunks = vec![];
        for i in 0..3 * 3 * 3 {
            let offset = index_to_vector3i_bounds(i, 3) + Vector3::from_value(-1);
            if let Ok(chunks_data) = world_data.lock() {
                if let Some(chunk_data ) = chunks_data.get(&(middle_chunk + offset)) {
                    chunks.push(Arc::clone(
                        chunk_data,
                    ))
                } else {
                    return None;
                }
            } else {
                println!("ERRRORORR in try new");
                return None;
            }
        }
        Some(Self { chunks })
    }

    pub fn is_all_voxels_same(&self) -> bool {
        let first_voxel = self.chunks[0].get_voxel_if_filled();
        let Some(voxel) = first_voxel else {
            return false;
        };
        for chunk in self.chunks[1..].iter() {
            let option = chunk.get_voxel_if_filled();
            if let Some(v) = option {
                if voxel != v {
                    return false;
                }
            } else {
                return false;
            }
        }
        true
    }

    pub fn get_voxel(&self, pos: Vector3<i32>) -> &u32 {
        let x = (pos.x + CHUNK_SIZE as i32) as u32;
        let y = (pos.y + CHUNK_SIZE as i32) as u32;
        let z = (pos.z + CHUNK_SIZE as i32) as u32;
        let (x_chunk, x) = ((x / CHUNK_SIZE as u32) as i32, (x % CHUNK_SIZE as u32) as i32);
        let (y_chunk, y) = ((y / CHUNK_SIZE as u32) as i32, (y % CHUNK_SIZE as u32) as i32);
        let (z_chunk, z) = ((z / CHUNK_SIZE as u32) as i32, (z % CHUNK_SIZE as u32) as i32);

        let chunk_index = vector3i_to_index(Vector3::new(x_chunk, y_chunk, z_chunk), 3);
        let chunk_data = &self.chunks[chunk_index];
        let i = vector3i_to_index(Vector3::new(x, y, z), CHUNK_SIZE as i32);
        chunk_data.get_voxel(i)
    }

    pub fn get_voxel_no_neighbour(&self, pos: Vector3<i32>) -> &u32 {
        let chunk_data = &self.chunks[13];
        let i = vector3i_to_index(pos, CHUNK_SIZE as i32);
        chunk_data.get_voxel(i)
    }

    pub fn get_adjacent_voxels(
        &self,
        pos: Vector3<i32>,
    ) -> (&u32, &u32, &u32, &u32) {
        let current = self.get_voxel(pos);
        let back = self.get_voxel(pos + Vector3::new(0, 0, -1));
        let left = self.get_voxel(pos + Vector3::new(-1, 0, 0));
        let down = self.get_voxel(pos + Vector3::new(0, -1, 0));
        (current, back, left, down)
    }

    pub fn get_von_neumann(&self, pos: Vector3<i32>) -> Option<Vec<(Direction, &u32)>> {
        let mut result = vec![];
        result.push((Direction::Back, self.get_voxel(pos + Vector3::new(0, 0, -1))));
        result.push((Direction::Forward, self.get_voxel(pos + Vector3::new(0, 0, 1))));
        result.push((Direction::Down, self.get_voxel(pos + Vector3::new(0, -1, 0))));
        result.push((Direction::Up, self.get_voxel(pos + Vector3::new(0, 1, 0))));
        result.push((Direction::Left, self.get_voxel(pos + Vector3::new(-1, 0, 0))));
        result.push((Direction::Right, self.get_voxel(pos + Vector3::new(1, 0, 0))));
        Some(result)
    }

    pub fn get_2(&self, pos: Vector3<i32>, offset: Vector3<i32>) -> (&u32, &u32) {
        let first = self.get_voxel(pos);
        let second = self.get_voxel(pos + offset);
        (first, second)
    }
}

#[inline]
pub fn index_to_vector3i_bounds(i: i32, bounds: i32) -> Vector3<i32> {
    let x = i % bounds;
    let y = (i / bounds) % bounds;
    let z = i / (bounds * bounds);
    Vector3::new(x, y, z)
}

#[inline]
pub fn index_to_vector3i_bounds_reverse(i: i32, bounds: i32) -> Vector3<i32> {
    let z = i % bounds;
    let y = (i / bounds) % bounds;
    let x = i / (bounds * bounds);
    Vector3::new(x, y, z)
}

#[inline]
pub fn vector3i_to_index(pos: Vector3<i32>, bounds: i32) -> usize {
    let x_i = pos.x % bounds;
    let y_i = pos.y * bounds;
    let z_i = pos.z * bounds * bounds;
    (x_i + y_i + z_i) as usize
}