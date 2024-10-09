use std::collections::HashMap;
use cgmath::Vector3;
use env_logger::Logger;
use rand::distributions::{Distribution, Uniform};
use rand::Rng;
use rand::rngs::ThreadRng;
use web_sys::wasm_bindgen::UnwrapThrowExt;
use wgpu::naga::MathFunction::Log;
use crate::{logger, voxels};
use crate::vertex_types::WorldMeshVertex;
use crate::voxels::{VoxelType, VoxelVector};
//z-
//red
//2, 1, 0, 0, 3, 2
//z+
//blue
//4, 5, 6, 6, 7, 4
//x+
//5, 1, 2, 2, 6, 5
//x-
//3, 0, 4, 4, 7, 3
//y+
//3, 7, 6, 6, 2, 3
//y-
//0, 1, 5, 5, 4, 0
pub struct ChunkRaw {
    pub vertices: Vec<WorldMeshVertex>,
    pub indices: Vec<u32>,
}

struct Chunk {
    voxels: [u8; 4096],
    position: Vector3<u32>,
}

impl Chunk {
    fn new(position: Vector3<u32>) -> Self
    {
        let mut voxels: [u8; 4096] = [0; 4096];
        let mut rng = ThreadRng::default();
        for i in 0..voxels.len()
        {
            voxels[i] = (rng.gen_range(0..3) as u8);
        }
        Chunk
        {
            voxels,
            position,
        }
    }

    fn solid(position: Vector3<u32>, id: u8) -> Self
    {
        let mut voxels: [u8; 4096] = [id; 4096];
        Chunk
        {
            voxels,
            position,
        }
    }

    fn to_raw(&self, voxel_map: &VoxelVector, index_size: u32) -> ChunkRaw
    {
        logger::log("Building raw chunk");
        let mut vertices: Vec<WorldMeshVertex> = vec![];
        let mut indices: Vec<u32> = vec![];

        if self.voxels.len() != 4096
        {
            log::error!("Voxels length does not match size");
        }
        let position = self.position * 16;
        for xi in 0..16
        {
            let x = xi as f32;
            let xpos = xi * 256;
            for yi in 0..16
            {
                let y = yi as f32;
                let ypos = yi * 16;
                for zi in 0..16
                {
                    let z = zi as f32;
                    let block_pos = xpos+ypos+zi;
                    let voxel: &u8 = self.voxels.get(block_pos).unwrap().into();
                    let voxel_type: &VoxelType = voxel_map.voxels.get(voxel).unwrap_throw();
                    vertices.push(WorldMeshVertex { //0
                        position: [0.0 + x + position.x as f32, 0.0 + y + position.y as f32, 0.0 + z + position.z as f32],
                        tex_coords: [0.0, 0.0],
                        color: voxel_type.tint,
                    });
                    vertices.push(WorldMeshVertex { //1
                        position: [1.0 + x + position.x as f32, 0.0 + y + position.y as f32, 0.0 + z + position.z as f32],
                        tex_coords: [0.0, 0.0],
                        color: voxel_type.tint,
                    });
                    vertices.push(WorldMeshVertex { //2
                        position: [1.0 + x + position.x as f32, 1.0 + y + position.y as f32, 0.0 + z + position.z as f32],
                        tex_coords: [0.0, 0.0],
                        color: voxel_type.tint,
                    });
                    vertices.push(WorldMeshVertex { //3
                        position: [0.0 + x + position.x as f32, 1.0 + y + position.y as f32, 0.0 + z + position.z as f32],
                        tex_coords: [0.0, 0.0],
                        color: voxel_type.tint,
                    });
                    vertices.push(WorldMeshVertex { //4
                        position: [0.0 + x + position.x as f32, 0.0 + y + position.y as f32, 1.0 + z + position.z as f32],
                        tex_coords: [0.0, 0.0],
                        color: voxel_type.tint,
                    });
                    vertices.push(WorldMeshVertex { //5
                        position: [1.0 + x + position.x as f32, 0.0 + y + position.y as f32, 1.0 + z + position.z as f32],
                        tex_coords: [0.0, 0.0],
                        color: voxel_type.tint,
                    });
                    vertices.push(WorldMeshVertex { //6
                        position: [1.0 + x + position.x as f32, 1.0 + y + position.y as f32, 1.0 + z + position.z as f32],
                        tex_coords: [0.0, 0.0],
                        color: voxel_type.tint,
                    });
                    vertices.push(WorldMeshVertex { //7
                        position: [0.0 + x + position.x as f32, 1.0 + y + position.y as f32, 1.0 + z + position.z as f32],
                        tex_coords: [0.0, 0.0],
                        color: voxel_type.tint,
                    });
                    let len: u32 = ((indices.len() as u32 + index_size) as f32 / 4.5) as u32;
                    indices.append(&mut vec![2 + len, 1 + len, 0 + len, 0 + len, 3 + len, 2 + len /* */, 4 + len, 5 + len, 6 + len, 6 + len, 7 + len, 4 + len  /* */, 5 + len, 1 + len, 2 + len, 2 + len, 6 + len, 5 + len  /* */, 3 + len, 0 + len, 4 + len, 4 + len, 7 + len, 3 + len  /* */, 3 + len, 7 + len, 6 + len, 6 + len, 2 + len, 3 + len /* */, 0 + len, 1 + len, 5 + len, 5 + len, 4 + len, 0 + len]);
                    }
            }
        }

        ChunkRaw
        {
            vertices,
            indices,
        }
    }
}

pub struct World
{
    chunks: HashMap<Vector3<u32>, Chunk>,
    voxel_vector: VoxelVector,
}

impl World
{

    pub fn get_raw_chunk
    (
        &mut self,
        position: &Vector3<u32>,
        index_size: u32,
    ) -> ChunkRaw
    {
        self.chunks.get(position).unwrap().to_raw(&self.voxel_vector, index_size)
    }
    pub fn generate_chunk
    (
        &mut self,
        position: Vector3<u32>
    )
    {
        logger::log_chunk(position);
        self.chunks.insert(position, Chunk::new(position));
    }

    pub fn set_chunk(&mut self, position: Vector3<u32>, id: u8) {
        logger::log_chunk(position);
        self.chunks.remove(&position);
        self.chunks.insert(position, Chunk::solid(position, id));
    }

    pub fn new() -> Self
    {
        World
        {
            chunks: HashMap::new(),
            voxel_vector: voxels::initialize(),
        }
    }
}