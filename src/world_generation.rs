use std::collections::HashMap;
use cgmath::Vector3;
use rand::Rng;
use rand::rngs::ThreadRng;
use web_sys::js_sys::Math::floor;
use web_sys::wasm_bindgen::UnwrapThrowExt;
use wgpu::naga::Expression::Math;
use wgpu::naga::MathFunction::Floor;
use wgpu::VertexBufferLayout;
use crate::{logger, voxels};
use crate::vertex_types::WorldMeshVertex;
use crate::voxels::{VoxelType, VoxelVector};

pub struct ChunkRaw {
    pub vertices: Vec<WorldMeshVertex>,
    pub indices: Vec<u32>,
}

pub const CHUNK_SIZE: f32 = 16.0;

pub struct Chunk {
    pub voxels: Vec<u8>,
    pub position: Vector3<i32>,
}

impl Chunk {
    fn new(position: Vector3<i32>) -> Self
    {
        let mut voxels: Vec<u8> = Vec::new();
        let mut rng = ThreadRng::default();
        for i in 0..(CHUNK_SIZE as usize * CHUNK_SIZE as usize * CHUNK_SIZE as usize)
        {
            voxels.push(rng.gen_range(0..3) as u8);
        }
        Chunk
        {
            voxels,
            position,
        }
    }

    fn solid(position: Vector3<i32>, id: u8) -> Self
    {
        let mut voxels: Vec<u8> = Vec::new();
        for i in 0..(CHUNK_SIZE as usize * CHUNK_SIZE as usize * CHUNK_SIZE as usize)
        {
            voxels.push(id);
        }
        Chunk
        {
            voxels,
            position,
        }
    }

    pub fn to_raw_translucent(&self, voxel_map: &VoxelVector) -> ChunkRaw
    {
        //logger::log("Building translucent raw chunk");
        let mut vertices: Vec<WorldMeshVertex> = vec![];
        let mut indices: Vec<u32> = vec![];

        if self.voxels.len() != CHUNK_SIZE as usize * CHUNK_SIZE as usize * CHUNK_SIZE as usize
        {
            log::error!("Voxels length does not match size");
        }
        let voxel_iterator = self.voxels.iter();
        for (i, voxel) in voxel_iterator.enumerate()
        {
            let voxel_type: &VoxelType = voxel_map.voxels.get(voxel).unwrap_throw();
            if voxel_type.translucent {
                let x = (i % CHUNK_SIZE as usize) as f32;
                let y = (i as f32/CHUNK_SIZE % CHUNK_SIZE).floor();
                let z = (i as f32/(CHUNK_SIZE * CHUNK_SIZE) % CHUNK_SIZE).floor();
                if x == 0f32 || voxel_map.voxels.get(&self.voxels[i - 1]).unwrap().translucent
                {
                    let len: u32 = vertices.len() as u32;
                    vertices.push(WorldMeshVertex { //0
                        position: [0.0 + x, 0.0 + y, 0.0 + z],
                        tex_coords: [0.0, 0.0],
                        color: voxel_type.tint,
                    });
                    vertices.push(WorldMeshVertex { //1
                        position: [0.0 + x, 0.0 + y, 1.0 + z],
                        tex_coords: [0.0, 0.0],
                        color: voxel_type.tint,
                    });
                    vertices.push(WorldMeshVertex { //2
                        position: [0.0 + x, 1.0 + y, 1.0 + z],
                        tex_coords: [0.0, 0.0],
                        color: voxel_type.tint,
                    });
                    vertices.push(WorldMeshVertex { //3
                        position: [0.0 + x, 1.0 + y, 0.0 + z],
                        tex_coords: [0.0, 0.0],
                        color: voxel_type.tint,
                    });
                    indices.extend([len + 0, len + 1, len + 2, len + 2, len + 3, len + 0])
                }
                if x == CHUNK_SIZE - 1.0|| voxel_map.voxels.get(&self.voxels[i + 1]).unwrap().translucent
                {
                    let len: u32 = vertices.len() as u32;
                    vertices.push(WorldMeshVertex { //5
                        position: [1.0 + x, 0.0 + y, 1.0 + z],
                        tex_coords: [0.0, 0.0],
                        color: voxel_type.tint,
                    });
                    vertices.push(WorldMeshVertex { //4
                        position: [1.0 + x, 0.0 + y, 0.0 + z],
                        tex_coords: [0.0, 0.0],
                        color: voxel_type.tint,
                    });
                    vertices.push(WorldMeshVertex { //7
                        position: [1.0 + x, 1.0 + y, 0.0 + z],
                        tex_coords: [0.0, 0.0],
                        color: voxel_type.tint,
                    });
                    vertices.push(WorldMeshVertex { //6
                        position: [1.0 + x, 1.0 + y, 1.0 + z],
                        tex_coords: [0.0, 0.0],
                        color: voxel_type.tint,
                    });
                    indices.extend([len + 0, len + 1, len + 2, len + 2, len + 3, len + 0])
                }
                if y == 0f32 || voxel_map.voxels.get(&self.voxels[i - CHUNK_SIZE as usize]).unwrap().translucent
                {
                    let len: u32 = vertices.len() as u32;
                    vertices.push(WorldMeshVertex { //4
                        position: [1.0 + x, 0.0 + y, 0.0 + z],
                        tex_coords: [0.0, 0.0],
                        color: voxel_type.tint,
                    });
                    vertices.push(WorldMeshVertex { //5
                        position: [1.0 + x, 0.0 + y, 1.0 + z],
                        tex_coords: [0.0, 0.0],
                        color: voxel_type.tint,
                    });
                    vertices.push(WorldMeshVertex { //1
                        position: [0.0 + x, 0.0 + y, 1.0 + z],
                        tex_coords: [0.0, 0.0],
                        color: voxel_type.tint,
                    });
                    vertices.push(WorldMeshVertex { //0
                        position: [0.0 + x, 0.0 + y, 0.0 + z],
                        tex_coords: [0.0, 0.0],
                        color: voxel_type.tint,
                    });

                    indices.extend([len + 0, len + 1, len + 2, len + 2, len + 3, len + 0])
                }
                if y == CHUNK_SIZE - 1.0 || voxel_map.voxels.get(&self.voxels[i + CHUNK_SIZE as usize]).unwrap().translucent
                {
                    let len: u32 = vertices.len() as u32;
                    vertices.push(WorldMeshVertex { //3
                        position: [0.0 + x, 1.0 + y, 0.0 + z],
                        tex_coords: [0.0, 0.0],
                        color: voxel_type.tint,
                    });
                    vertices.push(WorldMeshVertex { //2
                        position: [0.0 + x, 1.0 + y, 1.0 + z],
                        tex_coords: [0.0, 0.0],
                        color: voxel_type.tint,
                    });
                    vertices.push(WorldMeshVertex { //6
                        position: [1.0 + x, 1.0 + y, 1.0 + z],
                        tex_coords: [0.0, 0.0],
                        color: voxel_type.tint,
                    });
                    vertices.push(WorldMeshVertex { //7
                        position: [1.0 + x, 1.0 + y, 0.0 + z],
                        tex_coords: [0.0, 0.0],
                        color: voxel_type.tint,
                    });
                    indices.extend([len + 0, len + 1, len + 2, len + 2, len + 3, len + 0])
                }
                if z == 0f32 || voxel_map.voxels.get(&self.voxels[i - CHUNK_SIZE as usize*CHUNK_SIZE as usize]).unwrap().translucent
                {
                    let len: u32 = vertices.len() as u32;
                    vertices.push(WorldMeshVertex { //3
                        position: [0.0 + x, 1.0 + y, 0.0 + z],
                        tex_coords: [0.0, 0.0],
                        color: voxel_type.tint,
                    });
                    vertices.push(WorldMeshVertex { //7
                        position: [1.0 + x, 1.0 + y, 0.0 + z],
                        tex_coords: [0.0, 0.0],
                        color: voxel_type.tint,
                    });
                    vertices.push(WorldMeshVertex { //4
                        position: [1.0 + x, 0.0 + y, 0.0 + z],
                        tex_coords: [0.0, 0.0],
                        color: voxel_type.tint,
                    });
                    vertices.push(WorldMeshVertex { //0
                        position: [0.0 + x, 0.0 + y, 0.0 + z],
                        tex_coords: [0.0, 0.0],
                        color: voxel_type.tint,
                    });
                    indices.extend([len + 0, len + 1, len + 2, len + 2, len + 3, len + 0])
                }
                if z == CHUNK_SIZE-1.0 || voxel_map.voxels.get(&self.voxels[i + CHUNK_SIZE as usize*CHUNK_SIZE as usize]).unwrap().translucent
                {
                    let len: u32 = vertices.len() as u32;
                    vertices.push(WorldMeshVertex { //6
                        position: [1.0 + x, 1.0 + y, 1.0 + z],
                        tex_coords: [0.0, 0.0],
                        color: voxel_type.tint,
                    });
                    vertices.push(WorldMeshVertex { //2
                        position: [0.0 + x, 1.0 + y, 1.0 + z],
                        tex_coords: [0.0, 0.0],
                        color: voxel_type.tint,
                    });
                    vertices.push(WorldMeshVertex { //1
                        position: [0.0 + x, 0.0 + y, 1.0 + z],
                        tex_coords: [0.0, 0.0],
                        color: voxel_type.tint,
                    });
                    vertices.push(WorldMeshVertex { //5
                        position: [1.0 + x, 0.0 + y, 1.0 + z],
                        tex_coords: [0.0, 0.0],
                        color: voxel_type.tint,
                    });
                    indices.extend([len + 0, len + 1, len + 2, len + 2, len + 3, len + 0])
                }
            }
        }

        ChunkRaw
        {
            vertices,
            indices,
        }
    }

    pub fn to_raw_opaque(&self, voxel_map: &VoxelVector) -> ChunkRaw
    {
        logger::log("Building opaque raw chunk");
        let mut vertices: Vec<WorldMeshVertex> = vec![];
        let mut indices: Vec<u32> = vec![];
        if self.voxels.len() != CHUNK_SIZE as usize * CHUNK_SIZE as usize * CHUNK_SIZE as usize
        {
            log::error!("Voxels length does not match size");
        }
        let voxel_iterator = self.voxels.iter();
        for (i, voxel) in voxel_iterator.enumerate()
        {
            let voxel_type: &VoxelType = voxel_map.voxels.get(voxel).unwrap_throw();
            if !voxel_type.translucent {
                let x = (i % CHUNK_SIZE as usize) as f32;
                let y = ((i as f32/CHUNK_SIZE) % CHUNK_SIZE).floor();
                let z = (i as f32/(CHUNK_SIZE*CHUNK_SIZE) % CHUNK_SIZE).floor();
                if x == 0f32 || voxel_map.voxels.get(&self.voxels[i - 1]).unwrap().translucent
                {
                    let len: u32 = vertices.len() as u32;
                    vertices.push(WorldMeshVertex { //0
                        position: [0.0 + x, 0.0 + y, 0.0 + z],
                        tex_coords: [0.0, 0.0],
                        color: voxel_type.tint,
                    });
                    vertices.push(WorldMeshVertex { //1
                        position: [0.0 + x, 0.0 + y, 1.0 + z],
                        tex_coords: [0.0, 0.0],
                        color: voxel_type.tint,
                    });
                    vertices.push(WorldMeshVertex { //2
                        position: [0.0 + x, 1.0 + y, 1.0 + z],
                        tex_coords: [0.0, 0.0],
                        color: voxel_type.tint,
                    });
                    vertices.push(WorldMeshVertex { //3
                        position: [0.0 + x, 1.0 + y, 0.0 + z],
                        tex_coords: [0.0, 0.0],
                        color: voxel_type.tint,
                    });
                    indices.extend([len + 0, len + 1, len + 2, len + 2, len + 3, len + 0])
                }
                if x == CHUNK_SIZE - 1.0 || voxel_map.voxels.get(&self.voxels[i + 1]).unwrap().translucent
                {
                    let len: u32 = vertices.len() as u32;
                    vertices.push(WorldMeshVertex { //5
                        position: [1.0 + x, 0.0 + y, 1.0 + z],
                        tex_coords: [0.0, 0.0],
                        color: voxel_type.tint,
                    });
                    vertices.push(WorldMeshVertex { //4
                        position: [1.0 + x, 0.0 + y, 0.0 + z],
                        tex_coords: [0.0, 0.0],
                        color: voxel_type.tint,
                    });
                    vertices.push(WorldMeshVertex { //7
                        position: [1.0 + x, 1.0 + y, 0.0 + z],
                        tex_coords: [0.0, 0.0],
                        color: voxel_type.tint,
                    });
                    vertices.push(WorldMeshVertex { //6
                        position: [1.0 + x, 1.0 + y, 1.0 + z],
                        tex_coords: [0.0, 0.0],
                        color: voxel_type.tint,
                    });
                    indices.extend([len + 0, len + 1, len + 2, len + 2, len + 3, len + 0])
                }
                if y == 0f32 || voxel_map.voxels.get(&self.voxels[i - CHUNK_SIZE as usize]).unwrap().translucent
                {
                    let len: u32 = vertices.len() as u32;
                    vertices.push(WorldMeshVertex { //4
                    position: [1.0 + x, 0.0 + y, 0.0 + z],
                    tex_coords: [0.0, 0.0],
                    color: voxel_type.tint,
                    });
                    vertices.push(WorldMeshVertex { //5
                        position: [1.0 + x, 0.0 + y, 1.0 + z],
                        tex_coords: [0.0, 0.0],
                        color: voxel_type.tint,
                    });
                    vertices.push(WorldMeshVertex { //1
                        position: [0.0 + x, 0.0 + y, 1.0 + z],
                        tex_coords: [0.0, 0.0],
                        color: voxel_type.tint,
                    });
                    vertices.push(WorldMeshVertex { //0
                        position: [0.0 + x, 0.0 + y, 0.0 + z],
                        tex_coords: [0.0, 0.0],
                        color: voxel_type.tint,
                    });

                    indices.extend([len + 0, len + 1, len + 2, len + 2, len + 3, len + 0])
                }
                if y == CHUNK_SIZE - 1.0 || voxel_map.voxels.get(&self.voxels[i + CHUNK_SIZE as usize]).unwrap().translucent
                {
                    let len: u32 = vertices.len() as u32;
                    vertices.push(WorldMeshVertex { //3
                        position: [0.0 + x, 1.0 + y, 0.0 + z],
                        tex_coords: [0.0, 0.0],
                        color: voxel_type.tint,
                    });
                    vertices.push(WorldMeshVertex { //2
                        position: [0.0 + x, 1.0 + y, 1.0 + z],
                        tex_coords: [0.0, 0.0],
                        color: voxel_type.tint,
                    });
                    vertices.push(WorldMeshVertex { //6
                        position: [1.0 + x, 1.0 + y, 1.0 + z],
                        tex_coords: [0.0, 0.0],
                        color: voxel_type.tint,
                    });
                    vertices.push(WorldMeshVertex { //7
                        position: [1.0 + x, 1.0 + y, 0.0 + z],
                        tex_coords: [0.0, 0.0],
                        color: voxel_type.tint,
                    });
                    indices.extend([len + 0, len + 1, len + 2, len + 2, len + 3, len + 0])
                }
                if z == 0f32 || voxel_map.voxels.get(&self.voxels[i - (CHUNK_SIZE as usize * CHUNK_SIZE as usize)]).unwrap().translucent
                {
                    let len: u32 = vertices.len() as u32;
                    vertices.push(WorldMeshVertex { //3
                        position: [0.0 + x, 1.0 + y, 0.0 + z],
                        tex_coords: [0.0, 0.0],
                        color: voxel_type.tint,
                    });
                    vertices.push(WorldMeshVertex { //7
                        position: [1.0 + x, 1.0 + y, 0.0 + z],
                        tex_coords: [0.0, 0.0],
                        color: voxel_type.tint,
                    });
                    vertices.push(WorldMeshVertex { //4
                        position: [1.0 + x, 0.0 + y, 0.0 + z],
                        tex_coords: [0.0, 0.0],
                        color: voxel_type.tint,
                    });
                    vertices.push(WorldMeshVertex { //0
                        position: [0.0 + x, 0.0 + y, 0.0 + z],
                        tex_coords: [0.0, 0.0],
                        color: voxel_type.tint,
                    });
                    indices.extend([len + 0, len + 1, len + 2, len + 2, len + 3, len + 0])
                }
                if z == CHUNK_SIZE - 1.0 || voxel_map.voxels.get(&self.voxels[i + CHUNK_SIZE as usize * CHUNK_SIZE as usize]).unwrap().translucent
                {
                    let len: u32 = vertices.len() as u32;
                    vertices.push(WorldMeshVertex { //6
                        position: [1.0 + x, 1.0 + y, 1.0 + z],
                        tex_coords: [0.0, 0.0],
                        color: voxel_type.tint,
                    });
                    vertices.push(WorldMeshVertex { //2
                        position: [0.0 + x, 1.0 + y, 1.0 + z],
                        tex_coords: [0.0, 0.0],
                        color: voxel_type.tint,
                    });
                    vertices.push(WorldMeshVertex { //1
                        position: [0.0 + x, 0.0 + y, 1.0 + z],
                        tex_coords: [0.0, 0.0],
                        color: voxel_type.tint,
                    });
                    vertices.push(WorldMeshVertex { //5
                        position: [1.0 + x, 0.0 + y, 1.0 + z],
                        tex_coords: [0.0, 0.0],
                        color: voxel_type.tint,
                    });
                    indices.extend([len + 0, len + 1, len + 2, len + 2, len + 3, len + 0])
                }
            }
        }
        logger::log("Built opaque raw chunk");

        ChunkRaw
        {
            vertices,
            indices,
        }
    }
}

pub struct World
{
    pub chunks: HashMap<Vector3<i32>, Chunk>,
    pub voxel_vector: VoxelVector,
}

impl World
{
    pub fn generate_chunk
    (
        &mut self,
        position: Vector3<i32>
    )
    {
        logger::log_chunk(position);
        if self.chunks.contains_key(&position)
        {
            self.chunks.remove(&position);
        }
        self.chunks.insert(position, Chunk::new(position));
    }

    pub fn set_chunk(&mut self, position: Vector3<i32>, id: u8) {
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