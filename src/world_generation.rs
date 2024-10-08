use std::collections::HashMap;
use std::iter::Map;
use cgmath::Vector3;
use image::Rgba;
use web_sys::wasm_bindgen::UnwrapThrowExt;
use wgpu::Texture;
use crate::{logger, texture, vertex_types, voxels, wgpulib};
use crate::vertex_types::WorldMeshVertex;
use crate::voxels::{VoxelType, VoxelVector};
pub const VERTICES: &[vertex_types::WorldMeshVertex] = &[
    WorldMeshVertex { //0 red
        position: [0.0, 0.0, 0.0],
        tex_coords: [0.0, 0.0],
        color: [1.0, 0.0, 0.0],
    },
    WorldMeshVertex { //1 green
        position: [1.0, 0.0, 0.0],
        tex_coords: [0.0, 0.0],
        color: [0.0, 1.0, 0.0],
    },
    WorldMeshVertex { //2 blue
        position: [1.0, 1.0, 0.0],
        tex_coords: [0.0, 0.0],
        color: [1.0, 1.0, 0.0],
    },
    WorldMeshVertex { //3 purple
        position: [0.0, 1.0, 0.0],
        tex_coords: [0.0, 0.0],
        color: [0.0, 0.0, 1.0],
    },
    WorldMeshVertex { //4 yellow
        position: [0.0, 0.0, 1.0],
        tex_coords: [0.0, 0.0],
        color: [1.0, 0.0, 1.0],
    },
    WorldMeshVertex { //5 light blue
        position: [1.0, 0.0, 1.0],
        tex_coords: [0.0, 0.0],
        color: [0.0, 1.0, 1.0],
    },
    WorldMeshVertex { //6 white
        position: [1.0, 1.0, 1.0],
        tex_coords: [0.0, 0.0],
        color: [1.0, 1.0, 1.0],
    },
    WorldMeshVertex { //7 black
        position: [0.0, 1.0, 1.0],
        tex_coords: [0.0, 0.0],
        color: [0.0, 0.0, 0.0],
    },
];
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
pub const INDICES: &[u32] = &[2, 1, 0, 0, 3, 2 /* */, 4, 5, 6, 6, 7, 4  /* */, 5, 1, 2, 2, 6, 5  /* */, 3, 0, 4, 4, 7, 3  /* */, 3, 7, 6, 6, 2, 3 /* */, 0, 1, 5, 5, 4, 0];
pub struct ChunkRaw {
    pub vertices: Vec<WorldMeshVertex>,
    pub indices: Vec<u32>,
}

struct Chunk {
    voxels: Vec<u8>,
}

impl Chunk {
    fn new(position: Vector3<u32>) -> Self
    {
        let voxels: Vec<u8> = vec![0];
        Chunk
        {
            voxels,
        }
    }

    fn to_raw(&self, voxelMap: &VoxelVector) -> ChunkRaw
    {
        logger::log("Building raw chunk");
        let mut vertices: Vec<WorldMeshVertex> = vec![];
        let mut indices: Vec<u32> = vec![];

        for voxel in &self.voxels
        {
            let voxel_type: &VoxelType = voxelMap.voxels.get(voxel).unwrap_throw();
            vertices.push(WorldMeshVertex { //0
                position: [0.0, 0.0, 0.0],
                tex_coords: [0.0, 0.0],
                color: voxel_type.tint,
            });
            vertices.push(WorldMeshVertex { //1
                position: [1.0, 0.0, 0.0],
                tex_coords: [0.0, 0.0],
                color: voxel_type.tint,
            });
            vertices.push(WorldMeshVertex { //2
                position: [1.0, 1.0, 0.0],
                tex_coords: [0.0, 0.0],
                color: voxel_type.tint,
            });
            vertices.push(WorldMeshVertex { //3
                position: [0.0, 1.0, 0.0],
                tex_coords: [0.0, 0.0],
                color: voxel_type.tint,
            });
            vertices.push(WorldMeshVertex { //4
                position: [0.0, 0.0, 1.0],
                tex_coords: [0.0, 0.0],
                color: voxel_type.tint,
            });
            vertices.push(WorldMeshVertex { //5
                position: [1.0, 0.0, 1.0],
                tex_coords: [0.0, 0.0],
                color: voxel_type.tint,
            });
            vertices.push(WorldMeshVertex { //6
                position: [1.0, 1.0, 1.0],
                tex_coords: [0.0, 0.0],
                color: voxel_type.tint,
            });
            vertices.push(WorldMeshVertex { //7
                position: [0.0, 1.0, 1.0],
                tex_coords: [0.0, 0.0],
                color: voxel_type.tint,
            });

            indices.append(&mut vec![2, 1, 0, 0, 3, 2 /* */, 4, 5, 6, 6, 7, 4  /* */, 5, 1, 2, 2, 6, 5  /* */, 3, 0, 4, 4, 7, 3  /* */, 3, 7, 6, 6, 2, 3 /* */, 0, 1, 5, 5, 4, 0]);
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
    ) -> ChunkRaw
    {
        self.chunks.get(position).unwrap().to_raw(&self.voxel_vector)
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

    pub fn new() -> Self
    {
        World
        {
            chunks: HashMap::new(),
            voxel_vector: voxels::initialize(),
        }
    }
}