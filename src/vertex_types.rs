use std::fmt::Debug;
use wgpu::VertexFormat;
use crate::wgpulib::{PackedVec2, PackedVec3};

pub trait Vertex {
    fn desc() -> wgpu::VertexBufferLayout<'static>;
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct WorldMeshVertex {
    pub position: [f32; 3],
    pub tex_coords: [f32; 2],
    pub color: [f32; 3],
}


#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, Debug)]
pub struct ChunkMeshVertex {
    pub position: PackedVec3,  // 12 bytes for position
    pub tex_coords: PackedVec2, // 8 bytes for texture coordinates
}

impl ChunkMeshVertex {
    pub fn new(position: [f32; 3], tex_coords: [f32; 2]) -> Self {
        Self {position: PackedVec3 {
            x: position[0],
            y: position[1],
            z: position[2],
        }, tex_coords: PackedVec2 {
            x: tex_coords[0],
            y: tex_coords[1],
        }}
    }
}

impl Vertex for WorldMeshVertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: 32, // Step size between consecutive vertices
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,  // `position` starts at offset 0
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,  // vec3<f32> (16 bytes padding inclusive)
                },
                wgpu::VertexAttribute {
                    offset: 12, // `tex_coords` starts at offset 12, directly after `position`
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,  // vec2<f32> (8 bytes padding inclusive)
                },
            ]
        }
    }
}

impl Vertex for ChunkMeshVertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: 20,  // Total size: 12 bytes (PackedVec3) + 8 bytes (PackedVec2)
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // For PackedVec3: 3 floats for x, y, z
                wgpu::VertexAttribute {
                    offset: 0,  // `position` starts at byte 0
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,  // PackedVec3 -> 3 floats (x, y, z)
                },
                // For PackedVec2: 2 floats for texture coordinates (u, v)
                wgpu::VertexAttribute {
                    offset: 12, // `tex_coords` starts at byte 12
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,  // PackedVec2 -> 2 floats (u, v)
                },
            ],
        }
    }
}