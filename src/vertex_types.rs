use std::fmt::Debug;
use crate::wgpulib::{PackedVec2, PackedVec3};

pub trait Vertex {
    fn desc() -> wgpu::VertexBufferLayout<'static>;
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, Debug)]
pub struct ChunkMeshVertex {
    pub position: PackedVec3,  // 12 bytes for position
    pub tex_coords: PackedVec2, // 8 bytes for texture coordinates
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