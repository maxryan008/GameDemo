use cgmath::Vector3;
use crate::voxels::VoxelVector;
use crate::world_generation::World;

pub fn log(record: &str)
{
    println!("GameDemo: {}", record);
}

pub fn log_voxel_types(record: VoxelVector)
{
    for voxelType in record.voxels
    {
        println!("Id: {} - Name: {}", voxelType.0, voxelType.1.name);
    }
}

pub fn log_chunk(position: Vector3<u32>) {
    println!("GameDemo: Generating new chunk at {:?}", position);
}