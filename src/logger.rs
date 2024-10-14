use cgmath::Vector3;
use crate::voxels::VoxelVector;

pub fn log(record: &str)
{
    println!("GameDemo: {}", record);
}

pub fn log_raw(record: &str, value: Vector3<i32>)
{
    println!("GameDemo: {}{:?}", record, value);
}

pub fn log_voxel_types(record: VoxelVector)
{
    for voxel_type in record.voxels
    {
        println!("Id: {} - Name: {}", voxel_type.0, voxel_type.1.name);
    }
}

pub fn log_chunk(position: Vector3<i32>) {
    println!("GameDemo: Generating generate chunk at {:?}", position);
}