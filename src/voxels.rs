use std::collections::HashMap;
use std::hash::Hash;
use std::iter::Map;
use image::Rgba;
use crate::logger;

pub struct VoxelType
{
    pub name: String,
    pub texture: TexturePattern,
    pub tint: [f32; 3],
}

pub struct VoxelVector
{
    pub voxels: HashMap<u8, VoxelType>,
}

impl VoxelVector
{
    pub fn create(&mut self, id: u8, name: &str, tint: [f32; 3], pattern: TexturePattern)
    {
        self.voxels.insert(id, VoxelType::new(name.to_string(), pattern, tint));
    }

    pub fn new() -> Self
    {
        VoxelVector { voxels: HashMap::new()}
    }
}

impl VoxelType
{
    //texture 4x4 pixels
    pub fn new(name: String, pattern: TexturePattern, tint: [f32; 3],) -> Self
    {
        VoxelType
        {
            name,
            texture: pattern,
            tint,
        }
    }
}

pub struct TexturePattern
{
    pattern: [Rgba<u8>; 16],
}

impl TexturePattern
{
    pub fn new(rgb0: [u8; 4], rgb1: [u8; 4], rgb2: [u8; 4], rgb3: [u8; 4], rgb4: [u8; 4], rgb5: [u8; 4], rgb6: [u8; 4], rgb7: [u8; 4], rgb8: [u8; 4], rgb9: [u8; 4], rgb10: [u8; 4], rgb11: [u8; 4], rgb12: [u8; 4], rgb13: [u8; 4], rgb14: [u8; 4], rgb15: [u8; 4]) -> Self
    {
        TexturePattern
        {
            pattern: [Rgba {0: rgb0}, Rgba {0: rgb1}, Rgba {0: rgb2}, Rgba {0: rgb3}, Rgba {0: rgb4}, Rgba {0: rgb5}, Rgba {0: rgb6}, Rgba {0: rgb7}, Rgba {0: rgb8}, Rgba {0: rgb9}, Rgba {0: rgb10}, Rgba {0: rgb11}, Rgba {0: rgb12}, Rgba {0: rgb13}, Rgba {0: rgb14}, Rgba {0: rgb15}]
        }
    }
}

pub fn initialize() -> VoxelVector
{
    logger::log("Initializing voxels");
    let mut voxels: VoxelVector = VoxelVector::new();

    voxels.create(0, "red", [1.0, 0.0, 0.0], TexturePattern::new([1, 0, 0, 1],[1, 0, 0, 1],[1, 0, 0, 1],[1, 0, 0, 1],[1, 0, 0, 1],[1, 0, 0, 1],[1, 0, 0, 1],[1, 0, 0, 1],[1, 0, 0, 1],[1, 0, 0, 1],[1, 0, 0, 1],[1, 0, 0, 1],[1, 0, 0, 1],[1, 0, 0, 1],[1, 0, 0, 1],[1, 0, 0, 1]));
    voxels.create(1, "blue", [0.0, 0.0, 1.0], TexturePattern::new([0, 0, 1, 1],[0, 0, 1, 1],[0, 0, 1, 1],[0, 0, 1, 1],[0, 0, 1, 1],[0, 0, 1, 1],[0, 0, 1, 1],[0, 0, 1, 1],[0, 0, 1, 1],[0, 0, 1, 1],[0, 0, 1, 1],[0, 0, 1, 1],[0, 0, 1, 1],[0, 0, 1, 1],[0, 0, 1, 1],[0, 0, 1, 1]));
    voxels.create(2, "green", [0.0, 1.0, 0.0], TexturePattern::new([0, 0, 1, 1],[0, 0, 1, 1],[0, 0, 1, 1],[0, 0, 1, 1],[0, 0, 1, 1],[0, 0, 1, 1],[0, 0, 1, 1],[0, 0, 1, 1],[0, 0, 1, 1],[0, 0, 1, 1],[0, 0, 1, 1],[0, 0, 1, 1],[0, 0, 1, 1],[0, 0, 1, 1],[0, 0, 1, 1],[0, 0, 1, 1]));
    voxels.create(3, "purple", [1.0, 0.0, 1.0], TexturePattern::new([0, 0, 1, 1],[0, 0, 1, 1],[0, 0, 1, 1],[0, 0, 1, 1],[0, 0, 1, 1],[0, 0, 1, 1],[0, 0, 1, 1],[0, 0, 1, 1],[0, 0, 1, 1],[0, 0, 1, 1],[0, 0, 1, 1],[0, 0, 1, 1],[0, 0, 1, 1],[0, 0, 1, 1],[0, 0, 1, 1],[0, 0, 1, 1]));

    voxels
}