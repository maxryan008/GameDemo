use std::collections::HashMap;
use image::Rgba;
use crate::logger;

pub struct VoxelType
{
    pub name: String,
    pub texture: TexturePattern,
    pub translucent: bool,
    pub tint: [f32; 3],
    pub variants: u8,
    pub solid: bool,
}

pub struct VoxelVector
{
    pub voxels: HashMap<u32, VoxelType>,
}

impl VoxelVector
{
    pub fn create(&mut self, id: u32, name: &str, tint: [f32; 3], translucent: bool, solid: bool, variants: u8, pattern: TexturePattern)
    {
        self.voxels.insert(id, VoxelType::new(name.to_string(), pattern, tint, translucent, variants, solid));
    }

    pub fn get(&self, id: &u32) -> Option<&VoxelType>
    {
        self.voxels.get(id)
    }

    pub fn new() -> Self
    {
        VoxelVector { voxels: HashMap::new()}
    }
}

impl VoxelType
{
    //texture 4x4 pixels
    pub fn new(name: String, pattern: TexturePattern, tint: [f32; 3], translucent: bool, variants: u8, solid: bool) -> Self
    {
        VoxelType
        {
            name,
            translucent,
            texture: pattern,
            tint,
            variants,
            solid,
        }
    }

    pub fn is_solid(&self) -> bool
    {
        self.solid
    }
}

pub struct TexturePattern
{
    pub pattern: Vec<Rgba<u8>>,
}

impl TexturePattern
{
    pub fn new(rgb0: [u8; 4], rgb1: [u8; 4], rgb2: [u8; 4], rgb3: [u8; 4], rgb4: [u8; 4], rgb5: [u8; 4], rgb6: [u8; 4], rgb7: [u8; 4], rgb8: [u8; 4], rgb9: [u8; 4], rgb10: [u8; 4], rgb11: [u8; 4], rgb12: [u8; 4], rgb13: [u8; 4], rgb14: [u8; 4], rgb15: [u8; 4]) -> Self
    {
        TexturePattern
        {
            pattern: vec![Rgba {0: rgb0}, Rgba {0: rgb1}, Rgba {0: rgb2}, Rgba {0: rgb3}, Rgba {0: rgb4}, Rgba {0: rgb5}, Rgba {0: rgb6}, Rgba {0: rgb7}, Rgba {0: rgb8}, Rgba {0: rgb9}, Rgba {0: rgb10}, Rgba {0: rgb11}, Rgba {0: rgb12}, Rgba {0: rgb13}, Rgba {0: rgb14}, Rgba {0: rgb15}]
        }
    }

    pub fn from_vec(pattern: Vec<Rgba<u8>>) -> TexturePattern {
        TexturePattern
        {
            pattern,
        }
    }
}

pub fn initialize() -> VoxelVector
{
    logger::log("Initializing voxels");
    let mut voxels: VoxelVector = VoxelVector::new();

    voxels.create(0, "air", [0.0, 0.0, 0.0], false, false, 1, TexturePattern::new([0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0]));
    voxels.create(1, "red", [0.0, 0.0, 0.0], false, true, 1, TexturePattern::new([255, 0, 0, 255],[0, 255, 0, 255],[255, 0, 0, 255],[0, 255, 0, 255],[0, 255, 0, 255],[255, 0, 0, 255],[0, 255, 0, 255],[255, 0, 0, 255],[255, 0, 0, 255],[0, 255, 0, 255],[255, 0, 0, 255],[0, 255, 0, 255],[255, 0, 0, 255],[0, 255, 0, 255],[255, 0, 0, 255],[0, 255, 0, 255]));
    voxels.create(2, "blue", [0.0, 0.0, 0.0], false, true, 1, TexturePattern::new([0, 0, 255, 255],[0, 0, 255, 255],[0, 0, 255, 255],[0, 0, 255, 255],[0, 0, 255, 255],[0, 0, 255, 255],[0, 0, 255, 255],[0, 0, 255, 255],[0, 0, 255, 255],[0, 0, 255, 255],[0, 0, 255, 255],[0, 0, 255, 255],[0, 0, 255, 255],[0, 0, 255, 255],[0, 0, 255, 255],[0, 0, 255, 255]));
    voxels.create(3, "green", [0.0, 0.0, 0.0], false, true, 1, TexturePattern::new([0, 0, 255, 255],[0, 0, 255, 255],[0, 0, 255, 255],[0, 0, 255, 255],[0, 0, 255, 255],[0, 0, 255, 255],[0, 0, 255, 255],[0, 0, 255, 255],[0, 0, 255, 255],[0, 0, 255, 255],[0, 0, 255, 255],[0, 0, 255, 255],[0, 0, 255, 255],[0, 0, 255, 255],[0, 0, 255, 255],[0, 0, 255, 255]));
    voxels.create(4, "purple", [0.0, 0.0, 0.0], false, true, 1, TexturePattern::new([255, 0, 240, 255],[255, 0, 255, 255],[255, 0, 255, 255],[240, 0, 255, 255],[255, 0, 255, 255],[255, 0, 255, 255],[255, 0, 255, 255],[240, 0, 255, 240],[255, 0, 255, 255],[255, 0, 255, 255],[220, 0, 255, 255],[255, 0, 255, 255],[240, 0, 255, 255],[255, 0, 255, 255],[255, 0, 220, 255],[255, 0, 255, 255]));
    voxels.create(5, "dirt", [0.0, 0.0, 0.0], false, true , 3, TexturePattern::new([37, 21, 18, 255], [42, 23, 20, 255], [35, 20, 18, 255], [29, 16, 15, 255], [41, 24, 21, 255], [32, 18, 17, 255], [34, 20, 18, 255], [35, 20, 18, 255], [36, 21, 18, 255], [40, 23, 20, 255], [37, 21, 19, 255], [42, 23, 21, 255], [28, 16, 15, 255], [31, 18, 16, 255], [41, 24, 21, 255], [36, 20, 18, 255]));
    voxels.create(6, "grass", [-0.04, -0.01, -0.04], false, true, 10, TexturePattern::new([69, 98, 36, 255], [49, 67, 33, 255], [62, 87, 38, 255], [57, 77, 37, 255], [62, 87, 38, 255], [63, 89, 38, 255], [65, 93, 37, 255], [69, 101, 38, 255], [58, 80, 37, 255], [52, 72, 33, 255], [69, 100, 36, 255], [62, 87, 38, 255], [65, 92, 36, 255], [69, 101, 38, 255], [56, 76, 36, 255], [69, 101, 38, 255]));

    voxels
}