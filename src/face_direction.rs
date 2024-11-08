use std::ops::Neg;
use cgmath::Vector3;
use crate::lod::Lod;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum FaceDir {
    Up,
    Down,
    Left,
    Right,
    Forward,
    Back,
}

pub const FACES: [FaceDir; 6] = [FaceDir::Up, FaceDir::Down, FaceDir::Left, FaceDir::Right, FaceDir::Forward, FaceDir::Back];

impl FaceDir {
    pub fn normal_index(&self) -> u32 {
        match self {
            FaceDir::Left => 0u32,
            FaceDir::Right => 1u32,
            FaceDir::Down => 2u32,
            FaceDir::Up => 3u32,
            FaceDir::Forward => 4u32,
            FaceDir::Back => 5u32,
        }
    }

    pub fn air_sample_dir(&self) -> Vector3<i32> {
        match self {
            FaceDir::Up => Vector3::unit_y(),
            FaceDir::Down => Vector3::neg(Vector3::unit_y()),
            FaceDir::Left => Vector3::neg(Vector3::unit_x()),
            FaceDir::Right => Vector3::unit_x(),
            FaceDir::Forward => Vector3::neg(Vector3::unit_z()),
            FaceDir::Back => Vector3::unit_z(),
        }
    }

    pub fn world_to_sample(&self, axis: i32, x: i32, y: i32, _lod: &Lod) -> Vector3<i32> {
        match self {
            FaceDir::Up => Vector3::new(x, axis + 1, y),
            FaceDir::Down => Vector3::new(x, axis, y),
            FaceDir::Left => Vector3::new(axis, y, x),
            FaceDir::Right => Vector3::new(axis + 1, y, x),
            FaceDir::Forward => Vector3::new(x, y, axis),
            FaceDir::Back => Vector3::new(x, y, axis + 1),
        }
    }

    pub fn reverse_order(&self) -> bool {
        match self {
            FaceDir::Up => true,      //+1
            FaceDir::Down => false,   //-1
            FaceDir::Left => false,   //-1
            FaceDir::Right => true,   //+1
            FaceDir::Forward => true, //-1
            FaceDir::Back => false,   //+1
        }
    }

    pub fn negate_axis(&self) -> i32 {
        match self {
            FaceDir::Up => -1,     //+1
            FaceDir::Down => 0,    //-1
            FaceDir::Left => 0,    //-1
            FaceDir::Right => -1,  //+1
            FaceDir::Forward => 0, //-1
            FaceDir::Back => 1,    //+1
        }
    }
}