use cgmath::Vector3;

#[derive(Copy, Clone)]
pub enum Direction {
    Left,
    Right,
    Down,
    Up,
    Back,
    Forward,
}

impl Direction {
    pub fn get_normal(&self) -> i32 {
        match self {
            Direction::Left => 0i32,
            Direction::Right => 1i32,
            Direction::Down => 2i32,
            Direction::Up => 3i32,
            Direction::Back => 4i32,
            Direction::Forward => 5i32,
        }
    }
    pub fn get_opposite(self) -> Self {
        match self {
            Direction::Left => Direction::Right,
            Direction::Right => Direction::Left,
            Direction::Down => Direction::Up,
            Direction::Up => Direction::Down,
            Direction::Back => Direction::Forward,
            Direction::Forward => Direction::Back,
        }
    }
}

pub struct Quad {
    pub direction: Direction,
    pub corners: [[i32; 3]; 4],
    pub color: [f32; 3],
}

impl Quad {
    // the input position is assumed to be a voxel's (0,0,0) pos
    // therefore right / up / forward are offset by 1
    #[inline]
    pub fn from_direction(direction: Direction, pos: Vector3<i32>, color: [f32; 3]) -> Self {
        let corners = match direction {
            Direction::Left => [
                [pos.x, pos.y, pos.z],
                [pos.x, pos.y, pos.z + 1],
                [pos.x, pos.y + 1, pos.z + 1],
                [pos.x, pos.y + 1, pos.z],
            ],
            Direction::Right => [
                [pos.x, pos.y + 1, pos.z],
                [pos.x, pos.y + 1, pos.z + 1],
                [pos.x, pos.y, pos.z + 1],
                [pos.x, pos.y, pos.z],
            ],
            Direction::Down => [
                [pos.x, pos.y, pos.z],
                [pos.x + 1, pos.y, pos.z],
                [pos.x + 1, pos.y, pos.z + 1],
                [pos.x, pos.y, pos.z + 1],
            ],
            Direction::Up => [
                [pos.x, pos.y, pos.z + 1],
                [pos.x + 1, pos.y, pos.z + 1],
                [pos.x + 1, pos.y, pos.z],
                [pos.x, pos.y, pos.z],
            ],
            Direction::Back => [
                [pos.x, pos.y, pos.z],
                [pos.x, pos.y + 1, pos.z],
                [pos.x + 1, pos.y + 1, pos.z],
                [pos.x + 1, pos.y, pos.z],
            ],
            Direction::Forward => [
                [pos.x + 1, pos.y, pos.z],
                [pos.x + 1, pos.y + 1, pos.z],
                [pos.x, pos.y + 1, pos.z],
                [pos.x, pos.y, pos.z],
            ],
        };

        Self {
            corners,
            direction,
            color,
        }
    }
}