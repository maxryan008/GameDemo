use std::collections::HashMap;
use std::sync::{mpsc, Arc, Mutex};
use std::sync::mpsc::{Receiver, Sender};
use std::thread;
use cgmath::{Vector2, Vector3};
use rand::distributions::Uniform;
use rand::prelude::{SliceRandom, ThreadRng};
use rand::Rng;
use web_sys::wasm_bindgen::UnwrapThrowExt;
use crate::{logger, voxels, wgpulib, world_handler};
use crate::chunks_refs::ChunksRefs;
use crate::constants::{CHUNK_SIZE, CHUNK_SIZE2, CHUNK_SIZE3};
use crate::face_direction::FaceDir;
use crate::lod::Lod;
use crate::quad::{Direction, Quad};
use crate::vertex_types::WorldMeshVertex;
use crate::voxels::{VoxelType, VoxelVector};
use crate::wgpulib::{ChunkInstance, ChunkInstanceRaw};

pub struct ChunkRaw {
    pub vertices: Vec<WorldMeshVertex>,
    pub indices: Vec<u32>,
}

#[derive(Default, Debug)]
pub struct ChunkMesh {
    pub instance: Vec<u8>,
    pub indices: Vec<u32>,
    pub vertices: Vec<WorldMeshVertex>,
}
pub struct Queue<T> {
    queue: Vec<T>,
}

impl<T> Queue<T> {
    fn new() -> Self {
        Queue { queue: Vec::new() }
    }

    fn enqueue(&mut self, item: T) {
        self.queue.push(item)
    }

    fn dequeue(&mut self) -> Option<T> {
        if self.queue.is_empty() {
            None
        } else {
            Some(self.queue.remove(0))
        }
    }

    fn length(&self) -> usize {
        self.queue.len()
    }

    fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    fn drain(&mut self) -> Vec<T> {
        self.queue.drain(..).collect()
    }

    fn clear(&mut self) {
        self.queue.clear();
    }
}

pub struct ArcQueue<T> {
    queue: Arc<Mutex<Queue<T>>>,
}

impl<T> ArcQueue<T> {
    pub fn new() -> Self {
        ArcQueue {
            queue: Arc::new(Mutex::new(Queue::<T>::new())),
        }
    }

    pub fn drain(&self) -> Vec<T> {
        let mut queue = self.queue.lock().unwrap();
        queue.drain()
    }

    pub fn clear(&self) {
        let mut queue = self.queue.lock().unwrap();
        queue.clear();
    }

    pub fn enqueue(&self, item: T) {
        let mut queue = self.queue.lock().unwrap();
        queue.enqueue(item);
    }

    pub fn dequeue(&self) -> Option<T> {
        let mut queue = self.queue.lock().unwrap();
        queue.dequeue()
    }

    pub fn is_empty(&self) -> bool {
        let queue = self.queue.lock().unwrap();
        queue.is_empty()
    }

    pub fn clone(&self) -> ArcQueue<T> {
        ArcQueue {
            queue: Arc::clone(&self.queue),
        }
    }

    pub fn length(&self) -> usize {
        let queue = self.queue.lock().unwrap();
        queue.length()
    }
}

pub struct DataWorker<T> {
    queue: ArcQueue<T>,           // Shared queue for incoming items
    _handle: thread::JoinHandle<()>, // DataWorker thread handle
}

impl<T: Send + 'static> DataWorker<T> {
    // Function to create and start the worker, accepting a processing function
    pub fn new<F>(
        queue: ArcQueue<T>,
        process_item: F,
        voxel_vector: Arc<voxels::VoxelVector>,
        atlas_data: Arc<HashMap<u32, (Vec<Vector2<f32>>, f32, f32)>>,
        result_queue: ArcQueue<(Vector3<i32>, ChunkData)>,
        chunks_data: Arc<Mutex<HashMap<Vector3<i32>, Arc<ChunkData>>>>,
    ) -> Self
    where
        F: Fn(T, Arc<voxels::VoxelVector>, Arc<HashMap<u32, (Vec<Vector2<f32>>, f32, f32)>>, Arc<Mutex<HashMap<Vector3<i32>, Arc<ChunkData>>>>) -> (Vector3<i32>, ChunkData) + Send + 'static + Copy,
    {
        let queue_clone = queue.clone();
        let voxel_vector_clone = Arc::clone(&voxel_vector);
        let atlas_data_clone = Arc::clone(&atlas_data);
        let chunks_data_clone = Arc::clone(&chunks_data);
        let result_queue_clone = result_queue.clone();

        let handle = thread::spawn(move || loop {
            if let Some(item) = queue_clone.dequeue() {
                let result = process_item(
                    item,
                    Arc::clone(&voxel_vector_clone),
                    Arc::clone(&atlas_data_clone),
                    Arc::clone(&chunks_data_clone),
                );
                result_queue_clone.enqueue(result);
            } else {
                thread::sleep(std::time::Duration::from_millis(10));
            }
        });

        DataWorker {
            queue,
            _handle: handle,
        }
    }

    pub fn enqueue(&self, item: T) {
        self.queue.enqueue(item);
    }
}

pub struct MeshWorker<T> {
    queue: ArcQueue<T>,           // Shared queue for incoming items
    _handle: thread::JoinHandle<()>, // DataWorker thread handle
}

impl<T: Send + 'static + Clone> MeshWorker<T> {
    // Function to create and start the worker, accepting a processing function
    pub fn new<F>(
        queue: ArcQueue<T>,
        process_item: F,
        voxel_vector: Arc<voxels::VoxelVector>,
        atlas_data: Arc<HashMap<u32, (Vec<Vector2<f32>>, f32, f32)>>,
        result_queue: ArcQueue<(Vector3<i32>, Option<ChunkMesh>)>,
        chunks_data: Arc<Mutex<HashMap<Vector3<i32>, Arc<ChunkData>>>>,
    ) -> Self
    where
        F: Fn(T, Arc<voxels::VoxelVector>, Arc<HashMap<u32, (Vec<Vector2<f32>>, f32, f32)>>, Arc<Mutex<HashMap<Vector3<i32>, Arc<ChunkData>>>>) -> Option<(Vector3<i32>, Option<ChunkMesh>)> + Send + 'static + Copy,
    {
        let queue_clone = queue.clone();
        let voxel_vector_clone = Arc::clone(&voxel_vector);
        let atlas_data_clone = Arc::clone(&atlas_data);
        let chunks_data_clone = Arc::clone(&chunks_data);
        let result_queue_clone = result_queue.clone();

        let handle = thread::spawn(move || loop {
            if let Some(item) = queue_clone.dequeue() {
                let result = process_item(
                    item.clone(),
                    Arc::clone(&voxel_vector_clone),
                    Arc::clone(&atlas_data_clone),
                    Arc::clone(&chunks_data_clone),
                );

                if let Some(result_unwrapped) = result
                {
                    result_queue_clone.enqueue(result_unwrapped);
                } else {
                    queue_clone.enqueue(item);
                }
            } else {
                thread::sleep(std::time::Duration::from_millis(10));
            }
        });

        MeshWorker {
            queue,
            _handle: handle,
        }
    }

    pub fn enqueue(&self, item: T) {
        self.queue.enqueue(item);
    }
}

#[derive(Clone, Debug)]
pub struct ChunkData {
    pub voxels: Vec<u32>,
    pub chunk_pos: Vector3<i32>,
}


impl ChunkData {
    pub fn generate(chunk_pos: Vector3<i32>) -> Self
    {
        let mut voxels: Vec<u32> = Vec::new();
        let mut rng = ThreadRng::default();
        let types = vec![0,5,6];
        let voxel_type = types[rng.sample(Uniform::new(0, types.len()))];
        for i in 0..CHUNK_SIZE3 as usize
        {
            voxels.push(voxel_type);
        }
        ChunkData
        {
            voxels,
            chunk_pos,
        }
    }

    fn solid(chunk_pos: Vector3<i32>, id: u32) -> Self
    {
        let mut voxels: Vec<u32> = Vec::new();
        for i in 0..CHUNK_SIZE3 as usize
        {
            voxels.push(id);
        }
        ChunkData
        {
            voxels,
            chunk_pos,
        }
    }

    pub fn to_raw_opaque(&self, voxel_map: Arc<VoxelVector>, texture_map: Arc<HashMap<u32, (Vec<Vector2<f32>>, f32, f32)>>) -> world_handler::ChunkRaw {
        /*let mut rng = rand::thread_rng();
        logger::log("Building opaque raw chunk");
        let mut vertices: Vec<WorldMeshVertex> = vec![];
        let mut indices: Vec<u32> = vec![];
        if self.voxels.len() != CHUNK_SIZE3
        {
            log::error!("Voxels length does not match size");
        }
        let voxel_iterator = self.voxels.iter();
        for (i, voxel) in voxel_iterator.enumerate()
        {
            let voxel_type: &VoxelType = voxel_map.voxels.get(voxel).unwrap_throw();
            if !voxel_type.translucent {
                let mut tint = voxel_type.tint;
                let (mut points, w, h) = texture_map.get(voxel).unwrap_throw().clone();
                points.shuffle(&mut rng);
                let point = points.get(0).unwrap_throw();
                let tex_x = point.x;
                let tex_y = point.y;
                let x = (i as f32 % CHUNK_SIZE as f32) as f32;
                let y = ((i as f32/CHUNK_SIZE as f32) % CHUNK_SIZE as f32).floor();
                let z = (i as f32/(CHUNK_SIZE2 as f32) % CHUNK_SIZE as f32).floor();
                if x == 0f32 || voxel_map.voxels.get(&self.voxels[i - 1]).unwrap().translucent
                {
                    let len: u32 = vertices.len() as u32;
                    vertices.push(WorldMeshVertex { //0
                        position: [0.0 + x, 0.0 + y, 0.0 + z],
                        tex_coords: [tex_x, tex_y+h],
                        color: tint,
                    });
                    vertices.push(WorldMeshVertex { //1
                        position: [0.0 + x, 0.0 + y, 1.0 + z],
                        tex_coords: [tex_x+w, tex_y+h],
                        color: tint,
                    });
                    vertices.push(WorldMeshVertex { //2
                        position: [0.0 + x, 1.0 + y, 1.0 + z],
                        tex_coords: [tex_x+w, tex_y],
                        color: tint,
                    });
                    vertices.push(WorldMeshVertex { //3
                        position: [0.0 + x, 1.0 + y, 0.0 + z],
                        tex_coords: [tex_x, tex_y],
                        color: tint,
                    });
                    indices.extend([len + 0, len + 1, len + 2, len + 2, len + 3, len + 0])
                }
                if x == CHUNK_SIZE as f32- 1.0 || voxel_map.voxels.get(&self.voxels[i + 1]).unwrap().translucent
                {
                    let len: u32 = vertices.len() as u32;
                    vertices.push(WorldMeshVertex { //5
                        position: [1.0 + x, 0.0 + y, 1.0 + z],
                        tex_coords: [tex_x, tex_y+h],
                        color: tint,
                    });
                    vertices.push(WorldMeshVertex { //4
                        position: [1.0 + x, 0.0 + y, 0.0 + z],
                        tex_coords: [tex_x+w, tex_y+h],
                        color: tint,
                    });
                    vertices.push(WorldMeshVertex { //7
                        position: [1.0 + x, 1.0 + y, 0.0 + z],
                        tex_coords: [tex_x+w, tex_y],
                        color: tint,
                    });
                    vertices.push(WorldMeshVertex { //6
                        position: [1.0 + x, 1.0 + y, 1.0 + z],
                        tex_coords: [tex_x, tex_y],
                        color: tint,
                    });
                    indices.extend([len + 0, len + 1, len + 2, len + 2, len + 3, len + 0])
                }
                if y == 0f32 || voxel_map.voxels.get(&self.voxels[i - CHUNK_SIZE as usize]).unwrap().translucent
                {
                    let len: u32 = vertices.len() as u32;
                    vertices.push(WorldMeshVertex { //4
                        position: [1.0 + x, 0.0 + y, 0.0 + z],
                        tex_coords: [tex_x, tex_y+h],
                        color: tint,
                    });
                    vertices.push(WorldMeshVertex { //5
                        position: [1.0 + x, 0.0 + y, 1.0 + z],
                        tex_coords: [tex_x+w, tex_y+h],
                        color: tint,
                    });
                    vertices.push(WorldMeshVertex { //1
                        position: [0.0 + x, 0.0 + y, 1.0 + z],
                        tex_coords: [tex_x+w, tex_y],
                        color: tint,
                    });
                    vertices.push(WorldMeshVertex { //0
                        position: [0.0 + x, 0.0 + y, 0.0 + z],
                        tex_coords: [tex_x, tex_y],
                        color: tint,
                    });

                    indices.extend([len + 0, len + 1, len + 2, len + 2, len + 3, len + 0])
                }
                if y == CHUNK_SIZE as f32- 1.0 || voxel_map.voxels.get(&self.voxels[i + CHUNK_SIZE as usize]).unwrap().translucent
                {
                    let len: u32 = vertices.len() as u32;
                    vertices.push(WorldMeshVertex { //6
                        position: [1.0 + x, 1.0 + y, 1.0 + z],
                        tex_coords: [tex_x, tex_y+h],
                        color: tint,
                    });
                    vertices.push(WorldMeshVertex { //7
                        position: [1.0 + x, 1.0 + y, 0.0 + z],
                        tex_coords: [tex_x+w, tex_y+h],
                        color: tint,
                    });
                    vertices.push(WorldMeshVertex { //3
                        position: [0.0 + x, 1.0 + y, 0.0 + z],
                        tex_coords: [tex_x+w, tex_y],
                        color: tint,
                    });
                    vertices.push(WorldMeshVertex { //2
                        position: [0.0 + x, 1.0 + y, 1.0 + z],
                        tex_coords: [tex_x, tex_y],
                        color: tint,
                    });
                    indices.extend([len + 0, len + 1, len + 2, len + 2, len + 3, len + 0])
                }
                if z == 0f32 || voxel_map.voxels.get(&self.voxels[i - (CHUNK_SIZE2)]).unwrap().translucent
                {
                    let len: u32 = vertices.len() as u32;
                    vertices.push(WorldMeshVertex { //4
                        position: [1.0 + x, 0.0 + y, 0.0 + z],
                        tex_coords: [tex_x, tex_y+h],
                        color: tint,
                    });
                    vertices.push(WorldMeshVertex { //0
                        position: [0.0 + x, 0.0 + y, 0.0 + z],
                        tex_coords: [tex_x+w, tex_y+h],
                        color: tint,
                    });
                    vertices.push(WorldMeshVertex { //3
                        position: [0.0 + x, 1.0 + y, 0.0 + z],
                        tex_coords: [tex_x+w, tex_y],
                        color: tint,
                    });
                    vertices.push(WorldMeshVertex { //7
                        position: [1.0 + x, 1.0 + y, 0.0 + z],
                        tex_coords: [tex_x, tex_y],
                        color: tint,
                    });
                    indices.extend([len + 0, len + 1, len + 2, len + 2, len + 3, len + 0])
                }
                if z == CHUNK_SIZE as f32- 1.0 || voxel_map.voxels.get(&self.voxels[i + CHUNK_SIZE2]).unwrap().translucent
                {
                    let len: u32 = vertices.len() as u32;
                    vertices.push(WorldMeshVertex { //1
                        position: [0.0 + x, 0.0 + y, 1.0 + z],
                        tex_coords: [tex_x, tex_y+h],
                        color: tint,
                    });
                    vertices.push(WorldMeshVertex { //5
                        position: [1.0 + x, 0.0 + y, 1.0 + z],
                        tex_coords: [tex_x+w, tex_y+h],
                        color: tint,
                    });
                    vertices.push(WorldMeshVertex { //6
                        position: [1.0 + x, 1.0 + y, 1.0 + z],
                        tex_coords: [tex_x+w, tex_y],
                        color: tint,
                    });
                    vertices.push(WorldMeshVertex { //2
                        position: [0.0 + x, 1.0 + y, 1.0 + z],
                        tex_coords: [tex_x, tex_y],
                        color: tint,
                    });
                    indices.extend([len + 0, len + 1, len + 2, len + 2, len + 3, len + 0])
                }
            }
        }
        logger::log("Built opaque raw chunk");
*/
        ChunkRaw
        {
            vertices: Vec::new(),
            indices: Vec::new(),
        }
    }

    #[inline]
    pub fn get_voxel(&self, index: usize) -> &u32 {
        if self.voxels.len() == 1 {
            &self.voxels[0]
        } else {
            &self.voxels[index]
        }
    }

    #[inline]
    pub fn get_voxel_if_filled(&self) -> Option<&u32> {
        if self.voxels.len() == 1 {
            Some(&self.voxels[0])
        } else {
            None
        }
    }
}

fn push_face(mesh: &mut ChunkMesh, dir: Direction, vpos: Vector3<i32>, color: [f32; 3], voxel_type: u32) {
    let quad = Quad::from_direction(dir, vpos, color);
    for corner in quad.corners.into_iter() {
        mesh.vertices.push(WorldMeshVertex {
            position: [corner[0] as f32, corner[1] as f32, corner[2] as f32],
            tex_coords: [0.0,0.0],
            color,
        });
    }
}

#[inline]
pub fn make_vertex_u32(
    pos: Vector3<i32>,
    ao: u32,
    normal: u32,
    voxel_type: u32,
) -> u8 {
    (pos.x as u32
        | (pos.y as u32) << 6u32
        | (pos.z as u32) << 12u32
        | ao << 18u32
        | normal << 21u32
        | voxel_type << 25u32) as u8
    // | (normal as u32) << 18u32
    // | (texture_id) << 21u32
}

#[inline]
pub fn index_to_vector3i(i: i32) -> Vector3<i32> {
    let x = i % 32;
    let y = (i / 32) % 32;
    let z = i / (32 * 32);
    Vector3::new(x, y, z)
}

#[inline]
//todo make return u32
pub fn generate_indices(vertex_count: usize) -> Vec<u32> {
    let indices_count = vertex_count / 4;
    let mut indices = Vec::<u32>::with_capacity(indices_count);
    (0..indices_count).into_iter().for_each(|vert_index| {
        let vert_index = vert_index as u32 * 4u32;
        indices.push(vert_index);
        indices.push((vert_index + 1));
        indices.push((vert_index + 2));
        indices.push(vert_index);
        indices.push((vert_index + 2));
        indices.push((vert_index + 3));
    });
    indices
}

pub fn ambient_corner_voxels(
    chunks_refs: &ChunksRefs,
    direction: Direction,
    local_pos: Vector3<i32>,
    voxel_map: &Arc<VoxelVector>,
) -> [bool; 8] {
    #[rustfmt::skip]
    let mut positions = match direction {
        Direction::Left => [Vector3::new(-1,0,-1),Vector3::new(-1,-1,-1),Vector3::new(-1,-1,0),Vector3::new(-1,-1,1),Vector3::new(-1,0,1),Vector3::new(-1,1,1),Vector3::new(-1, 1, 0),Vector3::new(-1,1,-1),],
        Direction::Down => [Vector3::new(-1, -1, 0),Vector3::new(-1, -1, -1),Vector3::new(0, -1, -1), Vector3::new(1,-1,-1),Vector3::new(1,-1,0),Vector3::new(1, -1, 1),Vector3::new(0,-1,1),Vector3::new(-1,-1,1),],
        Direction::Back => [Vector3::new(0,-1,-1),Vector3::new(-1,-1,-1),Vector3::new(-1,0,-1),Vector3::new(-1,1,-1), Vector3::new(0,1,-1), Vector3::new(1,1,-1),Vector3::new(1,0,-1), Vector3::new(1,-1,-1)],

        Direction::Right => [Vector3::new(0,0,-1), Vector3::new(0,1,-1), Vector3::new(0,1,0), Vector3::new(0,1,1),Vector3::new(0,0,1),Vector3::new(0,-1,1),Vector3::new(0,-1,0),Vector3::new(0,-1,-1)],
        Direction::Up => [Vector3::new(-1,0,0),Vector3::new(-1,0,1),Vector3::new(0,0,1),Vector3::new(1,0,1),Vector3::new(1,0,0),Vector3::new(1,0,-1),Vector3::new(0,0,-1),Vector3::new(-1,0,-1),],
        Direction::Forward => [Vector3::new(0,-1,0),Vector3::new(1,-1,0),Vector3::new(1,0,0),Vector3::new(1,1,0),Vector3::new(0,1,0),Vector3::new(-1,1,0),Vector3::new(-1,0,0),Vector3::new(-1,-1,0),],
    };

    positions.iter_mut().for_each(|p| *p = local_pos + *p);

    let mut result = [false; 8];
    for i in 0..8 {
        result[i] = voxel_map.get(chunks_refs.get_voxel(positions[i])).unwrap().is_solid();
    }
    result
}

pub fn ambient_corner_voxels_cloned(
    chunks_refs: &ChunksRefs,
    direction: Direction,
    local_pos: Vector3<i32>,
    voxel_map: Arc<VoxelVector>,
) -> Option<[bool; 8]> {
    #[rustfmt::skip]
    let mut positions = match direction {
        Direction::Left => [Vector3::new(-1,0,-1),Vector3::new(-1,-1,-1),Vector3::new(-1,-1,0),Vector3::new(-1,-1,1),Vector3::new(-1,0,1),Vector3::new(-1,1,1),Vector3::new(-1, 1, 0),Vector3::new(-1,1,-1),],
        Direction::Down => [Vector3::new(-1, -1, 0),Vector3::new(-1, -1, -1),Vector3::new(0, -1, -1), Vector3::new(1,-1,-1),Vector3::new(1,-1,0),Vector3::new(1, -1, 1),Vector3::new(0,-1,1),Vector3::new(-1,-1,1),],
        Direction::Back => [Vector3::new(0,-1,-1),Vector3::new(-1,-1,-1),Vector3::new(-1,0,-1),Vector3::new(-1,1,-1), Vector3::new(0,1,-1), Vector3::new(1,1,-1),Vector3::new(1,0,-1), Vector3::new(1,-1,-1)],

        Direction::Right => [Vector3::new(0,0,-1), Vector3::new(0,1,-1), Vector3::new(0,1,0), Vector3::new(0,1,1),Vector3::new(0,0,1),Vector3::new(0,-1,1),Vector3::new(0,-1,0),Vector3::new(0,-1,-1)],
        Direction::Up => [Vector3::new(-1,0,0),Vector3::new(-1,0,1),Vector3::new(0,0,1),Vector3::new(1,0,1),Vector3::new(1,0,0),Vector3::new(1,0,-1),Vector3::new(0,0,-1),Vector3::new(-1,0,-1),],
        Direction::Forward => [Vector3::new(0,-1,0),Vector3::new(1,-1,0),Vector3::new(1,0,0),Vector3::new(1,1,0),Vector3::new(0,1,0),Vector3::new(-1,1,0),Vector3::new(-1,0,0),Vector3::new(-1,-1,0),],
    };

    positions.iter_mut().for_each(|p| *p = local_pos + *p);

    let mut result = [false; 8];
    for i in 0..8 {
        result[i] = voxel_map.get(chunks_refs.get_voxel(positions[i]))?.is_solid();
    }
    Some(result)
}

fn push_face_ao(
    chunks_refs: &ChunksRefs,
    mesh: &mut ChunkMesh,
    dir: Direction,
    vpos: Vector3<i32>,
    color: [f32; 3],
    voxel_type: u32,
    voxel_map: &Arc<VoxelVector>,
) {
    let ambient_corners = ambient_corner_voxels(&chunks_refs, dir, vpos, voxel_map);
    let quad = Quad::from_direction(dir, vpos, color);
    for (i, corner) in quad.corners.into_iter().enumerate() {
        let index = i * 2;

        let side_1 = ambient_corners[index] as u32;
        let side_2 = ambient_corners[(index + 2) % 8] as u32;
        let side_corner = ambient_corners[(index + 1) % 8] as u32;
        let mut ao_count = side_1 + side_2 + side_corner;
        // fully ambient occluded if both
        if side_1 == 1 && side_2 == 1 {
            ao_count = 3;
        }

        //todo fix ambient occlusion and also tex coords
        mesh.vertices.push(WorldMeshVertex {
            position: [corner[0] as f32, corner[1] as f32, corner[2] as f32],
            tex_coords: [0.0, 0.0],
            color,
        });
    }
}


