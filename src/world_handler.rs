use std::collections::HashMap;
use std::sync::{mpsc, Arc, Mutex};
use std::sync::mpsc::{Receiver, Sender};
use std::thread;
use cgmath::{Vector2, Vector3};
use rand::prelude::{SliceRandom, ThreadRng};
use rand::Rng;
use web_sys::wasm_bindgen::UnwrapThrowExt;
use crate::{logger, voxels, wgpulib, world_handler};
use crate::vertex_types::WorldMeshVertex;
use crate::voxels::{VoxelType, VoxelVector};

pub const CHUNK_SIZE: f32 = 96.0;
pub const CHUNK_SIZE_SQUARE: f32 = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;

pub struct ChunkRaw {
    pub vertices: Vec<WorldMeshVertex>,
    pub indices: Vec<u32>,
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
}

pub struct Worker<T> {
    queue: ArcQueue<T>,           // Shared queue for incoming items
    _handle: thread::JoinHandle<()>, // Worker thread handle
}

impl<T: Send + 'static> Worker<T> {
    // Function to create and start the worker, accepting a processing function
    pub fn new<F>(
        queue: ArcQueue<T>,
        process_item: F,
        voxel_vector: Arc<voxels::VoxelVector>,
        atlas_data: Arc<HashMap<u8, (Vec<Vector2<f32>>, f32, f32)>>,
        result_queue: ArcQueue<(Vector3<i32>, (Vec<wgpulib::ChunkInstanceRaw>, world_handler::ChunkRaw))>
    ) -> Self
    where
        F: Fn(T, Arc<voxels::VoxelVector>, Arc<HashMap<u8, (Vec<Vector2<f32>>, f32, f32)>>) -> (Vector3<i32>, (Vec<wgpulib::ChunkInstanceRaw>, world_handler::ChunkRaw)) + Send + 'static + Copy,
    {
        let queue_clone = queue.clone();
        let voxel_vector_clone = Arc::clone(&voxel_vector);
        let atlas_data_clone = Arc::clone(&atlas_data);
        let result_queue_clone = result_queue.clone();

        let handle = thread::spawn(move || loop {
            if let Some(item) = queue_clone.dequeue() {
                let result = process_item(
                    item,
                    Arc::clone(&voxel_vector_clone),
                    Arc::clone(&atlas_data_clone)
                );

                result_queue_clone.enqueue(result);
            } else {
                thread::sleep(std::time::Duration::from_millis(10));
            }
        });

        Worker {
            queue,
            _handle: handle,
        }
    }

    pub fn enqueue(&self, item: T) {
        self.queue.enqueue(item);
    }
}


pub struct Chunk {
    pub voxels: Vec<u8>,
    pub position: Vector3<i32>,
}


impl Chunk {
    pub fn new(position: Vector3<i32>) -> Self
    {
        let mut voxels: Vec<u8> = Vec::new();
        let mut rng = ThreadRng::default();
        let vtype = rng.gen_range(4..=5) as u8;
        for i in 0..CHUNK_SIZE_SQUARE as usize
        {
            voxels.push(vtype);
        }
        Chunk
        {
            voxels,
            position,
        }
    }

    fn solid(position: Vector3<i32>, id: u8) -> Self
    {
        let mut voxels: Vec<u8> = Vec::new();
        for i in 0..CHUNK_SIZE_SQUARE as usize
        {
            voxels.push(id);
        }
        Chunk
        {
            voxels,
            position,
        }
    }

    pub fn to_raw_opaque(&self, voxel_map: Arc<VoxelVector>, texture_map: Arc<HashMap<u8, (Vec<Vector2<f32>>, f32, f32)>>) -> world_handler::ChunkRaw {
        let mut rng = rand::thread_rng();
        logger::log("Building opaque raw chunk");
        let mut vertices: Vec<WorldMeshVertex> = vec![];
        let mut indices: Vec<u32> = vec![];
        if self.voxels.len() != CHUNK_SIZE as usize * CHUNK_SIZE as usize * CHUNK_SIZE as usize
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
                let x = (i % CHUNK_SIZE as usize) as f32;
                let y = ((i as f32/CHUNK_SIZE) % CHUNK_SIZE).floor();
                let z = (i as f32/(CHUNK_SIZE*CHUNK_SIZE) % CHUNK_SIZE).floor();
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
                if x == CHUNK_SIZE - 1.0 || voxel_map.voxels.get(&self.voxels[i + 1]).unwrap().translucent
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
                if y == CHUNK_SIZE - 1.0 || voxel_map.voxels.get(&self.voxels[i + CHUNK_SIZE as usize]).unwrap().translucent
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
                if z == 0f32 || voxel_map.voxels.get(&self.voxels[i - (CHUNK_SIZE as usize * CHUNK_SIZE as usize)]).unwrap().translucent
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
                if z == CHUNK_SIZE - 1.0 || voxel_map.voxels.get(&self.voxels[i + CHUNK_SIZE as usize * CHUNK_SIZE as usize]).unwrap().translucent
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

        ChunkRaw
        {
            vertices,
            indices,
        }
    }
}