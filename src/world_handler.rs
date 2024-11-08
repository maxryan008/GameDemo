use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;
use cgmath::{Vector3};
use rand::{thread_rng, Rng};
use crate::{voxels};
use crate::constants::{CHUNK_SIZE, CHUNK_SIZE3};
use crate::wgpulib::{RawChunkRenderData};

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
        result_queue: ArcQueue<(Vector3<i32>, ChunkData)>,
        chunks_data: Arc<Mutex<HashMap<Vector3<i32>, Arc<ChunkData>>>>,
    ) -> Self
    where
        F: Fn(T, Arc<voxels::VoxelVector>, Arc<Mutex<HashMap<Vector3<i32>, Arc<ChunkData>>>>) -> (Vector3<i32>, ChunkData) + Send + 'static + Copy,
    {
        let queue_clone = queue.clone();
        let voxel_vector_clone = Arc::clone(&voxel_vector);
        let chunks_data_clone = Arc::clone(&chunks_data);
        let result_queue_clone = result_queue.clone();

        let handle = thread::spawn(move || loop {
            if let Some(item) = queue_clone.dequeue() {
                let result = process_item(
                    item,
                    Arc::clone(&voxel_vector_clone),
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
        result_queue: ArcQueue<(Vector3<i32>, Option<RawChunkRenderData>)>,
        chunks_data: Arc<Mutex<HashMap<Vector3<i32>, Arc<ChunkData>>>>,
    ) -> Self
    where
        F: Fn(T, Arc<voxels::VoxelVector>, Arc<Mutex<HashMap<Vector3<i32>, Arc<ChunkData>>>>) -> Option<(Vector3<i32>, Option<RawChunkRenderData>)> + Send + 'static + Copy,
    {
        let queue_clone = queue.clone();
        let voxel_vector_clone = Arc::clone(&voxel_vector);
        let chunks_data_clone = Arc::clone(&chunks_data);
        let result_queue_clone = result_queue.clone();

        let handle = thread::spawn(move || loop {
            if let Some(item) = queue_clone.dequeue() {
                let result = process_item(
                    item.clone(),
                    Arc::clone(&voxel_vector_clone),
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
}


impl ChunkData {
    pub fn generate(chunk_pos: Vector3<i32>) -> Self
    {
        let mut voxels: Vec<u32> = Vec::new();
        let mut t = false;
        if thread_rng().gen_bool(1.0/1.1) {
            t = true;
        }
        let surface = 150;
        let under_surface = 125;
        let bedrock = 115;

        if t {
            for _ in 0..CHUNK_SIZE {
                for y in 0..CHUNK_SIZE {
                    for _ in 0..CHUNK_SIZE {
                        if (y + chunk_pos.y as usize * CHUNK_SIZE) < surface {
                            if (y + chunk_pos.y as usize * CHUNK_SIZE) < under_surface {
                                if (y + chunk_pos.y as usize * CHUNK_SIZE) < bedrock {
                                    voxels.push(7);
                                } else {
                                    voxels.push(5);
                                }
                            } else {
                                if thread_rng().gen_bool((y + chunk_pos.y as usize * CHUNK_SIZE - under_surface) as f64 / (surface - under_surface) as f64) {
                                    voxels.push(6);
                                } else {
                                    if (y + chunk_pos.y as usize * CHUNK_SIZE + 1) == surface {
                                        voxels.push(6);
                                    } else {
                                        voxels.push(5);
                                    }
                                }
                            }
                        } else {
                            voxels.push(0);
                        }
                    }
                }
            }
        } else {
            for _ in 0..CHUNK_SIZE3 {
                voxels.push(0);
            }
        }
        ChunkData
        {
            voxels,
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

pub fn run_game_updates() {

}


