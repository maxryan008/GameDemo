use std::collections::{HashMap, HashSet, VecDeque};
use std::process::exit;
use std::sync::{Arc, Mutex};
use cgmath::{Array, Rotation3, Vector2, Vector3, Zero};
use cgmath::num_traits::AsPrimitive;
use log::error;
use tobj::Mesh;
use wgpu::PrimitiveTopology;
use crate::chunks_refs::{vector3i_to_index, ChunksRefs};
use crate::constants::{ADJACENT_AO_DIRS, CHUNK_SIZE, CHUNK_SIZE_I32, CHUNK_SIZE_P};
use crate::face_direction::FaceDir;
use crate::lod::Lod;
use crate::voxels::VoxelVector;
use crate::wgpulib::{ChunkInstance, ChunkInstanceRaw, RawChunkRenderData};
use crate::{logger, voxels, world_handler};
use crate::vertex_types::WorldMeshVertex;
use crate::world_handler::{generate_indices, make_vertex_u32, ArcQueue, ChunkData, Queue, DataWorker, MeshWorker, ChunkRaw};
pub fn build_chunk_mesh(
    chunks_refs: &ChunksRefs,
    lod: Lod,
    voxel_map: &Arc<VoxelVector>,
    position: &Vector3<i32>,
) -> Option<RawChunkRenderData> {
    if chunks_refs.is_all_voxels_same() {
        return None;
    }

    let mut mesh = RawChunkRenderData {
        position: *position,
        rects: Vec::new(),
    };

    let mut axis_cols = [[[0u64; CHUNK_SIZE_P]; CHUNK_SIZE_P]; 3];

    let mut col_face_masks = [[[0u64; CHUNK_SIZE_P]; CHUNK_SIZE_P]; 6];

    #[inline]
    fn add_voxel_to_axis_cols(
        b: &u32,
        x: usize,
        y: usize,
        z: usize,
        axis_cols: &mut [[[u64; CHUNK_SIZE_P]; CHUNK_SIZE_P]; 3],
        voxel_map: &Arc<VoxelVector>,
    ) {
        if voxel_map.get(&b).unwrap().is_solid() {
            // x,z - y axis
            axis_cols[0][z][x] |= 1u64 << y as u64;
            // z,y - x axis
            axis_cols[1][y][z] |= 1u64 << x as u64;
            // x,y - z axis
            axis_cols[2][y][x] |= 1u64 << z as u64;
        }
    }

    // inner chunk voxels.
    let chunk = &*chunks_refs.chunks[vector3i_to_index(Vector3::new(1, 1, 1), 3)];
    assert!(chunk.voxels.len() == CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE || chunk.voxels.len() == 1);
    for z in 0..CHUNK_SIZE {
        for y in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                let i = match chunk.voxels.len() {
                    1 => 0,
                    _ => (z * CHUNK_SIZE + y) * CHUNK_SIZE + x,
                };
                add_voxel_to_axis_cols(&chunk.voxels[i], x + 1, y + 1, z + 1, &mut axis_cols, &voxel_map)
            }
        }
    }

    for z in [0, CHUNK_SIZE_P - 1] {
        for y in 0..CHUNK_SIZE_P {
            for x in 0..CHUNK_SIZE_P {
                let pos = Vector3::new(x as i32, y as i32, z as i32) - Vector3::new(1, 1, 1);
                add_voxel_to_axis_cols(chunks_refs.get_voxel(pos), x, y, z, &mut axis_cols, voxel_map);
            }
        }
    }
    for z in 0..CHUNK_SIZE_P {
        for x in [0, CHUNK_SIZE_P - 1] {
            for y in 0..CHUNK_SIZE_P {
                let pos = Vector3::new(x as i32, y as i32, z as i32) - Vector3::new(1, 1, 1);
                add_voxel_to_axis_cols(chunks_refs.get_voxel(pos), x, y, z, &mut axis_cols, voxel_map);
            }
        }
    }
    for z in 0..CHUNK_SIZE_P {
        for x in [0, CHUNK_SIZE_P - 1] {
            for y in 0..CHUNK_SIZE_P {
                let pos = Vector3::new(x as i32, y as i32, z as i32) - Vector3::new(1, 1, 1);
                add_voxel_to_axis_cols(chunks_refs.get_voxel(pos), x, y, z, &mut axis_cols, voxel_map);
            }
        }
    }

    // face culling
    for axis in 0..3 {
        for z in 0..CHUNK_SIZE_P {
            for x in 0..CHUNK_SIZE_P {
                // set if current is solid, and next is air
                let col = axis_cols[axis][z][x];
                // sample descending axis, and set true when air meets solid
                col_face_masks[2 * axis + 0][z][x] = col & !(col << 1);
                // sample ascending axis, and set true when air meets solid
                col_face_masks[2 * axis + 1][z][x] = col & !(col >> 1);
            }
        }
    }

    // greedy meshing planes for every axis (6)
    let mut data: [HashMap<u32, [u32; 32]>; 6] = [
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    ];

    // Process each axis and build binary planes for each face
    for axis in 0..6 {
        for z in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                let mut col = col_face_masks[axis][z + 1][x + 1];

                col >>= 1;
                col &= !(1 << CHUNK_SIZE as u64);

                while col != 0 {
                    let y = col.trailing_zeros();
                    col &= col - 1; // Clear the least significant set bit

                    // Get the voxel position and determine which face this belongs to
                    // let voxel_pos = match axis {
                    //     0 | 1 => Vector3::new(x as i32, y as i32, z as i32),  // Down, Up
                    //     2 | 3 => Vector3::new(y as i32, z as i32, x as i32),  // Left, Right
                    //     _     => Vector3::new(x as i32, z as i32, y as i32),  // Forward, Back
                    // };

                    // Mark the corresponding bit in the binary plane
                    let data = data[axis].entry(y).or_default();
                    data[x as usize] |= 1u32 << z as u32;
                }
            }
        }
    }

    // Now build quads for each face, working with solid blocks
    for (axis, axis_data) in data.iter().enumerate() {
        let face_dir = match axis {
            0 => FaceDir::Down,
            1 => FaceDir::Up,
            2 => FaceDir::Left,
            3 => FaceDir::Right,
            4 => FaceDir::Forward,
            _ => FaceDir::Back,
        };

        for (axis_pos, plane) in axis_data {
            // Perform greedy meshing on the plane for the current face
            let quads_from_axis = greedy_mesh_binary_plane(*plane, CHUNK_SIZE as u32);

            // For each quad, create a rect
            for quad in quads_from_axis {
                let x = quad.x;
                let y = quad.y;
                let width = quad.w;
                let height = quad.h;

                let mut block_ids = Vec::with_capacity((width * height) as usize);

                for i in 0..height {
                    for j in 0..width {
                        let block_id = match face_dir {
                            FaceDir::Down => {
                                // Bottom face (y = 0, facing downward)
                                let voxel_index = (x + j) as usize + (y + i) as usize * CHUNK_SIZE + *axis_pos as usize * CHUNK_SIZE * CHUNK_SIZE;
                                chunk.voxels[voxel_index as usize]
                            }
                            FaceDir::Up => {
                                // Top face (y = CHUNK_SIZE - 1, facing upward)
                                let voxel_index = (CHUNK_SIZE - 1 - (x + j) as usize)
                                    + (y + i) as usize * CHUNK_SIZE
                                    + (CHUNK_SIZE - 1) * CHUNK_SIZE * CHUNK_SIZE;
                                chunk.voxels[voxel_index as usize]
                            }
                            FaceDir::Left => {
                                // Left face (x = 0, facing left)
                                let voxel_index = *axis_pos as usize
                                    + (y + i) as usize * CHUNK_SIZE
                                    + (CHUNK_SIZE - 1 - (x + j) as usize) * CHUNK_SIZE * CHUNK_SIZE;
                                chunk.voxels[voxel_index as usize]
                            }
                            FaceDir::Right => {
                                // Right face (x = CHUNK_SIZE - 1, facing right)
                                let voxel_index = (CHUNK_SIZE - 1) + (y + i) as usize * CHUNK_SIZE + (x + j) as usize * CHUNK_SIZE * CHUNK_SIZE;
                                chunk.voxels[voxel_index as usize]
                            }
                            FaceDir::Forward => {
                                // Front face (z = 0, facing forward)
                                let voxel_index = (CHUNK_SIZE - 1 - (x + j) as usize)
                                    + *axis_pos as usize * CHUNK_SIZE
                                    + (y + i) as usize * CHUNK_SIZE * CHUNK_SIZE;
                                chunk.voxels[voxel_index as usize]
                            }
                            FaceDir::Back => {
                                // Back face (z = CHUNK_SIZE - 1, facing backward)
                                let voxel_index = (x + j) as usize + (CHUNK_SIZE - 1) * CHUNK_SIZE + (y + i) as usize * CHUNK_SIZE * CHUNK_SIZE;
                                chunk.voxels[voxel_index as usize]
                            }
                        };

                        // Push block ID for each voxel
                        block_ids.push(block_id);
                    }
                }

                let vertices = create_vertices(face_dir, *axis_pos, x, y, width, height);
                let indices = [0, 1, 2, 2, 3, 0];
                let tints = vec![[0.0, 0.0, 0.0]; width as usize * height as usize];

                let rect = Rect {
                    vertices,
                    indices,
                    blocks: block_ids,
                    tints,
                    width: width as u32,
                    height: height as u32,
                };

                mesh.rects.push(rect);
            }
        }
    }

    if mesh.rects.is_empty() {
        None
    } else {
        Some(mesh)
    }
}

// todo: compress further?
#[derive(Debug)]
pub struct GreedyQuad {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

fn create_vertices(
    face_dir: FaceDir,
    axis_pos: u32,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
) -> [[f32; 3]; 4] {
    let mut vertices = [[0.0; 3]; 4];

    match face_dir {
        FaceDir::Down => {
            // Bottom face (Y- axis, Z as up/down)
            //MATCHES
            vertices[0] = [x as f32, axis_pos as f32, y as f32];
            vertices[3] = [x as f32, axis_pos as f32, (y + height) as f32];
            vertices[2] = [(x + width) as f32, axis_pos as f32, (y + height) as f32];
            vertices[1] = [(x + width) as f32, axis_pos as f32, y as f32];
        }
        FaceDir::Up => {
            // Top face (Y+ axis, Z as up/down)
            let axis_pos = axis_pos + 1;
            //TODO flip
            vertices[1] = [x as f32, axis_pos as f32, y as f32];
            vertices[0] = [(x + width) as f32, axis_pos as f32, y as f32];
            vertices[3] = [(x + width) as f32, axis_pos as f32, (y + height) as f32];
            vertices[2] = [x as f32, axis_pos as f32, (y + height) as f32];
        }
        FaceDir::Left => {
            // Left face (X- axis, Y as up/down), adjust for CCW
            //Todo flip
            vertices[1] = [axis_pos as f32, y as f32, x as f32];
            vertices[0] = [axis_pos as f32, (y + height) as f32, x as f32];
            vertices[3] = [axis_pos as f32, (y + height) as f32, (x + width) as f32];
            vertices[2] = [axis_pos as f32, y as f32, (x + width) as f32];
        }
        FaceDir::Right => {
            // Right face (X+ axis, Y as up/down), adjust for CCW
            //MATCHES
            let axis_pos = axis_pos + 1;
            vertices[0] = [axis_pos as f32, y as f32, x as f32];
            vertices[1] = [axis_pos as f32, (y + height) as f32, x as f32];
            vertices[2] = [axis_pos as f32, (y + height) as f32, (x + width) as f32];
            vertices[3] = [axis_pos as f32, y as f32, (x + width) as f32];
        }
        FaceDir::Forward => {
            // Front face (Z- axis, X as left/right), adjust for CCW
            //Todo flip
            vertices[1] = [x as f32, y as f32, axis_pos as f32];
            vertices[0] = [(x + width) as f32, y as f32, axis_pos as f32];
            vertices[3] = [(x + width) as f32, (y + height) as f32, axis_pos as f32];
            vertices[2] = [x as f32, (y + height) as f32, axis_pos as f32];
        }
        FaceDir::Back => {
            // Back face (Z+ axis, X as left/right), adjust for CCW
            let axis_pos = axis_pos + 1;
            //MATCHES
            vertices[0] = [x as f32, y as f32, axis_pos as f32];
            vertices[1] = [(x + width) as f32, y as f32, axis_pos as f32];
            vertices[2] = [(x + width) as f32, (y + height) as f32, axis_pos as f32];
            vertices[3] = [x as f32, (y + height) as f32, axis_pos as f32];
        }
    }

    vertices
}

impl GreedyQuad {
    ///! compress this quad data into the input vertices vec
    pub fn append_vertices(
        &self,
        vertices: &mut Vec<WorldMeshVertex>,
        face_dir: FaceDir,
        axis: u32,
        lod: &Lod,
        ao: u32,
        voxel_type: u32,
    ) {
        // let negate_axis = face_dir.negate_axis();
        // let axis = axis as i32 + negate_axis;
        let axis = axis as i32;
        let jump = lod.jump_index();

        // pack ambient occlusion strength into vertex
        let v1ao = ((ao >> 0) & 1) + ((ao >> 1) & 1) + ((ao >> 3) & 1);
        let v2ao = ((ao >> 3) & 1) + ((ao >> 6) & 1) + ((ao >> 7) & 1);
        let v3ao = ((ao >> 5) & 1) + ((ao >> 8) & 1) + ((ao >> 7) & 1);
        let v4ao = ((ao >> 1) & 1) + ((ao >> 2) & 1) + ((ao >> 5) & 1);

        let pos1 = (face_dir.world_to_sample(axis as i32, self.x as i32, self.y as i32, &lod) * jump);
        let pos2 = (face_dir.world_to_sample(axis as i32, self.x as i32 + self.w as i32, self.y as i32, &lod, ) * jump);
        let pos3 = (face_dir.world_to_sample(axis as i32, self.x as i32 + self.w as i32, self.y as i32 + self.h as i32, &lod, ) * jump);
        let pos4 = (face_dir.world_to_sample(axis as i32, self.x as i32, self.y as i32 + self.h as i32, &lod, ) * jump);
        let v1 = WorldMeshVertex {
            position: [pos1.x as f32, pos1.y as f32, pos1.z as f32],
            tex_coords: [0.0, 0.0],
            color: [0.0, 0.0, 0.0],
        };
        let v2 = WorldMeshVertex {
            position: [pos2.x as f32, pos2.y as f32, pos2.z as f32],
            tex_coords: [0.0, 0.0],
            color: [0.0, 0.0, 0.0],
        };
        let v3 = WorldMeshVertex {
            position: [pos3.x as f32, pos3.y as f32, pos3.z as f32],
            tex_coords: [0.0, 0.0],
            color: [0.0, 0.0, 0.0],
        };
        let v4 = WorldMeshVertex {
            position: [pos4.x as f32, pos4.y as f32, pos4.z as f32],
            tex_coords: [0.0, 0.0],
            color: [0.0, 0.0, 0.0],
        };
        // the quad vertices to be added
        let mut new_vertices:VecDeque<WorldMeshVertex> = VecDeque::from([v1, v2, v3, v4]);

        // triangle vertex order is different depending on the facing direction
        // due to indices always being the same
        if face_dir.reverse_order() {
            // keep first index, but reverse the rest
            let o = new_vertices.split_off(1);
            o.into_iter().rev().for_each(|i| new_vertices.push_back(i));
        }

        // anisotropy flip
        if (v1ao > 0) ^ (v3ao > 0) {
            // right shift array, to swap triangle intersection angle
            let f = new_vertices.pop_front().unwrap();
            new_vertices.push_back(f);
        }

        vertices.extend(new_vertices);
    }
}

///! generate quads of a binary slice
///! lod not implemented atm
pub fn greedy_mesh_binary_plane(mut data: [u32; 32], lod_size: u32) -> Vec<GreedyQuad> {
    let mut greedy_quads = vec![];
    for row in 0..data.len() {
        let mut y = 0;
        while y < lod_size {
            // find first solid, "air/zero's" could be first so skip
            y += (data[row] >> y).trailing_zeros();
            if y >= lod_size {
                // reached top
                continue;
            }
            let h = (data[row] >> y).trailing_ones();
            // convert height 'num' to positive bits repeated 'num' times aka:
            // 1 = 0b1, 2 = 0b11, 4 = 0b1111
            let h_as_mask = u32::checked_shl(1, h).map_or(!0, |v| v - 1);
            let mask = h_as_mask << y;
            // grow horizontally
            let mut w = 1;
            while row + w < lod_size as usize {
                // fetch bits spanning height, in the next row
                let next_row_h = (data[row + w] >> y) & h_as_mask;
                if next_row_h != h_as_mask {
                    break; // can no longer expand horizontally
                }

                // nuke the bits we expanded into
                data[row + w] = data[row + w] & !mask;

                w += 1;
            }
            greedy_quads.push(GreedyQuad {
                y,
                w: w as u32,
                h,
                x: row as u32,
            });
            y += h;
        }
    }
    greedy_quads
}

pub struct ChunkModification(pub Vector3<i32>, pub u32);
pub struct WorldData {
    pub chunks_data: Arc<Mutex<HashMap<Vector3<i32>, Arc<ChunkData>>>>,
    pub load_data_queue: ArcQueue<Vector3<i32>>,
    pub load_mesh_queue: ArcQueue<Vector3<i32>>,
    pub finished_data_queue: ArcQueue<(Vector3<i32>, ChunkData)>,
    pub finished_mesh_queue: ArcQueue<(Vector3<i32>, Option<RawChunkRenderData>)>,
    pub data_workers: Vec<DataWorker<Vector3<i32>>>,
    pub mesh_workers: Vec<MeshWorker<Vector3<i32>>>,
    pub chunk_modifications: HashMap<Vector3<i32>, Vec<ChunkModification>>,
}

impl WorldData {
    pub fn unload_all_meshes(&mut self) {
        self.load_mesh_queue.clear();
        self.load_data_queue.clear();
    }
}

pub fn unload_data(mut world_data: WorldData) {
    for chunk_pos in world_data.finished_data_queue.drain() {
        world_data.chunks_data.lock().unwrap().remove(&chunk_pos.0);
    }
}

//Todo:: probably better to make unload mesh work in the state impl because it has more access to the opaque render hashmap

// pub fn unload_mesh(mut world_data: WorldData) {
//     //let mut retry = Vec::new();
//     for chunk_pos in world_data.finished_mesh_queue.drain() {
//         // let Some(chunk_id) = world_data.chunk_entities.remove(&chunk_pos.0) else {
//         //     continue;
//         // };
//         //todo make despawn chunk from chunk entitites?
//         // if let Some(mut entity_commands) = commands.get_entity(chunk_id) {
//         //     entity_commands.despawn();
//         // }
//         // world_data.remove(&chunk_pos);
//     }
//     //world_data.finished_mesh_queue.enqueue(&mut retry);
// }

pub fn start_data_tasks(
    mut world_data: &mut WorldData,
    voxel_vector: Arc<voxels::VoxelVector>,
    workers: u32,
) {
    logger::log("Starting Data Tasks");
    let mut data_workers = Vec::new();
    let arc_queue = ArcQueue::new();
    let finished_data_queue: ArcQueue<(Vector3<i32>, ChunkData)> = ArcQueue::new();
    let chunks_data: Arc<Mutex<HashMap<Vector3<i32>, Arc<ChunkData>>>> = world_data.chunks_data.clone();
    for _ in 0..workers {
        data_workers.push(DataWorker::new(
            arc_queue.clone(),
            |item: Vector3<i32>, voxel_vector: Arc<VoxelVector>, chunks_data: Arc<Mutex<HashMap<Vector3<i32>, Arc<ChunkData>>>>| -> (Vector3<i32>, ChunkData) {
                let chunk_data = ChunkData::generate(item);
                return (item, chunk_data);
            },
            voxel_vector.clone(),
            finished_data_queue.clone(),
            chunks_data.clone()
        ));
    };
    world_data.data_workers = data_workers;
    world_data.load_data_queue = arc_queue;
    world_data.finished_data_queue = finished_data_queue;
}

pub fn start_mesh_tasks(
    mut world_data: &mut WorldData,
    voxel_vector: Arc<voxels::VoxelVector>,
    workers: u32,
) {
    logger::log("Starting Mesh Tasks");
    let mut mesh_workers = Vec::new();
    let arc_queue = ArcQueue::new();
    let finished_mesh_queue: ArcQueue<(Vector3<i32>, Option<RawChunkRenderData>)> = ArcQueue::new();
    let chunks_data: Arc<Mutex<HashMap<Vector3<i32>, Arc<ChunkData>>>> = world_data.chunks_data.clone();
    for _ in 0..workers {
        mesh_workers.push(MeshWorker::new(
            arc_queue.clone(),
            |item: Vector3<i32>, voxel_vector: Arc<voxels::VoxelVector>, chunks_data: Arc<Mutex<HashMap<Vector3<i32>, Arc<ChunkData>>>>| -> Option<(Vector3<i32>, Option<RawChunkRenderData>)> {
                let Some(chunks_refs) = ChunksRefs::try_new(chunks_data, item) else {
                    //logger::log("Not enough chunk references!");
                    return None
                };
                let llod = Lod::L32;
                return Some((item, build_chunk_mesh(&chunks_refs, llod, &voxel_vector, &item)));
            },
            voxel_vector.clone(),
            finished_mesh_queue.clone(),
            chunks_data.clone()
        ));
    };
    world_data.mesh_workers = mesh_workers;
    world_data.load_mesh_queue = arc_queue;
    world_data.finished_mesh_queue = finished_mesh_queue;
}

pub fn start_modifications(mut world_data: WorldData) {
    for (pos, mods) in world_data.chunk_modifications.drain() {
        if let Ok(mut chunks_data) = world_data.chunks_data.lock() {
            let Some(chunk_data) = chunks_data.get_mut(&pos) else {
                continue;
            };
            let new_chunk_data = Arc::make_mut(chunk_data);
            let mut adj_chunk_set = HashSet::new();
            for ChunkModification(local_pos, block_type) in mods.into_iter() {
                let i = vector3i_to_index(local_pos, 32);
                if new_chunk_data.voxels.len() == 1 {
                    let mut voxels = vec![];
                    for _ in 0..CHUNK_SIZE_I32 * CHUNK_SIZE_I32 * CHUNK_SIZE_I32 {
                        voxels.push(new_chunk_data.voxels[0]);
                    }
                    new_chunk_data.voxels = voxels;
                }
                new_chunk_data.voxels[i] = block_type;
                if let Some(edge_chunk) = get_edging_chunk(local_pos) {
                    adj_chunk_set.insert(edge_chunk);
                }
            }
            for adj_chunk in adj_chunk_set.into_iter() {
                world_data.load_mesh_queue.enqueue(pos + adj_chunk);
            }
            world_data.load_mesh_queue.enqueue(pos);
        } else {
            continue;
        };
    }
}

#[inline]
pub fn get_edging_chunk(pos: Vector3<i32>) -> Option<Vector3<i32>> {
    let mut chunk_dir = Vector3::zero();
    if pos.x == 0 {
        chunk_dir.x = -1;
    } else if pos.x == 31 {
        chunk_dir.x = 1;
    }
    if pos.y == 0 {
        chunk_dir.y = -1;
    } else if pos.y == 31 {
        chunk_dir.y = 1;
    }
    if pos.z == 0 {
        chunk_dir.z = -1;
    } else if pos.z == 31 {
        chunk_dir.z = 1;
    }
    if chunk_dir == Vector3::zero() {
        None
    } else {
        Some(chunk_dir)
    }
}

#[derive(Debug, Clone)]
pub struct Rect  {
    pub vertices: [[f32; 3]; 4], // 4 vertices, each with x, y, z coordinates
    pub indices: [u32; 6],       // Indices for drawing 2 triangles (6 indices)
    pub blocks: Vec<u32>,        // List of the blocks that should be displayed on this rect
    pub tints: Vec<[f32; 3]>,    // List of tints that should be displayed on this rect
    pub width: u32,
    pub height: u32,
}