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
use crate::wgpulib::{ChunkInstance, ChunkInstanceRaw};
use crate::{logger, voxels, world_handler};
use crate::vertex_types::WorldMeshVertex;
use crate::world_handler::{generate_indices, make_vertex_u32, ArcQueue, ChunkData, ChunkMesh, Queue, DataWorker, MeshWorker};
pub fn build_chunk_mesh(
    chunks_refs: &ChunksRefs,
    lod: Lod,
    voxel_map: &Arc<VoxelVector>,
    position: &Vector3<i32>,
) -> Option<ChunkMesh> {
    if chunks_refs.is_all_voxels_same() {
        return None;
    }
    let mut mesh = ChunkMesh::default();

    let mut axis_cols = [[[0u64; CHUNK_SIZE_P]; CHUNK_SIZE_P]; 3];

    let mut col_face_masks = [[[0u64; CHUNK_SIZE_P]; CHUNK_SIZE_P]; 6];

    #[inline]
    fn add_voxel_to_axis_cols(
        b: &u32,
        x: usize,
        y: usize,
        z: usize,
        axis_cols: &mut [[[u64; 34]; 34]; 3],
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
    // key(voxel + ao) -> HashMap<axis(0-32), binary_plane>
    // note(leddoo): don't ask me how this isn't a massive blottleneck.
    //  might become an issue in the future, when there are more voxel types.
    //  consider using a single hashmap with key (axis, voxel_hash, y).
    let mut data: [HashMap<u32, HashMap<u32, [u32; 32]>>; 6];
    data = [
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    ];

    // find faces and build binary planes based on the voxel voxel+ao etc...
    for axis in 0..6 {
        for z in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                // skip padded by adding 1(for x padding) and (z+1) for (z padding)
                let mut col = col_face_masks[axis][z + 1][x + 1];

                // removes the right most padding value, because it's invalid
                col >>= 1;
                // removes the left most padding value, because it's invalid
                col &= !(1 << CHUNK_SIZE as u64);
                while col != 0 {
                    let y = col.trailing_zeros();
                    // clear least significant set bit
                    col &= col - 1;

                    // get the voxel position based on axis
                    let voxel_pos = match axis {
                        0 | 1 => Vector3::new(x as i32, y as i32, z as i32), // down,up
                        2 | 3 => Vector3::new(y as i32, z as i32, x as i32), // left, right
                        _ => Vector3::new(x as i32, z as i32, y as i32),     // forward, back
                    };

                    // calculate ambient occlusion
                    let mut ao_index = 0;
                    for (ao_i, ao_offset) in ADJACENT_AO_DIRS.iter().enumerate() {
                        // ambient occlusion is sampled based on axis(ascent or descent)
                        let ao_sample_offset = match axis {
                            0 => Vector3::new(ao_offset.x, -1, ao_offset.y), // down
                            1 => Vector3::new(ao_offset.x, 1, ao_offset.y),  // up
                            2 => Vector3::new(-1, ao_offset.y, ao_offset.x), // left
                            3 => Vector3::new(1, ao_offset.y, ao_offset.x),  // right
                            4 => Vector3::new(ao_offset.x, ao_offset.y, -1), // forward
                            _ => Vector3::new(ao_offset.x, ao_offset.y, 1),  // back
                        };
                        let ao_voxel_pos = voxel_pos + ao_sample_offset;
                        let ao_voxel = chunks_refs.get_voxel(ao_voxel_pos);
                        if voxel_map.get(ao_voxel)?.is_solid() {
                            ao_index |= 1u32 << ao_i;
                        }
                    }

                    let current_voxel = chunks_refs.get_voxel_no_neighbour(voxel_pos);
                    // let current_voxel = chunks_refs.get_voxel(voxel_pos);
                    // we can only greedy mesh same voxel types + same ambient occlusion
                    let voxel_hash = ao_index | ((current_voxel) << 9);
                    let data = data[axis]
                        .entry(voxel_hash)
                        .or_default()
                        .entry(y)
                        .or_default();
                    data[x as usize] |= 1u32 << z as u32;
                }
            }
        }
    }
    let mut vertices:Vec<WorldMeshVertex> = vec![];
    for (axis, voxel_ao_data) in data.into_iter().enumerate() {
        let facedir = match axis {
            0 => FaceDir::Down,
            1 => FaceDir::Up,
            2 => FaceDir::Left,
            3 => FaceDir::Right,
            4 => FaceDir::Forward,
            _ => FaceDir::Back,
        };
        for (voxel_ao, axis_plane) in voxel_ao_data.into_iter() {
            let ao = voxel_ao & 0b111111111;
            let voxel_type = voxel_ao >> 9;
            for (axis_pos, plane) in axis_plane.into_iter() {
                let quads_from_axis = greedy_mesh_binary_plane(plane, lod.size() as u32);
                quads_from_axis.into_iter().for_each(|q| {
                    q.append_vertices(&mut vertices, facedir, axis_pos, &Lod::L32, ao, voxel_type)
                });
            }
        }
    }
    let position = Vector3::new(position.x * CHUNK_SIZE as i32, position.y * CHUNK_SIZE as i32, position.z * CHUNK_SIZE as i32);
    let rotation = cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_z(), cgmath::Deg(0.0));
    let chunk_instance = ChunkInstance {
        position, rotation,
    };

    let binding = vec![chunk_instance].iter().map(ChunkInstance::to_raw).collect::<Vec<_>>();
    let t1: &[u8] = bytemuck::cast_slice(&*binding);
    mesh.instance = t1.to_vec();
    mesh.vertices.extend(vertices);
    if mesh.vertices.is_empty() {
        None
    } else {
        mesh.indices = generate_indices(mesh.vertices.len());
        //println!("vertices: {:?}", mesh.vertices.len());
        //println!("{:?}", mesh.indices);
        //println!("{:?}", mesh.instance);
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

        // let v1 = make_vertex_u32(
        //     face_dir.world_to_sample(axis as i32, self.x as i32, self.y as i32, &lod) * jump,
        //     v1ao,
        //     face_dir.normal_index(),
        //     voxel_type,
        // );
        // let v2 = make_vertex_u32(
        //     face_dir.world_to_sample(
        //         axis as i32,
        //         self.x as i32 + self.w as i32,
        //         self.y as i32,
        //         &lod,
        //     ) * jump,
        //     v2ao,
        //     face_dir.normal_index(),
        //     voxel_type,
        // );
        // let v3 = make_vertex_u32(
        //     face_dir.world_to_sample(
        //         axis as i32,
        //         self.x as i32 + self.w as i32,
        //         self.y as i32 + self.h as i32,
        //         &lod,
        //     ) * jump,
        //     v3ao,
        //     face_dir.normal_index(),
        //     voxel_type,
        // );
        // let v4 = make_vertex_u32(
        //     face_dir.world_to_sample(
        //         axis as i32,
        //         self.x as i32,
        //         self.y as i32 + self.h as i32,
        //         &lod,
        //     ) * jump,
        //     v4ao,
        //     face_dir.normal_index(),
        //     voxel_type,
        // );
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
    pub finished_mesh_queue: ArcQueue<(Vector3<i32>, Option<ChunkMesh>)>,
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
    atlas_data: Arc<HashMap<u32, (Vec<Vector2<f32>>, f32, f32)>>,
) {
    logger::log("Starting Data Tasks");
    let mut data_workers = Vec::new();
    let arc_queue = ArcQueue::new();
    let finished_data_queue: ArcQueue<(Vector3<i32>, ChunkData)> = ArcQueue::new();
    let chunks_data: Arc<Mutex<HashMap<Vector3<i32>, Arc<ChunkData>>>> = world_data.chunks_data.clone();
    for _ in 0..1 {
        data_workers.push(DataWorker::new(
            arc_queue.clone(),
            |item: Vector3<i32>, voxel_vector: Arc<VoxelVector>, atlas_data: Arc<HashMap<u32, (Vec<Vector2<f32>>, f32, f32)>>, chunks_data: Arc<Mutex<HashMap<Vector3<i32>, Arc<ChunkData>>>>| -> (Vector3<i32>, ChunkData) {
                let chunk_data = ChunkData::generate(item);
                return (item, chunk_data);
            },
            voxel_vector.clone(),
            atlas_data.clone(),
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
    atlas_data: Arc<HashMap<u32, (Vec<Vector2<f32>>, f32, f32)>>,
) {
    logger::log("Starting Mesh Tasks");
    let mut mesh_workers = Vec::new();
    let arc_queue = ArcQueue::new();
    let finished_mesh_queue: ArcQueue<(Vector3<i32>, Option<ChunkMesh>)> = ArcQueue::new();
    let chunks_data: Arc<Mutex<HashMap<Vector3<i32>, Arc<ChunkData>>>> = world_data.chunks_data.clone();
    for _ in 0..1 {
        mesh_workers.push(MeshWorker::new(
            arc_queue.clone(),
            |item: Vector3<i32>, voxel_vector: Arc<voxels::VoxelVector>, atlas_data: Arc<HashMap<u32, (Vec<Vector2<f32>>, f32, f32)>>, chunks_data: Arc<Mutex<HashMap<Vector3<i32>, Arc<ChunkData>>>>| -> Option<(Vector3<i32>, Option<ChunkMesh>)> {
                let Some(chunks_refs) = ChunksRefs::try_new(chunks_data, item) else {
                    //logger::log("Not enough chunk references!");
                    return None
                };
                let llod = Lod::L32;
                return Some((item, build_chunk_mesh(&chunks_refs, llod, &voxel_vector, &item)));
            },
            voxel_vector.clone(),
            atlas_data.clone(),
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
