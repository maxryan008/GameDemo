use std::collections::HashMap;
use std::{iter, thread};
use std::mem::replace;
use std::ops::{Add, Deref, Div, RangeFull};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use cgmath::prelude::*;
use std::sync::mpsc;
use cgmath::{Quaternion, Vector2, Vector3};
use image::{Rgba, RgbaImage};
use pollster::block_on;
use rand::{thread_rng, Rng};
use wgpu::{BindGroupLayout, Buffer, BufferAddress, BufferBindingType, BufferDescriptor, BufferSlice, CommandEncoder, Device, MaintainResult, Sampler, SurfaceConfiguration};
use wgpu::hal::empty::Encoder;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use winit::{
    event::*,
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};
use winit::monitor::MonitorHandle;
use winit::window::{CursorGrabMode, Fullscreen};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
use crate::{camera, logger, texture_packer, timeit, vertex_types, voxels, world_handler};
use crate::constants::{CHUNK_SIZE, TEXTURE_OUTPUT_SIZE};
use crate::face_direction::{FaceDir, FACES};
use crate::greedy_mesher::{start_data_tasks, start_mesh_tasks, start_modifications, unload_data, Rect, WorldData};
use crate::texture;
use crate::texture::Texture;
use crate::texture_packer::find_optimal_atlas_size;
use crate::vertex_types::{ChunkMeshVertex, Vertex, WorldMeshVertex};
use crate::voxels::{TexturePattern, VoxelVector};
use crate::world_handler::{ArcQueue, Queue, DataWorker};

const CULL_BACK_FACE: bool = true;
const FULL_SCREEN: bool = true;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PackedVec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PackedVec2 {
    pub x: f32,
    pub y: f32,
}

pub struct ChunkInstance {
    pub position: Vector3<i32>,
    pub rotation: Quaternion<f32>,
}

pub struct BufferedChunk {
    pub position: Vector3<i32>,
    pub chunk_texture: wgpu::Texture,
    pub vertex_buffer: Arc<Mutex<Buffer>>,
    pub index_buffer: Arc<Mutex<Buffer>>,
    pub instance_buffer: wgpu::Buffer,
    pub indices_length: u32,
}


pub struct BufferedChunkData {
    pub position: Vector3<i32>,
    pub instance: Buffer,
    pub vertex_buffer: Buffer,
    pub index_buffer: Buffer,
    pub chunk_texture: wgpu::Texture,
    pub indices_length: u32,
    pub chunk_texture_bind_group: wgpu::BindGroup,
}


#[derive(Clone)]
pub struct RawChunkRenderData {
    pub position: Vector3<i32>,
    pub rects: Vec<Rect>,
}


impl Default for RawChunkRenderData {
    fn default() -> Self {
        RawChunkRenderData {
            position: Vector3::from_value(0),
            rects: Vec::new(),
        }
    }
}


impl ChunkInstance {
    pub fn to_raw(&self) -> ChunkInstanceRaw {
        ChunkInstanceRaw {
            model: (cgmath::Matrix4::from_translation(Vector3::new(self.position.x as f32, self.position.y as f32, self.position.z as f32)) * cgmath::Matrix4::from(self.rotation)).into(),
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, Debug)]
pub struct ChunkInstanceRaw {
    model: [[f32; 4]; 4],
}

impl ChunkInstanceRaw {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<ChunkInstanceRaw>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 12]>() as wgpu::BufferAddress,
                    shader_location: 8,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_position: [f32; 4],
    view_proj: [[f32; 4]; 4],
}

impl CameraUniform {
    fn new() -> Self {
        Self {
            view_position: [0.0; 4],
            view_proj: cgmath::Matrix4::identity().into(),
        }
    }
    fn update_view_proj(&mut self, camera: &camera::Camera, projection: &camera::Projection) {
        self.view_position = camera.position.to_homogeneous().into();
        self.view_proj = (projection.calc_matrix() * camera.calc_matrix()).into()
    }
}

fn create_sampler(device: &wgpu::Device) -> wgpu::Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::ClampToEdge, // How to handle out-of-bounds sampling along the U axis
        address_mode_v: wgpu::AddressMode::ClampToEdge, // V axis
        address_mode_w: wgpu::AddressMode::ClampToEdge, // W axis (used for 3D textures)
        mag_filter: wgpu::FilterMode::Nearest, // How to interpolate when magnifying the texture
        min_filter: wgpu::FilterMode::Nearest, // How to interpolate when minifying the texture
        mipmap_filter: wgpu::FilterMode::Linear, // How to interpolate between mipmap levels
        lod_min_clamp: 0.0, // Level of detail (LOD) min clamp
        lod_max_clamp: 100.0, // LOD max clamp
        compare: None, // Optional depth comparison mode
        anisotropy_clamp: 1, // Optional anisotropic filtering
        label: Some("Texture Sampler"),
        border_color: None,
    })
}

struct State<'a> {
    window: &'a Window,
    surface: wgpu::Surface<'a>,
    device: Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    render_pipelines: (wgpu::RenderPipeline, wgpu::RenderPipeline),
    camera: camera::Camera,
    projection: camera::Projection,
    camera_controller: camera::CameraController,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    diffuse_texture: texture::Texture,
    diffuse_bind_group: wgpu::BindGroup,
    #[allow(dead_code)]
    depth_texture: texture::Texture,
    mouse_pressed: bool,
    world_data: WorldData,
    voxel_vector: Arc<voxels::VoxelVector>,
    atlas: RgbaImage,
    texture_map: Arc<HashMap<u32, (Vec<Vector2<u32>>, u32, u32)>>,
    buffered_chunks: HashMap<Vector3<i32>, BufferedChunkData>,
    unfinished_chunks: HashMap<Vector3<i32>, BufferedChunk>,
    texture_bind_group_layout: BindGroupLayout,
    texture_sampler: Sampler,
}

fn create_render_pipeline(
    device: &Device,
    layout: &wgpu::PipelineLayout,
    color_format: wgpu::TextureFormat,
    depth_format: Option<wgpu::TextureFormat>,
    vertex_layouts: &[wgpu::VertexBufferLayout],
    shader: wgpu::ShaderModuleDescriptor,
    polygon_mode: wgpu::PolygonMode,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(shader);
    let cull_face;
    if CULL_BACK_FACE
    {
        cull_face = Some(wgpu::Face::Back);
    } else {
        cull_face = None;
    }
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(&format!("{:?}", shader)),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: vertex_layouts,
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState {
                format: color_format,
                //TODO: Originally replace now blend
                blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: cull_face,
            polygon_mode,
            unclipped_depth: false,
            conservative: false,
        },
        depth_stencil: depth_format.map(|format| wgpu::DepthStencilState {
            format,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
        cache: None,
    })
}

impl<'a> State<'a> {
    async fn new(window: &'a Window) -> State<'a> {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            #[cfg(not(target_arch = "wasm32"))]
            backends: wgpu::Backends::PRIMARY,
            #[cfg(target_arch = "wasm32")]
            backends: wgpu::Backends::GL,
            ..Default::default()
        });

        let surface = instance.create_surface(window).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::POLYGON_MODE_LINE,
                    required_limits: if cfg!(target_arch = "wasm32") {
                        wgpu::Limits::downlevel_webgl2_defaults()
                    } else {
                        wgpu::Limits::default()
                    },
                    memory_hints: Default::default(),
                },
                None,
            )
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);

        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        //let world = world_generation::World::generate();

        let mut rng = thread_rng();
        let voxels = voxels::initialize();
        let mut images: Vec<(u32, RgbaImage)> = Vec::new();
        for voxel in voxels.voxels.iter().clone() {
            let id = voxel.0.clone();
            let voxel_type = voxel.1;
            let variants = voxel_type.variants;
            if variants == 1 {
                let texture = Texture::from_rgba(&voxel_type.texture, 4, 4);
                images.push((id, texture));
            } else {
                let texture = Texture::from_rgba(&voxel_type.texture, 4, 4);
                images.push((id, texture));
                for i in 1..variants
                {
                    let mut variant_pattern: Vec<Rgba<u8>> = voxel_type.texture.pattern.clone();
                    let pos1 = rng.gen_range(0..variant_pattern.len());
                    let pos2 = rng.gen_range(0..variant_pattern.len()-1);
                    let pos3 = rng.gen_range(0..variant_pattern.len()-2);
                    let pos4 = rng.gen_range(0..variant_pattern.len()-2);
                    let pos5 = rng.gen_range(0..variant_pattern.len()-1);
                    let pos6 = rng.gen_range(0..variant_pattern.len());
                    let pixel1 = variant_pattern.remove(pos1);
                    let pixel2 = variant_pattern.remove(pos2);
                    let pixel3 = variant_pattern.remove(pos3);
                    variant_pattern.insert(pos4, pixel1);
                    variant_pattern.insert(pos5, pixel2);
                    variant_pattern.insert(pos6, pixel3);
                    let texture = Texture::from_rgba(&TexturePattern::from_vec(variant_pattern), 4, 4);
                    images.push((id, texture));
                }
            }
        }
        let (atlas, texture_map) = find_optimal_atlas_size(images);
        atlas.save("texture_atlas.png").unwrap();

        let diffuse_texture =
            Texture::from_rgba_image(&device, &queue, &atlas, "atlas.png".into(), false).unwrap();

        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("Texture Bind Group Layout"),
            });

        let diffuse_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&diffuse_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler),
                },
            ],
            label: Some("diffuse_bind_group"),
        });

        let camera = camera::Camera::new((100.0, 100.0, 100.0), cgmath::Deg(-90.0), cgmath::Deg(-20.0));
        let projection =
            camera::Projection::new(config.width, config.height, cgmath::Deg(45.0), 0.1, 100.0);
        let camera_controller = camera::CameraController::new( 20.0, 1.0); //speed

        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera, &projection);

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("camera_bind_group_layout"),
            });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        });

        let depth_texture =
            texture::Texture::create_depth_texture(&device, &config, "depth_texture");

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[
                    &texture_bind_group_layout,
                    &camera_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

        let render_pipeline_fill = {
            create_render_pipeline(
                &device,
                &render_pipeline_layout,
                config.format,
                Some(texture::Texture::DEPTH_FORMAT),
                &[ChunkMeshVertex::desc(), ChunkInstanceRaw::desc()],
                wgpu::ShaderModuleDescriptor {
                    label: Some("Render Shader"),
                    source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
                },
                wgpu::PolygonMode::Fill,
            )
        };
        let render_pipeline_line = {
            create_render_pipeline(
                &device,
                &render_pipeline_layout,
                config.format,
                Some(texture::Texture::DEPTH_FORMAT),
                &[ChunkMeshVertex::desc(), ChunkInstanceRaw::desc()],
                wgpu::ShaderModuleDescriptor {
                    label: Some("Render Shader"),
                    source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
                },
                wgpu::PolygonMode::Line,
            )
        };

        let voxel_vector = Arc::new(voxels);

        let atlas_data: Arc<HashMap<u32, (Vec<Vector2<u32>>, u32, u32)>> = Arc::new(texture_map);

        let mut rng = rand::thread_rng();
        let seed: u32 = rng.gen();

        let mut world_data: WorldData = WorldData
        {
            seed,
            chunks_data: Arc::new(Mutex::new(HashMap::new())),
            load_data_queue: ArcQueue::new(),
            load_mesh_queue: ArcQueue::new(),
            finished_data_queue: ArcQueue::new(),
            finished_mesh_queue: ArcQueue::new(),
            data_workers: Vec::new(),
            mesh_workers: Vec::new(),
            chunk_modifications: HashMap::new(),
        };

        start_data_tasks(&mut world_data, voxel_vector.clone(), 3);
        start_mesh_tasks(&mut world_data, voxel_vector.clone(), 3);

        let buffered_chunks = HashMap::new();

        let unfinished_chunks = HashMap::new();

        let texture_sampler = create_sampler(&device);

        Self {
            window,
            surface,
            device,
            queue,
            config,
            render_pipelines: (render_pipeline_fill, render_pipeline_line),
            camera,
            projection,
            camera_controller,
            camera_buffer,
            camera_bind_group,
            camera_uniform,
            depth_texture,
            atlas,
            size,
            diffuse_texture,
            diffuse_bind_group,
            #[allow(dead_code)]
            mouse_pressed: false,
            world_data,
            texture_map: atlas_data,
            voxel_vector,
            buffered_chunks,
            unfinished_chunks,
            texture_bind_group_layout,
            texture_sampler,
        }
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.projection.resize(new_size.width, new_size.height);
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            self.depth_texture =
                texture::Texture::create_depth_texture(&self.device, &self.config, "depth_texture");
        }
    }
    fn input(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput {
                event:
                KeyEvent {
                    physical_key: PhysicalKey::Code(key),
                    state,
                    ..
                },
                ..
            } => self.camera_controller.process_keyboard(*key, *state),
            WindowEvent::MouseWheel { delta, .. } => {
                self.camera_controller.process_scroll(delta);
                true
            }
            WindowEvent::MouseInput {
                button: MouseButton::Left,
                state,
                ..
            } => {
                self.mouse_pressed = *state == ElementState::Pressed;
                true
            }
            _ => false,
        }
    }

    pub fn place_block_in_front(&mut self, block_type: u32, distance: f32) {
        let look_direction = self.camera.forward();

        let target_position = self.camera.position.to_vec() + (look_direction * distance);

        let block_position = target_position.map(|x| x.floor() as i32);

        self.world_data.queue_block_modification(block_position, block_type);
    }

    pub fn place_sphere(&mut self, block_type: u32, distance: f32, radius: f32) {
        let look_direction = self.camera.forward();
        let center_position = self.camera.position.to_vec() + (look_direction * distance);
        let center_block_position = center_position.map(|x| x.floor() as i32);

        let int_radius = radius.ceil() as i32;
        for x in -int_radius..=int_radius {
            for y in -int_radius..=int_radius {
                for z in -int_radius..=int_radius {
                    let offset = Vector3::new(x, y, z);
                    let block_position = center_block_position + offset;

                    let distance_from_center = offset.map(|c| c as f32).magnitude();

                    if distance_from_center <= radius {
                        self.world_data.queue_block_modification(block_position, block_type);
                    }
                }
            }
        }
    }

    pub fn break_sphere(&mut self, distance: f32, radius: f32) {
        let look_direction = self.camera.forward();
        let center_position = self.camera.position.to_vec() + (look_direction * distance);
        let center_block_position = center_position.map(|x| x.floor() as i32);

        let int_radius = radius.ceil() as i32;
        for x in -int_radius..=int_radius {
            for y in -int_radius..=int_radius {
                for z in -int_radius..=int_radius {
                    let offset = Vector3::new(x, y, z);
                    let block_position = center_block_position + offset;

                    let distance_from_center = offset.map(|c| c as f32).magnitude();

                    if distance_from_center <= radius {
                        self.world_data.queue_block_modification(block_position, 0);
                    }
                }
            }
        }
    }


    pub fn break_block_in_front(&mut self, distance: f32) {
        let look_direction = self.camera.forward();

        let target_position = self.camera.position.to_vec() + (look_direction * distance);

        let block_position = target_position.map(|x| x.floor() as i32);

        self.world_data.queue_block_modification(block_position, 0);
    }

    fn update(&mut self, dt: std::time::Duration) {
        //applies all modifications to world at once
        start_modifications(&mut self.world_data);
        let player_chunk_position = get_chunk_pos_from_world(self.camera.position.to_vec());
        //println!("chunk: {:?}", player_chunk_position);
        if self.camera_controller.amount_k_pressed > 0.0  {
            if !self.camera_controller.k_already_pressed {
                self.place_sphere(5, 20.0, 5.0);
                self.camera_controller.k_already_pressed = true;
            }
        } else {
            self.camera_controller.k_already_pressed = false;
        }
        if self.camera_controller.amount_j_pressed > 0.0  {
            if !self.camera_controller.j_already_pressed {
                self.break_sphere(20.0, 5.0);
                self.camera_controller.j_already_pressed = true;
            }
        } else {
            self.camera_controller.j_already_pressed = false;
        }
        self.camera_controller.update_camera(&mut self.camera, dt);
        self.camera_uniform
            .update_view_proj(&self.camera, &self.projection);
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );
        if let Some(result) = self.world_data.finished_data_queue.dequeue()
        {
            if let Ok(mut chunks_data) = self.world_data.chunks_data.lock() {
                chunks_data.insert(result.0, Arc::new(result.1.clone()));
            }
            self.world_data.load_mesh_queue.enqueue(result.0);
        }
        //println!("finished chunks: {:?}", self.world_data.finished_mesh_queue.length());
        if let Some(result) = self.world_data.finished_mesh_queue.dequeue() {
            let (position, chunk_mesh) = result;
            if let Some(chunk_mesh_result) = chunk_mesh
            {
                self.unfinished_chunks.insert(position, process_raw_chunk(chunk_mesh_result, &self.device, &self.queue, self.atlas.clone(), self.texture_map.clone(), self.world_data.seed));
            }
        }

        let mut calculated_chunks: Vec<Vector3<i32>> = Vec::new();

        for (position, buffered_chunk) in self.unfinished_chunks.iter_mut() {
            // Poll the GPU to check if this chunk's texture is ready
            self.device.poll(wgpu::Maintain::Poll);

            // Create the texture bind group dynamically from the chunk's texture
            let chunk_texture_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&buffered_chunk.chunk_texture.create_view(
                            &wgpu::TextureViewDescriptor::default(),
                        )),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.texture_sampler),
                    },
                ],
                label: Some("Chunk Texture Bind Group"),
            });

            // Move the data from unfinished_chunks to buffered_chunks
            let instance_buffer = replace(
                &mut buffered_chunk.instance_buffer,
                create_dummy_buffer(&self.device), // Replace with a dummy buffer
            );

            let vertex_buffer = replace(
                &mut *buffered_chunk.vertex_buffer.lock().unwrap(),
                create_dummy_buffer(&self.device), // Replace with a dummy buffer
            );

            let index_buffer = replace(
                &mut *buffered_chunk.index_buffer.lock().unwrap(),
                create_dummy_buffer(&self.device), // Replace with a dummy buffer
            );

            let chunk_texture = replace(
                &mut buffered_chunk.chunk_texture,
                self.device.create_texture(&wgpu::TextureDescriptor { // Replace with a dummy texture
                    size: wgpu::Extent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
                    label: Some("Dummy Texture"),
                    view_formats: &[],
                }),
            );

            // Copy buffers as needed
            let vertex_buffer_copy = copy_buffer(&self.device, &self.queue, &vertex_buffer, vertex_buffer.size(), vertex_buffer.usage());
            let index_buffer_copy = copy_buffer(&self.device, &self.queue, &index_buffer, index_buffer.size(), index_buffer.usage());
            // Insert into buffered_chunks
            self.buffered_chunks.insert(buffered_chunk.position, BufferedChunkData {
                position: buffered_chunk.position,
                instance: instance_buffer, // Use moved instance buffer
                vertex_buffer: vertex_buffer_copy, // Use copied vertex buffer
                index_buffer: index_buffer_copy,   // Use copied index buffer
                chunk_texture,                     // Move the texture directly
                indices_length: buffered_chunk.indices_length,
                chunk_texture_bind_group,
            });

            calculated_chunks.push(*position);
        }

        // After moving, remove the processed chunks
        for pos in calculated_chunks {
            self.unfinished_chunks.remove(&pos);
        }
    }

    fn unload_chunk(&mut self, position: Vector3<i32>) {
        if self.world_data.chunks_data.lock().unwrap().contains_key(&position) {
            logger::log_raw("Unloading chunk at", position);
            self.world_data.chunks_data.lock().unwrap().remove(&position);
            self.buffered_chunks.remove(&position);

            if self.unfinished_chunks.remove(&position).is_some() {
                logger::log_raw("Removed unfinished chunk at", position);
            }
        } else {
            logger::log_raw("Error: Attempted to unload non-existent chunk at", position);
        }
    }

    fn load_chunk(&mut self, position: Vector3<i32>)
    {
        self.world_data.load_data_queue.enqueue(position);
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            if self.camera_controller.amount_l_pressed > 0.0 {
                render_pass.set_pipeline(&self.render_pipelines.1);
            } else {
                render_pass.set_pipeline(&self.render_pipelines.0);
            }

            render_pass.set_bind_group(1, &self.camera_bind_group, &[]);
            // Render all buffered chunks
            //println!("Rendering {:?} chunks", self.buffered_chunks.len());
            for (position, chunk) in self.buffered_chunks.iter() {
                // Set the texture bind group for the chunk's texture
                render_pass.set_bind_group(0, &chunk.chunk_texture_bind_group, &[]);

                // Set the vertex and instance buffers
                render_pass.set_vertex_buffer(0, chunk.vertex_buffer.slice(..));
                //render_pass.set_vertex_buffer(0, vertices_flat_buffer.slice(..));
                render_pass.set_vertex_buffer(1, chunk.instance.slice(..));

                // Set the index buffer
                render_pass.set_index_buffer(chunk.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                //render_pass.set_index_buffer(indices_flat_buffer.slice(..), wgpu::IndexFormat::Uint32);
                // Draw the chunk with the correct number of indices and instances
                render_pass.draw_indexed(0..chunk.indices_length, 0, 0..1);
            }
        }

        self.queue.submit(iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub async fn run() {
    cfg_if::cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            std::panic::set_hook(Box::new(console_error_panic_hook::hook));
            console_log::init_with_level(log::Level::Info).expect("Couldn't initialize logger");
        } else {
            env_logger::init();
        }
    }

    let event_loop = EventLoop::new().unwrap();
    let title = env!("CARGO_PKG_NAME");
    let mut fullscreen = None;
    if FULL_SCREEN {
        fullscreen = Some(Fullscreen::Borderless(None));
    }
    let window = winit::window::WindowBuilder::new()
        .with_title(title)
        .with_fullscreen(fullscreen)
        .build(&event_loop)
        .expect("Failed to create window");

    let mut state = State::new(&window).await; // NEW!
    let mut last_render_time = instant::Instant::now();
    let world_size_cubed = 7;
    for x in 0..world_size_cubed
    {
        for y in 0..world_size_cubed
        {
            for z in 0..world_size_cubed
            {
                state.load_chunk(Vector3::new(x, y, z));
            }
        }
    }
    window.set_cursor_grab(CursorGrabMode::Locked).or_else(|_e| window.set_cursor_grab(CursorGrabMode::Confined)).unwrap();
    window.set_cursor_visible(false);

    event_loop.run(move |event, control_flow| {
        match event {
            Event::DeviceEvent {
                event: DeviceEvent::MouseMotion{ delta, },
                ..
            } => /* if state.mouse_pressed*/ {
                state.camera_controller.process_mouse(delta.0, delta.1)
            }
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == state.window().id() && !state.input(event) => {
                match event {
                    #[cfg(not(target_arch="wasm32"))]
                    WindowEvent::CloseRequested
                    | WindowEvent::KeyboardInput {
                        event:
                        KeyEvent {
                            state: ElementState::Pressed,
                            physical_key: PhysicalKey::Code(KeyCode::Escape),
                            ..
                        },
                        ..
                    } => control_flow.exit(),
                    WindowEvent::Resized(physical_size) => {
                        state.resize(*physical_size);
                    }
                    WindowEvent::RedrawRequested => {
                        state.window().request_redraw();
                        let now = instant::Instant::now();
                        let dt = now - last_render_time;
                        last_render_time = now;
                        //fps counter sort of
                        //println!("{:?}", 1000000/dt.as_micros());
                        state.update(dt);
                        match state.render() {
                            Ok(_) => {}
                            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => state.resize(state.size),
                            Err(wgpu::SurfaceError::OutOfMemory) => control_flow.exit(),
                            Err(wgpu::SurfaceError::Timeout) => log::warn!("Surface timeout"),
                        }
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }).unwrap();
}

pub fn process_raw_chunk(
    mut raw_chunk: RawChunkRenderData,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    atlas: RgbaImage,
    texture_map: Arc<HashMap<u32, (Vec<Vector2<u32>>, u32, u32)>>,
    seed: u32,
) -> BufferedChunk
{
    let atlas_size = atlas.dimensions();

    // 1. Upload atlas texture
    let atlas_texture = device.create_texture(&wgpu::TextureDescriptor {
        size: wgpu::Extent3d {
            width: atlas_size.0,
            height: atlas_size.1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::STORAGE_BINDING,
        label: Some("Atlas Texture"),
        view_formats: &[],
    });

    // Write atlas data to GPU
    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &atlas_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &atlas,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(4 * atlas_size.0),
            rows_per_image: Some(atlas_size.1),
        },
        wgpu::Extent3d {
            width: atlas_size.0,
            height: atlas_size.1,
            depth_or_array_layers: 1,
        },
    );

    // 2. Create output texture for stitched result
    let output_texture = device.create_texture(&wgpu::TextureDescriptor {
        size: wgpu::Extent3d {
            width: TEXTURE_OUTPUT_SIZE,
            height: TEXTURE_OUTPUT_SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING, // Add TEXTURE_BINDING here
        label: Some("Output Texture"),
        view_formats: &[],
    });

    // 4. Flatten the rect data and create buffers
    let mut vertices_flat: Vec<[f32; 3]> = vec![];
    let mut indices_flat: Vec<u32> = vec![];
    let mut blocks_flat: Vec<u32> = vec![];
    let mut tints_flat: Vec<[f32; 3]> = vec![];
    let mut rects_vertices_sizes: Vec<u32> = vec![];
    let mut rects_indices_sizes: Vec<u32> = vec![];
    let mut rects_blocks_sizes: Vec<u32> = vec![];
    let mut rects_tints_sizes: Vec<u32> = vec![];

    for rect in raw_chunk.rects.clone() {
        rects_vertices_sizes.push(vertices_flat.len() as u32);
        rects_indices_sizes.push(indices_flat.len() as u32);
        rects_blocks_sizes.push(blocks_flat.len() as u32);
        rects_tints_sizes.push(tints_flat.len() as u32);
        vertices_flat.extend(rect.vertices);
        indices_flat.extend(rect.indices);
        blocks_flat.extend(rect.blocks);
        tints_flat.extend(rect.tints);
    }
    rects_vertices_sizes.push(vertices_flat.len() as u32);
    rects_indices_sizes.push(indices_flat.len() as u32);
    rects_blocks_sizes.push(blocks_flat.len() as u32);
    rects_tints_sizes.push(tints_flat.len() as u32);


    //todo make width and height automatic
    //print_rect_directions(raw_chunk.rects.clone());
    let (packed_positions, rects_widths): (Vec<[u32;2]>, Vec<u32>) = pack_rects_in_rows(&raw_chunk.rects, TEXTURE_OUTPUT_SIZE / 4, TEXTURE_OUTPUT_SIZE / 4);

    // Upload the packed_positions to the GPU
    let positions_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Rect Positions Buffer"),
        contents: bytemuck::cast_slice(&packed_positions),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    // Create GPU buffers for the flattened data
    let vertices_flat_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Vertices Buffer"),
        contents: bytemuck::cast_slice(&vertices_flat),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    let indices_flat_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Indices Buffer"),
        contents: bytemuck::cast_slice(&indices_flat),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let blocks_flat_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Blocks Buffer"),
        contents: bytemuck::cast_slice(&blocks_flat),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let tints_flat_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Tints Buffer"),
        contents: bytemuck::cast_slice(&tints_flat),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let vertices_sizes_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Vertices Sizes Buffer"),
        contents: bytemuck::cast_slice(&rects_vertices_sizes),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let indices_sizes_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Indices Sizes Buffer"),
        contents: bytemuck::cast_slice(&rects_indices_sizes),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let blocks_sizes_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Blocks Sizes Buffer"),
        contents: bytemuck::cast_slice(&rects_blocks_sizes),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let tints_sizes_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Tints Sizes Buffer"),
        contents: bytemuck::cast_slice(&rects_tints_sizes),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let rects_widths_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Tints Sizes Buffer"),
        contents: bytemuck::cast_slice(&rects_widths),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // Texture map buffer (flattened texture map data)
    let mut sorted_keys: Vec<_> = texture_map.keys().cloned().collect();
    sorted_keys.sort();  // Sort the ids

    let mut texture_map_data: Vec<u32> = Vec::new();
    let mut cumulative_variant_lengths: Vec<u32> = Vec::new();
    let mut cumulative_sum: u32 = 0;

    for id in sorted_keys {
        // Get the value associated width the current id
        if let Some((variants, width, height)) = texture_map.get(&id) {
            // Flatten the variants into the texture_map_data
            for variant in variants {
                texture_map_data.extend_from_slice(&[variant.x, variant.y, *width, *height]);
            }

            // Calculate the cumulative variant length
            cumulative_sum += variants.len() as u32;
            cumulative_variant_lengths.push(cumulative_sum);
        }
    }

    let texture_map_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Texture Map Buffer"),
        contents: bytemuck::cast_slice(&texture_map_data),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let variant_lengths_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Variant Lengths Buffer"),
        contents: bytemuck::cast_slice(&cumulative_variant_lengths),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let seed_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Seed Buffer"),
        contents: bytemuck::cast_slice(&[seed]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // 5. Create output buffers with Arc and Mutex for async handling
    let mut vertex_output_buffer = Arc::new(Mutex::new(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Vertex Output Buffer"),
        size: (vertices_flat.len() * std::mem::size_of::<ChunkMeshVertex>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::VERTEX,
        mapped_at_creation: false,
    })));
    let mut index_output_buffer = Arc::new(Mutex::new(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Index Output Buffer"),
        size: (indices_flat.len() * std::mem::size_of::<u32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::INDEX,
        mapped_at_creation: false,
    })));

    let bind_group_layout_group_0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 7,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
        label: Some("Compute Bind Group Layout Group 0"),
    });

    let bind_group_layout_group_1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false }, // Writable buffer
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false }, // Writable buffer
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 7,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },

        ],
        label: Some("Compute Bind Group Layout Group 1"),
    });

    let bind_group_layout_group_2 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }
        ],
        label: Some("Compute Bind Group Layout Group 2"),
    });

    let pipeline_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute Pipeline Layout"),
            bind_group_layouts: &[
                &bind_group_layout_group_0,
                &bind_group_layout_group_1,
                &bind_group_layout_group_2,
            ],
            push_constant_ranges: &[],
        });

    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Compute Shader Module"),
        source: wgpu::ShaderSource::Wgsl(include_str!("compute_shader.wgsl").into()),
    });

    // Create compute pipeline
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: "main",
        compilation_options: Default::default(),
        label: Some("Texture Stitching Compute Pipeline"),
        cache: None,
    });

    let bind_group_0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &bind_group_layout_group_0,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: vertices_flat_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: indices_flat_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: blocks_flat_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: tints_flat_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: texture_map_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::TextureView(&atlas_texture.create_view(&wgpu::TextureViewDescriptor::default())),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: wgpu::BindingResource::TextureView(&output_texture.create_view(&wgpu::TextureViewDescriptor::default())),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: seed_buffer.as_entire_binding(),
            },
        ],
        label: Some("Compute Bind Group 0"),
    });

    let bind_group_1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &bind_group_layout_group_1,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: vertices_sizes_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: indices_sizes_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: blocks_sizes_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: tints_sizes_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: variant_lengths_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: vertex_output_buffer.clone().lock().unwrap().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: index_output_buffer.clone().lock().unwrap().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: rects_widths_buffer.as_entire_binding(),
            },
        ],
        label: Some("Compute Bind Group 1"),
    });

    let bind_group_2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &bind_group_layout_group_2,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: positions_buffer.as_entire_binding(),
            }
        ],
        label: Some("Compute Bind Group 2"),
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Compute Encoder"),
    });

    // Create and dispatch compute pass
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Compute Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&compute_pipeline);
        pass.set_bind_group(0, &bind_group_0, &[]);
        pass.set_bind_group(1, &bind_group_1, &[]);
        pass.set_bind_group(2, &bind_group_2, &[]);
        pass.dispatch_workgroups(raw_chunk.rects.len().div_ceil(10) as u32, 10, 1);
    }

    // Submit the commands to the GPU without blocking
    queue.submit(Some(encoder.finish()));



    let position = Vector3::new(raw_chunk.position.x * CHUNK_SIZE as i32, raw_chunk.position.y * CHUNK_SIZE as i32, raw_chunk.position.z * CHUNK_SIZE as i32);
    let rotation = cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_z(), cgmath::Deg(0.0));
    let chunk_instance = ChunkInstance {
        position, rotation,
    };

    let binding = vec![chunk_instance].iter().map(ChunkInstance::to_raw).collect::<Vec<_>>();
    let instance: &[u8] = bytemuck::cast_slice(&*binding);

    let instance_buffer = device.create_buffer_init(
        &wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: &instance,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_SRC,
        }
    );

    BufferedChunk {
        position: raw_chunk.position,
        chunk_texture: output_texture,
        instance_buffer,
        vertex_buffer: vertex_output_buffer.clone(),
        index_buffer: index_output_buffer.clone(),
        indices_length: raw_chunk.rects.len() as u32 * 6,
    }
}

// Function to pack rects and return their top-left positions
fn pack_rects_in_rows(rects: &Vec<Rect>, output_width: u32, output_height: u32) -> (Vec<[u32; 2]>, Vec<u32>) {
    let mut placed_positions = vec![[0; 2]; rects.len()]; // To store the top-left positions, initialized to 0
    let mut placed_widths = vec![0; rects.len()];
    let mut current_row_y = 0; // Start from the top row
    let mut current_row_x = 0; // Track the current X position in the row
    let mut max_row_height = 0; // Track the maximum height in the current row

    // Create a vector of rects along with their original indices
    let mut indexed_rects: Vec<(usize, &Rect)> = rects.iter().enumerate().collect();

    // Sort the rects by height (optional, helps to pack the tallest rects first)
    indexed_rects.sort_by_key(|(_, r)| std::cmp::Reverse(r.height));

    for (original_index, rect) in indexed_rects.iter() {
        // Check if the rect can fit in the current row
        if current_row_x + rect.width > output_width {
            // Move to the next row if the rect doesn't fit horizontally
            current_row_y += max_row_height;
            current_row_x = 0; // Reset the X position to the start of the row
            max_row_height = 0; // Reset max row height for the new row
        }

        // Check if we have enough height for the next row
        if current_row_y + rect.height > output_height {
            println!("Error: Could not place rect with width {} and height {}", rect.width, rect.height);
            continue; // Skip this rect if there's no more room
        }

        // Place the rect in the current row
        placed_positions[*original_index] = [current_row_x, current_row_y]; // Store in the original order
        placed_widths[*original_index] = rect.width;
        // Update the current row's X position and max row height
        current_row_x += rect.width;
        max_row_height = max_row_height.max(rect.height);
    }

    (placed_positions, placed_widths)
}

#[derive(Copy, Clone)]
pub struct RawRect {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

fn copy_texture(
    device: &wgpu::Device,
    src_texture: &wgpu::Texture,
) -> wgpu::Texture {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Texture Cloning Encoder"),
    });

    let texture_desc = wgpu::TextureDescriptor {
        size: wgpu::Extent3d {
            width: TEXTURE_OUTPUT_SIZE,
            height: TEXTURE_OUTPUT_SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
        label: Some("Destination Texture"),
        view_formats: &[],
    };

    // Create the destination texture
    let dst_texture = device.create_texture(&texture_desc);

    // Define source and destination texture copies
    let src_copy = wgpu::ImageCopyTexture {
        texture: src_texture,
        mip_level: 0,
        origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
        aspect: wgpu::TextureAspect::All,
    };

    let dst_copy = wgpu::ImageCopyTexture {
        texture: &dst_texture,
        mip_level: 0,
        origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
        aspect: wgpu::TextureAspect::All,
    };

    // Perform the copy operation
    encoder.copy_texture_to_texture(src_copy, dst_copy, texture_desc.size);

    dst_texture
}

fn copy_buffer(device: &wgpu::Device, queue: &wgpu::Queue, original_buffer: &wgpu::Buffer, size: wgpu::BufferAddress, usage: wgpu::BufferUsages) -> wgpu::Buffer {
    // Create a new buffer with the same size and usage as the original one
    let new_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Cloned Buffer"),
        size: size,
        usage: usage | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Submit a command to copy from the original buffer to the new buffer
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Buffer Copy Encoder"),
    });
    encoder.copy_buffer_to_buffer(original_buffer, 0, &new_buffer, 0, size);

    // Submit the commands to the queue
    queue.submit(Some(encoder.finish()));

    new_buffer
}

fn create_dummy_buffer(device: &wgpu::Device) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Dummy Buffer"),
        size: 1, // Minimal size
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

fn get_rect_direction(rect: &Rect) -> &'static str {
    let v = &rect.vertices;

    // Check if all vertices share the same X coordinate (Left or Right face)
    if v[0][0] == v[1][0] && v[0][0] == v[2][0] && v[0][0] == v[3][0] {
        if v[0][0] == 0.0 {
            "Left" // X = 0 face (Left face)
        } else {
            "Right" // X = CHUNK_SIZE face (Right face)
        }
    }
    // Check if all vertices share the same Y coordinate (Up or Down face)
    else if v[0][1] == v[1][1] && v[0][1] == v[2][1] && v[0][1] == v[3][1] {
        if v[0][1] == 0.0 {
            "Down" // Y = 0 face (Down face)
        } else {
            "Up" // Y = CHUNK_SIZE face (Up face)
        }
    }
    // Check if all vertices share the same Z coordinate (Front or Back face)
    else if v[0][2] == v[1][2] && v[0][2] == v[2][2] && v[0][2] == v[3][2] {
        if v[0][2] == 0.0 {
            "Front" // Z = 0 face (Front face)
        } else {
            "Back" // Z = CHUNK_SIZE face (Back face)
        }
    } else {
        "Unknown" // In case it does not match any expected face
    }
}

fn print_rect_directions(rects: Vec<Rect>) {
    for (i, rect) in rects.iter().enumerate() {
        let direction = get_rect_direction(rect);
        println!("Rect {:?} is facing {}", rect, direction);
    }
}

pub fn get_chunk_pos_from_world(world_pos: Vector3<f32>) -> Vector3<i32>
{
    Vector3::new((world_pos.x / CHUNK_SIZE as f32).floor() as i32, (world_pos.y / CHUNK_SIZE as f32).floor() as i32, (world_pos.z / CHUNK_SIZE as f32).floor() as i32)
}

pub fn vec3_i32_f32(v: Vector3<i32>) -> Vector3<f32>
{
    v.map(|x| x as f32)
}
