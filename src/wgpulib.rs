use std::collections::HashMap;
use std::{iter, thread};
use std::ops::Add;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use cgmath::prelude::*;
use std::sync::mpsc;
use cgmath::{Vector2, Vector3};
use image::{Rgba, RgbaImage};
use rand::{thread_rng, Rng};
use wgpu::{Buffer, Device};
use wgpu::util::DeviceExt;
use winit::{
    event::*,
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};
use winit::window::CursorGrabMode;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
use crate::{camera, logger, texture_packer, timeit, vertex_types, voxels, world_handler};
use crate::texture;
use crate::texture::Texture;
use crate::texture_packer::find_optimal_atlas_size;
use crate::vertex_types::{Vertex, WorldMeshVertex};
use crate::voxels::{TexturePattern, VoxelVector};
use crate::world_handler::{ArcQueue, Queue, Worker, CHUNK_SIZE};

const CULL_BACK_FACE: bool = true;

struct ChunkInstance {
    position: cgmath::Vector3<i32>,
    rotation: cgmath::Quaternion<f32>,
}
impl ChunkInstance {
    fn to_raw(&self) -> ChunkInstanceRaw {
        ChunkInstanceRaw {
            model: (cgmath::Matrix4::from_translation(Vector3::new(self.position.x as f32, self.position.y as f32, self.position.z as f32)) * cgmath::Matrix4::from(self.rotation)).into(),
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
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
    //world: world_generation::World,
    atlas: RgbaImage,

    voxel_vector: Arc<voxels::VoxelVector>,
    texture_map: Arc<HashMap<u8, (Vec<Vector2<f32>>, f32, f32)>>,
    opaque_mesh_buffer: HashMap<Vector3<i32>, (Buffer, Buffer, Buffer, u32)>,
    mesh_workers: Vec<Worker<world_handler::Chunk>>,
    mesh_result_queue: ArcQueue<(Vector3<i32>, (Vec<ChunkInstanceRaw>, world_handler::ChunkRaw))>,
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
        //let world = world_generation::World::new();

        let mut rng = thread_rng();
        let voxels = voxels::initialize();
        let mut images: Vec<(u8, RgbaImage)> = Vec::new();
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
                for i in 1..=variants
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
                label: Some("texture_bind_group_layout"),
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

        let camera = camera::Camera::new((0.0, 0.0, 1.0), cgmath::Deg(-90.0), cgmath::Deg(-20.0));
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
                &[WorldMeshVertex::desc(), ChunkInstanceRaw::desc()],
                wgpu::ShaderModuleDescriptor {
                    label: Some("Shader"),
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
                &[WorldMeshVertex::desc(), ChunkInstanceRaw::desc()],
                wgpu::ShaderModuleDescriptor {
                    label: Some("Shader"),
                    source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
                },
                wgpu::PolygonMode::Line,
            )
        };

        let voxel_vector = Arc::new(voxels);

        let atlas_data: Arc<HashMap<u8, (Vec<Vector2<f32>>, f32, f32)>> = Arc::new(texture_map);

        let mut mesh_workers = Vec::new();
        let arc_queue = ArcQueue::new();
        let result_queue: ArcQueue<(Vector3<i32>, (Vec<ChunkInstanceRaw>, world_handler::ChunkRaw))> = ArcQueue::new();
        for _ in 0..5 {
            mesh_workers.push(Worker::new(
                arc_queue.clone(),
                |item: world_handler::Chunk, voxel_vector: Arc<voxels::VoxelVector>, atlas_data: Arc<HashMap<u8, (Vec<Vector2<f32>>, f32, f32)>>| -> (Vector3<i32>, (Vec<ChunkInstanceRaw>, world_handler::ChunkRaw)) {
                    let raw_chunk = item.to_raw_opaque(voxel_vector, atlas_data);
                    let position = Vector3::new(
                        item.position.x * CHUNK_SIZE as i32,
                        item.position.y * CHUNK_SIZE as i32,
                        item.position.z * CHUNK_SIZE as i32
                    );
                    let chunk_instance = ChunkInstance {
                        position,
                        rotation: cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_z(), cgmath::Deg(0.0)),
                    };
                    let binding = vec![chunk_instance].iter().map(ChunkInstance::to_raw).collect::<Vec<_>>();
                    return (item.position, (binding, raw_chunk));
                },
                voxel_vector.clone(),
                atlas_data.clone(),
                result_queue.clone()
            ));
        }

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
            //world,
            mesh_result_queue: result_queue,
            opaque_mesh_buffer: HashMap::new(),
            mesh_workers,
            texture_map: atlas_data,
            voxel_vector,
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

    fn update(&mut self, dt: std::time::Duration) {
        self.camera_controller.update_camera(&mut self.camera, dt);
        self.camera_uniform
            .update_view_proj(&self.camera, &self.projection);
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );
        if let Some(result) = self.mesh_result_queue.dequeue() {
            let (position, (raw_instance, raw_chunk)) = result;
            if self.opaque_mesh_buffer.contains_key(&position) {
                let buffers = self.opaque_mesh_buffer.get(&result.0).unwrap();
                self.queue.write_buffer(
                    &buffers.0,
                    0,
                    bytemuck::cast_slice(&raw_instance),
                );
                self.queue.write_buffer(
                    &buffers.1,
                    0,
                    bytemuck::cast_slice(&raw_chunk.vertices),
                );
                self.queue.write_buffer(
                    &buffers.2,
                    0,
                    bytemuck::cast_slice(&raw_chunk.indices),
                );
            } else {
                let instance_buffer = self.device.create_buffer_init(
                    &wgpu::util::BufferInitDescriptor {
                        label: Some("Instance Buffer"),
                        contents: bytemuck::cast_slice(&raw_instance),
                        usage: wgpu::BufferUsages::VERTEX,
                    }
                );
                let vertex_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Vertex Buffer"),
                    contents: bytemuck::cast_slice(&raw_chunk.vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                });
                let index_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Index Buffer"),
                    contents:  bytemuck::cast_slice(&raw_chunk.indices),
                    usage: wgpu::BufferUsages::INDEX,
                });
                self.opaque_mesh_buffer.insert(position, (instance_buffer, vertex_buffer, index_buffer, raw_chunk.indices.len() as u32));
            }
        }
    }

    fn unload_chunk(&mut self, position: Vector3<i32>)
    {
        /*
        logger::log_raw("Unloaded chunk at ", position);
        if self.world.chunks.contains_key(&position)
        {
            self.world.chunks.remove(&position);
        }
        */
    }

    fn update_chunk(&mut self, position: Vector3<i32>)
    {
        /*
        let raw_chunk = self.world.chunks.get(&position).unwrap().to_raw_opaque(&self.world.voxel_vector, &self.texture_map);
        if self.opaque_mesh_buffer.contains_key(&position)
        {
            self.opaque_mesh_buffer.remove(&position);
        }
        let position = Vector3::new(position.x * world_generation::CHUNK_SIZE as i32, position.y * world_generation::CHUNK_SIZE as i32, position.z * world_generation::CHUNK_SIZE as i32);
        let rotation = cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_z(), cgmath::Deg(0.0));
        let chunk_instance = ChunkInstance {
            position, rotation,
        };
        let t1 = bytemuck::cast_slice(&*vec![chunk_instance].iter().map(ChunkInstance::to_raw).collect::<Vec<_>>());
        let instance_buffer = self.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Instance Buffer"),
                contents: t1,
                usage: wgpu::BufferUsages::VERTEX,
            }
        );
        let t2 = bytemuck::cast_slice(&raw_chunk.vertices);
        let vertex_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: t2,
            usage: wgpu::BufferUsages::VERTEX,
        });
        let t3 = bytemuck::cast_slice(&raw_chunk.indices);
        let index_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: t3,
            usage: wgpu::BufferUsages::INDEX,
        });
        self.opaque_mesh_buffer.insert(position, (instance_buffer, vertex_buffer, index_buffer, raw_chunk.indices.len() as u32));
        */
    }

    fn load_chunk(&mut self, position: Vector3<i32>)
    {
        self.mesh_workers.get(0).unwrap().enqueue(world_handler::Chunk::new(position));
        /*
        self.update_chunk(position);
        */
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

            if self.camera_controller.amount_l_pressed > 0.0
            {
                render_pass.set_pipeline(&self.render_pipelines.1);
            } else {
                render_pass.set_pipeline(&self.render_pipelines.0);
            }

            render_pass.set_bind_group(0, &self.diffuse_bind_group, &[]);
            render_pass.set_bind_group(1, &self.camera_bind_group, &[]);
            let mut total_triangles: u32 = 0;

            for (position, (instance_buffer, vertex_buffer, index_buffer, index_size)) in &self.opaque_mesh_buffer
            {
                total_triangles = total_triangles + index_size;
                render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                render_pass.set_vertex_buffer(1, instance_buffer.slice(..));
                render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..*index_size, 0, 0..1);
            }
            //println!("{:?}", total_triangles);
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
    let window = winit::window::WindowBuilder::new()
        .with_title(title)
        .build(&event_loop)
        .unwrap();

    let mut state = State::new(&window).await; // NEW!
    let mut last_render_time = instant::Instant::now();
    for x in 0..15
    {
        for y in 0..1
        {
            for z in 0..15
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