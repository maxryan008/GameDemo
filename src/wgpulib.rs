use std::collections::HashMap;
use std::iter;
use std::ops::Sub;
use cgmath::prelude::*;
use cgmath::Vector3;
use log::logger;
use wgpu::{Device, Queue};
use wgpu::naga::TypeInner::Vector;
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
use crate::{camera, logger, vertex_types, voxels, world_generation};
use crate::texture;
use crate::vertex_types::{Vertex, WorldMeshVertex};

const CULL_BACK_FACE: bool = false;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_position: [f32; 4],
    view_proj: [[f32; 4]; 4],
}

struct MeshBuffer {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
}

struct MeshBufferRaw {
    vertex_buffer: Vec<WorldMeshVertex>,
    index_buffer: Vec<u32>,
    num_indices: u32,
}

impl MeshBuffer {
    fn new(device: &Device) -> Self {
        MeshBuffer
        {
            vertex_buffer: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: &*vec![],
                usage: wgpu::BufferUsages::VERTEX,
            }),
            index_buffer: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Index Buffer"),
                contents: &*vec![],
                usage: wgpu::BufferUsages::INDEX,
            }),
            num_indices: 0,
        }
    }
}

impl MeshBufferRaw {
    pub fn new() -> Self {
        MeshBufferRaw
        {
            num_indices: 0,
            vertex_buffer: vec![],
            index_buffer: vec![],
        }
    }
}

struct BufferIndexer {
    //position key, u32 as order, u32 as size, u32 as vertex size
    by_position: HashMap<Vector3<u32>, (u32, u32, u32)>,
    //order key, position value, u32 as size, u32 as vertex size
    by_order: Vec<(Vector3<u32>, u32, u32)>,
    total_index_size: u32,
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
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    calculated_buffer: MeshBuffer,
    raw_buffer: MeshBufferRaw,
    buffer_indexer: BufferIndexer,
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
    world: world_generation::World,
}

fn create_render_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    color_format: wgpu::TextureFormat,
    depth_format: Option<wgpu::TextureFormat>,
    vertex_layouts: &[wgpu::VertexBufferLayout],
    shader: wgpu::ShaderModuleDescriptor,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(shader);
    let cullface;
    if CULL_BACK_FACE
    {
        cullface = Some(wgpu::Face::Back);
    } else {
        cullface = None
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
            cull_mode: cullface,
            polygon_mode: wgpu::PolygonMode::Fill,
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
                    required_features: wgpu::Features::empty(),
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

        let diffuse_bytes = include_bytes!("happy-tree.png");
        let diffuse_texture =
            texture::Texture::from_bytes(&device, &queue, diffuse_bytes, "happy-tree.png", false).unwrap();

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
        let camera_controller = camera::CameraController::new(100.0, 1.0);

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

        let render_pipeline = {
            create_render_pipeline(
                &device,
                &render_pipeline_layout,
                config.format,
                Some(texture::Texture::DEPTH_FORMAT),
                &[vertex_types::WorldMeshVertex::desc()],
                wgpu::ShaderModuleDescriptor {
                    label: Some("Shader"),
                    source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
                },
            )
        };

        let mut world = world_generation::World::new();

        let calculated_buffer: MeshBuffer = MeshBuffer::new(&device);
        let raw_buffer: MeshBufferRaw = MeshBufferRaw::new();
        let buffer_indexer: BufferIndexer = BufferIndexer
        {
            by_position: Default::default(),
            by_order: Default::default(),
            total_index_size: 0,
        };
        Self {
            window,
            surface,
            device,
            queue,
            config,
            render_pipeline,
            camera,
            projection,
            camera_controller,
            camera_buffer,
            camera_bind_group,
            camera_uniform,
            depth_texture,
            size,
            calculated_buffer,
            raw_buffer,
            buffer_indexer,
            diffuse_texture,
            diffuse_bind_group,
            #[allow(dead_code)]
            mouse_pressed: false,
            world,
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
    }

    fn rebuild_chunk(&mut self, position: Vector3<u32>) {
        if self.buffer_indexer.by_position.contains_key(&position) {
            //rebuilding old chunk
            let order: u32 = self.buffer_indexer.by_position.get(&position).unwrap().0;
            let mut indexer: u32 = 0;
            for i in 0..=order
            {
                indexer += self.buffer_indexer.by_order[i as usize].1;
            }
            let mut vertexer: u32 = 0;
            for i in 0..=order
            {
                vertexer += self.buffer_indexer.by_order[i as usize].2;
            }
            let chunk_raw = self.world.get_raw_chunk(&position,indexer);
            self.buffer_indexer.total_index_size += chunk_raw.indices.len() as u32 - self.buffer_indexer.by_order.get(order as usize).unwrap().1;
            self.raw_buffer.num_indices += chunk_raw.indices.len() as u32 - self.buffer_indexer.by_order.get(order as usize).unwrap().1;
            let (left_v, right_v) = self.raw_buffer.vertex_buffer.split_at(vertexer as usize);
            let (left_v, _) = left_v.split_at((vertexer - self.buffer_indexer.by_position.get(&position).unwrap().2) as usize);
            let (left_i, right_i) = self.raw_buffer.index_buffer.split_at(indexer as usize);
            let (left_i, _) = left_i.split_at((indexer - self.buffer_indexer.by_position.get(&position).unwrap().1) as usize);
            let mut left_v_vec = left_v.to_vec();
            let mut left_i_vec = left_i.to_vec();
            left_v_vec.extend_from_slice(&chunk_raw.vertices);
            left_i_vec.extend_from_slice(&chunk_raw.indices);
            left_v_vec.extend_from_slice(right_v);
            left_i_vec.extend_from_slice(right_i);
            self.buffer_indexer.by_position.remove(&position);
            self.buffer_indexer.by_position.insert(position,(order, chunk_raw.indices.len() as u32, chunk_raw.vertices.len() as u32));
            self.buffer_indexer.by_order.remove(order as usize);
            self.buffer_indexer.by_order.insert(order as usize,(position, chunk_raw.indices.len() as u32, chunk_raw.vertices.len() as u32));
            self.raw_buffer.index_buffer = left_i_vec;
            self.raw_buffer.vertex_buffer = left_v_vec;
        }else {
            //building new chunk
            let order: u32 = self.buffer_indexer.by_position.len() as u32;
            let chunk_raw = self.world.get_raw_chunk(&position,self.buffer_indexer.total_index_size);
            self.buffer_indexer.by_position.insert(position,(order, chunk_raw.indices.len() as u32, chunk_raw.vertices.len() as u32));
            self.buffer_indexer.by_order.insert(order as usize,(position, chunk_raw.indices.len() as u32, chunk_raw.vertices.len() as u32));
            self.buffer_indexer.total_index_size += chunk_raw.indices.len() as u32;
            self.raw_buffer.num_indices += chunk_raw.indices.len() as u32;
            self.raw_buffer.index_buffer.extend(&chunk_raw.indices);
            self.raw_buffer.vertex_buffer.extend(&chunk_raw.vertices);
        }

        self.submit_buffer();
    }

    fn unload_chunk(&mut self, position: Vector3<u32>)
    {
        if self.buffer_indexer.by_position.contains_key(&position) {
            let order: u32 = self.buffer_indexer.by_position.get(&position).unwrap().0;
            let mut indexer: u32 = 0;
            for i in 0..=order
            {
                indexer += self.buffer_indexer.by_order[i as usize].1;
            }
            let mut vertexer: u32 = 0;
            for i in 0..=order
            {
                vertexer += self.buffer_indexer.by_order[i as usize].2;
            }
            let index_size = self.buffer_indexer.by_position.get(&position).unwrap().1;
            let vertex_size = self.buffer_indexer.by_position.get(&position).unwrap().2;
            let (left_v, right_v) = self.raw_buffer.vertex_buffer.split_at(vertexer as usize);
            let (left_v, _) = left_v.split_at((vertexer - self.buffer_indexer.by_position.get(&position).unwrap().2) as usize);
            let (left_i, right_i) = self.raw_buffer.index_buffer.split_at(indexer as usize);
            let (left_i, _) = left_i.split_at((indexer - self.buffer_indexer.by_position.get(&position).unwrap().1) as usize);
            let mut left_v_vec = left_v.to_vec();
            let mut left_i_vec = left_i.to_vec();
            left_v_vec.extend_from_slice(right_v);
            left_i_vec.extend_from_slice(right_i);
            self.buffer_indexer.total_index_size -= self.buffer_indexer.by_order.get(order as usize).unwrap().1;
            self.buffer_indexer.by_position.remove(&position);
            self.raw_buffer.num_indices -= self.buffer_indexer.by_order.get(order as usize).unwrap().1;
            self.buffer_indexer.by_order.remove(order as usize);
            self.raw_buffer.index_buffer = left_i_vec;
            self.raw_buffer.vertex_buffer = left_v_vec;

            for i in order as usize..self.buffer_indexer.by_position.len() {
                println!("{:?}", i);
                println!("{:?}", &self.buffer_indexer.by_order[i].0);
                println!("{:?}", self.buffer_indexer.by_position.get(&self.buffer_indexer.by_order[i].0).unwrap().0);
                self.buffer_indexer.by_position.get_mut(&self.buffer_indexer.by_order[i].0).unwrap().0 = i as u32;
                for mut j in self.raw_buffer.index_buffer.iter_mut()
                {
                    j = &mut j.sub(vertex_size);
                }
            }

            self.submit_buffer();
        }
    }

    fn submit_buffer(&mut self)
    {
        //turn raw into buffer
        let mut vertex_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&self.raw_buffer.vertex_buffer),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let mut index_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&self.raw_buffer.index_buffer),
            usage: wgpu::BufferUsages::INDEX,
        });

        //submit buffer to renderer
        self.calculated_buffer = MeshBuffer
        {
            vertex_buffer,
            index_buffer,
            num_indices: self.raw_buffer.num_indices,
        }
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

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.diffuse_bind_group, &[]);
            render_pass.set_bind_group(1, &self.camera_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.calculated_buffer.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.calculated_buffer.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..self.calculated_buffer.num_indices, 0, 0..1);
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
            console_log::init_with_level(log::Level::Info).expect("Could't initialize logger");
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
    state.world.generate_chunk(Vector3::new(0, 0, 0));
    state.rebuild_chunk(Vector3::new(0, 0, 0));
    state.world.generate_chunk(Vector3::new(0, 1, 0));
    state.rebuild_chunk(Vector3::new(0, 1, 0));
    state.world.generate_chunk(Vector3::new(0, 1, 1));
    state.rebuild_chunk(Vector3::new(0, 1, 1));
    state.world.generate_chunk(Vector3::new(1, 3, 1));
    state.rebuild_chunk(Vector3::new(1, 3, 1));
    state.unload_chunk(Vector3::new(0, 0, 0));

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