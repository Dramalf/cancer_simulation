use console::style;
use dialoguer::{Select, theme::ColorfulTheme};
use utils::effect_hill;
use std::{iter, time::Duration};
use wgpu::util::DeviceExt;
use winit::{
    event::*,
    event_loop::EventLoop,
    window::{Window, WindowBuilder},
};
mod log;
use log::{info_log, status_log};
mod status;
use status::status_calculate;
mod cell;
mod utils;
use cell::{CellParams, GridUniforms, init_cell_grid,random_init_cancer, init_wbc,progressive_multi_jittered_sampling};
mod config;
use config::{Config, load_config};
mod save_csv;
use save_csv::SimulationHistory;

struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    window: &'static Window, // Use 'static lifetime
    compute_pipeline: wgpu::ComputePipeline,
    render_pipeline: wgpu::RenderPipeline,
    cell_data_prev_buffer: wgpu::Buffer,
    cell_data_curr_buffer: wgpu::Buffer,
    wbc_data_prev_buffer: wgpu::Buffer,
    wbc_data_curr_buffer: wgpu::Buffer,
    ctt_data_prev_buffer: wgpu::Buffer,
    ctt_data_curr_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    cell_params: CellParams,
    cell_params_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    swaped_bind_group: wgpu::BindGroup,
    grid_width: u32,
    frame_count: u32,
    max_frames: u32,
    immune_level: u32,
    cell_data_prev: Vec<u32>,
    updated_by_immune: bool,
    paused: bool,
    waiting_for_ctt_input: bool,        // Whether waiting for CTT input
    ctt_positions: Vec<(u32, u32)>,     // Store CTT positions
    mouse_position: Option<(f64, f64)>, // Mouse position field
    simulation_config: Config,
    need_swap: bool,
    debug_mode: bool,
    simulation_history: SimulationHistory,
}

impl State {
    async fn new(window: &'static Window, simulation_config: Config) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
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
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
                trace: wgpu::Trace::Off, // Trace path
            })
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb()) // Usually choose SRGB format
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0], // Usually Fifo
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // --- Grid data ---
        let mut cell_data_prev: Vec<u32> =
            vec![0; (simulation_config.grid_width * simulation_config.grid_width) as usize];
        if simulation_config.init_strategy == "random" {
            random_init_cancer(&mut cell_data_prev, simulation_config.grid_width, simulation_config.init_cancer_rate);
        } else if simulation_config.init_strategy == "progressive" {
            progressive_multi_jittered_sampling(&mut cell_data_prev, simulation_config.grid_width, simulation_config.init_cancer_rate);
        } else if simulation_config.init_strategy == "grid" {
            init_cell_grid(
                &mut cell_data_prev,
                simulation_config.grid_width,
                simulation_config.init_cancer_rate,
                simulation_config.init_cancer_grid_width,
                simulation_config.init_cancer_grid_num,
            );
        }
        let buffer_desc = |label, size: u64| wgpu::BufferDescriptor {
            label: Some(label),
            size: size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        };
        let grid_size = (simulation_config.grid_width * simulation_config.grid_width) as u64;

        let cell_data_curr_buffer = device.create_buffer(&buffer_desc(
            "Cell Data Curr Buffer",
            grid_size * std::mem::size_of::<u32>() as u64,
        ));
        let wbc_data_prev_buffer = device.create_buffer(&buffer_desc(
            "Wbc Data Prev Buffer",
            grid_size * std::mem::size_of::<u32>() as u64,
        ));
        let wbc_data_curr_buffer = device.create_buffer(&buffer_desc(
            "Wbc Data Curr Buffer",
            grid_size * std::mem::size_of::<f32>() as u64,
        ));
        let ctt_data_prev_buffer = device.create_buffer(&buffer_desc(
            "Ctt Data Prev Buffer",
            grid_size * std::mem::size_of::<f32>() as u64,
        ));
        let ctt_data_curr_buffer = device.create_buffer(&buffer_desc(
            "Ctt Data Curr Buffer",
            grid_size * std::mem::size_of::<f32>() as u64,
        ));
        
        // Initialize CTT and WBC buffers to 0
        let zero_data: Vec<f32> = vec![0.0; grid_size as usize];
        queue.write_buffer(&ctt_data_prev_buffer, 0, bytemuck::cast_slice(&zero_data));
        queue.write_buffer(&ctt_data_curr_buffer, 0, bytemuck::cast_slice(&zero_data));
        queue.write_buffer(&wbc_data_prev_buffer, 0, bytemuck::cast_slice(&zero_data));
        queue.write_buffer(&wbc_data_curr_buffer, 0, bytemuck::cast_slice(&zero_data));

        let cell_data_prev_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cell Data Prev Buffer"),
            contents: bytemuck::cast_slice(&cell_data_prev),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        // --- Uniforms ---
        let uniforms = GridUniforms {
            output_resolution: [size.width, size.height],
            grid_resolution: [simulation_config.grid_width, simulation_config.grid_width],
        };
        let cell_params = CellParams {
            cancer_transformation_prob: simulation_config.cancer_transformation_prob,
            cell_regeneration_prob: simulation_config.cell_regeneration_prob,
            wbc_degeneration_prob: simulation_config.wbc_degeneration_prob,
            wbc_regeneration_prob: simulation_config.wbc_regeneration_prob,
            time_stamp: 0,
            regen_invincible_time: simulation_config.regen_invincible_time,
            ctt_effect: simulation_config.ctt_effect,
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Size Buffer"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let cell_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cell Params Buffer"),
            contents: bytemuck::bytes_of(&cell_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // --- Shaders ---
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        // --- Bind Group Layout and Bind Group ---
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT |wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility:wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // uniforms
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // uniforms
                    binding: 7,
                    visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: cell_data_prev_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: cell_data_curr_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wbc_data_prev_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wbc_data_curr_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: ctt_data_prev_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: ctt_data_curr_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: cell_params_buffer.as_entire_binding(),
                },
            ],
        });
        let swaped_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Swaped Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: cell_data_curr_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: cell_data_prev_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wbc_data_curr_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wbc_data_prev_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: ctt_data_curr_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: ctt_data_prev_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: cell_params_buffer.as_entire_binding(),
                },
            ],
        });
        // --- Render Pipeline ---
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Compute Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader_module,
                entry_point: Some("vs_main"),
                buffers: &[], // 我们直接在顶点着色器中生成顶点
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_module,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent::REPLACE,
                        alpha: wgpu::BlendComponent::REPLACE,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                            front_face: wgpu::FrontFace::Ccw,  // Counter-clockwise as front face
            cull_mode: Some(wgpu::Face::Back), // Cull back faces
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &shader_module,
            entry_point: Some("cancer_transformation"),
            cache: None,
            compilation_options: Default::default(),
        });
        Self {
            config,
            window,
            surface,
            device,
            queue,
            size,
            render_pipeline,
            compute_pipeline,
            cell_data_prev_buffer,
            cell_data_curr_buffer,
            wbc_data_prev_buffer,
            wbc_data_curr_buffer,
            ctt_data_prev_buffer,
            ctt_data_curr_buffer,
            cell_params,
            uniform_buffer,
            cell_params_buffer,
            bind_group,
            swaped_bind_group,
            grid_width: simulation_config.grid_width,
            frame_count: 0,
            max_frames: simulation_config.num_frames,
            immune_level: 1,
            cell_data_prev,
            updated_by_immune: false,
            paused: false,
            waiting_for_ctt_input: false,
            ctt_positions: Vec::new(),
            mouse_position: None,
            simulation_config,
            need_swap: false,
            debug_mode: false,
            simulation_history: SimulationHistory::new(),
        }
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);

            // Update output_resolution in uniform buffer
            let uniforms = GridUniforms {
                output_resolution: [new_size.width, new_size.height],
                grid_resolution: [self.grid_width, self.grid_width],
            };
            self.queue
                .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
        }
    }

    fn show_menu(&mut self) {
        let options = vec![
            "Cancer Target Therapy", 
            "Continue Simulation", 
            "Debug",
            "Save Data to CSV",
        ];

        let selection = Select::with_theme(&ColorfulTheme::default())
            .with_prompt("Please select an operation")
            .items(&options)
            .default(0)
            .interact()
            .unwrap();

        match selection {
            0 => {
                self.waiting_for_ctt_input = true;
                self.ctt_positions.clear();
                println!(
                    "{}",
                    style(
                        "Please click on the canvas to select the CTT position, press Enter to end"
                    )
                    .green()
                );
            }
            1 => {
                self.paused = false;
                self.window.request_redraw();
            }
            2 => {
                self.debug_mode = true;
            }
            3 => {
                // 保存数据到CSV
                let timestamp = self.cell_params.time_stamp;
                let filename = format!("data/simulation_data_{}.csv", timestamp);
                if let Err(e) = self.simulation_history.export_to_csv(&filename) {
                    println!("{}", style(format!("Error saving data: {}", e)).red());
                } else {
                    println!("{}", style(format!("Data saved to {}", filename)).green());
                }
                self.show_menu(); // 返回菜单
            }
            _ => unreachable!(),
        }
    }

    fn input(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::CursorMoved { position, .. } => {
                self.mouse_position = Some((position.x, position.y));
                false
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => {
                if self.debug_mode {
                    if let Some((x, y)) = self.mouse_position {
                        let window_size = self.window.inner_size();
                        let grid_x =
                            (x as f32 / window_size.width as f32 * self.grid_width as f32) as u32;
                        let grid_y =
                            (y as f32 / window_size.height as f32 * self.grid_width as f32) as u32;

                        // 创建命令编码器来从GPU读取数据
                        self.read_cell_data_at_position(grid_x, grid_y);
                        return true;
                    }
                    return true;
                }
                if self.waiting_for_ctt_input {
                    if let Some((x, y)) = self.mouse_position {
                        let window_size = self.window.inner_size();
                        let grid_x =
                            (x as f32 / window_size.width as f32 * self.grid_width as f32) as u32;
                        let grid_y =
                            (y as f32 / window_size.height as f32 * self.grid_width as f32) as u32;
                        self.ctt_positions.push((grid_x, grid_y));
                        println!(
                            "{}",
                            style(format!("CTT position: ({}, {})", grid_x, grid_y)).cyan()
                        );
                    }
                    true
                } else {
                    // Toggle pause state
                    self.paused = !self.paused;
                    if self.paused {
                        self.show_menu();
                    } else {
                        println!("{}", style("Simulation continues...").green());
                    }
                    true
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: winit::keyboard::Key::Named(winit::keyboard::NamedKey::Enter),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                if self.waiting_for_ctt_input {
                    self.waiting_for_ctt_input = false;
                    println!(
                        "{}",
                        style(format!("CTT position list: {:?}", self.ctt_positions)).yellow()
                    );
                    self.paused = false; // Continue simulation instead of returning to menu
                    self.window.request_redraw();
                    true
                } else if self.debug_mode {
                    self.debug_mode = false;
                    self.show_menu(); // Return to menu
                    true
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    fn update(&mut self) {
        self.cell_params.time_stamp += 1;
        self.queue.write_buffer(
            &self.cell_params_buffer,
            0,
            bytemuck::bytes_of(&self.cell_params),
        );

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cell simulation encoder"),
            });
        let staging_cell_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cell staging_buffer"),
            size: (self.grid_width * self.grid_width) as u64 * std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging_wbc_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("WBC staging_buffer"),
            size: (self.grid_width * self.grid_width) as u64 * std::mem::size_of::<f32>() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging_ctt_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("CTT staging_buffer"),
            size: (self.grid_width * self.grid_width) as u64 * std::mem::size_of::<f32>() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.compute_pipeline);
            let bind_group = if self.need_swap {
                &self.swaped_bind_group
            } else {
                &self.bind_group
            };
            cpass.set_bind_group(0, bind_group, &[]);
            cpass.dispatch_workgroups(self.grid_width * self.grid_width, 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &self.cell_data_prev_buffer,
            0,
            &staging_cell_buffer,
            0,
            self.cell_data_prev_buffer.size(),
        );
        encoder.copy_buffer_to_buffer(
            &self.wbc_data_curr_buffer,
            0,
            &staging_wbc_buffer,
            0,
            self.wbc_data_curr_buffer.size(),
        );
        encoder.copy_buffer_to_buffer(
            &self.ctt_data_curr_buffer,
            0,
            &staging_ctt_buffer,
            0,
            self.ctt_data_curr_buffer.size(),
        );

        self.queue.submit(Some(encoder.finish()));
        let buffer_slice = staging_cell_buffer.slice(..);
        let (sender, receiver) = flume::bounded(1);
        buffer_slice.map_async(wgpu::MapMode::Read, move |r| sender.send(r).unwrap());
        self.device.poll(wgpu::PollType::wait()).unwrap();
        let _ = receiver.recv().unwrap();
        let wbc_buffer_slice = staging_wbc_buffer.slice(..);
        let (sender, receiver) = flume::bounded(1);
        wbc_buffer_slice.map_async(wgpu::MapMode::Read, move |r| sender.send(r).unwrap());
        self.device.poll(wgpu::PollType::wait()).unwrap();
        let _ = receiver.recv().unwrap();
        let ctt_buffer_slice = staging_ctt_buffer.slice(..);
        let (sender, receiver) = flume::bounded(1);
        ctt_buffer_slice.map_async(wgpu::MapMode::Read, move |r| sender.send(r).unwrap());
        self.device.poll(wgpu::PollType::wait()).unwrap();
        let _ = receiver.recv().unwrap();
        {
            let view: wgpu::BufferView<'_> = buffer_slice.get_mapped_range();
            let latest_cell_data = bytemuck::cast_slice(&view).to_vec();
            let wbc_view: wgpu::BufferView<'_> = wbc_buffer_slice.get_mapped_range();
            let mut latest_wbc_data = bytemuck::cast_slice(&wbc_view).to_vec();
            let ctt_view: wgpu::BufferView<'_> = ctt_buffer_slice.get_mapped_range();
            let latest_ctt_data = bytemuck::cast_slice(&ctt_view).to_vec();
            let status = status_calculate(&latest_cell_data, &latest_wbc_data, &latest_ctt_data);
            
            // Add to history
            
            if self.cell_params.time_stamp % 1 == 0 {
                // Update frequency can be adjusted as needed
                self.simulation_history.add_data_point(self.cell_params.time_stamp, &status);

                status_log(&status, self.cell_params.time_stamp);
            }
            let cancer_percent = status.cancer_percent;
            if self.ctt_positions.len() > 0 {
                // Create CTT data and write to buffer
                let mut ctt_data: Vec<f32> = vec![0.0; (self.grid_width * self.grid_width) as usize];
                for (x, y) in self.ctt_positions.iter() {
                    let index = (y * self.grid_width + x) as usize;
                    if index < ctt_data.len() {
                        ctt_data[index] = self.cell_params.ctt_effect as f32;
                    }
                }
                self.queue.write_buffer(
                    &self.ctt_data_curr_buffer,
                    0,
                    bytemuck::cast_slice(&ctt_data),
                );
                // Clear position list to avoid duplicate additions
                self.ctt_positions.clear();
            }
            // Assume immune system crashes when cancer cells exceed 20%
            println!("cancer_percent: {}", cancer_percent);
            println!("immune_level: {}", self.immune_level);
            if false &&cancer_percent < 20.0
                && cancer_percent > self.immune_level as f32 * 5.0
                && self.cell_params.time_stamp % 10 == 0
            {
                let mut wbc_prob = (2.0 * cancer_percent / 100.0).min(0.2);
                if wbc_prob > status.wbc_percent {
                    wbc_prob -= status.wbc_percent;
                }
                init_wbc(&mut latest_wbc_data, self.grid_width, wbc_prob);
                self.queue.write_buffer(
                    &self.wbc_data_curr_buffer,
                    0,
                    bytemuck::cast_slice(&latest_wbc_data),
                );

                self.immune_level = (cancer_percent / 5.0) as u32 + 1;
                info_log(
                    &format!("WBC target rate: {}", wbc_prob),
                    self.cell_params.time_stamp,
                );
            }
        }
        std::mem::swap(
            &mut self.cell_data_prev_buffer,
            &mut self.cell_data_curr_buffer,
        );
        std::mem::swap(
            &mut self.wbc_data_prev_buffer,
            &mut self.wbc_data_curr_buffer,
        );
        std::mem::swap(
            &mut self.ctt_data_prev_buffer,
            &mut self.ctt_data_curr_buffer,
        );
        self.need_swap = !self.need_swap;
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        std::thread::sleep(Duration::from_millis(
            self.simulation_config.sleep_time as u64,
        ));

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
                            // 清屏颜色
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.bind_group, &[]);
            render_pass.draw(0..6, 0..1); // 画6个顶点 (两个三角形组成一个正方形)
        }

        self.queue.submit(iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    fn read_cell_data_at_position(&mut self, grid_x: u32, grid_y: u32) {
        // Calculate 1D index
        let index = (grid_y * self.grid_width + grid_x) as usize;

        let staging_cell_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Debug staging buffer"),
            size: (self.grid_width * self.grid_width) as u64 * std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let staging_wbc_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Debug WBC staging buffer"),
            size: (self.grid_width * self.grid_width) as u64 * std::mem::size_of::<f32>() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let staging_ctt_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Debug CTT staging buffer"),
            size: (self.grid_width * self.grid_width) as u64 * std::mem::size_of::<f32>() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create command encoder to read data from GPU
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Debug read encoder"),
            });


        encoder.copy_buffer_to_buffer(
            &self.cell_data_prev_buffer,
            0,
            &staging_cell_buffer,
            0,
            self.cell_data_prev_buffer.size(),
        );

        encoder.copy_buffer_to_buffer(
            &self.wbc_data_prev_buffer,
            0,
            &staging_wbc_buffer,
            0,
            self.wbc_data_prev_buffer.size(),
        );

        encoder.copy_buffer_to_buffer(
            &self.ctt_data_prev_buffer,
            0,
            &staging_ctt_buffer,
            0,
            self.ctt_data_prev_buffer.size(),
        );

        // Submit commands
        self.queue.submit(Some(encoder.finish()));

        // Read cell data
        let cell_buffer_slice = staging_cell_buffer.slice(..);
        let (sender, receiver) = flume::bounded(1);
        cell_buffer_slice.map_async(wgpu::MapMode::Read, move |r| sender.send(r).unwrap());
        self.device.poll(wgpu::PollType::wait()).unwrap();
        let _ = receiver.recv().unwrap();

        // Read WBC data
        let wbc_buffer_slice = staging_wbc_buffer.slice(..);
        let (sender, receiver) = flume::bounded(1);
        wbc_buffer_slice.map_async(wgpu::MapMode::Read, move |r| sender.send(r).unwrap());
        self.device.poll(wgpu::PollType::wait()).unwrap();
        let _ = receiver.recv().unwrap();

        // 读取CTT数据
        let ctt_buffer_slice = staging_ctt_buffer.slice(..);
        let (sender, receiver) = flume::bounded(1);
        ctt_buffer_slice.map_async(wgpu::MapMode::Read, move |r| sender.send(r).unwrap());
        self.device.poll(wgpu::PollType::wait()).unwrap();
        let _ = receiver.recv().unwrap();

        // 解析并显示数据
        {
            let cell_view = cell_buffer_slice.get_mapped_range();
            let cell_data: &[u32] = bytemuck::cast_slice(&cell_view);

            let wbc_view = wbc_buffer_slice.get_mapped_range();
            let wbc_data: &[f32] = bytemuck::cast_slice(&wbc_view);

            let ctt_view = ctt_buffer_slice.get_mapped_range();
            let ctt_data: &[f32] = bytemuck::cast_slice(&ctt_view);

            // 确保索引在有效范围内
            if index < cell_data.len() {
                // 解析单元格类型和状态
                let cell_value = cell_data[index];
                let cell_type = cell_value & 0xFF; // 假设类型存储在低8位
                let cell_state = (cell_value >> 8) & 0xFF; // 假设状态存储在接下来的8位

                let cell_type_name = match cell_type {
                    0 => "Normal Cell",
                    1 => "Cancer Cell",
                    2 => "Dead Cell",
                    3 => "Regenerated Cell",
                    _ => "Unknown Type",
                };

                println!("\n{}", style("--- DEBUG INFO ---").yellow().bold());
                println!("Position: ({}, {})", grid_x, grid_y);
                println!("Index: {}", index);
                println!("Cell Value: {}", cell_value);
                println!("Cell Type: {} ({})", cell_type, cell_type_name);
                println!("Cell State: {}", cell_state);
                if index < wbc_data.len() {
                    println!("WBC Value: {:.6}", wbc_data[index]);
                    println!("WBC effect: {}",effect_hill(wbc_data[index],0.5f32,1.0f32));

                }

                if index < ctt_data.len() {
                    println!("CTT Value: {:.6}", ctt_data[index]);
                }

                println!("{}", style("----------------").yellow().bold());
            } else {
                println!("Index {} out of range (size: {})", index, cell_data.len());
            }
        }

        // 解除映射，丢弃临时缓冲区
        drop(staging_cell_buffer);
        drop(staging_wbc_buffer);
        drop(staging_ctt_buffer);
    }
}

fn main() {
    let simulation_config = load_config();
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    let window = WindowBuilder::new()
        .with_title(format!(
            "WGPU Grid Renderer ({}*{})",
            simulation_config.grid_width, simulation_config.grid_width
        ))
        .with_inner_size(winit::dpi::LogicalSize::new(600.0, 600.0)) // 窗口尺寸，每个单元格6x6像素
        .build(&event_loop)
        .unwrap();

    // 将 window 放入 Box 中并泄漏，以获得 'static lifetime
    // 这是一种简化方法，对于更复杂的应用，你可能需要 Arc<Window> 或其他方式管理生命周期
    let static_window: &'static Window = Box::leak(Box::new(window));

    let mut state = pollster::block_on(State::new(static_window, simulation_config));

    event_loop
        .run(move |event, elwt| {
            match event {
                Event::WindowEvent {
                    ref event,
                    window_id,
                } if window_id == state.window().id() => {
                    if !state.input(event) {
                        match event {
                            WindowEvent::CloseRequested
                            | WindowEvent::KeyboardInput {
                                event:
                                    KeyEvent {
                                        logical_key:
                                            winit::keyboard::Key::Named(
                                                winit::keyboard::NamedKey::Escape,
                                            ),
                                        ..
                                    },
                                ..
                            } => elwt.exit(),
                            WindowEvent::Resized(physical_size) => {
                                state.resize(*physical_size);
                            }
                            WindowEvent::RedrawRequested => {
                                if !state.paused {
                                    state.update();
                                    match state.render() {
                                        Ok(_) => {}
                                        Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                                        Err(wgpu::SurfaceError::OutOfMemory) => elwt.exit(),
                                        Err(e) => eprintln!("渲染时发生错误: {:?}", e),
                                    }
                                    state.frame_count += 1;
                                    if state.max_frames != 0 && state.frame_count >= state.max_frames {
                                        elwt.exit();
                                    } else {
                                        state.window().request_redraw();
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
                Event::AboutToWait => {
                    if !state.paused && state.frame_count < state.max_frames {
                        state.window().request_redraw();
                    }
                }
                _ => {}
            }
        })
        .unwrap();
}
