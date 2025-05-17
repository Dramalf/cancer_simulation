use rand::Rng;
use std::{io::Write, iter, time::Duration};
use std::io::stdout;
use wgpu::util::DeviceExt;
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

use serde::Deserialize;
use std::fs;
use crossterm::{
    ExecutableCommand, // 用于执行各种终端命令
    cursor::{MoveTo, MoveDown, SavePosition, RestorePosition}, // 光标移动命令
    terminal::{Clear, ClearType}, // 清除屏幕或行命令
    style::{Print, SetForegroundColor, Color, ResetColor}, // 样式和打印命令
};
use dialoguer::{theme::ColorfulTheme, Select};
use console::{style, Term};

const STATUS_LINE: u16 = 0;

fn status_log(status: &str) -> Result<(), std::io::Error> {
    let mut stdout = stdout();

    // 1. 保存当前光标位置（可选，但更健壮）
    stdout.execute(SavePosition)?;

    // 2. 移动光标到状态行的开头 (crossterm MoveTo 是 0-based)
    stdout.execute(MoveTo(0, STATUS_LINE.saturating_sub(1)))?; // 防止 STATUS_LINE=0 导致下溢

    // 3. 清除当前行内容
    stdout.execute(Clear(ClearType::CurrentLine))?;

    // 4. 打印新的状态信息，可以加上颜色区分
    stdout.execute(SetForegroundColor(Color::Yellow))?; // 设置黄色
    stdout.execute(Print(format!("Status: {}", status)))?;
    stdout.execute(ResetColor)?; // 重置颜色

    // 5. 将光标移动到状态行下方一行（日志的起始行）的开头
    // 这样后续的 println! 就会从这里开始输出
    stdout.execute(MoveTo(0, STATUS_LINE))?;

    // 6. 恢复光标位置（如果之前保存了）
    stdout.execute(RestorePosition)?;

    // 7. 刷新输出，确保立即显示
    stdout.flush()?;

    Ok(())
}
fn info_log(log_message: &str) -> Result<(), std::io::Error> {
    let mut stdout = stdout();
    // 直接打印日志内容，加上回车换行确保在新行开始，并使用 \r\n 确保跨平台兼容
    stdout.execute(Print(format!("{}\r\n", log_message)))?;
    stdout.execute(SavePosition)?;

    // Flush 输出
    stdout.flush()?;
    Ok(())
}
#[derive(Deserialize, Debug)]
struct Config {
    grid_width: u32,
    num_frames: u32,
    init_cancer_rate: f32,
    init_cancer_grid_width: u32,
    init_cancer_grid_num: u32,
    cancer_transformation_prob: f32,
    sleep_time: u32,
}

fn load_config() -> Config {
    let config_data = fs::read_to_string("config.json").expect("Unable to read config file");
    serde_json::from_str(&config_data).expect("Unable to parse config file")
}

enum CellType {
    NormalCell = 0,
    CancerCell = 1,
    DeadCell = 2,
    RegeneratedCell = 3,

    WhiteBloodCell = 4,
    TargetedTherapy = 9,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    output_resolution: [u32; 2], // [width, height]
    grid_resolution: [u32; 2],   // [width, height]
    _padding: [u32; 2], // WGSL vec2<u32> 之后需要填充以满足vec4对齐（如果后面还有其他成员）
                        // 或者确保整个ubo大小是16字节的倍数，这里[u32;2] + [u32;2] = 16 bytes，是OK的
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CellParams {
    cancer_transformation_prob: f32,
    cell_regeneration_prob: f32,
    wbc_degeneration_prob: f32,
    wbc_regeneration_prob: f32,
    time_stamp: u32,
}

struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    window: &'static Window, // 使用 'static lifetime
    compute_pipeline: wgpu::ComputePipeline,
    render_pipeline: wgpu::RenderPipeline,
    grid_data_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    cell_params: CellParams,
    cell_params_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    grid_width: u32,
    frame_count: u32,
    max_frames: u32,
    immune_level: u32,
    grid_data: Vec<u32>,
    need_wbc: bool,
    paused: bool,
    waiting_for_CTT_input: bool,  // 是否等待靶向药输入
    CTT_positions: Vec<(u32, u32)>,  // 存储靶向药位置
    mouse_position: Option<(f64, f64)>,  // 添加鼠标位置字段
    simulation_config: Config,
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
            .find(|f| f.is_srgb()) // 通常选择 SRGB 格式
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0], // 通常是 Fifo
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // --- 网格数据 ---

        let mut grid_data: Vec<u32> =
            vec![0; (simulation_config.grid_width * simulation_config.grid_width) as usize];
        init_cell_grid(
            &mut grid_data,
            simulation_config.grid_width,
            simulation_config.init_cancer_rate,
            simulation_config.init_cancer_grid_width,
            simulation_config.init_cancer_grid_num,
        );

        // 将 u8 数据转换为 u32 数据，因为 WGSL 中 storage buffer array<u8> 的支持和对齐可能复杂

        let grid_data_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Grid Data Buffer"),
            contents: bytemuck::cast_slice(&grid_data),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC, // COPY_DST 如果后续要更新
        });

        // --- Uniforms ---
        let uniforms = Uniforms {
            output_resolution: [size.width, size.height],
            grid_resolution: [simulation_config.grid_width, simulation_config.grid_width],
            _padding: [0, 0], // 确保对齐
        };
        let cell_params = CellParams {
            cancer_transformation_prob: simulation_config.cancer_transformation_prob,
            cell_regeneration_prob: 0.01,
            wbc_degeneration_prob: 0.01,
            wbc_regeneration_prob: 0.001,
            time_stamp: 0,
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

        // --- 着色器 ---
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        // --- Bind Group Layout 和 Bind Group ---
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    // grid_data
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE, // 或 COMPUTE | FRAGMENT 如果也用于计算
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // uniforms
                    binding: 1,
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
                    binding: 2,
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
                    resource: grid_data_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
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
                front_face: wgpu::FrontFace::Ccw,  // 逆时针为正面
                cull_mode: Some(wgpu::Face::Back), // 剔除背面
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
            layout: Some(&pipeline_layout),
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
            grid_data_buffer,
            cell_params,
            uniform_buffer,
            cell_params_buffer,
            bind_group,
            grid_width: simulation_config.grid_width,
            frame_count: 0,
            max_frames: simulation_config.num_frames,
            immune_level: 1,
            grid_data,
            need_wbc: false,
            paused: false,
            waiting_for_CTT_input: false,
            CTT_positions: Vec::new(),
            mouse_position: None,
            simulation_config,
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

            // 更新 uniform buffer 中的 output_resolution
            let uniforms = Uniforms {
                output_resolution: [new_size.width, new_size.height],
                grid_resolution: [self.grid_width, self.grid_width],
                _padding: [0, 0],
            };
            self.queue
                .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
        }
    }

    fn show_menu(&mut self) {
        let term = Term::stdout();
        term.clear_screen().unwrap();
        
        let options = vec!["增加白细胞", "选择化疗位置", "继续模拟"];
        
        let selection = Select::with_theme(&ColorfulTheme::default())
            .with_prompt("请选择操作")
            .items(&options)
            .default(0)
            .interact()
            .unwrap();

        match selection {
            0 => {
                println!("{}", style("TODO: 增加白细胞").yellow());
                self.show_menu(); // 返回菜单
            }
            1 => {
                self.waiting_for_CTT_input = true;
                self.CTT_positions.clear();
                println!("{}", style("请点击画布选择靶向药位置，按回车结束").green());
            }
            2 => {
                self.paused = false;
                self.window.request_redraw();
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
                if self.waiting_for_CTT_input {
                    if let Some((x, y)) = self.mouse_position {
                        let window_size = self.window.inner_size();
                        let grid_x = (x as f32 / window_size.width as f32 * self.grid_width as f32) as u32;
                        let grid_y = (y as f32 / window_size.height as f32 * self.grid_width as f32) as u32;
                        self.CTT_positions.push((grid_x, grid_y));
                        println!("{}", style(format!("添加靶向药位置: ({}, {})", grid_x, grid_y)).cyan());
                    }
                    true
                } else {
                    // 切换暂停状态
                    self.paused = !self.paused;
                    if self.paused {
                        self.show_menu();
                    } else {
                        println!("{}", style("模拟继续...").green());
                    }
                    true
                }
            }
            WindowEvent::KeyboardInput {
                event: KeyEvent {
                    logical_key: winit::keyboard::Key::Named(winit::keyboard::NamedKey::Enter),
                    state: ElementState::Pressed,
                    ..
                },
                ..
            } => { 
                if self.waiting_for_CTT_input {
                    self.waiting_for_CTT_input = false;
                    println!("{}", style(format!("靶向药位置列表: {:?}", self.CTT_positions)).yellow());
                    self.show_menu(); // 返回菜单
                    true
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    fn update(&mut self) {
     

        // 后续可以在这里更新 grid_data_buffer 的内容
        // 例如，每帧或根据某些逻辑修改 self.grid_data_u32，然后:
        // self.queue.write_buffer(&self.grid_data_buffer, 0, bytemuck::cast_slice(&self.grid_data_u32_updated));

        self.cell_params.time_stamp += 1;
        self.queue.write_buffer(
            &self.cell_params_buffer,
            0,
            bytemuck::bytes_of(&self.cell_params),
        );
        if self.need_wbc {
            self.queue.write_buffer(
                &self.grid_data_buffer,
                0,
                bytemuck::cast_slice(&self.grid_data),
            );
            self.need_wbc = false;
        }
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("wave encoder"),
            });
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_buffer"),
            size: (self.grid_width * self.grid_width) as u64 * std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.compute_pipeline);
            cpass.set_bind_group(0, &self.bind_group, &[]);
            cpass.dispatch_workgroups(self.grid_width * self.grid_width, 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &self.grid_data_buffer,
            0,
            &staging_buffer,
            0,
            self.grid_data_buffer.size(),
        );

        self.queue.submit(Some(encoder.finish()));
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = flume::bounded(1);
        buffer_slice.map_async(wgpu::MapMode::Read, move |r| sender.send(r).unwrap());
        self.device.poll(wgpu::PollType::wait()).unwrap();
        let _ = receiver.recv().unwrap();
        {
            let view: wgpu::BufferView<'_> = buffer_slice.get_mapped_range();
            let current_time_data: &[u32] = bytemuck::cast_slice(&view);
            let mut grid_data: Vec<u32> = current_time_data.to_vec();
            let percents = status_print(current_time_data, self.cell_params.time_stamp);
            let cancer_percent = percents[0];
            let max_boost = 1.4;
            let adjusted_prob =
                0.1 * (1.0 + (max_boost - 1.0) * cancer_percent / (20.0 + cancer_percent));
            self.cell_params.cancer_transformation_prob = adjusted_prob;
            if self.CTT_positions.len() > 0 {
                for (x, y) in self.CTT_positions.iter() {
                    grid_data[(x + y * self.grid_width) as usize] = CellType::TargetedTherapy as u32;
                }
            }
            if cancer_percent > self.immune_level as f32 * 5.0 {
                init_wbc(
                    &mut grid_data,
                    self.grid_width,
                    (0.4 * self.immune_level as f32).min(0.8),
                );
                self.immune_level = (cancer_percent / 10.0) as u32 + 1;
                self.need_wbc = true;
                let _ = info_log(&format!("免疫系统启动 免疫等级: {}", self.immune_level));
            }
            self.grid_data = grid_data;
        }
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

}

fn main() {
    let mut stdout = stdout();

    // --- 在应用程序开始时清空终端 ---
    // 1. 清空整个屏幕
    stdout.execute(Clear(ClearType::All)).unwrap();
    // 2. 将光标移动到屏幕左上角 (0, 0)
    stdout.execute(MoveTo(0, 0)).unwrap();
    // 确保清空和移动命令被发送
    stdout.flush().unwrap();
    for _ in 0..11 {
        info_log(&format!("init")).unwrap();
    }

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

fn init_wbc(cell_grid: &mut Vec<u32>, grid_width: u32, init_wbc_rate: f32) {
    let mut rng = rand::rng();
    let total_wbc_cells = ((init_wbc_rate * (grid_width * grid_width) as f32) as u32)
        .min(count_value(cell_grid, CellType::NormalCell as u32) as u32);
    let mut wbc_cells_placed = 0;

    while wbc_cells_placed < total_wbc_cells {
        let x = rng.random_range(0..grid_width);
        let y = rng.random_range(0..grid_width);
        let index = (y * grid_width + x) as usize;
        if cell_grid[index] == 0 || cell_grid[index] == 3 {
            cell_grid[index] = 4;
            wbc_cells_placed += 1;
        }
    }
}

fn init_cell_grid(
    cell_grid: &mut Vec<u32>,
    grid_width: u32,
    init_cancer_rate: f32,
    init_cancer_grid_width: u32,
    init_cancer_grid_num: u32,
) {
    let mut rng = rand::rng();

    // Calculate total number of cancer cells needed
    let total_cancer_cells = (init_cancer_rate * (grid_width * grid_width) as f32) as u32;
    let mut cancer_cells_placed = 0;

    // Generate random center points for cancer clusters
    let mut center_points = Vec::new();
    for _ in 0..init_cancer_grid_num {
        let x =
            rng.random_range(init_cancer_grid_width / 2..grid_width - init_cancer_grid_width / 2);
        let y =
            rng.random_range(init_cancer_grid_width / 2..grid_width - init_cancer_grid_width / 2);
        center_points.push((x, y));
    }

    // Place cancer cells until we reach the target number
    while cancer_cells_placed < total_cancer_cells {
        // Randomly select one of the center points
        let (center_x, center_y) = center_points[rng.random_range(0..center_points.len())];

        // Generate random position within the square around the center point
        let x = center_x as i32
            + rng.random_range(
                -(init_cancer_grid_width as i32) / 2..=init_cancer_grid_width as i32 / 2,
            );
        let y = center_y as i32
            + rng.random_range(
                -(init_cancer_grid_width as i32) / 2..=init_cancer_grid_width as i32 / 2,
            );

        // Check if position is within grid bounds
        if x >= 0 && x < grid_width as i32 && y >= 0 && y < grid_width as i32 {
            let index = (y * grid_width as i32 + x) as usize;
            // Check if this position is not already a cancer cell
            if cell_grid[index] == 0 {
                cell_grid[index] = 1; // Mark as cancer cell
                cancer_cells_placed += 1;
            }
        }
    }
}

fn status_print(data: &[u32], time_stamp: u32) -> [f32; 2] {
    let total_cells = data.len();
    let cancer_cells = count_value(data, CellType::CancerCell as u32);
    let normal_cells = count_value(data, CellType::NormalCell as u32);
    let regenerated_cells = count_value(data, CellType::RegeneratedCell as u32);
    let wbc_cells = data
        .iter()
        .filter(|&&x| x >= CellType::WhiteBloodCell as u32)
        .count();
    let dead_cells = count_value(data, CellType::DeadCell as u32);

    let cancer_percent = cancer_cells as f32 / total_cells as f32 * 100.0;
    let normal_percent = normal_cells as f32 / total_cells as f32 * 100.0;
    let wbc_percent = wbc_cells as f32 / total_cells as f32 * 100.0;
    let dead_percent = dead_cells as f32 / total_cells as f32 * 100.0;
    let regenerated_percent = regenerated_cells as f32 / total_cells as f32 * 100.0;

    if time_stamp % 10 == 0 {  // 更新频率可以根据需要调整
        let _ = status_log(&format!("Frame: {}
--------------------------------
| Cell Type   │ Count  │ Percent   │
--------------------------------
| Cancer      │ {:<6} │ {:>6.2}%   │
| Normal      │ {:<6} │ {:>6.2}%   │
| WBC         │ {:<6} │ {:>6.2}%   │
| Regenerated │ {:<6} │ {:>6.2}%   │
| Dead        │ {:<6} │ {:>6.2}%   │
--------------------------------
        ", time_stamp, cancer_cells, cancer_percent, normal_cells, normal_percent, wbc_cells, wbc_percent, regenerated_cells, regenerated_percent, dead_cells, dead_percent));
        // 移动到顶部
        // print!("\x1B[H");
        // // 清除状态区域（11行）
        // for _ in 0..11 {
        //     print!("\x1B[K\n");
        // }
        // // 再次移动到顶部
        // print!("\x1B[H");
        
        // // 打印状态信息
        // println!("Frame: {}", time_stamp);
        // println!("┌─────────────┬────────┬───────────┐");
        // println!("│ Cell Type   │ Count  │ Percent   │");
        // println!("├─────────────┼────────┼───────────┤");
        // println!("│ Total       │ {:<6} │           │", total_cells);
        // println!(
        //     "│ Cancer      │ {:<6} │ {:>6.2}%   │",
        //     cancer_cells, cancer_percent
        // );
        // println!(
        //     "│ Dead        │ {:<6} │ {:>6.2}%   │",
        //     dead_cells, dead_percent
        // );
        // println!(
        //     "│ Normal      │ {:<6} │ {:>6.2}%   │",
        //     normal_cells, normal_percent
        // );
        // println!(
        //     "│ WBC         │ {:<6} │ {:>6.2}%   │",
        //     wbc_cells, wbc_percent
        // );
        // println!(
        //     "│ Regenerated │ {:<6} │ {:>6.2}%   │",
        //     regenerated_cells, regenerated_percent
        // );
        // println!("└─────────────┴────────┴───────────┘");
        // println!("\x1B[u");
        // 确保输出立即显示
    }
    
    [cancer_percent, normal_percent]
}

fn count_value(data: &[u32], value: u32) -> usize {
    data.iter().filter(|&&x| x == value).count()
}
