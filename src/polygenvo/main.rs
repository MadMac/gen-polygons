use genevo::{operator::prelude::*, population::*, prelude::*, random::Rng, types::fmt::Display};
use rand::prelude::*;
use std::iter;

use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

use wgpu::util::DeviceExt;

const NUM_VERTICES: i16 = 33;
const POPULATION_SIZE: usize = 100;
const GENERATION_LIMIT: u64 = 10000;

struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    clear_color: wgpu::Color,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    num_vertices: u32,
    vertices: Vec<Vertex>,
    smaa_target: smaa::SmaaTarget
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable, PartialEq, PartialOrd)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 4],
}

type Vertices = Vec<Vertex>;

struct Pictures;

impl GenomeBuilder<Vertices> for Pictures {
    fn build_genome<R>(&self, _: usize, rng: &mut R) -> Vertices
    where
        R: Rng + Sized,
    {
        (0..NUM_VERTICES)
            .map(|_| Vertex {
                position: [rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0), 0.0],
                color: [rng.gen(), rng.gen(), rng.gen(), rng.gen()],
            })
            .collect()
    }
}

impl Vertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

#[derive(Clone, Debug)]
struct FitnessCalc;

impl FitnessFunction<Vertices, usize> for FitnessCalc {
    fn fitness_of(&self, vertices: &Vertices) -> usize {
        let mut result: f32 = 0.0;
        //println!("{:?}", self.vertices);
        for (_, ver_q) in vertices.iter().enumerate() {
            //println!("Vertex:  {:?}", ver_q);
            for i in 0..3 {
                result += ver_q.position[i];
            }

            for i in 0..4 {
                result += ver_q.color[i];
            }
        }

        result as usize
    }

    fn average(&self, values: &[usize]) -> usize {
        (values.iter().sum::<usize>() as f32 / values.len() as f32 + 0.5).floor() as usize
    }

    fn highest_possible_fitness(&self) -> usize {
        240
    }

    fn lowest_possible_fitness(&self) -> usize {
        0
    }
}

impl BreederValueMutation for Vertex {
    fn breeder_mutated(value: Self, range: &Vertex, adjustment: f64, sign: i8) -> Self {
        // println!("{} {}", adjustment, sign);
        Vertex {
            position: [
                value.position[0]
                    + (range.position[0] as f32 * adjustment as f32 * sign as f32) as f32,
                value.position[1]
                    + (range.position[1] as f32 * adjustment as f32 * sign as f32) as f32,
                0.0,
            ],
            color: [
                value.color[0] + (range.color[0] * adjustment as f32 * sign as f32),
                value.color[1] + (range.color[1] * adjustment as f32 * sign as f32),
                value.color[2] + (range.color[2] * adjustment as f32 * sign as f32),
                value.color[3] + (range.color[3] * adjustment as f32 * sign as f32),
            ],
        }
    }
}

impl RandomValueMutation for Vertex {
    fn random_mutated<R>(value: Self, min_value: &Vertex, max_value: &Self, rng: &mut R) -> Self
    where
        R: Rng + Sized,
    {
        Vertex {
            position: [rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0), 0.0],
            color: [rng.gen(), rng.gen(), rng.gen(), rng.gen()],
        }
    }
}

impl State {
    async fn new(window: &Window) -> Self {
        let size = window.inner_size();

        // The instance is a handle to our GPU
        // BackendBit::PRIMARY => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(wgpu::Backends::PRIMARY);
        let surface = unsafe { instance.create_surface(window) };
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
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::default(),
                },
                None, // Trace path
            )
            .await
            .unwrap();

        let swapchain_format = surface
            .get_preferred_format(&adapter)
            .unwrap_or(wgpu::TextureFormat::Bgra8UnormSrgb);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface.get_preferred_format(&adapter).unwrap(),
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
        };

        let smaa_target = smaa::SmaaTarget::new(
            &device,
            &queue,
            window.inner_size().width,
            window.inner_size().height,
            swapchain_format,
            smaa::SmaaMode::Smaa1X,
        );

        surface.configure(&device, &config);

        // let swapchain_format = adapter.get_swap_chain_preferred_format(&surface).unwrap();

        // let sc_desc = wgpu::SwapChainDescriptor {
        //     usage: wgpu::TextureUsage::RENDER_ATTACHMENT,
        //     format: swapchain_format,
        //     width: size.width,
        //     height: size.height,
        //     present_mode: wgpu::PresentMode::Fifo,
        // };
        // let swap_chain = device.create_swap_chain(&surface, &sc_desc);

        let clear_color = wgpu::Color::BLACK;

        let shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",        // 1.
                buffers: &[Vertex::desc()], // 2.
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent::OVER,
                        alpha: wgpu::BlendComponent::OVER,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                }],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList, // 1.
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw, // 2.
                cull_mode: Some(wgpu::Face::Back),
                // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None, // 1.
            multisample: wgpu::MultisampleState {
                count: 1,                        // 2.
                mask: !0,                        // 3.
                alpha_to_coverage_enabled: true, // 4.
            },
            multiview: None,
        });

        let num_vertices = 0;
        let vertices = Vec::with_capacity(0);

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        Self {
            surface,
            device,
            queue,
            clear_color,
            config,
            size,
            render_pipeline,
            vertex_buffer,
            num_vertices,
            vertices,
            smaa_target
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        self.size = new_size;
        self.config.width = new_size.width;
        self.config.height = new_size.height;
        self.surface.configure(&self.device, &self.config);
        self.smaa_target.resize(&self.device, new_size.width, new_size.height);
        // self.sc_desc.width = new_size.width;
        // self.sc_desc.height = new_size.height;
        // self.swap_chain = self.device.create_swap_chain(&self.surface, &self.sc_desc);
    }

    fn input(&mut self, _event: &WindowEvent) -> bool {
        false
    }

    fn update(&mut self) {}

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output_frame = self.surface.get_current_texture().unwrap();
        let output_view = output_frame.texture.create_view(&Default::default());
        let frame = self.smaa_target.start_frame(&self.device, &self.queue, &output_view);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[wgpu::RenderPassColorAttachment {
                    view: &*frame,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.clear_color),
                        store: true,
                    },
                }],
                depth_stencil_attachment: None,
            });
            render_pass.set_pipeline(&self.render_pipeline); // 2.
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.draw(0..self.num_vertices, 0..1);
        }

        self.queue.submit(iter::once(encoder.finish()));
        frame.resolve();
        output_frame.present();

        Ok(())
    }
}

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).unwrap();
    use futures::executor::block_on;

    // Since main can't be async, we're going to need to block
    let mut state = block_on(State::new(&window));
    let mut is_sim_running = true;
    let mut current_result: Vec<Vertex> = Vec::with_capacity(0);

    println!("Running genevoalgo");

    println!("Making initial population");
    let initial_population: Population<Vertices> = build_population()
        .with_genome_builder(Pictures)
        .of_size(POPULATION_SIZE)
        .uniform_at_random();
    println!("Initial population done");
    println!("{:?}", initial_population);

    let mut picture_sim = simulate(
        genetic_algorithm()
            .with_evaluation(FitnessCalc)
            .with_selection(MaximizeSelector::new(0.7, 2))
            .with_crossover(UniformCrossBreeder::new())
            .with_mutation(BreederValueMutator::new(
                0.5,
                Vertex {
                    position: [0.05, 0.05, 0.05],
                    color: [0.05, 0.05, 0.05, 0.0],
                },
                3,
                Vertex {
                    position: [-1.0, -1.0, -1.0],
                    color: [0.0, 0.0, 0.0, 0.0],
                },
                Vertex {
                    position: [1.0, 1.0, 1.0],
                    color: [1.0, 1.0, 1.0, 1.0],
                },
            ))
            .with_reinsertion(ElitistReinserter::new(FitnessCalc, false, 0.7))
            .with_initial_population(initial_population)
            .build(),
    )
    .until(or(
        FitnessLimit::new(FitnessCalc.highest_possible_fitness()),
        GenerationLimit::new(GENERATION_LIMIT),
    ))
    .build();

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == window.id() => {
                if !state.input(event) {
                    match event {
                        WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                        WindowEvent::KeyboardInput { input, .. } => match input {
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::Escape),
                                ..
                            } => *control_flow = ControlFlow::Exit,
                            _ => {}
                        },
                        WindowEvent::Resized(physical_size) => {
                            state.resize(*physical_size);
                        }
                        WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                            // new_inner_size is &mut so w have to dereference it twice
                            state.resize(**new_inner_size);
                        }
                        _ => {}
                    }
                }
            }
            Event::RedrawRequested(_) => {
                state.update();
                match state.render() {
                    Ok(_) => {}
                    // Recreate the swap_chain if lost
                    Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                    // The system is out of memory, we should probably quit
                    Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                    // All other errors (Outdated, Timeout) should be resolved by the next frame
                    Err(e) => eprintln!("{:?}", e),
                }
            }
            Event::MainEventsCleared => {
                window.request_redraw();
            }
            _ => {}
        }
        let amount_of_polygons = 100;
        state.vertices = Vec::with_capacity(0);

        if is_sim_running {
            let result = picture_sim.step();
            match result {
                Ok(SimResult::Intermediate(step)) => {
                    let evaluated_population = step.result.evaluated_population;
                    let best_solution = step.result.best_solution;
                    println!(
                        "Step: generation: {}, average_fitness: {}, \
                     best fitness: {}, duration: {}, processing_time: {}",
                        step.iteration,
                        evaluated_population.average_fitness(),
                        best_solution.solution.fitness,
                        step.duration.fmt(),
                        step.processing_time.fmt()
                    );
                    current_result = best_solution.solution.genome;
                    // println!("      {:?}", best_solution.solution.genome);
                    //                println!("| population: [{}]", result.population.iter().map(|g| g.as_text())
                    //                    .collect::<Vec<String>>().join("], ["));
                }
                Ok(SimResult::Final(step, processing_time, duration, stop_reason)) => {
                    let best_solution = step.result.best_solution;
                    println!("{}", stop_reason);
                    println!(
                        "Final result after {}: generation: {}, \
                     best solution with fitness {} found in generation {}, processing_time: {}",
                        duration.fmt(),
                        step.iteration,
                        best_solution.solution.fitness,
                        best_solution.generation,
                        processing_time.fmt()
                    );
                    is_sim_running = false;
                    println!("Best solution:     {:?}", best_solution.solution.genome);
                    current_result = best_solution.solution.genome;
                }
                Err(error) => {
                    println!("{}", error);
                    is_sim_running = false;
                }
            }
        }
        state.vertices = current_result.to_owned();
        // for _n in 0..amount_of_polygons * 3 {
        //     let vertex = Vertex {
        //         position: [rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0), 0.0],
        //         color: [rng.gen(), rng.gen(), rng.gen(), rng.gen()],
        //     };
        //     state.vertices.push(vertex);
        // }
        state.vertex_buffer = state
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(&state.vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
        state.num_vertices = state.vertices.len() as u32;
        //println!("{:?}", state.vertices);
    });
}
