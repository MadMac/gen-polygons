use futures::executor::block_on;
use genevo::{operator::prelude::*, population::*, prelude::*, random::Rng, types::fmt::Display};
use rand::prelude::*;
use std::iter;
use std::num::NonZeroU32;
// use dssim::*;
use image::{ImageBuffer, Rgba};
use rgb::*;
use std::fmt;
use std::path::Path;
use std::time::Instant;

use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

use wgpu::util::DeviceExt;

const NUM_VERTICES: i16 = 450;
const POPULATION_SIZE: usize = 200;
const GENERATION_LIMIT: u64 = 10000;

#[derive(Debug)]
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
    output_buffer: wgpu::Buffer,
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
            .map(|_| {
                Vertex {
                position: [rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0), 0.0],
                color: [rng.gen(), rng.gen(), rng.gen(), rng.gen()],
            }
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

struct GoalImage {
    goal_image: image::ImageBuffer<image::Rgba<u8>, std::vec::Vec<u8>>,
}

impl Clone for GoalImage {
    fn clone(&self) -> GoalImage {
        GoalImage {
            goal_image: self.goal_image.clone(),
        }
    }
}

impl fmt::Debug for GoalImage {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Empty debug")
    }
}

#[derive(Clone, Debug)]
struct FitnessCalc<'a> {
    goal_image: GoalImage,
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
}

fn substract_rgba(first: &image::Rgba<u8>, second: &image::Rgba<u8>) -> usize {
    let mut diff = 0;
    diff += (first[0] as i32 - second[0] as i32).abs() as usize;
    diff += (first[1] as i32 - second[1] as i32).abs() as usize;
    diff += (first[2] as i32 - second[2] as i32).abs() as usize;
    diff += (first[3] as i32 - second[3] as i32).abs() as usize;
    diff
}

fn generate_sign() -> i8 {
    let mut rng = thread_rng();
    let sign: f32 = rng.gen();
    if sign >= 0.5 {
        1
    } else {
        -1
    }
}

impl FitnessFunction<Vertices, usize> for FitnessCalc<'_> {
    fn fitness_of(&self, vertices: &Vertices) -> usize {
        // let instance = wgpu::Instance::new(wgpu::Backends::VULKAN);
        // let elapsed = now.elapsed();
        // println!("Elapsed 0: {:.2?}", elapsed);
        // let adapter = block_on(instance
        //     .request_adapter(&wgpu::RequestAdapterOptions {
        //         power_preference: wgpu::PowerPreference::HighPerformance,
        //         compatible_surface: None,
        //         force_fallback_adapter: false,
        //     })).unwrap();
        // let (device, queue) = block_on(adapter
        //     .request_device(&Default::default(), None)).unwrap();
        let device = self.device;
        let queue = self.queue;

        let texture_size = 512u32;
        let texture_desc = wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width: texture_size,
                height: texture_size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::RENDER_ATTACHMENT,
            label: None,
        };
        let texture = device.create_texture(&texture_desc);
        let texture_view = texture.create_view(&Default::default());

        // we need to store this for later
        let u32_size = std::mem::size_of::<u32>() as u32;

        let output_buffer_size = (u32_size * texture_size * texture_size) as wgpu::BufferAddress;
        let output_buffer_desc = wgpu::BufferDescriptor {
            size: output_buffer_size,
            usage: wgpu::BufferUsages::COPY_DST
                // this tells wpgu that we want to read this buffer from the cpu
                | wgpu::BufferUsages::MAP_READ,
            label: None,
            mapped_at_creation: false,
        };
        let output_buffer = device.create_buffer(&output_buffer_desc);

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
        let clear_color = wgpu::Color::BLACK;

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[wgpu::ColorTargetState {
                    format: texture_desc.format,
                    blend: Some(wgpu::BlendState {
                        alpha: wgpu::BlendComponent::OVER,
                        color: wgpu::BlendComponent::OVER,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                }],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                polygon_mode: wgpu::PolygonMode::Fill,
                // Requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // Requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: true,
            },
            // If the pipeline will be used with a multiview render pass, this
            // indicates how many array layers the attachments will have.
            multiview: None,
        });

        let num_vertices = vertices.len() as u32;

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::MAP_READ
                | wgpu::BufferUsages::COPY_DST,
        });

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        {
            let render_pass_desc = wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[wgpu::RenderPassColorAttachment {
                    view: &texture_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear_color),
                        store: true,
                    },
                }],
                depth_stencil_attachment: None,
            };
            let mut render_pass = encoder.begin_render_pass(&render_pass_desc);

            render_pass.set_pipeline(&render_pipeline);
            render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
            render_pass.draw(0..num_vertices, 0..1);
        }
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                aspect: wgpu::TextureAspect::All,
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            wgpu::ImageCopyBuffer {
                buffer: &output_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: NonZeroU32::new(u32_size * texture_size),
                    rows_per_image: NonZeroU32::new(texture_size),
                },
            },
            texture_desc.size,
        );

        queue.submit(Some(encoder.finish()));

        let mut fitness_result: usize = 0;

        // We need to scope the mapping variables so that we can
        // unmap the buffer
        {
            let buffer_slice = output_buffer.slice(..);

            // NOTE: We have to create the mapping THEN device.poll() before await
            // the future. Otherwise the application will freeze.
            let mapping = buffer_slice.map_async(wgpu::MapMode::Read);
            device.poll(wgpu::Maintain::Wait);
            block_on(mapping).unwrap();
            let data = buffer_slice.get_mapped_range();

            let buffer =
                ImageBuffer::<Rgba<u8>, _>::from_raw(texture_size, texture_size, data).unwrap();
            for i in buffer.enumerate_pixels() {
                fitness_result +=
                    substract_rgba(self.goal_image.goal_image.get_pixel(i.0, i.1), i.2);
            }
            fitness_result =
                (10000000.0 - (fitness_result as f32 / 267386880.0) * 10000000.0) as usize;

            // let mut rng = thread_rng();
            // let id: u32 = rng.gen_range(0..100);
            // buffer.save(String::from("triangles/image") + &id.to_string() + ".png").unwrap();
        }
        output_buffer.unmap();
        // let elapsed = now.elapsed();
        // println!("Elapsed 3: {:.2?}", elapsed);
        // let mut result: f32 = 0.0;
        // //println!("{:?}", self.vertices);
        // for (_, ver_q) in vertices.iter().enumerate() {
        //     //println!("Vertex:  {:?}", ver_q);
        //     for i in 0..3 {
        //         result += ver_q.position[i];
        //     }

        //     for i in 0..4 {
        //         result += ver_q.color[i];
        //     }
        // }

        // let elapsed = now.elapsed();
        // println!("Elapsed 3: {:.2?}", elapsed);
        fitness_result
    }

    fn average(&self, values: &[usize]) -> usize {
        (values.iter().sum::<usize>() as f32 / values.len() as f32 + 0.5).floor() as usize
    }

    fn highest_possible_fitness(&self) -> usize {
        10000000
    }

    fn lowest_possible_fitness(&self) -> usize {
        0
    }
}

impl BreederValueMutation for Vertex {
    fn breeder_mutated(value: Self, range: &Vertex, adjustment: f64, sign: i8) -> Self {
        // println!("{}", adjustment);
        let mut rng = thread_rng();
        Vertex {
            position: [
                value.position[0]
                    + (range.position[0] as f32
                        * rng.gen_range(0.0..adjustment) as f32
                        * generate_sign() as f32) as f32,
                value.position[1]
                    + (range.position[1] as f32
                        * rng.gen_range(0.0..adjustment) as f32
                        * generate_sign() as f32) as f32,
                0.0,
            ],
            color: [
                value.color[0]
                    + (range.color[0]
                        * rng.gen_range(0.0..adjustment) as f32
                        * generate_sign() as f32),
                value.color[1]
                    + (range.color[1]
                        * rng.gen_range(0.0..adjustment) as f32
                        * generate_sign() as f32),
                value.color[2]
                    + (range.color[2]
                        * rng.gen_range(0.0..adjustment) as f32
                        * generate_sign() as f32),
                value.color[3]
                    + (range.color[3]
                        * rng.gen_range(0.0..adjustment) as f32
                        * generate_sign() as f32),
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

async fn save_buffer(
    vertex_buffer: &wgpu::Buffer,
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
) {
    let buffer_slice = vertex_buffer.slice(..);
    let mapping = buffer_slice.map_async(wgpu::MapMode::Read);
    device.poll(wgpu::Maintain::Wait);
    mapping.await.unwrap();
    let data = buffer_slice.get_mapped_range();
    println!("{:?}", data);
    let buffer = ImageBuffer::<Rgba<u8>, _>::from_raw(config.width, config.height, data).unwrap();
    buffer.save("image.png").unwrap();
}

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new();
    // let window = WindowBuilder::new().build(&event_loop).unwrap();
    // window.set_inner_size(winit::dpi::LogicalSize::new(512.0, 512.0));
    // window.set_resizable(false);

    // Since main can't be async, we're going to need to block
    // let mut state = block_on(State::new(&window));
    let mut is_sim_running = true;
    let mut current_result: Vec<Vertex> = Vec::with_capacity(0);

    println!("Running genevoalgo");

    println!("Making initial population");
    let initial_population: Population<Vertices> = build_population()
        .with_genome_builder(Pictures)
        .of_size(POPULATION_SIZE)
        .uniform_at_random();
    println!("Initial population done");
    // println!("{:?}", initial_population);
    let instance = wgpu::Instance::new(wgpu::Backends::GL);
    let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::default(),
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .unwrap();
    let (device, queue) = block_on(adapter.request_device(&Default::default(), None)).unwrap();

    // let dssim = Dssim::new();
    let goal_image_image = image::open("goal.png").unwrap().into_rgba8();
    let goal_image = GoalImage {
        goal_image: goal_image_image,
    };
    // let goal_image = image::open("goal.png").unwrap();

    // let img = image::open("goal.png").unwrap();
    // let img_asrgb = img.into_rgba8();
    // for i in img_asrgb.enumerate_pixels() {
    //     println!("Pixel: {:?}", i.0);
    // }
    let adjustment_size = 0.2;
    let mut picture_sim = simulate(
        genetic_algorithm()
            .with_evaluation(FitnessCalc {
                goal_image: goal_image.clone(),
                device: &device,
                queue: &queue,
            })
            .with_selection(MaximizeSelector::new(0.8, 1))
            .with_crossover(UniformCrossBreeder::new())
            .with_mutation(BreederValueMutator::new(
                0.1,
                Vertex {
                    position: [adjustment_size, adjustment_size, adjustment_size],
                    color: [
                        adjustment_size,
                        adjustment_size,
                        adjustment_size,
                        adjustment_size,
                    ],
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
            .with_reinsertion(ElitistReinserter::new(
                FitnessCalc {
                    goal_image: goal_image.clone(),
                    device: &device,
                    queue: &queue,
                },
                false,
                0.7,
            ))
            .with_initial_population(initial_population)
            .build(),
    )
    .until(FitnessLimit::new(
        FitnessCalc {
            goal_image: goal_image.clone(),
            device: &device,
            queue: &queue,
        }
        .highest_possible_fitness(),
    ))
    .build();

    // event_loop.run(move |event, _, control_flow| {
    //     match event {
    //         Event::WindowEvent {
    //             ref event,
    //             window_id,
    //         } if window_id == window.id() => {
    //             if !state.input(event) {
    //                 match event {
    //                     WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
    //                     WindowEvent::KeyboardInput { input, .. } => match input {
    //                         KeyboardInput {
    //                             state: ElementState::Pressed,
    //                             virtual_keycode: Some(VirtualKeyCode::Escape),
    //                             ..
    //                         } => *control_flow = ControlFlow::Exit,
    //                         _ => {}
    //                     },
    //                     WindowEvent::Resized(physical_size) => {
    //                         state.resize(*physical_size);
    //                     }
    //                     WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
    //                         // new_inner_size is &mut so w have to dereference it twice
    //                         state.resize(**new_inner_size);
    //                     }
    //                     _ => {}
    //                 }
    //             }
    //         }
    //         Event::RedrawRequested(_) => {
    //             state.update();
    //             match state.render() {
    //                 Ok(_) => {}
    //                 // Recreate the swap_chain if lost
    //                 Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
    //                 // The system is out of memory, we should probably quit
    //                 Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
    //                 // All other errors (Outdated, Timeout) should be resolved by the next frame
    //                 Err(e) => eprintln!("{:?}", e),
    //             }
    //         }
    //         Event::MainEventsCleared => {
    //             window.request_redraw();
    //         }
    //         _ => {}
    //     }
    let amount_of_polygons = 100;
    let mut current_best_fitness: usize = 0;
    // state.vertices = Vec::with_capacity(0);

    while is_sim_running {
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
                // let instance = wgpu::Instance::new(wgpu::Backends::VULKAN);
                // let adapter = block_on(instance
                //     .request_adapter(&wgpu::RequestAdapterOptions {
                //         power_preference: wgpu::PowerPreference::HighPerformance,
                //         compatible_surface: None,
                //         force_fallback_adapter: false,
                //     })).unwrap();
                // let (device, queue) = block_on(adapter
                //     .request_device(&Default::default(), None)).unwrap();
                if best_solution.solution.fitness > current_best_fitness {
                    current_best_fitness = best_solution.solution.fitness;
                    let texture_size = 512u32;
                    let texture_desc = wgpu::TextureDescriptor {
                        size: wgpu::Extent3d {
                            width: texture_size,
                            height: texture_size,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::Rgba8UnormSrgb,
                        usage: wgpu::TextureUsages::COPY_SRC
                            | wgpu::TextureUsages::RENDER_ATTACHMENT,
                        label: None,
                    };
                    let texture = device.create_texture(&texture_desc);
                    let texture_view = texture.create_view(&Default::default());

                    // we need to store this for later
                    let u32_size = std::mem::size_of::<u32>() as u32;

                    let output_buffer_size =
                        (u32_size * texture_size * texture_size) as wgpu::BufferAddress;
                    let output_buffer_desc = wgpu::BufferDescriptor {
                        size: output_buffer_size,
                        usage: wgpu::BufferUsages::COPY_DST
                // this tells wpgu that we want to read this buffer from the cpu
                | wgpu::BufferUsages::MAP_READ,
                        label: None,
                        mapped_at_creation: false,
                    };
                    let output_buffer = device.create_buffer(&output_buffer_desc);

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

                    let clear_color = wgpu::Color::BLACK;

                    let render_pipeline =
                        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                            label: Some("Render Pipeline"),
                            layout: Some(&render_pipeline_layout),
                            vertex: wgpu::VertexState {
                                module: &shader,
                                entry_point: "vs_main",
                                buffers: &[Vertex::desc()],
                            },
                            fragment: Some(wgpu::FragmentState {
                                module: &shader,
                                entry_point: "fs_main",
                                targets: &[wgpu::ColorTargetState {
                                    format: texture_desc.format,
                                    blend: Some(wgpu::BlendState {
                                        alpha: wgpu::BlendComponent::OVER,
                                        color: wgpu::BlendComponent::OVER,
                                    }),
                                    write_mask: wgpu::ColorWrites::ALL,
                                }],
                            }),
                            primitive: wgpu::PrimitiveState {
                                topology: wgpu::PrimitiveTopology::TriangleList,
                                strip_index_format: None,
                                front_face: wgpu::FrontFace::Ccw,
                                cull_mode: Some(wgpu::Face::Back),
                                // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                                polygon_mode: wgpu::PolygonMode::Fill,
                                // Requires Features::DEPTH_CLIP_CONTROL
                                unclipped_depth: false,
                                // Requires Features::CONSERVATIVE_RASTERIZATION
                                conservative: false,
                            },
                            depth_stencil: None,
                            multisample: wgpu::MultisampleState {
                                count: 1,
                                mask: !0,
                                alpha_to_coverage_enabled: true,
                            },
                            // If the pipeline will be used with a multiview render pass, this
                            // indicates how many array layers the attachments will have.
                            multiview: None,
                        });

                    let num_vertices = current_result.len() as u32;

                    let vertex_buffer =
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Vertex Buffer"),
                            contents: bytemuck::cast_slice(&current_result),
                            usage: wgpu::BufferUsages::VERTEX
                                | wgpu::BufferUsages::MAP_READ
                                | wgpu::BufferUsages::COPY_DST,
                        });

                    let mut encoder = device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

                    {
                        let render_pass_desc = wgpu::RenderPassDescriptor {
                            label: Some("Render Pass"),
                            color_attachments: &[wgpu::RenderPassColorAttachment {
                                view: &texture_view,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(clear_color),
                                    store: true,
                                },
                            }],
                            depth_stencil_attachment: None,
                        };
                        let mut render_pass = encoder.begin_render_pass(&render_pass_desc);

                        render_pass.set_pipeline(&render_pipeline);
                        render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                        render_pass.draw(0..num_vertices, 0..1);
                    }

                    encoder.copy_texture_to_buffer(
                        wgpu::ImageCopyTexture {
                            aspect: wgpu::TextureAspect::All,
                            texture: &texture,
                            mip_level: 0,
                            origin: wgpu::Origin3d::ZERO,
                        },
                        wgpu::ImageCopyBuffer {
                            buffer: &output_buffer,
                            layout: wgpu::ImageDataLayout {
                                offset: 0,
                                bytes_per_row: NonZeroU32::new(u32_size * texture_size),
                                rows_per_image: NonZeroU32::new(texture_size),
                            },
                        },
                        texture_desc.size,
                    );

                    queue.submit(Some(encoder.finish()));
                    // We need to scope the mapping variables so that we can
                    // unmap the buffer
                    {
                        let buffer_slice = output_buffer.slice(..);

                        // NOTE: We have to create the mapping THEN device.poll() before await
                        // the future. Otherwise the application will freeze.
                        let mapping = buffer_slice.map_async(wgpu::MapMode::Read);
                        device.poll(wgpu::Maintain::Wait);
                        block_on(mapping).unwrap();

                        let data = buffer_slice.get_mapped_range();

                        let buffer =
                            ImageBuffer::<Rgba<u8>, _>::from_raw(texture_size, texture_size, data)
                                .unwrap();
                        buffer
                            .save(
                                String::from("triangles/image")
                                    + &step.iteration.to_string()
                                    + ".png",
                            )
                            .unwrap();
                    }
                    output_buffer.unmap();
                }
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
                println!("Error: {}", error);
                is_sim_running = false;
            }
        }
    }
    // state.vertices = current_result.to_owned();
    // for _n in 0..amount_of_polygons * 3 {
    //     let vertex = Vertex {
    //         position: [rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0), 0.0],
    //         color: [rng.gen(), rng.gen(), rng.gen(), rng.gen()],
    //     };
    //     state.vertices.push(vertex);
    // }
    // state.vertex_buffer = state
    //     .device
    //     .create_buffer_init(&wgpu::util::BufferInitDescriptor {
    //         label: Some("Vertex Buffer"),
    //         contents: bytemuck::cast_slice(&state.vertices),
    //         usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::MAP_READ
    //     });

    //     let u32_size = std::mem::size_of::<u32>() as u32;

    // let output_buffer_size = (u32_size * state.config.width * state.config.height) as wgpu::BufferAddress;
    // let output_buffer_desc = wgpu::BufferDescriptor {
    //     size: output_buffer_size,
    //     usage: wgpu::BufferUsages::COPY_DST
    //         // this tells wpgu that we want to read this buffer from the cpu
    //         | wgpu::BufferUsages::MAP_READ,
    //     label: None,
    //     mapped_at_creation: false,
    // };
    // state.output_buffer = state.device.create_buffer(&output_buffer_desc);

    // state.num_vertices = state.vertices.len() as u32;
    //println!("{:?}", state.vertices);
    // });
}
