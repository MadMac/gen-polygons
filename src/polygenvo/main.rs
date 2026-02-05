use futures::executor::block_on;
use genevo::{operator::prelude::*, population::*, prelude::*, random::Rng, types::fmt::Display};
use rand::prelude::*;
use std::collections::VecDeque;
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

const INITIAL_VERTICES: i16 = 150;
const MAX_VERTICES: i16 = 450;
const VERTICES_INCREMENT: i16 = 50;
const POPULATION_SIZE: usize = 25;
const GENERATION_LIMIT: u64 = 500;
const PHASE_DURATION: u64 = 50;

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

struct Pictures {
    current_vertices: i16,
}

impl Pictures {
    fn new() -> Self {
        Pictures {
            current_vertices: INITIAL_VERTICES,
        }
    }
}

impl GenomeBuilder<Vertices> for Pictures {
    fn build_genome<R>(&self, _: usize, rng: &mut R) -> Vertices
    where
        R: Rng + Sized,
    {
        (0..self.current_vertices)
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

#[derive(Debug)]
struct FitnessCalc<'a> {
    goal_image: GoalImage,
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    texture_size: u32,
    texture_desc: wgpu::TextureDescriptor<'a>,
    render_pipeline: wgpu::RenderPipeline,
    texture: wgpu::Texture,
    texture_view: wgpu::TextureView,
    output_buffer: wgpu::Buffer,
    u32_size: u32,
    // GPU compute resources for accelerated fitness
    compute_pipeline: Option<wgpu::ComputePipeline>,
    compute_bind_group: Option<wgpu::BindGroup>,
    fitness_buffer: Option<wgpu::Buffer>,
    goal_texture: Option<wgpu::Texture>,
    goal_texture_view: Option<wgpu::TextureView>,
    use_gpu_fitness: bool,
}

impl<'a> Clone for FitnessCalc<'a> {
    fn clone(&self) -> Self {
        // Recreate the render pipeline since it doesn't implement Clone
        let shader = self.device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let render_pipeline_layout =
            self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[],
                push_constant_ranges: &[],
            });

        let render_pipeline = self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
                    format: self.texture_desc.format,
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
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: true,
            },
            multiview: None,
        });

        // Create new reusable resources for the clone
        let texture = self.device.create_texture(&self.texture_desc);
        let texture_view = texture.create_view(&Default::default());

        let u32_size = std::mem::size_of::<u32>() as u32;
        let output_buffer_size = (u32_size * self.texture_size * self.texture_size) as wgpu::BufferAddress;
        let output_buffer_desc = wgpu::BufferDescriptor {
            size: output_buffer_size,
            usage: wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::MAP_READ,
            label: None,
            mapped_at_creation: false,
        };
        let output_buffer = self.device.create_buffer(&output_buffer_desc);

        FitnessCalc {
            goal_image: self.goal_image.clone(),
            device: self.device,
            queue: self.queue,
            texture_size: self.texture_size,
            texture_desc: self.texture_desc.clone(),
            render_pipeline,
            texture,
            texture_view,
            output_buffer,
            u32_size,
            compute_pipeline: None,
            compute_bind_group: None,
            fitness_buffer: None,
            goal_texture: None,
            goal_texture_view: None,
            use_gpu_fitness: false,
        }
    }
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

impl<'a> FitnessCalc<'a> {
    pub fn new(
        goal_image: GoalImage,
        device: &'a wgpu::Device,
        queue: &'a wgpu::Queue,
    ) -> Self {
        let texture_size = goal_image.goal_image.width();
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

        // Create reusable resources
        let u32_size = std::mem::size_of::<u32>() as u32;
        let texture = device.create_texture(&texture_desc);
        let texture_view = texture.create_view(&Default::default());

        let bytes_per_row = (u32_size * texture_size + 255) & !255;
        let output_buffer_size = (bytes_per_row * texture_size) as wgpu::BufferAddress;
        let output_buffer_desc = wgpu::BufferDescriptor {
            size: output_buffer_size,
            usage: wgpu::BufferUsages::COPY_DST
                // this tells wpgu that we want to read this buffer from the cpu
                | wgpu::BufferUsages::MAP_READ,
            label: None,
            mapped_at_creation: false,
        };
        let output_buffer = device.create_buffer(&output_buffer_desc);

        // Initialize GPU compute resources
        let (compute_pipeline_opt, compute_bind_group_opt, fitness_buffer_opt, goal_texture_opt, goal_texture_view_opt) = 
            Self::init_gpu_compute(&device, &queue, &goal_image, texture_size);
        
        let use_gpu_fitness = compute_pipeline_opt.is_some();
        
        // Unwrap the options or use None values
        let compute_pipeline = compute_pipeline_opt;
        let compute_bind_group = compute_bind_group_opt;
        let fitness_buffer = fitness_buffer_opt;
        let goal_texture = goal_texture_opt;
        let goal_texture_view = goal_texture_view_opt;
        
        FitnessCalc {
            goal_image,
            device,
            queue,
            texture_size,
            texture_desc,
            render_pipeline,
            texture,
            texture_view,
            output_buffer,
            u32_size,
            compute_pipeline,
            compute_bind_group,
            fitness_buffer,
            goal_texture,
            goal_texture_view,
            use_gpu_fitness,
        }
    }
}

impl<'a> FitnessCalc<'a> {
    fn init_gpu_compute(
        device: &'a wgpu::Device,
        queue: &'a wgpu::Queue,
        goal_image: &GoalImage,
        texture_size: u32,
    ) -> (
        Option<wgpu::ComputePipeline>,
        Option<wgpu::BindGroup>,
        Option<wgpu::Buffer>,
        Option<wgpu::Texture>,
        Option<wgpu::TextureView>,
    ) {
        // Load compute shader
        let shader_source = include_str!("fitness.wgsl");
        
        let shader_module = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: Some("Fitness Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });
        
        // Create compute pipeline
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Fitness Compute Pipeline"),
            layout: None,
            module: &shader_module,
            entry_point: "main",
        });
        
        // Create goal texture
        let goal_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Goal Texture"),
            size: wgpu::Extent3d {
                width: texture_size,
                height: texture_size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        });
        
        let goal_texture_view = goal_texture.create_view(&Default::default());
        
        // Upload goal image data to GPU
        let goal_image_data = goal_image.goal_image.as_raw();
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &goal_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            goal_image_data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(std::num::NonZeroU32::new(texture_size * 4).unwrap()),
                rows_per_image: Some(std::num::NonZeroU32::new(texture_size).unwrap()),
            },
            wgpu::Extent3d {
                width: texture_size,
                height: texture_size,
                depth_or_array_layers: 1,
            },
        );
        
        // Create fitness buffer - using a struct instead of array for storage class
        let fitness_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fitness Buffer"),
            size: std::mem::size_of::<u32>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        // Create bind group
        let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Fitness Params"),
            contents: bytemuck::cast_slice(&[texture_size, texture_size, 2u32, 0u32]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Fitness Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &params_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&goal_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&goal_texture_view), // Use goal texture view as placeholder
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &fitness_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        });
        
        (
            Some(compute_pipeline),
            Some(bind_group),
            Some(fitness_buffer),
            Some(goal_texture),
            Some(goal_texture_view),
        )
    }
}

impl FitnessFunction<Vertices, usize> for FitnessCalc<'_> {
    fn fitness_of(&self, vertices: &Vertices) -> usize {
        let device = self.device;
        let queue = self.queue;
        let clear_color = wgpu::Color::BLACK;
        // Use reusable resources
        let texture_view = &self.texture_view;
        let output_buffer = &self.output_buffer;

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

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
            render_pass.draw(0..num_vertices, 0..1);
        }
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                aspect: wgpu::TextureAspect::All,
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            wgpu::ImageCopyBuffer {
                buffer: &output_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: NonZeroU32::new((self.u32_size * self.texture_size + 255) & !255),
                    rows_per_image: NonZeroU32::new(self.texture_size),
                },
            },
            self.texture_desc.size,
        );

        queue.submit(Some(encoder.finish()));

        let mut fitness_result: usize = 0;

        // Try to use GPU compute for fitness calculation if available
        if self.use_gpu_fitness {
            if let (Some(compute_pipeline), Some(compute_bind_group), Some(fitness_buffer), Some(goal_texture), Some(goal_texture_view)) = 
                (&self.compute_pipeline, &self.compute_bind_group, &self.fitness_buffer, &self.goal_texture, &self.goal_texture_view) {
                
                // Copy the rendered texture to the goal texture format for compute shader
                let mut compute_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                
                // Dispatch compute shader
                {
                    let mut compute_pass = compute_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Fitness Compute Pass"),
                    });
                    compute_pass.set_pipeline(compute_pipeline);
                    compute_pass.set_bind_group(0, compute_bind_group, &[]);
                    
                    // Dispatch with workgroup size matching the shader (8x8)
                    let workgroup_size_x = (self.texture_size + 7) / 8; // ceil(texture_size / 8)
                    let workgroup_size_y = (self.texture_size + 7) / 8;
                    compute_pass.dispatch(workgroup_size_x, workgroup_size_y, 1);
                }
                
                // Submit compute work
                queue.submit(Some(compute_encoder.finish()));
                
                // Read back the fitness result from GPU
                let fitness_buffer_slice = fitness_buffer.slice(..);
                let fitness_mapping = fitness_buffer_slice.map_async(wgpu::MapMode::Read);
                device.poll(wgpu::Maintain::Wait);
                block_on(fitness_mapping).unwrap();
                let fitness_data = fitness_buffer_slice.get_mapped_range();
                
                // Convert bytes to fitness value
                if fitness_data.len() >= 4 {
                    let fitness_bytes: [u8; 4] = [fitness_data[0], fitness_data[1], fitness_data[2], fitness_data[3]];
                    fitness_result = u32::from_le_bytes(fitness_bytes) as usize;
                }
                
                drop(fitness_data);
                
                // Apply bonuses for complexity
                if vertices.len() > 100 {
                    fitness_result += (vertices.len() as f32 * 10.0) as usize;
                } else {
                    fitness_result += (vertices.len() as f32 * 5.0) as usize;
                }
            }
        } else {
            // Fallback to CPU-based fitness calculation
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
                    ImageBuffer::<Rgba<u8>, _>::from_raw(self.texture_size, self.texture_size, data).unwrap();
                // Ultra-fast fitness calculation with aggressive sampling
                let mut sampled_pixels = 0;
                let goal_width = self.goal_image.goal_image.width();
                let goal_height = self.goal_image.goal_image.height();
                
                // Aggressive sampling - larger steps for much faster evaluation
                let sample_step = if vertices.len() < 100 { 4 } else { 3 };
                
                // Use a smaller, representative region for very fast evaluation
                let sample_end_x = (self.texture_size * 3/4) as u32;
                let sample_end_y = (self.texture_size * 3/4) as u32;
                
                for y in (0..sample_end_y).step_by(sample_step) {
                    for x in (0..sample_end_x).step_by(sample_step) {
                        // Ensure we don't access out of bounds pixels
                        let goal_x = x.min(goal_width - 1);
                        let goal_y = y.min(goal_height - 1);
                        let goal_pixel = self.goal_image.goal_image.get_pixel(goal_x, goal_y);
                        let current_pixel = buffer.get_pixel(x, y);
                        
                        // Simplified difference calculation - focus on RGB only
                        let diff = (
                            (goal_pixel[0] as i32 - current_pixel[0] as i32).abs() +
                            (goal_pixel[1] as i32 - current_pixel[1] as i32).abs() +
                            (goal_pixel[2] as i32 - current_pixel[2] as i32).abs()
                        ) as usize;
                        
                        fitness_result += diff;
                        sampled_pixels += 1;
                    }
                }
                
                // Fast normalization with linear scaling
                let max_possible_diff = sampled_pixels * (255 * 3); // RGB only
                let fitness_normalized = 1.0 - (fitness_result as f32 / max_possible_diff as f32);
                fitness_result = (fitness_normalized * 1000000.0) as usize;
                
                // Strongly reward solutions with more triangles to prevent premature simplification
                // This encourages the algorithm to maintain complexity
                if vertices.len() > 100 {
                    fitness_result += (vertices.len() as f32 * 10.0) as usize; // 10x bonus
                } else {
                    fitness_result += (vertices.len() as f32 * 5.0) as usize; // 5x bonus
                }

                // let mut rng = thread_rng();
                // let id: u32 = rng.gen_range(0..100);
                // buffer.save(String::from("triangles/image") + &id.to_string() + ".png").unwrap();
            }
            output_buffer.unmap();
        }
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
    let genome_builder = Pictures::new();
    let initial_population: Population<Vertices> = build_population()
        .with_genome_builder(genome_builder)
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
    let adjustment_size = 0.1;
    let fitness_calc = FitnessCalc::new(goal_image.clone(), &device, &queue);
    
    // Adaptive mutation tracking
    let mut fitness_history: VecDeque<usize> = VecDeque::with_capacity(10);
    let mut current_mutation_rate = 0.2;
    
    let mut picture_sim = simulate(
        genetic_algorithm()
            .with_evaluation(fitness_calc.clone())
            .with_selection(MaximizeSelector::new(0.6, 1))  // Less selective pressure to maintain diversity
            .with_crossover(UniformCrossBreeder::new())
            .with_mutation(BreederValueMutator::new(
                current_mutation_rate,  // Dynamic mutation rate
                Vertex {
                    position: [adjustment_size, adjustment_size, adjustment_size],
                    color: [
                        adjustment_size,
                        adjustment_size,
                        adjustment_size,
                        adjustment_size,
                    ],
                },
                3,  // More adjustment steps for finer mutations
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
                fitness_calc.clone(),
                false,
                0.7,
            ))
            .with_initial_population(initial_population)
            .build(),
    )
    .until(FitnessLimit::new(
        fitness_calc.highest_possible_fitness(),
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
                // Track fitness history for adaptive mutation
                fitness_history.push_back(best_solution.solution.fitness);
                
                // Adaptive mutation strategy
                if fitness_history.len() >= 2 {
                    let recent_improvement = best_solution.solution.fitness as i32 - 
                        fitness_history[fitness_history.len() - 2] as i32;
                    let improvement_rate = recent_improvement as f32 / 
                        fitness_history[fitness_history.len() - 2] as f32;
                    
                    // Adjust mutation rate based on improvement
                    if improvement_rate < 0.005 {
                        // Stagnating - increase mutation
                        current_mutation_rate = (current_mutation_rate * 1.2).min(0.4);
                    } else if improvement_rate > 0.02 {
                        // Improving well - decrease mutation
                        current_mutation_rate = (current_mutation_rate * 0.8).max(0.1);
                    }
                    // Else keep current rate
                }
                
                // Log adaptive mutation rate changes
                if fitness_history.len() >= 2 {
                    let recent_improvement = best_solution.solution.fitness as i32 - 
                        fitness_history[fitness_history.len() - 2] as i32;
                    let improvement_rate = recent_improvement as f32 / 
                        fitness_history[fitness_history.len() - 2] as f32;
                    
                    if improvement_rate < 0.005 || improvement_rate > 0.02 {
                        println!("Adaptive mutation rate adjusted to: {:.3}", current_mutation_rate);
                    }
                }
                
                // Only save images occasionally to improve performance
                if best_solution.solution.fitness > current_best_fitness && 
                   (step.iteration % 100 == 0 || step.iteration < 5) {
                    current_best_fitness = best_solution.solution.fitness;
                    let texture_size = goal_image.goal_image.width();
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
                    let bytes_per_row = (u32_size * texture_size + 255) & !255;

                    let output_buffer_size =
                        (bytes_per_row * texture_size) as wgpu::BufferAddress;
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
                                bytes_per_row: NonZeroU32::new((u32_size * texture_size + 255) & !255),
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
                
                // Early termination - convergence detection
                if step.iteration > 50 && fitness_history.len() >= 10 {
                    let max_fitness = *fitness_history.iter().max().unwrap_or(&0);
                    let min_fitness = *fitness_history.iter().min().unwrap_or(&0);
                    let fitness_range = max_fitness as f32 - min_fitness as f32;
                    
                    // If fitness range is very small, we've converged
                    if fitness_range < 5000.0 && current_mutation_rate > 0.3 {
                        println!("Convergence detected at generation {}. Final fitness: {}", 
                                step.iteration, best_solution.solution.fitness);
                        break;
                    }
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
