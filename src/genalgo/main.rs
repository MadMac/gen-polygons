use oxigen::prelude::*;

#[derive(Clone)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 4],
}

#[derive(Clone)]
struct Picture {
	vertices: [Vertex; 20]
}

fn main() {
	let problem_size = 60; // Has to be divisible by 3
	let population_size = 2_i32.pow(problem_size as u32) as usize;
	let (solutions, generation, _progress, _population) = GeneticExecution::<bool, Picture>::new()
		.population_size()
		.run();
	println!("Running genalgo...");
}