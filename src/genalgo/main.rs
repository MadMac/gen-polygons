use oxigen::prelude::*;
use rand::prelude::*;
use std::fs::File;
use std::fmt::Display;

#[derive(Clone, Debug, PartialEq)]
struct Vertex {
	position: [f64; 3],
	color: [f64; 4],
}

#[derive(Clone, Debug)]
struct Picture {
	//vertices: [Vertex; 20]
	vertices: Vec<Vertex>,
}

impl Display for Picture {
	fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
		write!(f, "{:?}", self.vertices)
	}
}

impl Genotype<Vertex> for Picture {
	type ProblemSize = usize;

	fn iter(&self) -> std::slice::Iter<Vertex> {
		self.vertices.iter()
	}
	fn into_iter(self) -> std::vec::IntoIter<Vertex> {
		self.vertices.into_iter()
	}
	fn from_iter<I: Iterator<Item = Vertex>>(&mut self, genes: I) {
		self.vertices = genes.collect();
	}

	// Generates the initial values
	fn generate(size: &Self::ProblemSize) -> Self {
		println!("Run generate!");
		let mut rng = thread_rng();

		let mut picture_vertices = Vec::with_capacity(0);
		for _i in 0..*size {
			let vertex = Vertex {
				position: [rng.gen_range(-1.0, 1.0), rng.gen_range(-1.0, 1.0), 0.0],
				color: [rng.gen(), rng.gen(), rng.gen(), rng.gen()],
			};
			picture_vertices.push(vertex);
		}
		println!("{:?}", picture_vertices);
		Picture {
			vertices: picture_vertices,
		}
	}

	fn fitness(&self) -> f64 {
		let mut result: f64 = 0.0;
		//println!("{:?}", self.vertices);
		for (_, ver_q) in self.vertices.iter().enumerate() {
			//println!("Vertex:  {:?}", ver_q);
			for i in 0..2 {
				result += ver_q.position[i];
			}
		}
		println!("Fitness: {}", result);
		result
	}

	fn mutate(&mut self, rgen: &mut SmallRng, index: usize) {
		
		for (i, ver_q) in self.iter().enumerate() {
			//println!("Vertex:  {:?}", ver_q);
			let rand_index = rgen.gen_range(0, 3);
			let rand_value = rgen.gen_range(-1, 1) as f64;
			let mut vertex = ver_q.to_owned();
			vertex.position[rand_index] = rand_value;
			self.vertices[i] = vertex;
			println!("MUTATE: {} {}", rand_index, rand_value);
		}
		
	}

	fn is_solution(&self, fitness: f64) -> bool {
		fitness as f64 == 2.15 as f64
	}
}

fn main() {
	println!("Running genalgo...");
	let progress_log = File::create("progress.csv").expect("Error creating progress log file");
    let population_log = File::create("population.txt").expect("Error creating population log file");

		
	let problem_size: u8 = 4;
	let log2 = (f64::from(problem_size) * 4_f64).log2().ceil();
	let population_size = 2_i32.pow(problem_size as u32) as usize;
	println!("POPULATION: {}", population_size);

	let (solutions, generation, _progress, _population) =
		GeneticExecution::<Vertex, Picture>::new()
			.population_size(population_size)
			.genotype_size(20)
			.mutation_rate(Box::new(MutationRates::Linear(SlopeParams {
                start: f64::from(problem_size) / (8_f64 + 2_f64 * log2) / 100_f64,
                bound: 0.005,
                coefficient: -0.0002,
            })))
            .selection_rate(Box::new(SelectionRates::Linear(SlopeParams {
                start: log2 - 2_f64,
                bound: log2 / 1.5,
                coefficient: -0.0005,
            })))
			.select_function(Box::new(SelectionFunctions::Cup))
        	.crossover_function(Box::new(CrossoverFunctions::UniformCross))
			.progress_log(10, progress_log)
            .population_log(2000, population_log)
			.run();
			
	println!("Finished in the generation {}", generation);
	for sol in &solutions {
		println!("{}", sol.fitness());
	}
}
