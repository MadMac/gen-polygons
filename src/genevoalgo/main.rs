use genevo::{population::*, prelude::FitnessFunction, random::Rng};

const NUM_VERTICES: i16 = 20;
const POPULATION_SIZE: usize = 100;

#[derive(Clone, Debug, PartialEq, Copy)]
struct Vertex {
	position: [f64; 3],
	color: [f64; 4],
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

#[derive(Clone, Debug)]
struct FitnessCalc;

impl FitnessFunction<Vertices, usize> for FitnessCalc {
	fn fitness_of(&self, vertices: &Vertices) -> usize {
		10
	}

	fn average(&self, values: &[usize]) -> usize {
		(values.iter().sum::<usize>() as f32 / values.len() as f32 + 0.5).floor() as usize
	}

	fn highest_possible_fitness(&self) -> usize {
		100
	}

	fn lowest_possible_fitness(&self) -> usize {
		0
	}
}

fn main() {
	println!("Running genevoalgo");

	println!("Making initial population");
	let initial_population: Population<Vertices> = build_population()
		.with_genome_builder(Pictures)
		.of_size(POPULATION_SIZE)
		.uniform_at_random();
	println!("Initial population done");
	// println!("{:?}", initial_population);
}
