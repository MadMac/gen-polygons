use genevo::{operator::prelude::*, population::*, prelude::*, random::Rng, types::fmt::Display};

const NUM_VERTICES: i16 = 34;
const POPULATION_SIZE: usize = 100;
const GENERATION_LIMIT: u64 = 10000;

#[derive(Clone, Debug, PartialEq, PartialOrd)]
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
		let mut result: f64 = 0.0;
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
		100
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
				value.position[0] + (range.position[0] as f64 * adjustment * sign as f64) as f64,
				value.position[1] + (range.position[1] as f64 * adjustment * sign as f64) as f64,
				0.0,
			],
			color: [
				value.color[0] + (range.color[0] * adjustment * sign as f64),
				value.color[1] + (range.color[1] * adjustment * sign as f64),
				value.color[2] + (range.color[2] * adjustment * sign as f64),
				value.color[3] + (range.color[3] * adjustment * sign as f64),
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

fn main() {
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

	loop {
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
                // println!("      {:?}", best_solution.solution.genome);
                //                println!("| population: [{}]", result.population.iter().map(|g| g.as_text())
                //                    .collect::<Vec<String>>().join("], ["));
            },
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
                println!("Best solution:     {:?}", best_solution.solution.genome);
                break;
            },
            Err(error) => {
                println!("{}", error);
                break;
            },
        }
    }
}
