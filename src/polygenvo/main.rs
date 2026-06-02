//! polygenvo — approximate `goal.png` by evolving a population of coloured
//! triangles with a GPU-evaluated (1+λ)-ES. See the module docs for each layer:
//! `goal`/`genome` (representation), `fitness`/`gpu` (GPU evaluation),
//! `variation` (operators), `es` (search driver).

mod es;
mod fitness;
mod genome;
mod goal;
mod gpu;
mod variation;
#[cfg(test)]
mod test_support;

use futures::executor::block_on;

fn main() {
    env_logger::init();
    let goal = goal::load_goal_image("goal.png");
    let (device, queue) = block_on(gpu::init_wgpu());
    let result = es::run_es(device, queue, goal, es::EsConfig::production());
    println!(
        "Done. Initial fitness: {}, final fitness: {}, steps: {}",
        result.initial_fitness, result.final_fitness, result.steps_run
    );
}
