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
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

fn main() {
    env_logger::init();

    // `--infinite`: drop the MAX_STEPS ceiling and run until Ctrl-C. A signal
    // handler flips a shared flag the ES loop checks each step, so the run stops
    // cleanly and still writes its final snapshot/summary (rather than the
    // process being hard-killed mid-step).
    let infinite = std::env::args().any(|a| a == "--infinite");

    let goal = goal::load_goal_image("goal.png");
    let (device, queue) = block_on(gpu::init_wgpu());

    let mut cfg = es::EsConfig::production();
    if infinite {
        let stop = Arc::new(AtomicBool::new(false));
        cfg.max_steps = u64::MAX;
        cfg.stop_flag = Some(stop.clone());
        ctrlc::set_handler(move || {
            eprintln!("\nCtrl-C received — finishing the current step, then stopping…");
            stop.store(true, Ordering::Relaxed);
        })
        .expect("failed to install Ctrl-C handler");
        println!("Running in --infinite mode; press Ctrl-C to stop.");
    }

    let result = es::run_es(device, queue, goal, cfg);
    println!(
        "Done. Initial fitness: {}, final fitness: {}, steps: {}",
        result.initial_fitness, result.final_fitness, result.steps_run
    );
}
