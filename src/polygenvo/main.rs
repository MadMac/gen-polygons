//! polygenvo — approximate `goal.png` by evolving a population of coloured
//! triangles with a GPU-evaluated (1+λ)-ES. See the module docs for each layer:
//! `goal`/`genome` (representation), `fitness`/`gpu` (GPU evaluation),
//! `variation` (operators), `es` (search driver).

mod es;
mod fitness;
mod genome;
mod gradient;
mod goal;
mod gpu;
mod variation;
mod window;
#[cfg(test)]
mod test_support;
#[cfg(test)]
mod softras_ref;

use futures::executor::block_on;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Value of a `--flag <value>` (or `--flag=value`) command-line argument, if present.
fn arg_value(flag: &str) -> Option<String> {
    let mut args = std::env::args();
    while let Some(a) = args.next() {
        if a == flag {
            return args.next();
        }
        if let Some(v) = a.strip_prefix(&format!("{flag}=")) {
            return Some(v.to_string());
        }
    }
    None
}

fn main() {
    env_logger::init();

    // `--infinite`: drop the MAX_STEPS ceiling and run until Ctrl-C. A signal
    // handler flips a shared flag the ES loop checks each step, so the run stops
    // cleanly and still writes its final snapshot/summary (rather than the
    // process being hard-killed mid-step).
    let infinite = std::env::args().any(|a| a == "--infinite");

    // `--show-window`: open a live window that renders the current best
    // candidate as the run progresses. Closing the window stops the run
    // gracefully (the same final-snapshot/summary path as Ctrl-C).
    let show_window = std::env::args().any(|a| a == "--show-window");

    // `--goal <path>`: image to approximate. Defaults to goal.png.
    let goal_path = arg_value("--goal").unwrap_or_else(|| "goal.png".to_string());
    let goal = goal::load_goal_image(&goal_path);

    // In windowed mode the device must be compatible with the window surface, so
    // `window::init_window` brings up both the window and the device together;
    // the headless path keeps using `gpu::init_wgpu`. Either way the ES and the
    // viewer share one device/queue.
    let mut window_init = show_window.then(|| window::init_window(goal.pixels.width()));
    let (device, queue) = match &window_init {
        Some(w) => (w.device.clone(), w.queue.clone()),
        None => block_on(gpu::init_wgpu()),
    };

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

    let observer = window_init
        .as_mut()
        .map(|w| &mut w.observer as &mut dyn es::StepObserver);
    let result = es::run_es(device, queue, goal, cfg, observer);
    println!(
        "Done. Initial fitness: {}, final fitness: {}, steps: {}",
        result.initial_fitness, result.final_fitness, result.steps_run
    );
}
