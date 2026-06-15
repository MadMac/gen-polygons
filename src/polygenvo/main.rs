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
mod persistence;
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

/// Parsed command-line options for `polygenvo`.
struct Cli {
    /// `--infinite`: drop the MAX_STEPS ceiling and run until Ctrl-C. A signal
    /// handler flips a shared flag the ES loop checks each step, so the run stops
    /// cleanly and still writes its final snapshot/summary (rather than the
    /// process being hard-killed mid-step).
    infinite: bool,
    /// `--show-window`: open a live window that renders the current best
    /// candidate as the run progresses. Closing it stops the run gracefully (the
    /// same final-snapshot/summary path as Ctrl-C).
    show_window: bool,
    /// `--gradient-polish`: every PolishCfg.every_k accepted improvements, run an
    /// on-device gradient polish of all triangle positions+colors and keep it
    /// only if the hard ΔE2000 renderer confirms it beats the parent.
    gradient_polish: bool,
    /// `--goal <path>`: image to approximate. Defaults to goal.png.
    goal: String,
    /// `--list-sessions`: list all saved sessions and exit.
    list_sessions: bool,
    /// `--load <id>`: load a saved session by ID and continue.
    load: Option<i64>,
    /// `--checkpoint-interval <n>`: auto-save every N accepted improvements.
    /// 0 means disabled (default).
    checkpoint_interval: Option<u64>,
    /// `--label <text>`: label for a new session (when saving).
    label: Option<String>,
}

impl Cli {
    fn parse() -> Cli {
        let flag = |name: &str| std::env::args().any(|a| a == name);
        Cli {
            infinite: flag("--infinite"),
            show_window: flag("--show-window"),
            gradient_polish: flag("--gradient-polish"),
            goal: arg_value("--goal").unwrap_or_else(|| "goal.png".to_string()),
            list_sessions: flag("--list-sessions"),
            load: arg_value("--load").and_then(|s| s.parse().ok()),
            checkpoint_interval: arg_value("--checkpoint-interval").and_then(|s| s.parse().ok()),
            label: arg_value("--label"),
        }
    }
}

fn main() {
    env_logger::init();

    let cli = Cli::parse();

    // Initialize database
    let mut db = persistence::init_default_db().expect("failed to initialize database");

    // Handle --list-sessions
    if cli.list_sessions {
        let sessions = persistence::list_sessions(&db).expect("failed to list sessions");
        if sessions.is_empty() {
            println!("No sessions found.");
            return;
        }
        for s in sessions {
            let label = s.label.as_deref().unwrap_or("<unnamed>");
            let fitness_pct = (s.current_fitness as f64 / 1_000_000.0) * 100.0;
            println!(
                "ID: {}, Label: {}, Fitness: {:.2}%, Triangles: {}, Phase: {}, Steps: {}, Updated: {}",
                s.id, label, fitness_pct, s.triangle_count, s.phase_index, s.steps_run, s.updated_at
            );
        }
        return;
    }

    // Load goal image
    let goal = goal::load_goal_image(&cli.goal);

    // Load session if requested
    let initial_state = cli.load.map(|id| {
        persistence::load_session(&mut db, id).expect("failed to load session")
    });

    // Get or create session ID for saving
    let session_id = if let Some(ref cp) = initial_state {
        // Loading existing session - use its ID
        cp.session_id
    } else if cli.checkpoint_interval.is_some() || cli.label.is_some() {
        // Creating new session - create entry now with label and goal info
        let mut checkpoint = persistence::Checkpoint::new(cli.label.clone());
        checkpoint.goal_width = goal.pixels.width();
        checkpoint.goal_height = goal.pixels.height();
        checkpoint.goal_pixels = goal.pixels.to_vec();
        Some(persistence::save_session(
            &mut db,
            &checkpoint
        ).expect("failed to create new session"))
    } else {
        // No persistence requested
        None
    };

    // In windowed mode the device must be compatible with the window surface, so
    // `window::init_window` brings up both the window and the device together;
    // the headless path keeps using `gpu::init_wgpu`. Either way the ES and the
    // viewer share one device/queue.
    let mut window_init = cli.show_window.then(|| window::init_window(goal.pixels.width()));
    let (device, queue) = match &window_init {
        Some(w) => (w.device.clone(), w.queue.clone()),
        None => block_on(gpu::init_wgpu()),
    };

    let mut cfg = es::EsConfig::production();
    cfg.polish.enabled = cli.gradient_polish;
    if cli.gradient_polish {
        println!("Gradient polish enabled (every {} accepted improvements).", cfg.polish.every_k);
    }
    cfg.checkpoint_interval = cli.checkpoint_interval;
    cfg.initial_state = initial_state.clone();
    
    if cli.infinite {
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
    
    // Get the label for the current session (if any)
    let session_label = initial_state.as_ref().and_then(|cp| cp.label.clone())
        .or(cli.label.clone());
    
    let result = es::run_es(device, queue, goal, cfg, observer, Some(&mut db), session_id, session_label);
    println!(
        "Done. Initial fitness: {}, final fitness: {}, steps: {}",
        result.initial_fitness, result.final_fitness, result.steps_run
    );
}
