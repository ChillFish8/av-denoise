//! The `av-denoise` command line tool.

mod cli;
mod frame_index;
mod pipeline;
mod progress;
mod warm_start;
mod y4m_format;

use clap::Parser;
use tracing_subscriber::EnvFilter;

use self::cli::{Args, Command, InputSource, RunOptions, run_list_devices};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// Scene workers used when `--workers` is not given.
const DEFAULT_WORKERS: usize = 2;

/// Frame budget in bytes used when `--frame-budget` is not given, 1 GiB.
const DEFAULT_FRAME_BUDGET_BYTES: u64 = 1 << 30;

/// Runs the scene-parallel pipeline on an input, with defaults for unset worker and budget flags.
fn run_input(
    opts: &RunOptions,
    input: &InputSource,
    workers: Option<usize>,
    frame_budget: Option<u64>,
) -> Result<(), anyhow::Error> {
    let workers = workers.unwrap_or(DEFAULT_WORKERS);
    let frame_budget = frame_budget.unwrap_or(DEFAULT_FRAME_BUDGET_BYTES);

    tracing::info!(input = %input, "reading input");

    pipeline::run(opts, input, workers, frame_budget)
}

fn main() -> anyhow::Result<()> {
    // SAFETY: still single-threaded, no other thread can race the env mutation.
    unsafe { av_denoise::raise_codegen_stack_limit() };

    let args = Args::parse();

    if std::env::var("RUST_LOG").is_err() {
        // `list-devices` prints only a table, and the backends log at info level while they
        // start, so it runs quieter than a denoising run.
        let default_filter = match args.command {
            Command::ListDevices => "warn",
            _ => "info",
        };

        // SAFETY: still single-threaded, no other thread can race the env mutation.
        unsafe { std::env::set_var("RUST_LOG", default_filter) };
    }

    let env_filter = EnvFilter::from_default_env();
    let log_writer = progress::tracing_writer();
    tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .with_writer(log_writer)
        .init();

    // Listing devices compiles no kernels, so it skips the kernel cache.
    if matches!(args.command, Command::ListDevices) {
        let table = run_list_devices(&args.accelerators);
        print!("{table}");
        return Ok(());
    }

    // The first CubeCL client locks the global config, so the cache must be installed before
    // any denoiser is created.
    match av_denoise::install_compilation_cache() {
        Ok(Some(path)) => tracing::info!(?path, "caching compiled kernels"),
        Ok(None) => tracing::info!(
            "kernel caching is off, every run recompiles. Unset {} to turn it back on.",
            av_denoise::COMPILATION_CACHE_ENV,
        ),
        Err(error) => {
            let error = anyhow::Error::new(error).context("unable to install the kernel cache");
            return Err(error);
        },
    }

    let (opts, input, workers, frame_budget) = match &args.command {
        Command::Nlmeans(nlm) => (
            nlm.build_options(&args)?,
            &nlm.common.input,
            nlm.common.workers,
            nlm.common.frame_budget,
        ),
        Command::Nl4d(nl4d) => (
            nl4d.build_options(&args)?,
            &nl4d.common.input,
            nl4d.common.workers,
            nl4d.common.frame_budget,
        ),
        // Handled above, before any denoising options are built.
        Command::ListDevices => unreachable!(),
    };

    run_input(&opts, input, workers, frame_budget)
}
