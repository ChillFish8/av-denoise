use clap::Parser;
use tracing_subscriber::EnvFilter;

mod cli;
mod frame_index;
mod pipeline;
mod progress;
mod warm_start;
mod y4m_format;

use cli::{Args, Command, InputSource, RunOptions, run_list_devices};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// Scene workers used when `--workers` is not given.
const DEFAULT_WORKERS: usize = 2;

/// Frame budget in bytes used when `--frame-budget` is not given. (1GB)
const DEFAULT_FRAME_BUDGET_BYTES: u64 = 1 << 30;

/// Routes an input to the scene-parallel pipeline.
///
/// A path opens with ffms2. A pipe reads a y4m stream.
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
        // `list-devices` prints a table and nothing else, and the
        // backends chatter at info level while they start up, so it
        // starts quieter than a denoising run.
        let default = match args.command {
            Command::ListDevices => "warn",
            _ => "info",
        };
        unsafe { std::env::set_var("RUST_LOG", default) };
    }

    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_writer(progress::tracing_writer())
        .init();

    // Listing devices compiles no kernels, so it runs before the cache
    // is installed and skips it entirely.
    if matches!(args.command, Command::ListDevices) {
        print!("{}", run_list_devices(&args.accelerators));
        return Ok(());
    }

    // Point CubeCL at a kernel cache. This has to run before
    // HostDenoiser::create, because the first CubeCL client locks the global
    // config the moment it is built.
    match av_denoise::install_compilation_cache() {
        Ok(Some(path)) => tracing::info!(?path, "caching compiled kernels"),
        Ok(None) => tracing::info!(
            "kernel caching is off, every run recompiles. Unset {} to turn it back on.",
            av_denoise::COMPILATION_CACHE_ENV,
        ),
        Err(err) => return Err(anyhow::Error::new(err).context("unable to install the kernel cache")),
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
