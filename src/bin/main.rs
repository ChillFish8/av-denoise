use clap::Parser;
use tracing_subscriber::EnvFilter;

mod cli;
mod file_mode;
mod ingest;
mod progress;
mod stream_mode;

use cli::{Args, Command, InputSource};
use ingest::CliOptions;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// Scene workers used when `--workers` is not given.
const DEFAULT_WORKERS: usize = 2;

/// Routes an input source to the pipeline that fits it.
///
/// A path is opened with ffms2 and split across scenes. Anything piped
/// streams y4m frame by frame with no scene detection.
fn run_input(opts: &CliOptions, input: &InputSource, workers: Option<usize>) -> Result<(), anyhow::Error> {
    match input {
        InputSource::File(path) => file_mode::run_file(opts, path, workers.unwrap_or(DEFAULT_WORKERS)),
        stream @ (InputSource::Stdin | InputSource::Fd(_)) => {
            if workers.is_some() {
                tracing::warn!("--workers is ignored for piped input, which cannot be split by scene");
            }

            tracing::info!(input = %stream, "reading a y4m stream");

            stream_mode::run_stream(opts, stream.open_reader()?)
        },
    }
}

fn main() -> anyhow::Result<()> {
    // cubecl spawns its per-device worker thread without asking for a
    // stack size, so it gets Rust's default 2 MiB. GPU kernel codegen
    // runs on that thread, and at a large --search-radius the (2R+1)^2
    // unrolled body of the windowed NLM kernels in
    // src/nlmeans/kernels/fused.rs overflows that stack.
    //
    // RUST_MIN_STACK is cached the first time it is read, so it has to
    // be set here, before any GPU thread spawns.
    if std::env::var_os("RUST_MIN_STACK").is_none() {
        // SAFETY: still single-threaded, no other thread can race the env mutation.
        unsafe { std::env::set_var("RUST_MIN_STACK", "16777216") };
    }

    let args = Args::parse();

    if std::env::var("RUST_LOG").is_err() {
        unsafe { std::env::set_var("RUST_LOG", "info") };
    }

    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_writer(progress::tracing_writer())
        .init();

    // Honour AV_DENOISE_COMPILATION_CACHE. This has to run before
    // Denoiser::create, because the first CubeCL client locks the global
    // config the moment it is built.
    match av_denoise::apply_compilation_cache_env() {
        Ok(Some(path)) => {
            tracing::info!(?path, "AV_DENOISE_COMPILATION_CACHE override active")
        },
        Ok(None) => {},
        Err(_) => anyhow::bail!("unable to apply AV_DENOISE_COMPILATION_CACHE, this is a bug."),
    }

    let (opts, input, workers) = match &args.command {
        Command::Nlmeans(nlm) => (nlm.build_options(&args)?, &nlm.input, nlm.workers),
        Command::Nl4d(nl4d) => (nl4d.build_options(&args)?, &nl4d.nlm.input, nl4d.nlm.workers),
    };

    run_input(&opts, input, workers)
}
