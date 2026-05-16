use av_denoise::accelerate::{Accelerator, get_default_accelerators};
use clap::Parser;
use strum_macros::EnumString;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(Debug, Copy, Clone, Default, EnumString)]
/// The denoising algorithm to use.
///
/// Currently, only nlmeans is supported.
pub enum Algorithm {
    #[default]
    /// The Non-Local Means algorithm.
    Nlmeans,
}

#[derive(Debug, Parser)]
/// Fast and efficient video denoising
struct Args {
    #[arg(short, long, default_value = "nlmeans")]
    /// The denoising algorithm to apply to the frames.
    ///
    /// Currently, only "nlmeans" is available.
    pub algorithm: Algorithm,
    #[arg(short, long, value_delimiter = ',', default_values_t = get_default_accelerators())]
    /// The hardware accelerators to perform the computation
    ///
    /// Accelerators should be ordered from the highest priority to the lowest priority,
    /// the system will attempt to use each accelerator sequentially one after the other
    /// until it finds an accelerator that works for the host hardware.
    ///
    /// If no accelerator can be found, the application will error.
    ///
    /// By default, this will be all accelerators the binary was compiled with.
    pub accelerators: Vec<Accelerator>,
    #[arg(short, long)]
    /// Run the compute operations on a specific device.
    ///
    /// If this is left unset, the system will select the default device.
    pub device: Option<String>,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    if std::env::var("RUST_LOG").is_err() {
        unsafe { std::env::set_var("RUST_LOG", "info") };
    }

    tracing_subscriber::fmt::init();

    todo!("implement connection to lib");

    Ok(())
}
