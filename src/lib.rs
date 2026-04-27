mod models;
pub mod source;
mod model;

use anyhow::Context;
use clap::Parser;

use crate::models::Accelerator;

#[derive(Debug, Parser)]
/// Required denoising parameters.
pub struct Options {
    #[arg(short, long, value_delimiter = ',', default_values_t = models::get_default_accelerators())]
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
    #[arg(long, short, default_value = "10")]
    /// The number of frames to buffer and group into a batch before uploading
    /// and processing on the accelerator.
    ///
    /// Until a certain point, the higher the batch size the better your efficiency and performance,
    /// at the cost of increasing the memory usage of the accelerator, i.e. VRAM.
    pub batch_size: usize,
}

/// Run denoiser on the given input.
pub fn run<I>(opts: Options, input: I) -> anyhow::Result<()>
where
    I: source::InputSource,
{
    tracing::info!("initialising with accelerators: {:?}", opts.accelerators);

    run_pipeline(opts, input).context("run pipeline")?;

    Ok(())
}

fn run_pipeline<I>(_opts: Options, _input: I) -> anyhow::Result<()>
where
    I: source::InputSource,
{

    todo!()
}
