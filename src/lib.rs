mod batcher;
mod model;
mod models;
mod sniff;
pub mod source;

use std::path::{Path, PathBuf};
use std::time::Instant;
use anyhow::Context;
use burn::backend;
use burn::prelude::*;
use burn::tensor::TensorData;
use burn::tensor::ops::PadMode;
use clap::Parser;

use crate::batcher::TILE_SIZE;
use crate::models::Accelerator;
use crate::source::BitDepth;


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
    #[arg(long, short, default_value = "models/")]
    /// The directory path that holds the models.
    ///
    /// The system expects specific file names for each model:
    ///
    /// - `rt_ldr.bpk` -> Full-sized SDR denoising.
    /// - `rt_ldr_small.bpk` -> Lighter weight model for SDR denoising.
    ///
    /// HDR model files are not yet generated in `src/model/`, so RGB48 input currently
    /// returns a clear unsupported-model error.
    pub model_path: PathBuf,
    #[arg(long)]
    /// Use the small variants of the models rather than their full-sized counterparts.
    ///
    /// These models are faster and use less resources, and are typically "good enough".
    pub use_small: bool,
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

fn run_pipeline<I>(opts: Options, input: I) -> anyhow::Result<()>
where
    I: source::InputSource,
{
    let model_name = match (input.bit_depth(), opts.use_small) {
        (BitDepth::Eight, false) => "rt_ldr.bpk",
        (BitDepth::Eight, true) => "rt_ldr_small.bpk",
        (BitDepth::Ten, _) => anyhow::bail!(
            "RGB48 input is not supported yet because the HDR Burn models have not been generated"
        ),
    };

    let model_path = opts.model_path.join(model_name);
    if !model_path.exists() {
        anyhow::bail!(
            "model path {} does not exist, or cannot be read.",
            model_path.display()
        );
    }
        
    todo!()
}
