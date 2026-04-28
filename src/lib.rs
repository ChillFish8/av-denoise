mod models;
pub mod source;
mod model;
mod sniff;

use anyhow::Context;
use clap::Parser;
use burn::prelude::*;
use burn::backend;

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

fn run_pipeline<I>(opts: Options, input: I) -> anyhow::Result<()>
where
    I: source::InputSource,
{
    let Some(best_accelerator) = sniff::sniff_best_accelerator(&opts.accelerators) else {
        anyhow::bail!(
            "no specified accelerator is able to run on the host hardware. \
        Please check you have any runtime dependencies installed like NVCC for CUDA."
        );
    };

    match best_accelerator {
        #[cfg(feature = "cuda")]
        Accelerator::Cuda => {
            dispatch_pipeline::<backend::cuda::Cuda, I>(opts, input)
                .context("dispatch pipeline on CUDA runtime")
        },
        #[cfg(feature = "rocm")]
        Accelerator::Rocm => dispatch_pipeline::<backend::rocm::Rocm, I>(opts, input)
            .context("dispatch pipeline on ROCM runtime"),
        #[cfg(feature = "vulkan")]
        Accelerator::Vulkan => {
            dispatch_pipeline::<backend::wgpu::Vulkan, I>(opts, input)
                .context("dispatch pipeline on VULKAN runtime")
        },
        #[cfg(feature = "metal")]
        Accelerator::Metal => {
            dispatch_pipeline::<backend::wgpu::Metal, I>(opts, input)
                .context("dispatch pipeline on METAL runtime")
        },
        #[cfg(feature = "cpu")]
        Accelerator::Cpu => dispatch_pipeline::<backend::cpu::Cpu, I>(opts, input)
            .context("dispatch pipeline on CPU runtime"),
    }
}

fn dispatch_pipeline<B, I>(opts: Options, input: I) -> anyhow::Result<()>
where
    B: Backend + 'static,
    I: source::InputSource + Send + 'static,
{
    let _width = input.width();
    let _height = input.height();
    let _bit_depth = input.bit_depth();


    Ok(())
}

