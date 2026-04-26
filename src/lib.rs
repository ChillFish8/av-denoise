mod models;
mod sniff;
pub mod source;

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
}

/// Run denoiser on the given input.
pub fn run<I>(opts: Options, input: I) -> anyhow::Result<()>
where
    I: source::InputSource,
{
    tracing::info!("initialising with accelerators: {:?}", opts.accelerators);

    Ok(())
}

fn run_evaluation_pipeline<I>(opts: Options, input: I) -> anyhow::Result<()>
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
            dispatch_evaluation_pipeline::<cubecl::cude::CudaRuntime, I>(opts, input)
                .context("dispatch pipeline on CUDA runtime")
        },
        #[cfg(feature = "rocm")]
        Accelerator::Rocm => {
            dispatch_evaluation_pipeline::<cubecl::hip::HipRuntime, I>(opts, input)
                .context("dispatch pipeline on ROCM runtime")
        },
        #[cfg(feature = "vulkan")]
        Accelerator::Vulkan => {
            dispatch_evaluation_pipeline::<cubecl::wgpu::WgpuRuntime, I>(opts, input)
                .context("dispatch pipeline on VULKAN runtime")
        },
        #[cfg(feature = "metal")]
        Accelerator::Metal => {
            dispatch_evaluation_pipeline::<cubecl::wgpu::WgpuRuntime, I>(opts, input)
                .context("dispatch pipeline on METAL runtime")
        },
        #[cfg(feature = "cpu")]
        Accelerator::Cpu => {
            dispatch_evaluation_pipeline::<cubecl::cpu::CpuRuntime, I>(opts, input)
                .context("dispatch pipeline on CPU runtime")
        },
    }
}

fn dispatch_evaluation_pipeline<R, I>(opts: Options, input: I) -> anyhow::Result<()>
where
    R: cubecl::Runtime + 'static,
    I: source::InputSource + Send + 'static,
{
    let width = input.width();
    let height = input.height();
    let bit_depth = input.bit_depth();
    let device = <R::Device as Default>::default();
    let client = R::client(&device);

    todo!()
}
