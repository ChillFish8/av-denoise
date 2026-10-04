pub mod accelerate;
pub mod device;
pub mod enumerate;
mod probe;
pub mod sniff;

use av_denoise_core::{Engine, Geometry, Nl4d, Nl4dOptions, Nlmeans, NlmeansAlgorithm};
use cubecl::Runtime;
use cubecl::prelude::ComputeClient;

use self::accelerate::Accelerator;
pub use self::device::Device;
use self::sniff::sniff_best_accelerator;
use crate::host::DenoiserError;
use crate::host::io::{ClientIo, PlaneIo};

/// Which engine to build, and for what planes.
#[derive(Debug, Clone, Copy)]
pub enum EngineSpec {
    Nlmeans {
        algorithm: NlmeansAlgorithm,
        geometry: Geometry,
    },
    Nl4d {
        options: Nl4dOptions,
        geometry: Geometry,
    },
}

/// An engine and the plane I/O for the runtime it was built on.
pub(crate) struct BuiltEngine {
    pub engine: Box<dyn Engine>,
    pub io: Box<dyn PlaneIo>,
    pub accelerator: Accelerator,
}

fn build_on<R: Runtime>(
    client: ComputeClient<R>,
    spec: EngineSpec,
    accelerator: Accelerator,
) -> Result<BuiltEngine, DenoiserError> {
    let engine: Box<dyn Engine> = match spec {
        EngineSpec::Nlmeans { algorithm, geometry } => {
            let engine = Nlmeans::new(&client, algorithm, geometry)?;
            Box::new(engine)
        },
        EngineSpec::Nl4d { options, geometry } => {
            let engine = Nl4d::new(&client, options, geometry)?;
            Box::new(engine)
        },
    };

    let io = ClientIo::new(client);

    Ok(BuiltEngine {
        engine,
        io: Box::new(io),
        accelerator,
    })
}

/// Builds `spec` on the first accelerator in `accelerators` that works.
pub(crate) fn build_engine(
    accelerators: &[Accelerator],
    device: &Device,
    spec: EngineSpec,
) -> Result<BuiltEngine, DenoiserError> {
    let accelerator = sniff_best_accelerator(accelerators, device);
    let accelerator = accelerator.ok_or(DenoiserError::NoAcceleratorAvailable)?;

    match accelerator {
        #[cfg(feature = "cuda")]
        Accelerator::Cuda => {
            let cuda_device = device.to_cuda()?;
            let client = <cubecl::cuda::CudaRuntime as Runtime>::client(&cuda_device);
            build_on(client, spec, accelerator)
        },
        #[cfg(feature = "rocm")]
        Accelerator::Rocm => {
            let amd_device = device.to_amd()?;
            let client = <cubecl::hip::HipRuntime as Runtime>::client(&amd_device);
            build_on(client, spec, accelerator)
        },
        #[cfg(feature = "vulkan")]
        Accelerator::Vulkan => {
            let wgpu_device = device.to_wgpu()?;
            let client = <cubecl::wgpu::WgpuRuntime as Runtime>::client(&wgpu_device);
            build_on(client, spec, accelerator)
        },
        #[cfg(feature = "metal")]
        Accelerator::Metal => {
            let wgpu_device = device.to_wgpu()?;
            let client = <cubecl::wgpu::WgpuRuntime as Runtime>::client(&wgpu_device);
            build_on(client, spec, accelerator)
        },
        // Keeps the match exhaustive on docs.rs, where `cfg(docsrs)` widens the `Accelerator` enum to
        // include variants whose backend feature is not enabled. Never reached at runtime.
        #[cfg(docsrs)]
        #[expect(
            unreachable_patterns,
            reason = "the arm only keeps the match exhaustive on docs.rs"
        )]
        _ => unreachable!(),
    }
}
