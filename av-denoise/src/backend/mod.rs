pub mod accelerate;
mod describe;
pub mod device;
pub mod enumerate;
mod probe;
pub mod sniff;

use av_denoise_core::tune::{DeviceIdentity, DeviceLimits, DeviceProfile};
use av_denoise_core::{Engine, Geometry, Nl4d, Nl4dOptions, Nlmeans, NlmeansAlgorithm};
use cubecl::Runtime;
use cubecl::prelude::ComputeClient;

use self::accelerate::Accelerator;
pub use self::device::Device;
use self::sniff::sniff_best_accelerator;
use crate::host::DenoiserError;
use crate::host::io::{ClientIo, PlaneIo};
use crate::tune::{KernelRequest, engine_kernels};

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
    pub device: DeviceProfile,
}

fn profile_of<R: Runtime>(client: &ComputeClient<R>, identity: DeviceIdentity) -> DeviceProfile {
    let limits = DeviceLimits::from_client(client);
    DeviceProfile { identity, limits }
}

/// Work to run on the client of whichever accelerator [run_on_best] picks.
trait OnClient {
    type Output;

    fn run<R: Runtime>(
        self,
        client: ComputeClient<R>,
        accelerator: Accelerator,
        identity: DeviceIdentity,
    ) -> Result<Self::Output, DenoiserError>;
}

/// Builds an engine for `spec` on the client.
struct BuildEngine<'a> {
    spec: EngineSpec,
    request: &'a KernelRequest<'a>,
}

impl OnClient for BuildEngine<'_> {
    type Output = BuiltEngine;

    fn run<R: Runtime>(
        self,
        client: ComputeClient<R>,
        accelerator: Accelerator,
        identity: DeviceIdentity,
    ) -> Result<BuiltEngine, DenoiserError> {
        let device = profile_of(&client, identity);
        let kernels = engine_kernels(self.request, &device);

        let engine: Box<dyn Engine> = match self.spec {
            EngineSpec::Nlmeans { algorithm, geometry } => {
                let engine = Nlmeans::with_kernels(&client, algorithm, geometry, kernels)?;
                Box::new(engine)
            },
            EngineSpec::Nl4d { options, geometry } => {
                let engine = Nl4d::with_kernels(&client, options, geometry, kernels)?;
                Box::new(engine)
            },
        };

        let io = ClientIo::new(client);

        Ok(BuiltEngine {
            engine,
            io: Box::new(io),
            accelerator,
            device,
        })
    }
}

/// Reads the client's device profile.
struct DescribeDevice;

impl OnClient for DescribeDevice {
    type Output = DeviceProfile;

    fn run<R: Runtime>(
        self,
        client: ComputeClient<R>,
        _accelerator: Accelerator,
        identity: DeviceIdentity,
    ) -> Result<DeviceProfile, DenoiserError> {
        let profile = profile_of(&client, identity);
        Ok(profile)
    }
}

/// Opens a client and identifies its device on the first accelerator in `accelerators` that works,
/// then runs `task` on it.
fn run_on_best<T: OnClient>(
    accelerators: &[Accelerator],
    device: &Device,
    task: T,
) -> Result<T::Output, DenoiserError> {
    let accelerator = sniff_best_accelerator(accelerators, device);
    let accelerator = accelerator.ok_or(DenoiserError::NoAcceleratorAvailable)?;

    match accelerator {
        #[cfg(feature = "cuda")]
        Accelerator::Cuda => {
            let cuda_device = device.to_cuda()?;
            let client = <cubecl::cuda::CudaRuntime as Runtime>::client(&cuda_device);
            let identity = describe::describe_cuda(cuda_device.index, &client);
            task.run(client, accelerator, identity)
        },
        #[cfg(feature = "rocm")]
        Accelerator::Rocm => {
            let amd_device = device.to_amd()?;
            let client = <cubecl::hip::HipRuntime as Runtime>::client(&amd_device);
            let identity = describe::describe_rocm(amd_device.index, &client);
            task.run(client, accelerator, identity)
        },
        #[cfg(feature = "vulkan")]
        Accelerator::Vulkan => {
            let wgpu_device = device.to_wgpu()?;
            let client = <cubecl::wgpu::WgpuRuntime as Runtime>::client(&wgpu_device);
            let backend = *client.info();
            let identity = describe::describe_wgpu(&wgpu_device, backend, &client);
            task.run(client, accelerator, identity)
        },
        #[cfg(feature = "metal")]
        Accelerator::Metal => {
            let wgpu_device = device.to_wgpu()?;
            let client = <cubecl::wgpu::WgpuRuntime as Runtime>::client(&wgpu_device);
            let backend = *client.info();
            let identity = describe::describe_wgpu(&wgpu_device, backend, &client);
            task.run(client, accelerator, identity)
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

/// Builds `spec` on the first accelerator in `accelerators` that works.
pub(crate) fn build_engine(
    accelerators: &[Accelerator],
    device: &Device,
    spec: EngineSpec,
    request: &KernelRequest<'_>,
) -> Result<BuiltEngine, DenoiserError> {
    let task = BuildEngine { spec, request };
    run_on_best(accelerators, device, task)
}

/// The profile of the device [build_engine] picks for `accelerators` and `device`.
pub fn selected_device(
    accelerators: &[Accelerator],
    device: &Device,
) -> Result<DeviceProfile, DenoiserError> {
    run_on_best(accelerators, device, DescribeDevice)
}
