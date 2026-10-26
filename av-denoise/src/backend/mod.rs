pub mod accelerate;
mod describe;
pub mod device;
pub mod enumerate;
#[cfg(feature = "vulkan")]
mod helpers;
mod placement;
mod probe;
mod resolve;
pub mod sniff;
#[cfg(test)]
mod tests;

use av_denoise_core::tune::{DeviceIdentity, DeviceLimits, DeviceProfile};
use av_denoise_core::{Engine, Geometry, Nl4d, Nl4dOptions, Nlmeans, NlmeansAlgorithm};
use cubecl::prelude::Client;

use self::accelerate::Accelerator;
pub use self::device::Device;
pub(crate) use self::placement::DevicePlacement;
use self::resolve::resolve;
#[cfg(feature = "vulkan")]
use self::resolve::vulkan_device;
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

fn profile_of(client: &Client, identity: DeviceIdentity) -> DeviceProfile {
    let limits = DeviceLimits::from_client(client);
    DeviceProfile { identity, limits }
}

/// Work to run on the client of whichever accelerator [run_on_best] picks.
trait OnClient {
    type Output;

    fn run(
        self,
        client: Client,
        accelerator: Accelerator,
        identity: DeviceIdentity,
    ) -> Result<Self::Output, DenoiserError>;

    /// Runs before [OnClient::run] when the client is on Vulkan.
    #[cfg(feature = "vulkan")]
    fn on_vulkan(&self, _device: &cubecl::wgpu::WgpuDevice) {}
}

/// Builds an engine for `spec` on the client.
struct BuildEngine<'a> {
    spec: EngineSpec,
    request: &'a KernelRequest<'a>,
}

impl OnClient for BuildEngine<'_> {
    type Output = BuiltEngine;

    #[cfg(feature = "vulkan")]
    fn on_vulkan(&self, device: &cubecl::wgpu::WgpuDevice) {
        if let Some(ledger) = &self.request.setup.ledger {
            helpers::install_compile_helpers(ledger, device);
        }
    }

    fn run(
        self,
        client: Client,
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

    fn run(
        self,
        client: Client,
        _accelerator: Accelerator,
        identity: DeviceIdentity,
    ) -> Result<DeviceProfile, DenoiserError> {
        let profile = profile_of(&client, identity);
        Ok(profile)
    }
}

/// Opens a client and identifies its device on the first accelerator in `accelerators` that works,
/// then runs `task` on it.
///
/// The identity always describes `device`, so a client on an own device from `placement` loads the
/// same stored tuning results as the shared one.
fn run_on_best<T: OnClient>(
    accelerators: &[Accelerator],
    device: &Device,
    placement: &DevicePlacement,
    task: T,
) -> Result<T::Output, DenoiserError> {
    let Some(accelerator) = sniff_best_accelerator(accelerators, device) else {
        let error = unavailable_error(accelerators, device);
        return Err(error);
    };

    let requested = device.clone();
    let selected = resolve(accelerator, requested).map_err(anyhow::Error::from)?;

    let placed = placement.place(&selected);
    let client = placed.client();
    let identity = describe::describe(&selected, &client);

    #[cfg(feature = "vulkan")]
    if let Some(wgpu_device) = vulkan_device(&selected) {
        task.on_vulkan(wgpu_device);
    }

    task.run(client, accelerator, identity)
}

/// Why no accelerator in `accelerators` runs on `device`.
///
/// When every one of them rejects the device itself, the first rejection names the forms it takes.
fn unavailable_error(accelerators: &[Accelerator], device: &Device) -> DenoiserError {
    let mut rejection = None;
    for accelerator in accelerators {
        let requested = device.clone();
        match resolve(*accelerator, requested) {
            Ok(_) => return DenoiserError::NoAcceleratorAvailable,
            Err(error) => {
                rejection.get_or_insert(error);
            },
        }
    }

    match rejection {
        Some(error) => DenoiserError::Other(error.into()),
        None => DenoiserError::NoAcceleratorAvailable,
    }
}

/// Builds `spec` on the first accelerator in `accelerators` that works.
pub(crate) fn build_engine(
    accelerators: &[Accelerator],
    device: &Device,
    placement: &DevicePlacement,
    spec: EngineSpec,
    request: &KernelRequest<'_>,
) -> Result<BuiltEngine, DenoiserError> {
    let task = BuildEngine { spec, request };
    run_on_best(accelerators, device, placement, task)
}

/// The profile of the device [build_engine] picks for `accelerators` and `device`.
pub fn selected_device(
    accelerators: &[Accelerator],
    device: &Device,
) -> Result<DeviceProfile, DenoiserError> {
    let placement = DevicePlacement::shared();
    run_on_best(accelerators, device, &placement, DescribeDevice)
}
