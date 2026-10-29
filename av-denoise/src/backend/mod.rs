pub mod accelerate;
#[cfg(feature = "cuda")]
mod cuda_check;
pub mod device;
pub mod enumerate;
mod placement;
mod probe;
mod resolve;
pub mod sniff;
#[cfg(test)]
mod tests;

#[cfg(feature = "vulkan")]
use av_denoise_core::tune::install_compile_helpers;
use av_denoise_core::tune::{
    DeviceIdentity,
    DeviceLimits,
    DeviceProfile,
    EngineKernels,
    TuningConfig,
    build_engine as build_settled_engine,
    describe,
};
use av_denoise_core::{Engine, EngineSpec, Nl4d, Nlmeans};
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
use crate::tune::{KernelRequest, default_store_dir, engine_kernels};

/// An engine and the plane I/O for the runtime it was built on.
pub(crate) struct BuiltEngine {
    pub engine: Box<dyn Engine>,
    pub io: Box<dyn PlaneIo>,
    pub accelerator: Accelerator,
    pub device: DeviceProfile,
    /// The device the engine was asked for, which names its stored tuning results.
    pub selected: cubecl::Device,
    pub client: Client,
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
        selected: cubecl::Device,
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
            install_compile_helpers(ledger, device);
        }
    }

    fn run(
        self,
        selected: cubecl::Device,
        client: Client,
        accelerator: Accelerator,
        identity: DeviceIdentity,
    ) -> Result<BuiltEngine, DenoiserError> {
        let device = profile_of(&client, identity);
        let setup = self.request.setup;

        let engine = if setup.ledger.is_none() && setup.profile_override.is_none() {
            let config = TuningConfig {
                mode: self.request.tuning,
                store: default_store_dir(),
                deterministic: self.request.deterministic,
                ..TuningConfig::default()
            };
            build_settled_engine(&selected, &client, self.spec, &config)?
        } else {
            let kernels = engine_kernels(self.request, &device);
            build_tuned_engine(&client, self.spec, kernels)?
        };

        let io = ClientIo::new(client.clone());

        Ok(BuiltEngine {
            engine,
            io: Box::new(io),
            accelerator,
            device,
            selected,
            client,
        })
    }
}

/// Builds `spec` with `kernels`, which carry a tuning ledger or a profile override.
fn build_tuned_engine(
    client: &Client,
    spec: EngineSpec,
    kernels: EngineKernels,
) -> Result<Box<dyn Engine>, DenoiserError> {
    match spec {
        EngineSpec::Nlmeans { algorithm, geometry } => {
            let engine = Nlmeans::with_kernels(client, algorithm, geometry, kernels)?;
            Ok(Box::new(engine))
        },
        EngineSpec::Nl4d { options, geometry } => {
            let engine = Nl4d::with_kernels(client, options, geometry, kernels)?;
            Ok(Box::new(engine))
        },
    }
}

/// Reads the client's device profile.
struct DescribeDevice;

impl OnClient for DescribeDevice {
    type Output = DeviceProfile;

    fn run(
        self,
        _selected: cubecl::Device,
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
    let identity = describe(&selected, &client);

    #[cfg(feature = "vulkan")]
    if let Some(wgpu_device) = vulkan_device(&selected) {
        task.on_vulkan(wgpu_device);
    }

    task.run(selected, client, accelerator, identity)
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
