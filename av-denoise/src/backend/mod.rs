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

use av_denoise_core::tune::{
    DeviceIdentity,
    DeviceLimits,
    DeviceProfile,
    TuningConfig,
    TuningMode,
    build_pass_engine as build_settled_engine,
    describe,
};
use av_denoise_core::{PassEngine, PassSpec};
use cubecl::prelude::Client;

use self::accelerate::Accelerator;
pub use self::device::Device;
pub(crate) use self::placement::DevicePlacement;
use self::resolve::resolve;
use self::sniff::sniff_best_accelerator;
use crate::host::DenoiserError;
use crate::host::io::{ClientIo, PlaneIo};
use crate::tune::default_store_dir;

/// An engine and the plane I/O for the runtime it was built on.
pub(crate) struct BuiltEngine {
    pub engine: Box<dyn PassEngine>,
    pub io: Box<dyn PlaneIo>,
    pub accelerator: Accelerator,
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
}

/// Builds an engine for `spec` on the client from the stored tuning results.
struct BuildEngine {
    spec: PassSpec,
    tuning: TuningMode,
    deterministic: bool,
}

impl OnClient for BuildEngine {
    type Output = BuiltEngine;

    fn run(
        self,
        selected: cubecl::Device,
        client: Client,
        accelerator: Accelerator,
        _identity: DeviceIdentity,
    ) -> Result<BuiltEngine, DenoiserError> {
        let config = TuningConfig {
            mode: self.tuning,
            store: default_store_dir(),
            deterministic: self.deterministic,
            ..TuningConfig::default()
        };
        let engine = build_settled_engine(&selected, &client, self.spec, &config)?;

        let io = ClientIo::new(client.clone());

        Ok(BuiltEngine {
            engine,
            io: Box::new(io),
            accelerator,
            selected,
            client,
        })
    }
}

/// Hands back the device and the client opened on it.
struct OpenClient;

impl OnClient for OpenClient {
    type Output = (cubecl::Device, Client);

    fn run(
        self,
        selected: cubecl::Device,
        client: Client,
        _accelerator: Accelerator,
        _identity: DeviceIdentity,
    ) -> Result<(cubecl::Device, Client), DenoiserError> {
        Ok((selected, client))
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
    spec: PassSpec,
    tuning: TuningMode,
    deterministic: bool,
) -> Result<BuiltEngine, DenoiserError> {
    let task = BuildEngine {
        spec,
        tuning,
        deterministic,
    };
    run_on_best(accelerators, device, placement, task)
}

/// The device [build_engine] picks for `accelerators` and `device`, with its shared client.
pub(crate) fn selected_client(
    accelerators: &[Accelerator],
    device: &Device,
) -> Result<(cubecl::Device, Client), DenoiserError> {
    let placement = DevicePlacement::shared();
    run_on_best(accelerators, device, &placement, OpenClient)
}

/// The profile of the device [build_engine] picks for `accelerators` and `device`.
pub fn selected_device(
    accelerators: &[Accelerator],
    device: &Device,
) -> Result<DeviceProfile, DenoiserError> {
    let placement = DevicePlacement::shared();
    run_on_best(accelerators, device, &placement, DescribeDevice)
}
