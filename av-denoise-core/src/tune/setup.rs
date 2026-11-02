use std::sync::Arc;

use cubecl::prelude::*;

use super::config::{TuningConfig, TuningMode};
use super::device::{DeviceLimits, DeviceProfile};
use super::identity::describe;
use super::ledger::TuneLedger;
use super::legality::{DeviceClass, Legality};
use super::plan::PipelineProfile;
use super::priority::{PriorityTable, priority_for};
use super::profile_key::profile_key;
use super::selector::EngineKernels;
use super::store::resolve;
use crate::engine::{Engine, EngineSpec};
use crate::error::Error;

/// How the engines of one group are wired for tuning.
#[derive(Debug, Clone, Default)]
pub(crate) struct KernelSetup {
    /// Shared tuning state, present when the engines may tune in place.
    pub ledger: Option<Arc<TuneLedger>>,
    /// A profile to build whatever the store says, used to compare profiles.
    pub profile_override: Option<PipelineProfile>,
}

/// The plan, profile, ledger, legality rules and family priority an engine on `device` is built with.
///
/// [TuningMode::Off] runs every baseline under the device's legality rules. Otherwise the stored
/// result for the device picks each key's candidate and the plane's profile, and seeds the ledger
/// when one is attached.
pub(crate) fn engine_kernels(
    setup: &KernelSetup,
    config: &TuningConfig,
    profile_key: &str,
    device: &DeviceProfile,
) -> EngineKernels {
    let legality = Legality::for_device(device);
    let priority = device_priority(device);

    if config.mode == TuningMode::Off {
        return EngineKernels {
            legality,
            priority,
            ..EngineKernels::default()
        };
    }

    let stored = resolve(config.store.as_deref(), &device.identity);
    let plan = stored.kernel_plan(config.deterministic);
    let stored_profile = stored.profile_for(profile_key, config.deterministic);
    let profile = setup.profile_override.unwrap_or(stored_profile);
    let ledger = setup.ledger.clone();

    if let Some(ledger) = &ledger {
        ledger.seed(&plan);
    }

    EngineKernels {
        plan,
        profile,
        ledger,
        legality,
        priority,
    }
}

/// The family and pipeline profile order for `device`.
pub(crate) fn device_priority(device: &DeviceProfile) -> &'static PriorityTable {
    let class = DeviceClass::of(&device.identity);
    priority_for(class, &device.identity.backend)
}

/// Builds `spec` on `client` from the store's picks, with no tuning ledger.
///
/// `device` is the device `client` runs on, and names the stored result to read. Each pass reads
/// the profile stored under its own profile key. With [TuningMode::Off] every launch site runs its
/// baseline under the device's legality rules.
pub fn build_engine(
    device: &cubecl::Device,
    client: &Client,
    spec: EngineSpec,
    config: &TuningConfig,
) -> Result<Box<dyn Engine>, Error> {
    let profile = device_profile(device, client);
    let setup = KernelSetup::default();

    spec.build(client, |pass| {
        let key = profile_key(pass);
        engine_kernels(&setup, config, &key, &profile)
    })
}

fn device_profile(device: &cubecl::Device, client: &Client) -> DeviceProfile {
    let identity = describe(device, client);
    let limits = DeviceLimits::from_client(client);

    DeviceProfile { identity, limits }
}
