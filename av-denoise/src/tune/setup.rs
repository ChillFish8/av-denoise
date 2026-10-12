use std::sync::Arc;

use av_denoise_core::tune::{DeviceProfile, EngineKernels, PipelineProfile, TuneLedger};

use super::mode::TuningMode;
use super::resolve;

/// How the engines of one denoiser are wired for tuning.
#[derive(Debug, Clone, Default)]
pub(crate) struct KernelSetup {
    /// Shared tuning state, present when the denoiser may tune in place.
    pub ledger: Option<Arc<TuneLedger>>,
    /// A profile to build whatever the store says, used to compare profiles.
    pub profile_override: Option<PipelineProfile>,
}

/// Everything the engine builder needs to pick kernels for one plane.
#[derive(Debug, Clone, Copy)]
pub(crate) struct KernelRequest<'a> {
    pub tuning: TuningMode,
    pub deterministic: bool,
    pub profile_key: &'a str,
    pub setup: &'a KernelSetup,
}

/// The plan, profile and ledger an engine on `device` is built with.
///
/// [TuningMode::Off] runs every baseline. Otherwise the stored result for the device picks each
/// key's candidate and the plane's profile, and seeds the ledger when one is attached.
pub(crate) fn engine_kernels(request: &KernelRequest<'_>, device: &DeviceProfile) -> EngineKernels {
    if request.tuning == TuningMode::Off {
        return EngineKernels::default();
    }

    let stored = resolve(&device.identity);
    let plan = stored.kernel_plan(request.deterministic);
    let stored_profile = stored.profile_for(request.profile_key, request.deterministic);
    let profile = request.setup.profile_override.unwrap_or(stored_profile);
    let ledger = request.setup.ledger.clone();

    if let Some(ledger) = &ledger {
        ledger.seed(&plan);
    }

    EngineKernels {
        plan,
        profile,
        ledger,
    }
}
