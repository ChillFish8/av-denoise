pub(crate) mod collab;
pub(crate) mod nlm_window;
pub(crate) mod regularise;
#[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
mod tests;

use std::fmt;

use cubecl::config::autotune::AutotuneLevel;
use cubecl::config::{CubeClRuntimeConfig, RuntimeConfig};
use cubecl::ir::HardwareProperties;
use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::tune::TuneGroup;

/// Identifies the device a tuning result belongs to.
///
/// Built from the backend and the hardware limits that shape a launch, so identical cards share
/// results and each backend on one card tunes on its own.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct TuneId {
    backend: &'static str,
    streaming_multiprocessors: Option<u32>,
    plane_size_min: u32,
    plane_size_max: u32,
    max_shared_memory: usize,
    max_units_per_cube: u32,
}

impl TuneId {
    pub(crate) fn new<R: Runtime>(client: &ComputeClient<R>) -> Self {
        let hardware = &client.properties().hardware;

        Self {
            backend: R::name(client),
            streaming_multiprocessors: hardware.num_streaming_multiprocessors,
            plane_size_min: hardware.plane_size_min,
            plane_size_max: hardware.plane_size_max,
            max_shared_memory: hardware.max_shared_memory_size,
            max_units_per_cube: hardware.max_units_per_cube,
        }
    }
}

impl fmt::Display for TuneId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let units = self.streaming_multiprocessors.unwrap_or(0);

        write!(
            f,
            "{}-sm{}-plane{}x{}-smem{}-units{}",
            self.backend,
            units,
            self.plane_size_min,
            self.plane_size_max,
            self.max_shared_memory,
            self.max_units_per_cube,
        )
    }
}

/// Whether a launch of `units` threads using `shared_bytes` of shared memory fits the device.
pub(crate) fn fits(hardware: &HardwareProperties, units: u32, shared_bytes: usize) -> bool {
    let units_fit = units <= hardware.max_units_per_cube;
    let shared_fits = shared_bytes <= hardware.max_shared_memory_size;
    units_fit && shared_fits
}

/// A new buffer of `bytes` zeros.
pub(crate) fn zeroed<R: Runtime>(client: &ComputeClient<R>, bytes: usize) -> Handle {
    let zeros = vec![0u8; bytes];
    client.create_from_slice(&zeros)
}

/// The priority of the group every candidate other than candidate 0 belongs to.
///
/// The minimal autotune level skips the group, which leaves the default launches.
pub(crate) fn priority_for(level: &AutotuneLevel) -> i8 {
    match level {
        AutotuneLevel::Minimal => -1,
        _ => 0,
    }
}

/// The group priority for the autotune level the runtime is configured with.
pub(crate) fn alternatives_priority() -> i8 {
    let config = CubeClRuntimeConfig::get();
    priority_for(&config.autotune.level)
}

pub(crate) fn alternatives<K>() -> TuneGroup<K> {
    TuneGroup::new("alternatives", |_key| alternatives_priority())
}
