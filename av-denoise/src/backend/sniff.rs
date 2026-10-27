//! Picking a backend that actually works on this machine

use super::accelerate::Accelerator;
use super::device::Device;
use super::probe::open_client;
use super::resolve::resolve;

/// Returns the first accelerator, in order of preference, whose client can be built and synchronised
/// on `device`.
///
/// cubecl kernels are fully asynchronous, so a successful `client.sync()` is enough to prove the
/// backend works and no test kernel is needed. The probe runs on `device` rather than the backend's
/// default, because opening a client on another card tests the wrong hardware and pays that card's
/// first-time driver initialisation.
///
/// An accelerator that cannot express `device` at all is treated as unavailable, which is the answer
/// building on it would give one step earlier. So is a backend whose driver libraries are missing,
/// however loudly it fails.
///
/// ```no_run
/// use av_denoise::Device;
/// use av_denoise::accelerate::get_default_accelerators;
/// use av_denoise::sniff::sniff_best_accelerator;
///
/// let preferred = get_default_accelerators();
/// match sniff_best_accelerator(&preferred, &Device::Default) {
///     Some(accelerator) => println!("running on {accelerator}"),
///     None => println!("no usable backend on this machine"),
/// }
/// ```
pub fn sniff_best_accelerator(enable: &[Accelerator], device: &Device) -> Option<Accelerator> {
    for accelerator in enable {
        let requested = device.clone();
        let Ok(selected) = resolve(*accelerator, requested) else {
            continue;
        };

        if open_client(*accelerator, &selected).is_some() {
            return Some(*accelerator);
        }
    }

    None
}
