use cubecl::device::DeviceId;
use cubecl::prelude::*;

use crate::accelerate::Accelerator;
use crate::device::Device;
use crate::probe::open_client;

/// What one backend reports about this machine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendDevices {
    /// The backend that was asked.
    pub accelerator: Accelerator,
    /// Whether the backend started at all.
    ///
    /// A build can enable a backend the machine has no driver for, and
    /// that backend reports no devices because it never ran, not
    /// because the machine has no hardware.
    pub available: bool,
    /// The devices the backend can see, in the order it lists them.
    ///
    /// Always empty when `available` is false.
    pub devices: Vec<Device>,
}

/// Asks each backend in `enable` which devices it can see.
///
/// Backends are reported in the order given, including the ones that
/// could not start, so a caller can tell "no such hardware" apart from
/// "no such driver".
pub fn enumerate_devices(enable: &[Accelerator]) -> Vec<BackendDevices> {
    enable
        .iter()
        .map(|accelerator| match accelerator {
            #[cfg(feature = "cuda")]
            Accelerator::Cuda => match Device::Default.to_cuda() {
                Ok(dev) => query_runtime::<cubecl::cuda::CudaRuntime>(*accelerator, &dev),
                Err(_) => unavailable(*accelerator),
            },
            #[cfg(feature = "rocm")]
            Accelerator::Rocm => match Device::Default.to_amd() {
                Ok(dev) => query_runtime::<cubecl::hip::HipRuntime>(*accelerator, &dev),
                Err(_) => unavailable(*accelerator),
            },
            #[cfg(feature = "vulkan")]
            Accelerator::Vulkan => match Device::Default.to_wgpu() {
                Ok(dev) => query_runtime::<cubecl::wgpu::WgpuRuntime>(*accelerator, &dev),
                Err(_) => unavailable(*accelerator),
            },
            #[cfg(feature = "metal")]
            Accelerator::Metal => match Device::Default.to_wgpu() {
                Ok(dev) => query_runtime::<cubecl::wgpu::WgpuRuntime>(*accelerator, &dev),
                Err(_) => unavailable(*accelerator),
            },
            // Keeps the match exhaustive on docs.rs, where `cfg(docsrs)`
            // widens the `Accelerator` enum to include variants whose
            // backend feature is not enabled. Never reached at runtime.
            #[cfg(docsrs)]
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        })
        .collect()
}

fn unavailable(accelerator: Accelerator) -> BackendDevices {
    BackendDevices {
        accelerator,
        available: false,
        devices: Vec::new(),
    }
}

/// Opens a client on `device` and lists what that backend can see.
///
/// A backend that cannot open a client at all, because its driver
/// libraries are missing, is reported as unavailable rather than
/// allowed to take the process down. See [`probe`](crate::probe).
fn query_runtime<R: Runtime>(accelerator: Accelerator, device: &R::Device) -> BackendDevices {
    let Some(client) = open_client::<R>(accelerator, device) else {
        return unavailable(accelerator);
    };

    // Type ids 0 to 3 are the device kinds `Device` can name. Anything
    // else the runtime reports is hardware this tool cannot select.
    //
    // Not every backend filters by the type id it is given. ROCm and
    // CUDA report their whole device list for each one, so the same
    // device comes back on every pass and is kept only once.
    let mut devices: Vec<Device> = Vec::new();
    for type_id in 0..=3 {
        for id in client.enumerate_devices(type_id) {
            if let Some(device) = to_device(id)
                && !devices.contains(&device)
            {
                devices.push(device);
            }
        }
    }

    BackendDevices {
        accelerator,
        available: true,
        devices,
    }
}

/// Maps a cubecl device id onto the selector that names it.
///
/// The type ids come from cubecl's own ordering of device kinds.
/// Backends that report a kind this tool cannot select return `None`.
fn to_device(id: DeviceId) -> Option<Device> {
    let index = id.index_id as usize;
    match id.type_id {
        0 => Some(Device::Discrete { index }),
        1 => Some(Device::Integrated { index }),
        2 => Some(Device::Virtual { index }),
        3 => Some(Device::Cpu),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_kinds_map_from_type_ids() {
        assert_eq!(
            to_device(DeviceId::new(0, 1)),
            Some(Device::Discrete { index: 1 }),
        );
        assert_eq!(
            to_device(DeviceId::new(1, 0)),
            Some(Device::Integrated { index: 0 }),
        );
        assert_eq!(to_device(DeviceId::new(2, 2)), Some(Device::Virtual { index: 2 }),);
        assert_eq!(to_device(DeviceId::new(3, 0)), Some(Device::Cpu));
    }

    #[test]
    fn unknown_type_ids_are_skipped() {
        assert_eq!(to_device(DeviceId::new(4, 0)), None);
    }

    #[test]
    fn no_backends_lists_nothing() {
        assert!(enumerate_devices(&[]).is_empty());
    }

    #[cfg(feature = "vulkan")]
    #[test]
    fn vulkan_reports_at_least_one_device() {
        let reported = enumerate_devices(&[Accelerator::Vulkan]);
        assert_eq!(reported.len(), 1);

        let vulkan = &reported[0];
        assert_eq!(vulkan.accelerator, Accelerator::Vulkan);
        assert!(vulkan.available, "the vulkan backend did not start");
        assert!(
            !vulkan.devices.is_empty(),
            "the vulkan backend started but listed no devices",
        );

        let mut unique = vulkan.devices.clone();
        unique.dedup();
        assert_eq!(
            unique.len(),
            vulkan.devices.len(),
            "a device was listed more than once: {:?}",
            vulkan.devices,
        );
    }
}
