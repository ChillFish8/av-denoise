//! Listing the devices each backend can see

use cubecl::device::DeviceId;
use cubecl::prelude::*;

use super::accelerate::Accelerator;
use super::device::Device;
use super::probe::open_client;

/// What one backend reports about this machine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendDevices {
    /// The backend that was asked.
    pub accelerator: Accelerator,
    /// Whether the backend started at all.
    ///
    /// A backend the machine has no driver for reports no devices because it never ran, not because
    /// the machine has no hardware.
    pub available: bool,
    /// The devices the backend can see, in the order it lists them.
    ///
    /// Always empty when `available` is false.
    pub devices: Vec<Device>,
}

/// Asks each backend in `enable` which devices it can see.
///
/// Backends are reported in the order given, including the ones that could not start, so a caller
/// can tell "no such hardware" apart from "no such driver".
pub fn enumerate_devices(enable: &[Accelerator]) -> Vec<BackendDevices> {
    enable
        .iter()
        .map(|accelerator| match accelerator {
            #[cfg(feature = "cuda")]
            Accelerator::Cuda => match Device::Default.to_cuda() {
                Ok(cuda_device) => query_runtime::<cubecl::cuda::CudaRuntime>(*accelerator, &cuda_device),
                Err(_) => unavailable(*accelerator),
            },
            #[cfg(feature = "rocm")]
            Accelerator::Rocm => match Device::Default.to_amd() {
                Ok(amd_device) => query_runtime::<cubecl::hip::HipRuntime>(*accelerator, &amd_device),
                Err(_) => unavailable(*accelerator),
            },
            #[cfg(feature = "vulkan")]
            Accelerator::Vulkan => match Device::Default.to_wgpu() {
                Ok(wgpu_device) => query_runtime::<cubecl::wgpu::WgpuRuntime>(*accelerator, &wgpu_device),
                Err(_) => unavailable(*accelerator),
            },
            #[cfg(feature = "metal")]
            Accelerator::Metal => match Device::Default.to_wgpu() {
                Ok(wgpu_device) => query_runtime::<cubecl::wgpu::WgpuRuntime>(*accelerator, &wgpu_device),
                Err(_) => unavailable(*accelerator),
            },
            // Keeps the match exhaustive on docs.rs, where `cfg(docsrs)` widens the `Accelerator` enum to
            // include variants whose backend feature is not enabled. Never reached at runtime.
            #[cfg(docsrs)]
            #[expect(
                unreachable_patterns,
                reason = "the arm only keeps the match exhaustive on docs.rs"
            )]
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
/// A backend that cannot open a client at all, because its driver libraries are missing, is reported
/// as unavailable rather than allowed to take the process down.
fn query_runtime<R: Runtime>(accelerator: Accelerator, device: &R::Device) -> BackendDevices {
    let Some(client) = open_client::<R>(accelerator, device) else {
        return unavailable(accelerator);
    };

    // Type ids 0 to 3 are the device kinds `Device` can name, and anything else is hardware this tool
    // cannot select. ROCm and CUDA report their whole device list for every type id, so a device that
    // comes back on several passes is kept only once.
    let mut devices: Vec<Device> = Vec::new();
    for type_id in 0..=3 {
        for device_id in client.enumerate_devices(type_id) {
            if let Some(device) = to_device(device_id)
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
/// The type ids come from cubecl's own ordering of device kinds. A kind this tool cannot select
/// returns `None`.
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
        let discrete = to_device(DeviceId::new(0, 1));
        let integrated = to_device(DeviceId::new(1, 0));
        let virtual_gpu = to_device(DeviceId::new(2, 2));
        let cpu = to_device(DeviceId::new(3, 0));

        assert_eq!(discrete, Some(Device::Discrete { index: 1 }));
        assert_eq!(integrated, Some(Device::Integrated { index: 0 }));
        assert_eq!(virtual_gpu, Some(Device::Virtual { index: 2 }));
        assert_eq!(cpu, Some(Device::Cpu));
    }

    #[test]
    fn unknown_type_ids_are_skipped() {
        let unknown = to_device(DeviceId::new(4, 0));

        assert_eq!(unknown, None);
    }

    #[test]
    fn no_backends_lists_nothing() {
        let reported = enumerate_devices(&[]);

        assert!(reported.is_empty());
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
