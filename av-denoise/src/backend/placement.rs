#[cfg(feature = "vulkan")]
use std::sync::OnceLock;
#[cfg(all(test, feature = "vulkan"))]
use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(feature = "vulkan")]
use cubecl::future::block_on;
#[cfg(feature = "vulkan")]
use cubecl::wgpu::{RuntimeOptions, WgpuDevice, WgpuSetup, init_device, vulkan};

#[cfg(feature = "vulkan")]
use super::describe::find_adapter;

/// Whether a denoiser's engines share the process's client or open a device of their own.
#[derive(Debug)]
pub(crate) struct DevicePlacement {
    #[cfg_attr(
        not(feature = "vulkan"),
        expect(dead_code, reason = "only the Vulkan runtime opens an own device")
    )]
    own: bool,
    #[cfg(feature = "vulkan")]
    opened: OnceLock<Option<OwnDevice>>,
    /// How many clients were handed the own device.
    #[cfg(all(test, feature = "vulkan"))]
    handed_out: AtomicUsize,
}

/// A wgpu device opened for one denoiser, and the adapter it was opened on.
#[cfg(feature = "vulkan")]
#[derive(Debug)]
pub(crate) struct OwnDevice {
    pub device: WgpuDevice,
    pub adapter: wgpu::AdapterInfo,
}

impl DevicePlacement {
    /// Every engine opens the client CubeCL keeps for the selected device.
    pub(crate) fn shared() -> Self {
        Self::new(false)
    }

    /// On Vulkan, the engines open one wgpu device of their own on the selected adapter.
    ///
    /// The device has its own queue, so its work never waits behind another denoiser's. Every
    /// other runtime shares the client as [Self::shared] does.
    pub(crate) fn own() -> Self {
        Self::new(true)
    }

    fn new(own: bool) -> Self {
        Self {
            own,
            #[cfg(feature = "vulkan")]
            opened: OnceLock::new(),
            #[cfg(all(test, feature = "vulkan"))]
            handed_out: AtomicUsize::new(0),
        }
    }

    /// The device the engines open their client on, given the device the user selected.
    ///
    /// The own device is opened on the first call and reused by every later one. When it cannot be
    /// opened, the engines fall back to `selected`.
    #[cfg(feature = "vulkan")]
    pub(crate) fn vulkan_device(&self, selected: &WgpuDevice) -> WgpuDevice {
        if !self.own {
            return selected.clone();
        }

        let opened = self.opened.get_or_init(|| open_own_device(selected));
        let Some(own) = opened else {
            return selected.clone();
        };

        #[cfg(test)]
        self.handed_out.fetch_add(1, Ordering::Relaxed);

        own.device.clone()
    }

    /// The device opened by [Self::vulkan_device], if one was.
    #[cfg(all(test, feature = "vulkan"))]
    pub(crate) fn opened(&self) -> Option<&OwnDevice> {
        self.opened.get()?.as_ref()
    }

    /// How many clients [Self::vulkan_device] handed out on the own device.
    #[cfg(all(test, feature = "vulkan"))]
    pub(crate) fn handed_out(&self) -> usize {
        self.handed_out.load(Ordering::Relaxed)
    }
}

/// A new Vulkan device on the adapter CubeCL picks for `selected`.
#[cfg(feature = "vulkan")]
fn open_own_device(selected: &WgpuDevice) -> Option<OwnDevice> {
    let found = find_adapter(selected, wgpu::Backend::Vulkan);
    let Some((instance, adapter)) = found else {
        tracing::warn!(
            ?selected,
            "no adapter for an own device, so the shared device is used"
        );
        return None;
    };

    let request = vulkan::request_vulkan_device(&adapter);
    let requested = block_on(request);
    let Some((device, queue)) = requested else {
        tracing::warn!(
            ?selected,
            "cannot open an own device, so the shared device is used"
        );
        return None;
    };

    let info = adapter.get_info();
    let setup = WgpuSetup {
        instance,
        adapter,
        device,
        queue,
        backend: wgpu::Backend::Vulkan,
    };
    let options = RuntimeOptions::default();
    let device = init_device(setup, options);
    let own = OwnDevice {
        device,
        adapter: info,
    };

    tracing::debug!(
        device = ?own.device,
        adapter = own.adapter.name,
        bus = own.adapter.device_pci_bus_id,
        "opened an own device"
    );

    Some(own)
}

#[cfg(all(test, feature = "vulkan"))]
mod tests {
    use cubecl::wgpu::{Vulkan, init_setup};

    use super::*;
    use crate::backend::accelerate::Accelerator;
    use crate::backend::{DescribeDevice, Device, run_on_best};

    /// The adapter fields that tell two GPUs apart.
    fn adapter_fields(info: &wgpu::AdapterInfo) -> (String, String, u32, u32) {
        let name = info.name.clone();
        let bus = info.device_pci_bus_id.clone();
        (name, bus, info.vendor, info.device)
    }

    /// Needs a fresh process, since `init_setup` panics once the default client is registered.
    #[test]
    fn two_own_placements_open_two_devices_on_the_shared_clients_adapter() {
        let accelerators = [Accelerator::Vulkan];
        let selected = Device::Default;
        let wgpu_device = selected.to_wgpu().expect("a wgpu device");

        // CubeCL picks the shared client's adapter itself here, so the check below can fail.
        let options = RuntimeOptions::default();
        let shared_setup = init_setup::<Vulkan>(&wgpu_device, options);
        let shared_info = shared_setup.adapter.get_info();
        let shared_adapter = adapter_fields(&shared_info);

        let shared = DevicePlacement::shared();
        let first = DevicePlacement::own();
        let second = DevicePlacement::own();
        let shared_profile = run_on_best(&accelerators, &selected, &shared, DescribeDevice).expect("shared");
        let first_profile = run_on_best(&accelerators, &selected, &first, DescribeDevice).expect("first");
        let second_profile = run_on_best(&accelerators, &selected, &second, DescribeDevice).expect("second");

        let first_device = first.opened().expect("the first placement opened a device");
        let second_device = second.opened().expect("the second placement opened a device");
        assert!(shared.opened().is_none());
        assert_ne!(first_device.device, second_device.device);

        for opened in [first_device, second_device] {
            let own_adapter = adapter_fields(&opened.adapter);
            assert_eq!(own_adapter, shared_adapter);
        }

        let shared_hash = shared_profile.identity_hash();
        let first_hash = first_profile.identity_hash();
        assert_eq!(first_profile, shared_profile);
        assert_eq!(second_profile, shared_profile);
        assert_eq!(first_hash, shared_hash);
    }

    #[test]
    fn an_own_placement_reuses_its_device() {
        let selected = WgpuDevice::DefaultDevice;
        let placement = DevicePlacement::own();

        let first = placement.vulkan_device(&selected);
        let second = placement.vulkan_device(&selected);
        let handed_out = placement.handed_out();

        assert_eq!(first, second);
        assert_ne!(first, selected);
        assert_eq!(handed_out, 2);
    }
}
