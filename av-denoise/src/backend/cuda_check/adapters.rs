use av_denoise_core::tune::NVIDIA_VENDOR_ID;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum NvidiaPresence {
    Present,
    Absent,
    /// No hardware adapters were listed, so nothing can be said either way.
    Unknown,
}

/// An adapter wgpu can see.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Adapter {
    pub(super) vendor_id: u32,
    /// Whether the adapter is a software renderer, which says nothing about the installed GPUs.
    pub(super) software: bool,
}

pub(super) fn nvidia_presence(adapters: &[Adapter]) -> NvidiaPresence {
    let mut hardware = adapters.iter().filter(|adapter| !adapter.software).peekable();
    if hardware.peek().is_none() {
        return NvidiaPresence::Unknown;
    }

    if hardware.any(|adapter| adapter.vendor_id == NVIDIA_VENDOR_ID) {
        return NvidiaPresence::Present;
    }

    NvidiaPresence::Absent
}

/// Every adapter wgpu can see, or none without the `vulkan` feature.
#[cfg(feature = "vulkan")]
pub(super) fn list_adapters() -> Vec<Adapter> {
    let descriptor = wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    };
    let instance = wgpu::Instance::new(descriptor);
    let listing = instance.enumerate_adapters(wgpu::Backends::PRIMARY);
    let adapters = cubecl::future::block_on(listing);

    adapters
        .iter()
        .map(|adapter| {
            let info = adapter.get_info();
            Adapter {
                vendor_id: info.vendor,
                software: info.device_type == wgpu::DeviceType::Cpu,
            }
        })
        .collect()
}

#[cfg(not(feature = "vulkan"))]
pub(super) fn list_adapters() -> Vec<Adapter> {
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hardware(vendor_id: u32) -> Adapter {
        Adapter {
            vendor_id,
            software: false,
        }
    }

    fn software(vendor_id: u32) -> Adapter {
        Adapter {
            vendor_id,
            software: true,
        }
    }

    #[test]
    fn an_nvidia_adapter_is_present() {
        let adapters = [hardware(0x1002), hardware(0x10DE)];
        assert_eq!(nvidia_presence(&adapters), NvidiaPresence::Present);
    }

    #[test]
    fn other_vendors_only_is_absent() {
        let adapters = [hardware(0x1002), hardware(0x8086)];
        assert_eq!(nvidia_presence(&adapters), NvidiaPresence::Absent);
    }

    #[test]
    fn no_adapters_is_unknown() {
        assert_eq!(nvidia_presence(&[]), NvidiaPresence::Unknown);
    }

    #[test]
    fn software_adapters_only_is_unknown() {
        let adapters = [software(0x10005), software(0x1414)];
        assert_eq!(nvidia_presence(&adapters), NvidiaPresence::Unknown);
    }

    #[test]
    fn software_beside_other_hardware_is_absent() {
        let adapters = [software(0x10005), hardware(0x1002)];
        assert_eq!(nvidia_presence(&adapters), NvidiaPresence::Absent);
    }

    #[test]
    fn nvidia_hardware_among_software_is_present() {
        let adapters = [software(0x10005), hardware(0x10DE)];
        assert_eq!(nvidia_presence(&adapters), NvidiaPresence::Present);
    }
}
