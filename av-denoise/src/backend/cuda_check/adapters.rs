use av_denoise_core::tune::NVIDIA_VENDOR_ID;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum NvidiaPresence {
    Present,
    Absent,
    /// No adapters were listed, so nothing can be said either way.
    Unknown,
}

pub(super) fn nvidia_presence(vendor_ids: &[u32]) -> NvidiaPresence {
    if vendor_ids.is_empty() {
        return NvidiaPresence::Unknown;
    }

    if vendor_ids.contains(&NVIDIA_VENDOR_ID) {
        return NvidiaPresence::Present;
    }

    NvidiaPresence::Absent
}

/// The PCI vendor ID of every adapter wgpu can see, or none without the `vulkan` feature.
#[cfg(feature = "vulkan")]
pub(super) fn adapter_vendor_ids() -> Vec<u32> {
    let descriptor = wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    };
    let instance = wgpu::Instance::new(descriptor);
    let listing = instance.enumerate_adapters(wgpu::Backends::PRIMARY);
    let adapters = cubecl::future::block_on(listing);

    adapters.iter().map(|adapter| adapter.get_info().vendor).collect()
}

#[cfg(not(feature = "vulkan"))]
pub(super) fn adapter_vendor_ids() -> Vec<u32> {
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_nvidia_adapter_is_present() {
        assert_eq!(nvidia_presence(&[0x1002, 0x10DE]), NvidiaPresence::Present);
    }

    #[test]
    fn other_vendors_only_is_absent() {
        assert_eq!(nvidia_presence(&[0x1002, 0x8086]), NvidiaPresence::Absent);
    }

    #[test]
    fn no_adapters_is_unknown() {
        assert_eq!(nvidia_presence(&[]), NvidiaPresence::Unknown);
    }
}
