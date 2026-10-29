use cubecl::prelude::Client;
use cubecl::wgpu::{WgpuDevice, WgpuDeviceKind};

use super::{fallback_identity, remembered_identity};
use crate::tune::DeviceIdentity;

/// The variable CubeCL reads to point its default device at a specific adapter.
const DEFAULT_DEVICE_ENV: &str = "CUBECL_WGPU_DEFAULT_DEVICE";

/// The identity of the adapter CubeCL picks for `device` on `backend`.
///
/// The adapter is found the way CubeCL finds it, by kind and index, so the name matches the
/// device the client runs on.
pub(super) fn describe_wgpu(device: &WgpuDevice, backend: wgpu::Backend, client: &Client) -> DeviceIdentity {
    let memo_key = format!("wgpu|{backend:?}|{device:?}");

    remembered_identity(memo_key, || {
        let backend_name = format!("{backend:?}").to_ascii_lowercase();
        let info = adapter_info(device, backend);

        match info {
            Some(info) => {
                let driver = format!("{} {}", info.driver, info.driver_info);

                DeviceIdentity {
                    backend: backend_name,
                    adapter: info.name,
                    vendor_id: info.vendor,
                    device_id: info.device,
                    driver: driver.trim().to_string(),
                    os: std::env::consts::OS.to_string(),
                    compute_capability: None,
                }
            },
            None => fallback_identity(&backend_name, client),
        }
    })
}

fn adapter_info(device: &WgpuDevice, backend: wgpu::Backend) -> Option<wgpu::AdapterInfo> {
    let (_, adapter) = find_adapter(device, backend)?;
    Some(adapter.get_info())
}

/// The adapter CubeCL picks for `device` on `backend`, with the instance it was found on.
pub fn find_adapter(device: &WgpuDevice, backend: wgpu::Backend) -> Option<(wgpu::Instance, wgpu::Adapter)> {
    let descriptor = wgpu::InstanceDescriptor {
        backends: backend.into(),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    };
    let instance = wgpu::Instance::new(descriptor);
    let selected = default_override(device);

    let adapter = match selected {
        WgpuDeviceKind::DiscreteGpu(index) => {
            pick_adapter(&instance, backend, wgpu::DeviceType::DiscreteGpu, index)
        },
        WgpuDeviceKind::IntegratedGpu(index) => {
            pick_adapter(&instance, backend, wgpu::DeviceType::IntegratedGpu, index)
        },
        WgpuDeviceKind::VirtualGpu(index) => {
            pick_adapter(&instance, backend, wgpu::DeviceType::VirtualGpu, index)
        },
        WgpuDeviceKind::Cpu => pick_adapter(&instance, backend, wgpu::DeviceType::Cpu, 0),
        _ => {
            let options = wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
                apply_limit_buckets: false,
            };
            let request = instance.request_adapter(&options);
            let requested = cubecl::future::block_on(request);
            requested.ok()
        },
    }?;

    Some((instance, adapter))
}

/// The `index`-th adapter of `kind`.
///
/// Adapters of kind `Other` form a fallback list CubeCL also counts from, used when there are
/// not enough adapters of `kind`.
fn pick_adapter(
    instance: &wgpu::Instance,
    backend: wgpu::Backend,
    kind: wgpu::DeviceType,
    index: usize,
) -> Option<wgpu::Adapter> {
    let backends: wgpu::Backends = backend.into();
    let listing = instance.enumerate_adapters(backends);
    let adapters = cubecl::future::block_on(listing);
    let mut matching = Vec::new();
    let mut other = Vec::new();

    for adapter in adapters {
        let device_type = adapter.get_info().device_type;

        if device_type == wgpu::DeviceType::Other {
            other.push(adapter);
        } else if device_type == kind {
            matching.push(adapter);
        }
    }

    if index < matching.len() {
        return matching.into_iter().nth(index);
    }

    other.into_iter().nth(index)
}

/// `device`'s kind, or the adapter [DEFAULT_DEVICE_ENV] names when `device` is the default.
fn default_override(device: &WgpuDevice) -> WgpuDeviceKind {
    if !matches!(device.kind, WgpuDeviceKind::DefaultDevice) {
        return device.kind.clone();
    }

    let Ok(text) = std::env::var(DEFAULT_DEVICE_ENV) else {
        return device.kind.clone();
    };

    let parsed = parse_override(&text);
    parsed.unwrap_or_else(|| device.kind.clone())
}

pub(crate) fn parse_override(text: &str) -> Option<WgpuDeviceKind> {
    if text == "Cpu" {
        return Some(WgpuDeviceKind::Cpu);
    }

    let (kind, rest) = text.split_once('(')?;
    let index_text = rest.strip_suffix(')')?;
    let index: usize = index_text.parse().ok()?;

    match kind {
        "DiscreteGpu" => Some(WgpuDeviceKind::DiscreteGpu(index)),
        "IntegratedGpu" => Some(WgpuDeviceKind::IntegratedGpu(index)),
        "VirtualGpu" => Some(WgpuDeviceKind::VirtualGpu(index)),
        _ => None,
    }
}
