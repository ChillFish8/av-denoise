# av-denoise-core

GPU denoising engines that read and write CubeCL plane handles.

An `Engine` cleans whole frames of planes, built from an `EngineSpec` that names the algorithm, the options for
each plane group and the frame `Geometry`. Push one frame of planes at a time. Each push returns how many
frames are ready, and every ready frame is written into planes you own with `Engine::emit_into` before
the next push. At the end of a stream, `Engine::finish` returns the tail count.

`Nl4dOptions::default()` uses a `temporal_radius` of 2 and calibrated defaults for the rest.

```rust,no_run
use av_denoise_core::{
    ChromaLayout,
    ChromaSiting,
    DevicePlane,
    Engine,
    EngineSpec,
    Geometry,
    Nl4dOptions,
    SampleFormat,
    TuningConfig,
    build_engine,
};
use cubecl::wgpu::{WgpuBackend, WgpuDevice, WgpuDeviceKind};

# fn main() -> Result<(), av_denoise_core::Error> {
let any_backend = WgpuDevice::new(WgpuDeviceKind::DefaultDevice);
let wgpu_device = any_backend.on(WgpuBackend::Vulkan);
let device = cubecl::Device::from(wgpu_device);
let client = device.client();
let geometry = Geometry {
    width: 1920,
    height: 1080,
    chroma: ChromaLayout::Gray,
    input: SampleFormat::U16 { depth: 10 },
    output: SampleFormat::U16 { depth: 10 },
    chroma_siting: ChromaSiting::Unknown,
};
let spec = EngineSpec::Nl4d {
    luma: Some(Nl4dOptions::default()),
    chroma: None,
    geometry,
};
let mut engine = build_engine(&device, &client, spec, &TuningConfig::default())?;

let input = client.empty(1920 * 1080 * 2);
let output = client.empty(1920 * 1080 * 2);
let input_planes = [DevicePlane::new(&input, 1920, 1080)];
let output_planes = [DevicePlane::new(&output, 1920, 1080)];

let ready = engine.push(&input_planes)?;
for _ in 0..ready {
    engine.emit_into(&output_planes)?;
}

let tail = engine.finish()?;
for _ in 0..tail {
    engine.emit_into(&output_planes)?;
}
# Ok(())
# }
```
