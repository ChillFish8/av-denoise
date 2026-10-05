# av-denoise-core

GPU denoising engines that read and write CubeCL plane handles.

`Nlmeans` and `Nl4d` implement `Engine`. Push one frame of planes at a time. Each push returns how many
frames are ready, and every ready frame is written into planes you own with `Engine::emit_into` before the
next push. At the end of a stream, `Engine::finish` returns the tail count.

`Nl4dOptions::default()` uses the base preset's `temporal_radius` and calibrated defaults for the rest.

```rust,no_run
use av_denoise_core::{ChannelMode, DevicePlane, Engine, Geometry, Nl4d, Nl4dOptions, SampleFormat};
use cubecl::Runtime;
use cubecl::wgpu::WgpuRuntime;

# fn main() -> Result<(), av_denoise_core::Error> {
let device = <WgpuRuntime as Runtime>::Device::default();
let client = WgpuRuntime::client(&device);
let geometry = Geometry {
    width: 1920,
    height: 1080,
    channels: ChannelMode::Luma,
    input: SampleFormat::U16 { depth: 10 },
    output: SampleFormat::U16 { depth: 10 },
};
let mut engine = Nl4d::new(&client, Nl4dOptions::default(), geometry)?;

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
