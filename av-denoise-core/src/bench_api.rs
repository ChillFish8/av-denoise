//! Internal items and host helpers shared by the benches and tests. Not a stable interface.

pub mod collab {
    pub use crate::collab::*;
}

pub mod engine_kernels {
    pub use crate::engine::kernels::*;
}

pub mod harness {
    pub use crate::nl4d::harness::*;
}

pub mod kernels {
    pub use crate::nlmeans::kernels::*;
}

pub const MAX_GRID_1D: u32 = crate::nlmeans::MAX_GRID_1D;

pub mod motion {
    pub use crate::nlmeans::motion::*;
}

pub mod nl4d_kernels {
    pub use crate::nl4d::kernels::*;
}

pub mod prefilter {
    pub use crate::nlmeans::prefilter::*;
}

use std::future::Future;
use std::pin::Pin;
use std::str::FromStr;

use anyhow::Context;
use cubecl::bytes::Bytes;
use cubecl::prelude::*;
use cubecl::server::{Handle, ServerError};

use crate::engine::{DevicePlane, EgressSource, SampleFormat, egress};
pub use crate::nl4d::denoiser::Nl4dDenoiser;
pub use crate::nl4d::params::Nl4dParams;
pub use crate::nl4d::snapshot::MotionSnapshot;
use crate::nlmeans::ChannelMode;
pub use crate::nlmeans::denoiser::{GpuOutput, NlmDenoiser};
pub use crate::nlmeans::params::NlmParams;

pub const BLOCK_X: u32 = crate::nlmeans::BLOCK_X;
pub const BLOCK_Y: u32 = crate::nlmeans::BLOCK_Y;
pub const NOISE_CURVE_BINS: usize = crate::nlmeans::NOISE_CURVE_BINS;

/// A readback in flight, resolving to one buffer per handle read.
pub type ReadFuture = Pin<Box<dyn Future<Output = Result<Vec<Bytes>, ServerError>> + Send>>;

/// Starts reading `handle` back. Nothing is read until the future is first polled.
pub fn start_read<R: Runtime>(client: &ComputeClient<R>, handle: Handle) -> ReadFuture {
    let client = client.clone();
    let future = async move { client.read_async(vec![handle]).await };

    Box::pin(future)
}

/// Blocks on a readback and copies the frame out as f32.
pub fn wait_read(read: ReadFuture) -> Vec<f32> {
    let buffers = cubecl::future::block_on(read).expect("readback failed");
    let samples = f32::from_bytes(&buffers[0]);

    samples.to_vec()
}

/// Pushes interleaved host frames and reads denoised frames back as interleaved f32.
pub trait HostIo {
    /// Uploads one frame of `width * height * channels` values and pushes it.
    fn push_frame(&mut self, frame: &[f32]);

    /// Denoises the next frame, or returns `None` while the window is still filling.
    fn denoise(&mut self) -> Result<Option<Vec<f32>>, anyhow::Error>;

    /// Hands every frame the stream still holds to `sink`, then starts a fresh stream.
    fn flush(&mut self, sink: impl FnMut(&[f32])) -> Result<(), anyhow::Error>;
}

impl<R: Runtime> HostIo for NlmDenoiser<R> {
    fn push_frame(&mut self, frame: &[f32]) {
        let (width, height, channels) = self.frame_shape();
        let handles = upload_frame(self.compute_client(), frame, width, height, channels);
        let planes = device_planes(&handles, width, height);

        let pushed = self.push_planes(&planes, SampleFormat::F32);
        pushed.expect("frame push failed");
    }

    fn denoise(&mut self) -> Result<Option<Vec<f32>>, anyhow::Error> {
        let Some(output) = self.denoise_submit_gpu()? else {
            return Ok(None);
        };

        let frame = self.read_back(&output.handle)?;
        Ok(Some(frame))
    }

    fn flush(&mut self, mut sink: impl FnMut(&[f32])) -> Result<(), anyhow::Error> {
        let target = self.flush_target();

        for _ in 0..target {
            let output = loop {
                let step = self.flush_step_gpu()?;
                if let Some(output) = step {
                    break output;
                }
            };

            let frame = self.read_back(&output.handle)?;
            sink(&frame);
        }

        self.reset_stream_state();

        Ok(())
    }
}

impl<R: Runtime> NlmDenoiser<R> {
    fn read_back(&self, frame: &Handle) -> Result<Vec<f32>, anyhow::Error> {
        let shape = self.frame_shape();
        read_frame(self.compute_client(), self.placeholder(), frame, shape)
    }
}

impl<R: Runtime> HostIo for Nl4dDenoiser<R> {
    fn push_frame(&mut self, frame: &[f32]) {
        let (width, height, channels) = self.frame_shape();
        let handles = upload_frame(self.compute_client(), frame, width, height, channels);
        let planes = device_planes(&handles, width, height);

        let pushed = self.push_planes(&planes, SampleFormat::F32);
        pushed.expect("frame push failed");
    }

    fn denoise(&mut self) -> Result<Option<Vec<f32>>, anyhow::Error> {
        let Some(region) = self.submit_passes()? else {
            return Ok(None);
        };

        let handle = self.read_region(region);
        let frame = self.read_back(&handle)?;
        Ok(Some(frame))
    }

    fn flush(&mut self, mut sink: impl FnMut(&[f32])) -> Result<(), anyhow::Error> {
        let regions = self.finish_passes()?;

        for region in regions {
            let handle = self.read_region(region);
            let frame = self.read_back(&handle)?;
            sink(&frame);
        }

        self.reset_stream();

        Ok(())
    }
}

impl<R: Runtime> Nl4dDenoiser<R> {
    fn read_back(&self, frame: &Handle) -> Result<Vec<f32>, anyhow::Error> {
        let shape = self.frame_shape();
        read_frame(self.compute_client(), self.placeholder(), frame, shape)
    }
}

/// Splits an interleaved frame into one uploaded f32 plane per channel.
fn upload_frame<R: Runtime>(
    client: &ComputeClient<R>,
    frame: &[f32],
    width: u32,
    height: u32,
    channels: ChannelMode,
) -> Vec<Handle> {
    let channel_count = channels.count() as usize;
    let expected = width as usize * height as usize * channel_count;

    assert_eq!(
        frame.len(),
        expected,
        "frame size mismatch: expected {expected}, got {}",
        frame.len()
    );

    let mut handles = Vec::with_capacity(channel_count);

    for channel in 0..channel_count {
        let samples: Vec<f32> = frame
            .iter()
            .skip(channel)
            .step_by(channel_count)
            .copied()
            .collect();
        let bytes = f32::as_bytes(&samples);
        let handle = client.create_from_slice(bytes);
        handles.push(handle);
    }

    handles
}

fn device_planes(handles: &[Handle], width: u32, height: u32) -> Vec<DevicePlane<'_>> {
    handles
        .iter()
        .map(|handle| DevicePlane::new(handle, width, height))
        .collect()
}

/// Egresses an internal interleaved frame into f32 planes and reads them back interleaved.
fn read_frame<R: Runtime>(
    client: &ComputeClient<R>,
    placeholder: &Handle,
    frame: &Handle,
    shape: (u32, u32, ChannelMode),
) -> Result<Vec<f32>, anyhow::Error> {
    let (width, height, channels) = shape;
    let pixels = width * height;
    let channel_count = channels.count() as usize;
    let plane_bytes = pixels as usize * size_of::<f32>();

    let handles: Vec<Handle> = (0..channel_count).map(|_| client.empty(plane_bytes)).collect();
    let planes = device_planes(&handles, width, height);
    let source = EgressSource {
        frame,
        pixels,
        channels: channels.count(),
        stored_ch: channels.storage_count(),
    };

    egress(client, source, &planes, SampleFormat::F32, placeholder);

    let mut plane_samples = Vec::with_capacity(channel_count);

    for handle in handles {
        let bytes = client.read_one(handle).context("plane readback failed")?;
        let samples = f32::from_bytes(&bytes).to_vec();
        plane_samples.push(samples);
    }

    let mut interleaved = Vec::with_capacity(pixels as usize * channel_count);

    for pixel in 0..pixels as usize {
        for samples in &plane_samples {
            interleaved.push(samples[pixel]);
        }
    }

    Ok(interleaved)
}

/// A `--device` selector, such as `default` or `discrete:1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Default,
    Discrete(usize),
    Integrated(usize),
    Virtual(usize),
    Cpu,
}

impl FromStr for Device {
    type Err = String;

    fn from_str(selector: &str) -> Result<Self, Self::Err> {
        let (kind, suffix) = match selector.split_once(':') {
            Some((kind, index)) => (kind, Some(index)),
            None => (selector, None),
        };

        if matches!(kind, "default" | "cpu") && suffix.is_some() {
            return Err(format!("device kind '{kind}' takes no index, got '{selector}'"));
        }

        let index_text = suffix.unwrap_or("0");
        let index = index_text.parse::<usize>();
        let index = index.map_err(|_| format!("invalid device index '{index_text}' in '{selector}'"));

        match kind {
            "default" => Ok(Device::Default),
            "cpu" => Ok(Device::Cpu),
            "discrete" => Ok(Device::Discrete(index?)),
            "integrated" => Ok(Device::Integrated(index?)),
            "virtual" => Ok(Device::Virtual(index?)),
            other => Err(format!(
                "unknown device kind '{other}', expected default, discrete[:N], integrated[:N], virtual[:N], or cpu"
            )),
        }
    }
}

#[cfg(any(feature = "vulkan", feature = "metal"))]
impl Device {
    pub fn to_wgpu(self) -> Result<cubecl::wgpu::WgpuDevice, anyhow::Error> {
        use cubecl::wgpu::WgpuDevice;

        let device = match self {
            Device::Default => WgpuDevice::DefaultDevice,
            Device::Discrete(index) => WgpuDevice::DiscreteGpu(index),
            Device::Integrated(index) => WgpuDevice::IntegratedGpu(index),
            Device::Virtual(index) => WgpuDevice::VirtualGpu(index),
            Device::Cpu => WgpuDevice::Cpu,
        };

        Ok(device)
    }
}
