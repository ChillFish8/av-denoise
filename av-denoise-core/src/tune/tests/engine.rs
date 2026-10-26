use cubecl::prelude::*;
use cubecl::server::Handle;

use super::fixtures::{hashed_unit, upload};
use super::runtime::test_client;
use crate::engine::{DevicePlane, Engine, Geometry, SampleFormat};
use crate::nlmeans::{ChannelMode, DenoisingMode, NlmTuning, NlmeansAlgorithm, NlmeansOptions};
use crate::tune::gpu::read_f32;
use crate::tune::{EngineKernels, KernelPlan};
use crate::{Nl4d, Nl4dOptions, Nlmeans};

/// `count` frames of a smooth ramp plus per-frame hashed noise, one `Vec` per plane.
pub(crate) fn noisy_frames(width: u32, height: u32, planes: usize, count: usize) -> Vec<Vec<Vec<f32>>> {
    let pixels = width * height;

    (0..count as u32)
        .map(|frame| {
            (0..planes as u32)
                .map(|plane| {
                    (0..pixels)
                        .map(|index| {
                            let column = (index % width) as f32 / width as f32;
                            let ramp = 0.3 + 0.4 * column;
                            let seed = 100 + frame * 8 + plane;
                            let grain = (hashed_unit(index, seed) - 0.5) * 0.15;
                            ramp + grain
                        })
                        .collect()
                })
                .collect()
        })
        .collect()
}

fn emit_frame(
    client: &Client,
    engine: &mut dyn Engine,
    width: u32,
    height: u32,
    planes: usize,
) -> Result<Vec<f32>, anyhow::Error> {
    let pixels = (width * height) as usize;
    let handles: Vec<Handle> = (0..planes)
        .map(|_| client.empty(pixels * size_of::<f32>()))
        .collect();
    let device_planes: Vec<DevicePlane<'_>> = handles
        .iter()
        .map(|handle| DevicePlane::new(handle, width, height))
        .collect();
    engine.emit_into(&device_planes)?;

    let mut flat = Vec::with_capacity(pixels * planes);
    for handle in &handles {
        let values = read_f32(client, handle, pixels)?;
        flat.extend(values);
    }

    Ok(flat)
}

/// Pushes `frames` through `engine` as f32 planes and returns every frame it emits, planes flattened.
pub(crate) fn run_engine(
    client: &Client,
    engine: &mut dyn Engine,
    width: u32,
    height: u32,
    frames: &[Vec<Vec<f32>>],
) -> Result<Vec<Vec<f32>>, anyhow::Error> {
    let planes = frames.first().map_or(0, Vec::len);
    let mut outputs = Vec::new();

    for frame in frames {
        let handles: Vec<Handle> = frame.iter().map(|plane| upload(client, plane)).collect();
        let device_planes: Vec<DevicePlane<'_>> = handles
            .iter()
            .map(|handle| DevicePlane::new(handle, width, height))
            .collect();
        let ready = engine.push(&device_planes)?;

        for _ in 0..ready {
            let emitted = emit_frame(client, engine, width, height, planes)?;
            outputs.push(emitted);
        }
    }

    let tail = engine.finish()?;
    for _ in 0..tail {
        let emitted = emit_frame(client, engine, width, height, planes)?;
        outputs.push(emitted);
    }

    Ok(outputs)
}

fn f32_geometry(width: u32, height: u32, channels: ChannelMode) -> Geometry {
    Geometry {
        width,
        height,
        channels,
        input: SampleFormat::F32,
        output: SampleFormat::F32,
    }
}

/// Denoises `frames` with an NLMeans engine that runs `plan`.
pub(crate) fn nlmeans_outputs(
    algorithm: NlmeansAlgorithm,
    channels: ChannelMode,
    width: u32,
    height: u32,
    frames: &[Vec<Vec<f32>>],
    plan: KernelPlan,
) -> Vec<Vec<f32>> {
    let client = test_client();
    let geometry = f32_geometry(width, height, channels);
    let kernels = EngineKernels {
        plan,
        ..EngineKernels::default()
    };
    let mut engine = Nlmeans::with_kernels(&client, algorithm, geometry, kernels).expect("the engine builds");

    run_engine(&client, &mut engine, width, height, frames).expect("the frames denoise")
}

/// Denoises `frames` with an nl4d engine that runs `plan`.
pub(crate) fn nl4d_outputs(
    options: Nl4dOptions,
    channels: ChannelMode,
    width: u32,
    height: u32,
    frames: &[Vec<Vec<f32>>],
    plan: KernelPlan,
) -> Vec<Vec<f32>> {
    let kernels = EngineKernels {
        plan,
        ..EngineKernels::default()
    };

    nl4d_outputs_with(options, channels, width, height, frames, kernels)
}

/// Denoises `frames` with an nl4d engine built on `kernels`.
pub(crate) fn nl4d_outputs_with(
    options: Nl4dOptions,
    channels: ChannelMode,
    width: u32,
    height: u32,
    frames: &[Vec<Vec<f32>>],
    kernels: EngineKernels,
) -> Vec<Vec<f32>> {
    let client = test_client();
    let geometry = f32_geometry(width, height, channels);
    let mut engine = Nl4d::with_kernels(&client, options, geometry, kernels).expect("the engine builds");

    run_engine(&client, &mut engine, width, height, frames).expect("the frames denoise")
}

/// A temporal NLMeans run with radius 1, which reaches both windowed kernels.
pub(crate) fn temporal_fast() -> NlmeansAlgorithm {
    let options = NlmeansOptions {
        mode: DenoisingMode::Temporal { radius: 1 },
        tuning: NlmTuning {
            search_radius: Some(3),
            patch_radius: Some(2),
            ..NlmTuning::default()
        },
        ..NlmeansOptions::default()
    };

    NlmeansAlgorithm::Fast(options)
}
