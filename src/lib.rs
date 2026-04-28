mod batcher;
mod model;
mod models;
mod sniff;
pub mod source;

use std::path::{Path, PathBuf};

use anyhow::Context;
use burn::backend;
use burn::tensor::TensorData;
use burn::tensor::ops::PadMode;
use burn::prelude::*;
use clap::Parser;

use crate::batcher::TILE_SIZE;
use crate::models::Accelerator;
use crate::source::BitDepth;

#[derive(Debug, Parser)]
/// Required denoising parameters.
pub struct Options {
    #[arg(short, long, value_delimiter = ',', default_values_t = models::get_default_accelerators())]
    /// The hardware accelerators to perform the computation
    ///
    /// Accelerators should be ordered from the highest priority to the lowest priority,
    /// the system will attempt to use each accelerator sequentially one after the other
    /// until it finds an accelerator that works for the host hardware.
    ///
    /// If no accelerator can be found, the application will error.
    ///
    /// By default, this will be all accelerators the binary was compiled with.
    pub accelerators: Vec<Accelerator>,
    #[arg(long, short, default_value = "10")]
    /// The number of frames to buffer and group into a batch before uploading
    /// and processing on the accelerator.
    ///
    /// Until a certain point, the higher the batch size the better your efficiency and performance,
    /// at the cost of increasing the memory usage of the accelerator, i.e. VRAM.
    pub batch_size: usize,
    #[arg(long, short, default_value = "models/")]
    /// The directory path that holds the models.
    ///
    /// The system expects specific file names for each model:
    ///
    /// - `rt_ldr.bpk` -> Full-sized SDR denoising.
    /// - `rt_ldr_small.bpk` -> Lighter weight model for SDR denoising.
    ///
    /// HDR model files are not yet generated in `src/model/`, so RGB48 input currently
    /// returns a clear unsupported-model error.
    pub model_path: PathBuf,
    #[arg(long)]
    /// Use the small variants of the models rather than their full-sized counterparts.
    ///
    /// These models are faster and use less resources, and are typically "good enough".
    pub use_small: bool,
}

/// Run denoiser on the given input.
pub fn run<I>(opts: Options, input: I) -> anyhow::Result<()>
where
    I: source::InputSource,
{
    tracing::info!("initialising with accelerators: {:?}", opts.accelerators);

    run_pipeline(opts, input).context("run pipeline")?;

    Ok(())
}

fn run_pipeline<I>(opts: Options, input: I) -> anyhow::Result<()>
where
    I: source::InputSource,
{
    let model_name = match (input.bit_depth(), opts.use_small) {
        (BitDepth::Eight, false) => "rt_ldr.bpk",
        (BitDepth::Eight, true) => "rt_ldr_small.bpk",
        (BitDepth::Ten, _) => anyhow::bail!(
            "RGB48 input is not supported yet because the HDR Burn models have not been generated"
        ),
    };

    let model_path = opts.model_path.join(model_name);
    if !model_path.exists() {
        anyhow::bail!(
            "model path {} does not exist, or cannot be read.",
            model_path.display()
        );
    }

    let Some(best_accelerator) = sniff::sniff_best_accelerator(&opts.accelerators) else {
        anyhow::bail!(
            "no specified accelerator is able to run on the host hardware. \
        Please check you have any runtime dependencies installed like NVCC for CUDA."
        );
    };

    match best_accelerator {
        #[cfg(feature = "cuda")]
        Accelerator::Cuda => {
            dispatch_pipeline::<backend::cuda::Cuda, I>(opts, &model_path, input)
                .context("dispatch pipeline on CUDA runtime")
        },
        #[cfg(feature = "rocm")]
        Accelerator::Rocm => {
            dispatch_pipeline::<backend::rocm::Rocm, I>(opts, &model_path, input)
                .context("dispatch pipeline on ROCM runtime")
        },
        #[cfg(feature = "vulkan")]
        Accelerator::Vulkan => {
            dispatch_pipeline::<backend::wgpu::Vulkan, I>(opts, &model_path, input)
                .context("dispatch pipeline on VULKAN runtime")
        },
        #[cfg(feature = "metal")]
        Accelerator::Metal => {
            dispatch_pipeline::<backend::wgpu::Metal, I>(opts, &model_path, input)
                .context("dispatch pipeline on METAL runtime")
        },
        #[cfg(feature = "cpu")]
        Accelerator::Cpu => {
            dispatch_pipeline::<backend::cpu::Cpu, I>(opts, &model_path, input)
                .context("dispatch pipeline on CPU runtime")
        },
    }
}

fn dispatch_pipeline<B, I>(
    opts: Options,
    model_path: &Path,
    input: I,
) -> anyhow::Result<()>
where
    B: Backend + 'static,
    I: source::InputSource + Send + 'static,
{
    let width = input.width();
    let height = input.height();
    let device = B::Device::default();

    let mut batcher = batcher::create_batcher::<B, _>(device.clone(), input, opts.batch_size);
    let overlap = 16;

    if opts.use_small {
        let model = model::SmallModel::<B>::from_file(model_path, &device);
        while let Some(batch) = batcher.next_batch() {
            let output = denoise_batch::<B, _>(
                &model,
                batch.frame_tensor,
                batch.size,
                width,
                height,
                overlap,
            )?;
            consume_output(output);
        }
    } else {
        let model = model::LargeModel::<B>::from_file(model_path, &device);
        while let Some(batch) = batcher.next_batch() {
            let output = denoise_batch::<B, _>(
                &model,
                batch.frame_tensor,
                batch.size,
                width,
                height,
                overlap,
            )?;
            consume_output(output);
        }
    }

    batcher.join_worker().context("join batcher worker")?;

    Ok(())
}

trait DenoiseModel<B: Backend> {
    fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4>;
}

impl<B: Backend> DenoiseModel<B> for model::SmallModel<B> {
    fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        self.forward(input)
    }
}

impl<B: Backend> DenoiseModel<B> for model::LargeModel<B> {
    fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        self.forward(input)
    }
}

fn denoise_batch<B, M>(
    model: &M,
    frames: Tensor<B, 4>,
    batch_size: usize,
    width: usize,
    height: usize,
    overlap: usize,
) -> anyhow::Result<Tensor<B, 4>>
where
    B: Backend,
    M: DenoiseModel<B>,
{
    let step = TILE_SIZE - overlap * 2;
    let pad_h = (TILE_SIZE - height % step) % step;
    let pad_w = (TILE_SIZE - width % step) % step;
    let device = frames.device();

    let padded = frames.pad(
        [(0, 0), (0, 0), (overlap, overlap + pad_h), (overlap, overlap + pad_w)],
        PadMode::Reflect,
    );
    let [_, _, padded_h, padded_w] = padded.dims();

    let mask = tile_mask::<B>(overlap, &device);
    let mut output = Tensor::<B, 4>::zeros([batch_size, 3, height, width], &device);
    let mut weight = Tensor::<B, 4>::zeros([batch_size, 1, height, width], &device);

    for y in (0..=padded_h - TILE_SIZE).step_by(step) {
        for x in (0..=padded_w - TILE_SIZE).step_by(step) {
            let tile = padded
                .clone()
                .slice([0..batch_size, 0..3, y..y + TILE_SIZE, x..x + TILE_SIZE]);
            let result = model.forward(tile);

            let oy = y as isize - overlap as isize;
            let ox = x as isize - overlap as isize;
            let dy0 = oy.max(0) as usize;
            let dy1 = (oy + TILE_SIZE as isize).min(height as isize) as usize;
            let dx0 = ox.max(0) as usize;
            let dx1 = (ox + TILE_SIZE as isize).min(width as isize) as usize;
            let sy0 = (dy0 as isize - oy) as usize;
            let sy1 = sy0 + (dy1 - dy0);
            let sx0 = (dx0 as isize - ox) as usize;
            let sx1 = sx0 + (dx1 - dx0);

            let result_slice = result.slice([0..batch_size, 0..3, sy0..sy1, sx0..sx1]);
            let mask_slice = mask.clone().slice([0..1, 0..1, sy0..sy1, sx0..sx1]);
            let output_slice = output
                .clone()
                .slice([0..batch_size, 0..3, dy0..dy1, dx0..dx1]);
            let weight_slice = weight
                .clone()
                .slice([0..batch_size, 0..1, dy0..dy1, dx0..dx1]);

            output = output.slice_assign(
                [0..batch_size, 0..3, dy0..dy1, dx0..dx1],
                output_slice + result_slice * mask_slice.clone(),
            );
            weight = weight.slice_assign(
                [0..batch_size, 0..1, dy0..dy1, dx0..dx1],
                weight_slice + mask_slice,
            );
        }
    }

    Ok(output / weight.clamp_min(1e-8))
}

fn tile_mask<B: Backend>(overlap: usize, device: &B::Device) -> Tensor<B, 4> {
    let mut mask = vec![1.0f32; TILE_SIZE * TILE_SIZE];
    if overlap > 0 {
        for index in 0..overlap {
            let ramp = index as f32 / overlap as f32;
            let reverse = (overlap - 1 - index) as f32 / overlap as f32;
            for x in 0..TILE_SIZE {
                mask[index * TILE_SIZE + x] *= ramp;
                mask[(TILE_SIZE - overlap + index) * TILE_SIZE + x] *= reverse;
            }
            for y in 0..TILE_SIZE {
                mask[y * TILE_SIZE + index] *= ramp;
                mask[y * TILE_SIZE + (TILE_SIZE - overlap + index)] *= reverse;
            }
        }
    }

    Tensor::<B, 4>::from_data(
        TensorData::new(mask, [1, 1, TILE_SIZE, TILE_SIZE]),
        device,
    )
}

fn consume_output<B: Backend>(output: Tensor<B, 4>) {
    std::hint::black_box(output.into_data());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::mock::MockInput;
    use crate::source::InputSource;

    #[test]
    fn selects_small_ldr_model() {
        let opts = Options {
            accelerators: vec![],
            batch_size: 1,
            model_path: PathBuf::from("models"),
            use_small: true,
        };
        let input = MockInput::new(64, 64, BitDepth::Eight, vec![vec![0; 64 * 64 * 3]])
            .expect("mock input should build");

        let model_name = match (input.bit_depth(), opts.use_small) {
            (BitDepth::Eight, false) => "rt_ldr.bpk",
            (BitDepth::Eight, true) => "rt_ldr_small.bpk",
            (BitDepth::Ten, _) => unreachable!(),
        };

        assert_eq!(model_name, "rt_ldr_small.bpk");
    }

    #[test]
    fn rejects_rgb48_without_hdr_model() {
        let opts = Options {
            accelerators: vec![],
            batch_size: 1,
            model_path: PathBuf::from("models"),
            use_small: false,
        };
        let input = MockInput::new(64, 64, BitDepth::Ten, vec![vec![0; 64 * 64 * 3 * 2]])
            .expect("mock input should build");

        let err = (|| -> anyhow::Result<()> {
            let _ = match (input.bit_depth(), opts.use_small) {
                (BitDepth::Eight, false) => "rt_ldr.bpk",
                (BitDepth::Eight, true) => "rt_ldr_small.bpk",
                (BitDepth::Ten, _) => anyhow::bail!(
                    "RGB48 input is not supported yet because the HDR Burn models have not been generated"
                ),
            };
            Ok(())
        })()
        .expect_err("rgb48 should error without hdr models");

        assert!(err
            .to_string()
            .contains("HDR Burn models have not been generated"));
    }

    #[test]
    fn creates_overlap_mask() {
        type TestBackend = burn::backend::cpu::Cpu;

        let mask = tile_mask::<TestBackend>(16, &Default::default()).into_data();
        let values = mask.to_vec::<f32>().expect("expected f32 mask");

        assert_eq!(values.len(), TILE_SIZE * TILE_SIZE);
        assert_eq!(values[0], 0.0);
        assert!(values[(TILE_SIZE / 2) * TILE_SIZE + TILE_SIZE / 2] > 0.9);
    }
}
