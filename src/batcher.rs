use std::sync::mpsc;

use anyhow::Context;
use burn::prelude::*;
use burn::tensor::TensorData;

use crate::source::{BitDepth, FrameSource, InputSource, VideoFrameBuffer};

pub const TILE_SIZE: usize = 64;

/// The input batcher takes an input source and produces normalized tile batches
/// ready for Burn inference.
pub fn create_batcher<B, I>(device: B::Device, source: I, batch_size: usize) -> InputBatchStream<B>
where
    B: Backend + 'static,
    I: InputSource + Send + 'static,
{
    let (tx, batches) = mpsc::sync_channel(4);

    let worker_handle = std::thread::Builder::new()
        .name("av-eval-batcher".into())
        .spawn(move || consume_input_source::<I>(source, batch_size, tx))
        .expect("spawn batcher worker thread");

    InputBatchStream {
        device,
        worker_handle,
        batches,
    }
}

/// A stream of input frame tiles bundled into batches.
pub struct InputBatchStream<B: Backend> {
    device: B::Device,
    worker_handle: std::thread::JoinHandle<anyhow::Result<()>>,
    batches: mpsc::Receiver<WorkerFrameBatch>,
}

impl<B: Backend> InputBatchStream<B> {
    pub fn next_batch(&mut self) -> Option<FrameBatch<B>> {
        let batch = self.batches.recv().ok()?;
        let frame_tensor = Tensor::<B, 4>::from_data(batch.frame_tensor, &self.device).permute([0, 3, 1, 2]);

        Some(FrameBatch {
            size: batch.size,
            frame_tensor,
        })
    }

    pub fn join_worker(self) -> anyhow::Result<()> {
        self.worker_handle.join().expect("worker thread panicked")
    }
}

fn consume_input_source<I>(
    source: I,
    batch_size: usize,
    batches: mpsc::SyncSender<WorkerFrameBatch>,
) -> anyhow::Result<()>
where
    I: InputSource + Send + 'static,
{
    let width = source.width();
    let height = source.height();
    let bit_depth = source.bit_depth();
    let single_frame_size = frame_bytes(width, height, bit_depth)?;
    let mut frame_source = source.into_frame_source()?;
    let mut has_frame = true;

    while has_frame {
        let mut frames = Vec::with_capacity(batch_size * width * height * 3);
        let mut num_frames = 0;

        while num_frames < batch_size {
            let mut frame = vec![0u8; single_frame_size];
            has_frame = frame_source.step_next_frame(VideoFrameBuffer::new(&mut frame))?;
            if !has_frame {
                break;
            }

            frames.extend(decode_frame_to_f32(&frame, bit_depth)?);
            num_frames += 1;
        }

        if num_frames == 0 {
            break;
        }

        let batch = WorkerFrameBatch {
            size: num_frames,
            frame_tensor: TensorData::new(frames, [num_frames, height, width, 3]),
        };

        if batches.send(batch).is_err() {
            break;
        }
    }

    Ok(())
}

fn frame_bytes(width: usize, height: usize, bit_depth: BitDepth) -> anyhow::Result<usize> {
    width
        .checked_mul(height)
        .and_then(|n| n.checked_mul(3))
        .and_then(|n| n.checked_mul(bit_depth.bytes_per_sample()))
        .context("calculate packed RGB frame size")
}

fn decode_frame_to_f32(frame: &[u8], bit_depth: BitDepth) -> anyhow::Result<Vec<f32>> {
    Ok(match bit_depth {
        BitDepth::Eight => frame
            .iter()
            .map(|value| *value as f32 / 255.0)
            .collect(),
        BitDepth::Ten => frame
            .chunks_exact(2)
            .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 1023.0)
            .collect(),
    })
}

/// A batch of input frames to be processed on the selected backend.
pub struct FrameBatch<B: Backend> {
    pub size: usize,
    pub frame_tensor: Tensor<B, 4>,
}

struct WorkerFrameBatch {
    size: usize,
    frame_tensor: TensorData,
}

impl<B: Backend> Clone for FrameBatch<B> {
    fn clone(&self) -> Self {
        Self {
            size: self.size,
            frame_tensor: self.frame_tensor.clone(),
        }
    }
}

#[cfg(all(test, feature = "cpu"))]
mod tests {
    use super::*;
    use crate::source::mock::MockInput;

    type TestBackend = burn::backend::cpu::Cpu;
    const TEST_WIDTH: usize = 2;
    const TEST_HEIGHT: usize = 2;

    #[test]
    fn batches_multiple_rgb24_frames() {
        let frame_1 = sequential_rgb24_frame(1);
        let frame_2 = sequential_rgb24_frame(2);
        let frame_3 = sequential_rgb24_frame(3);
        let source = MockInput::new(
            TEST_WIDTH,
            TEST_HEIGHT,
            BitDepth::Eight,
            vec![frame_1.clone(), frame_2.clone(), frame_3.clone()],
        )
        .expect("mock input should build");

        let mut batcher = create_batcher::<TestBackend, _>(Default::default(), source, 2);

        let first_batch = batcher.next_batch().expect("expected first batch");
        assert_eq!(first_batch.size, 2);
        assert_eq!(first_batch.frame_tensor.dims(), [2, 3, TEST_HEIGHT, TEST_WIDTH]);

        let first_data = first_batch.frame_tensor.clone().permute([0, 2, 3, 1]).into_data();
        let first_values = first_data.to_vec::<f32>().expect("expected f32 output");
        assert_eq!(first_values.len(), 2 * TEST_WIDTH * TEST_HEIGHT * 3);
        assert_eq!(first_values[0], frame_1[0] as f32 / 255.0);
        assert_eq!(first_values[TEST_WIDTH * TEST_HEIGHT * 3], frame_2[0] as f32 / 255.0);

        let second_batch = batcher.next_batch().expect("expected trailing batch");
        assert_eq!(second_batch.size, 1);
        assert_eq!(second_batch.frame_tensor.dims(), [1, 3, TEST_HEIGHT, TEST_WIDTH]);

        assert!(batcher.next_batch().is_none());
        batcher.join_worker().expect("worker should exit cleanly");
    }

    #[test]
    fn batches_rgb48_frames() {
        let frame_1 = sequential_rgb48_frame(1);
        let frame_2 = sequential_rgb48_frame(2);
        let source = MockInput::new(
            TEST_WIDTH,
            TEST_HEIGHT,
            BitDepth::Ten,
            vec![frame_1.clone(), frame_2.clone()],
        )
        .expect("mock input should build");

        let mut batcher = create_batcher::<TestBackend, _>(Default::default(), source, 2);

        let batch = batcher.next_batch().expect("expected batch");
        assert_eq!(batch.size, 2);
        assert_eq!(batch.frame_tensor.dims(), [2, 3, TEST_HEIGHT, TEST_WIDTH]);

        let data = batch.frame_tensor.permute([0, 2, 3, 1]).into_data();
        let values = data.to_vec::<f32>().expect("expected f32 output");
        assert_eq!(values[0], 1.0 / 1023.0);
        assert_eq!(values[TEST_WIDTH * TEST_HEIGHT * 3], 2.0 / 1023.0);

        assert!(batcher.next_batch().is_none());
        batcher.join_worker().expect("worker should exit cleanly");
    }

    fn sequential_rgb24_frame(seed: u8) -> Vec<u8> {
        let len = TEST_WIDTH * TEST_HEIGHT * 3;
        (0..len)
            .map(|offset| seed.wrapping_add(offset as u8))
            .collect()
    }

    fn sequential_rgb48_frame(seed: u16) -> Vec<u8> {
        let len = TEST_WIDTH * TEST_HEIGHT * 3;
        let mut frame = Vec::with_capacity(len * 2);
        for offset in 0..len {
            frame.extend_from_slice(&(seed + offset as u16).to_le_bytes());
        }
        frame
    }
}
