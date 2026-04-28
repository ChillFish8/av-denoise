use std::sync::mpsc;

use anyhow::Context;
use burn::prelude::*;
use burn::tensor::TensorData;

use crate::source::{BitDepth, FrameSource, InputSource, VideoFrameBuffer};

pub const TILE_SIZE: usize = 64;

/// The input worker takes an input source and produces normalized frames ready for
/// Burn inference.
pub fn create_batcher<B, I>(device: B::Device, source: I) -> InputBatchStream<B>
where
    B: Backend + 'static,
    I: InputSource + Send + 'static,
{
    let (tx, frames) = mpsc::sync_channel(4);

    let worker_handle = std::thread::Builder::new()
        .name("av-eval-batcher".into())
        .spawn(move || consume_input_source::<I>(source, tx))
        .expect("spawn batcher worker thread");

    InputBatchStream {
        device,
        worker_handle,
        frames,
    }
}

/// A stream of input frames.
pub struct InputBatchStream<B: Backend> {
    device: B::Device,
    worker_handle: std::thread::JoinHandle<anyhow::Result<()>>,
    frames: mpsc::Receiver<WorkerFrame>,
}

impl<B: Backend> InputBatchStream<B> {
    pub fn next_frame(&mut self) -> Option<FrameBatch<B>> {
        let frame = self.frames.recv().ok()?;
        let frame_tensor = Tensor::<B, 4>::from_data(frame.frame_tensor, &self.device)
            .permute([0, 3, 1, 2]);

        Some(FrameBatch { frame_tensor })
    }

    pub fn join_worker(self) -> anyhow::Result<()> {
        self.worker_handle.join().expect("worker thread panicked")
    }
}

fn consume_input_source<I>(
    source: I,
    frames: mpsc::SyncSender<WorkerFrame>,
) -> anyhow::Result<()>
where
    I: InputSource + Send + 'static,
{
    let width = source.width();
    let height = source.height();
    let bit_depth = source.bit_depth();
    let single_frame_size = frame_bytes(width, height, bit_depth)?;
    let mut frame_source = source.into_frame_source()?;

    loop {
        let mut frame = vec![0u8; single_frame_size];
        if !frame_source.step_next_frame(VideoFrameBuffer::new(&mut frame))? {
            break;
        }

        let frame = WorkerFrame {
            frame_tensor: TensorData::new(
                decode_frame_to_f32(&frame, bit_depth)?,
                [1, height, width, 3],
            ),
        };

        if frames.send(frame).is_err() {
            break;
        }
    }

    Ok(())
}

fn frame_bytes(
    width: usize,
    height: usize,
    bit_depth: BitDepth,
) -> anyhow::Result<usize> {
    width
        .checked_mul(height)
        .and_then(|n| n.checked_mul(3))
        .and_then(|n| n.checked_mul(bit_depth.bytes_per_sample()))
        .context("calculate packed RGB frame size")
}

fn decode_frame_to_f32(frame: &[u8], bit_depth: BitDepth) -> anyhow::Result<Vec<f32>> {
    Ok(match bit_depth {
        BitDepth::Eight => frame.iter().map(|value| *value as f32 / 255.0).collect(),
        BitDepth::Ten => frame
            .chunks_exact(2)
            .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 1023.0)
            .collect(),
    })
}

/// A decoded input frame to be processed on the selected backend.
pub struct FrameBatch<B: Backend> {
    pub frame_tensor: Tensor<B, 4>,
}

struct WorkerFrame {
    frame_tensor: TensorData,
}

impl<B: Backend> Clone for FrameBatch<B> {
    fn clone(&self) -> Self {
        Self {
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

        let mut batcher = create_batcher::<TestBackend, _>(Default::default(), source);

        let first_batch = batcher.next_frame().expect("expected first frame");
        assert_eq!(
            first_batch.frame_tensor.dims(),
            [1, 3, TEST_HEIGHT, TEST_WIDTH]
        );

        let first_data = first_batch
            .frame_tensor
            .clone()
            .permute([0, 2, 3, 1])
            .into_data();
        let first_values = first_data.to_vec::<f32>().expect("expected f32 output");
        assert_eq!(first_values.len(), TEST_WIDTH * TEST_HEIGHT * 3);
        assert_eq!(first_values[0], frame_1[0] as f32 / 255.0);

        let second_batch = batcher.next_frame().expect("expected second frame");
        assert_eq!(
            second_batch.frame_tensor.dims(),
            [1, 3, TEST_HEIGHT, TEST_WIDTH]
        );
        let second_data = second_batch
            .frame_tensor
            .clone()
            .permute([0, 2, 3, 1])
            .into_data();
        let second_values = second_data.to_vec::<f32>().expect("expected f32 output");
        assert_eq!(second_values[0], frame_2[0] as f32 / 255.0);

        let third_batch = batcher.next_frame().expect("expected third frame");
        assert_eq!(
            third_batch.frame_tensor.dims(),
            [1, 3, TEST_HEIGHT, TEST_WIDTH]
        );
        let third_data = third_batch.frame_tensor.permute([0, 2, 3, 1]).into_data();
        let third_values = third_data.to_vec::<f32>().expect("expected f32 output");
        assert_eq!(third_values[0], frame_3[0] as f32 / 255.0);

        assert!(batcher.next_frame().is_none());
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

        let mut batcher = create_batcher::<TestBackend, _>(Default::default(), source);

        let batch = batcher.next_frame().expect("expected frame");
        assert_eq!(batch.frame_tensor.dims(), [1, 3, TEST_HEIGHT, TEST_WIDTH]);

        let data = batch.frame_tensor.clone().permute([0, 2, 3, 1]).into_data();
        let values = data.to_vec::<f32>().expect("expected f32 output");
        assert_eq!(values[0], 1.0 / 1023.0);

        let second = batcher.next_frame().expect("expected second frame");
        let second_data = second.frame_tensor.permute([0, 2, 3, 1]).into_data();
        let second_values = second_data.to_vec::<f32>().expect("expected f32 output");
        assert_eq!(second_values[0], 2.0 / 1023.0);

        assert!(batcher.next_frame().is_none());
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
