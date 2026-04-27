use std::sync::mpsc;

use anyhow::{Context, ensure};
use cubecl::bytes::Bytes;
use cubecl::server::Allocation;

use crate::source::{FrameSource, InputSource, VideoFrameBuffer};

/// The input batcher takes an input source and produces
/// batches of frames loaded onto the accelerator arenas.
///
/// Input sources are consumed in a worker thread to ensure
/// proper performance and the accelerator is saturated.
pub fn create_batcher<R, I>(
    client: cubecl::client::ComputeClient<R>,
    source: I,
    batch_size: usize,
) -> InputBatchStream
where
    R: cubecl::Runtime + 'static,
    I: InputSource + Send + 'static,
{
    let (tx, batches) = mpsc::sync_channel(4);

    let worker_handle = std::thread::Builder::new()
        .name("av-eval-batcher".into())
        .spawn(move || consume_input_source(client, source, batch_size, tx))
        .expect("spawn batcher worker thread");

    InputBatchStream {
        worker_handle,
        batches,
    }
}

/// A stream of input frames which are pre-uploaded to the accelerator arena
/// and bundled into a batch of a given size.
pub struct InputBatchStream {
    /// The worker thread handle.
    worker_handle: std::thread::JoinHandle<anyhow::Result<()>>,
    /// Incoming batches read for processing.
    batches: mpsc::Receiver<FrameBatch>,
}

impl InputBatchStream {
    /// Receive a new batch of frames from the batcher.
    ///
    /// Waits for a new batch if one is not already available.
    pub fn next_batch(&mut self) -> Option<FrameBatch> {
        self.batches.recv().ok()
    }

    /// Join the worker thread, returning the result.
    pub fn join_worker(self) -> anyhow::Result<()> {
        self.worker_handle.join().expect("worker thread panicked")
    }
}

fn consume_input_source<R, I>(
    client: cubecl::client::ComputeClient<R>,
    source: I,
    batch_size: usize,
    batches: mpsc::SyncSender<FrameBatch>,
) -> anyhow::Result<()>
where
    R: cubecl::Runtime + 'static,
    I: InputSource + Send + 'static,
{
    let width = source.width();
    let height = source.height();
    let bit_depth = source.bit_depth();
    let mut frame_source = source.into_frame_source()?;
    let bytes_per_sample = bit_depth.bytes_per_sample();

    ensure!(
        width.is_multiple_of(2) && height.is_multiple_of(2),
        "YUV420 requires even frame width and height"
    );

    let luma_stride = width
        .checked_mul(height)
        .and_then(|r| r.checked_mul(bytes_per_sample))
        .context("calculate luma plane stride")?;

    let chroma_stride = (width / 2)
        .checked_mul(height / 2)
        .and_then(|r| r.checked_mul(bytes_per_sample))
        .context("calculate chroma plane stride")?;

    let single_frame_size = chroma_stride
        .checked_mul(2)
        .and_then(|r| luma_stride.checked_add(r))
        .context("calculate single YUV frame size")?;

    let single_frame_samples = chroma_stride
        .checked_mul(2)
        .and_then(|r| luma_stride.checked_add(r))
        .and_then(|r| r.checked_div(bytes_per_sample))
        .context("calculate single packed YUV420 frame sample count")?;

    let batch_bytes = single_frame_size
        .checked_mul(batch_size)
        .context("calculate batch memory allocation size")?;

    let mut has_frame = true;
    while has_frame {
        let mut frame_batch: Vec<u8> = vec![0; batch_bytes];

        let mut buffer_offset = 0;
        let mut num_frames = 0;
        for _ in 0..batch_size {
            let frame_end = buffer_offset + single_frame_size;
            let frame_buffer = VideoFrameBuffer::new(
                &mut frame_batch[buffer_offset..frame_end],
                luma_stride,
                chroma_stride,
            );

            has_frame = frame_source.step_next_frame(frame_buffer)?;
            if !has_frame {
                break;
            }

            num_frames += 1;
            buffer_offset += single_frame_size;
        }

        if num_frames == 0 {
            break;
        }

        frame_batch.truncate(buffer_offset);

        let frame_batch_bytes = Bytes::from_bytes_vec(frame_batch);

        let frame_tensor = client.create_tensor(
            frame_batch_bytes,
            &[num_frames, single_frame_samples],
            bytes_per_sample,
        );

        let batch = FrameBatch {
            size: num_frames,
            frame_tensor,
        };

        if batches.send(batch).is_err() {
            break;
        }
    }

    Ok(())
}

/// A batch of frames to be processed on the accelerators.
pub struct FrameBatch {
    /// The size of the batch.
    pub size: usize,
    /// The allocated tensor which holds a batch of YUV420 frames.
    pub frame_tensor: Allocation,
}

impl Clone for FrameBatch {
    fn clone(&self) -> Self {
        Self {
            size: self.size,
            frame_tensor: Allocation {
                handle: self.frame_tensor.handle.clone(),
                strides: self.frame_tensor.strides.clone(),
            },
        }
    }
}

#[cfg(all(test, feature = "cpu"))]
mod tests {
    use cubecl::Runtime;

    use super::*;
    use crate::source::BitDepth;
    use crate::source::mock::MockInput;

    #[test]
    fn batches_multiple_8bit_yuv420_frames() {
        let frame_1 = vec![1, 2, 3, 4, 10, 11];
        let frame_2 = vec![5, 6, 7, 8, 12, 13];
        let frame_3 = vec![9, 10, 11, 12, 14, 15];
        let source = MockInput::new(
            2,
            2,
            BitDepth::Eight,
            vec![frame_1.clone(), frame_2.clone(), frame_3.clone()],
        )
        .expect("mock input should build");
        let client = cpu_client();

        let mut batcher = create_batcher(client.clone(), source, 2);

        let first_batch = batcher.next_batch().expect("expected first batch");
        assert_eq!(first_batch.size, 2);
        assert_eq!(first_batch.frame_tensor.strides.len(), 2);
        assert_eq!(
            read_batch_bytes(&client, &first_batch, BitDepth::Eight),
            [frame_1.clone(), frame_2.clone()].concat()
        );

        let second_batch = batcher.next_batch().expect("expected trailing batch");
        assert_eq!(second_batch.size, 1);
        assert_eq!(second_batch.frame_tensor.strides.len(), 2);
        assert_eq!(
            read_batch_bytes(&client, &second_batch, BitDepth::Eight),
            frame_3
        );

        assert!(batcher.next_batch().is_none());
        batcher.join_worker().expect("worker should exit cleanly");
    }

    #[test]
    fn batches_10bit_yuv420_frames() {
        let frame_1 = vec![1, 0, 2, 0, 3, 0, 4, 0, 9, 0, 11, 0];
        let frame_2 = vec![5, 0, 6, 0, 7, 0, 8, 0, 13, 0, 15, 0];
        let source =
            MockInput::new(2, 2, BitDepth::Ten, vec![frame_1.clone(), frame_2.clone()])
                .expect("mock input should build");
        let client = cpu_client();

        let mut batcher = create_batcher(client.clone(), source, 2);

        let batch = batcher.next_batch().expect("expected batch");
        assert_eq!(batch.size, 2);
        assert_eq!(batch.frame_tensor.strides.len(), 2);
        assert_eq!(
            read_batch_bytes(&client, &batch, BitDepth::Ten),
            [frame_1, frame_2].concat()
        );

        assert!(batcher.next_batch().is_none());
        batcher.join_worker().expect("worker should exit cleanly");
    }

    #[test]
    fn errors_on_odd_dimensions() {
        let source =
            MockInput::new(3, 2, BitDepth::Eight, vec![vec![1, 2, 3, 4, 5, 6, 7, 8]])
                .expect("mock input should build");
        let client = cpu_client();

        let mut batcher = create_batcher(client, source, 1);

        assert!(batcher.next_batch().is_none());

        let err = batcher
            .join_worker()
            .expect_err("worker should reject odd dimensions");
        assert!(
            err.to_string()
                .contains("YUV420 requires even frame width and height")
        );
    }

    fn cpu_client() -> cubecl::client::ComputeClient<cubecl::cpu::CpuRuntime> {
        let device = <cubecl::cpu::CpuRuntime as Runtime>::Device::default();
        cubecl::cpu::CpuRuntime::client(&device)
    }

    fn read_batch_bytes(
        client: &cubecl::client::ComputeClient<cubecl::cpu::CpuRuntime>,
        batch: &FrameBatch,
        bit_depth: BitDepth,
    ) -> Vec<u8> {
        let shape = [batch.size, samples_per_frame(2, 2, bit_depth)];
        let descriptor = batch.frame_tensor.handle.copy_descriptor(
            &shape,
            &batch.frame_tensor.strides,
            bit_depth.bytes_per_sample(),
        );

        client.read_one_tensor(descriptor).to_vec()
    }

    fn samples_per_frame(width: usize, height: usize, bit_depth: BitDepth) -> usize {
        let bytes_per_sample = bit_depth.bytes_per_sample();
        let luma_bytes = width * height * bytes_per_sample;
        let chroma_bytes = (width / 2) * (height / 2) * bytes_per_sample;
        (luma_bytes + chroma_bytes * 2) / bytes_per_sample
    }
}
