use std::collections::BTreeSet;
use std::sync::Arc;
use std::thread;

use av_decoders::{Decoder, DecoderError};
use av_denoise::Depth;
use v_frame::frame::Frame;

use super::convert::{DecodedFrame, SourcePixel};
use super::source::{OpenedSource, SourceInfo};

/// Frames the decode thread may run ahead of the scene splitter.
pub const PREFETCH_FRAMES: usize = 4;

pub type FrameMsg = Result<DecodedFrame, anyhow::Error>;

/// Hands the decode thread what it needs to start streaming frames.
pub struct DecodeStart {
    pub permits: crossbeam_channel::Receiver<()>,
    pub frames: crossbeam_channel::Sender<FrameMsg>,
}

/// The thread that owns the decoder.
///
/// The decoder cannot move between threads, so it is opened on the thread that reads it.
pub struct DecodeThread {
    handle: thread::JoinHandle<()>,
    start: crossbeam_channel::Sender<DecodeStart>,
}

impl DecodeThread {
    /// Spawns the thread and waits for it to open the input.
    pub fn spawn<F>(opener: F) -> Result<(DecodeThread, SourceInfo), anyhow::Error>
    where
        F: FnOnce() -> Result<OpenedSource, anyhow::Error> + Send + 'static,
    {
        let (info_tx, info_rx) = crossbeam_channel::bounded(1);
        let (start_tx, start_rx) = crossbeam_channel::bounded::<DecodeStart>(1);
        let handle = thread::spawn(move || run_decode_thread(opener, info_tx, start_rx));

        let decode_thread = DecodeThread {
            handle,
            start: start_tx,
        };

        match info_rx.recv() {
            Ok(Ok(info)) => Ok((decode_thread, info)),
            Ok(Err(err)) => Err(err),
            Err(_) => {
                decode_thread.join()?;
                anyhow::bail!("the decoder thread stopped before reading the input header")
            },
        }
    }

    /// Starts streaming. Returns the frame channel's receiving end.
    pub fn start(&self, permits: crossbeam_channel::Receiver<()>) -> crossbeam_channel::Receiver<FrameMsg> {
        let (frames_tx, frames_rx) = crossbeam_channel::bounded(PREFETCH_FRAMES);
        let start = DecodeStart {
            permits,
            frames: frames_tx,
        };

        // A failed send means the thread already exited, which `join` reports.
        let _ = self.start.send(start);

        frames_rx
    }

    /// Waits for the thread, turning a panic into an error.
    pub fn join(self) -> Result<(), anyhow::Error> {
        drop(self.start);

        self.handle
            .join()
            .map_err(|panic| anyhow::anyhow!("decoder thread panicked: {panic:?}"))
    }
}

fn run_decode_thread<F>(
    opener: F,
    info_tx: crossbeam_channel::Sender<Result<SourceInfo, anyhow::Error>>,
    start_rx: crossbeam_channel::Receiver<DecodeStart>,
) where
    F: FnOnce() -> Result<OpenedSource, anyhow::Error>,
{
    let opened = match opener() {
        Ok(opened) => opened,
        Err(err) => {
            let _ = info_tx.send(Err(err));
            return;
        },
    };

    let info = opened.info.clone();
    let _ = info_tx.send(Ok(info));

    // The caller dropping its handle before starting means the run gave up.
    let Ok(start) = start_rx.recv() else {
        return;
    };

    let OpenedSource {
        decoder,
        phantom,
        info,
    } = opened;
    let result = match info.layout.depth {
        Depth::Eight => pump_decoder::<u8>(decoder, &phantom, &start),
        Depth::Ten | Depth::Twelve => pump_decoder::<u16>(decoder, &phantom, &start),
    };

    if let Err(err) = result {
        let _ = start.frames.send(Err(err));
    }
}

fn pump_decoder<T: SourcePixel>(
    mut decoder: Decoder,
    phantom: &BTreeSet<usize>,
    start: &DecodeStart,
) -> Result<(), anyhow::Error> {
    let frames = std::iter::from_fn(move || read_frame::<T>(&mut decoder));

    pump_frames(frames, phantom, &start.permits, &start.frames)
}

/// Reads the next frame, ending the iterator at end of file.
fn read_frame<T: SourcePixel>(decoder: &mut Decoder) -> Option<Result<Frame<T>, anyhow::Error>> {
    match decoder.read_video_frame::<T>() {
        Ok(frame) => Some(Ok(frame)),
        Err(DecoderError::EndOfFile) => None,
        Err(err) => Some(Err(err.into())),
    }
}

/// Forwards decoded frames, skipping phantom ones and taking a permit per frame sent.
///
/// Returns quietly when the receiving end hangs up.
pub fn pump_frames<T, I>(
    frames: I,
    phantom: &BTreeSet<usize>,
    permits: &crossbeam_channel::Receiver<()>,
    out: &crossbeam_channel::Sender<FrameMsg>,
) -> Result<(), anyhow::Error>
where
    T: SourcePixel,
    I: Iterator<Item = Result<Frame<T>, anyhow::Error>>,
{
    // Every frame is read, phantom or not, because the decoder walks the file in order.
    for (file_index, frame) in frames.enumerate() {
        let frame = frame?;

        if phantom.contains(&file_index) {
            continue;
        }

        permits
            .recv()
            .map_err(|_| anyhow::anyhow!("the coordinator stopped before the stream finished"))?;

        let shared = Arc::new(frame);
        let decoded = T::into_decoded(shared);

        if out.send(Ok(decoded)).is_err() {
            return Ok(());
        }
    }

    Ok(())
}
