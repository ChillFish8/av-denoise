use std::collections::BTreeMap;
use std::io::stdout;
use std::thread;
use std::time::Duration;

use av_decoders::Rational32;
use av_denoise::{FrameLayout, Planes};
use indicatif::ProgressBar;
use y4m::Frame as Y4mFrame;

use crate::progress::{self, denoise_progress_bar};
use crate::y4m_format::subsampling_to_y4m;

pub struct OutputMsg {
    pub global_idx: u64,
    pub planes: Planes,
}

pub fn spawn_coordinator(
    layout: FrameLayout,
    framerate: Rational32,
    rx: crossbeam_channel::Receiver<OutputMsg>,
    total_frames: usize,
    visible: bool,
    permits: crossbeam_channel::Sender<()>,
) -> thread::JoinHandle<Result<(), anyhow::Error>> {
    thread::spawn(move || run_coordinator(layout, framerate, rx, total_frames, visible, permits))
}

pub fn run_coordinator(
    layout: FrameLayout,
    framerate: Rational32,
    rx: crossbeam_channel::Receiver<OutputMsg>,
    total_frames: usize,
    visible: bool,
    permits: crossbeam_channel::Sender<()>,
) -> Result<(), anyhow::Error> {
    let stdout = stdout();
    let lock = stdout.lock();

    // No `XCOLORRANGE=` tag is emitted here. `av_decoders::VideoDetails`
    // doesn't surface the source's color range for any of its backends
    // (ffms2 included), so there's nothing to forward.
    let mut encoder = y4m::encode(
        layout.width as usize,
        layout.height as usize,
        y4m::Ratio::new((*framerate.numer()) as usize, (*framerate.denom()) as usize),
    )
    .with_colorspace(subsampling_to_y4m(layout.subsampling, layout.depth))
    .write_header(lock)?;

    // Counts frames written to the output, which lags the frames read by
    // the depth of the worker pipelines. Emitted frames are the honest
    // measure of progress, because the count stalls whenever whatever
    // consumes our stdout stops reading.
    let pb = denoise_progress_bar(total_frames, visible);

    // The first frame only lands once a worker has compiled its
    // kernels, which takes seconds. A steady tick draws the bar right
    // away and keeps its elapsed time moving until then.
    pb.enable_steady_tick(Duration::from_millis(250));

    let result = emit_frames(&mut encoder, &rx, total_frames as u64, &pb, &permits);

    progress::finish(&pb);

    result
}

/// Reorders worker output by frame index and writes it out, updating
/// `pb` as frames are emitted.
///
/// Returns once every frame has been written. If the workers all
/// disconnect before `total` frames have landed this errors, naming how
/// many were written and how many were expected.
pub fn emit_frames<W: std::io::Write>(
    encoder: &mut y4m::Encoder<W>,
    rx: &crossbeam_channel::Receiver<OutputMsg>,
    total: u64,
    pb: &ProgressBar,
    permits: &crossbeam_channel::Sender<()>,
) -> Result<(), anyhow::Error> {
    let mut pending: BTreeMap<u64, Planes> = BTreeMap::new();
    let mut next_emit: u64 = 0;

    while next_emit < total {
        let msg = match rx.recv() {
            Ok(m) => m,
            Err(_) => break,
        };

        pending.insert(msg.global_idx, msg.planes);

        while let Some(planes) = pending.remove(&next_emit) {
            let frame = Y4mFrame::new([&planes.y, &planes.u, &planes.v], None);
            encoder.write_frame(&frame)?;
            next_emit += 1;

            // Returning the permit is what lets the decoder run further
            // ahead. The send never blocks, because permits held plus
            // permits waiting is always the channel's capacity. The
            // result is discarded because it fails once the dispatcher
            // has already errored out and dropped its receiver.
            let _ = permits.send(());
        }

        pb.set_position(next_emit);
    }

    if next_emit != total {
        anyhow::bail!(
            "wrote {next_emit} frames but expected {total}. Every worker disconnected \
             before the stream finished, so a frame index was likely lost"
        );
    }

    Ok(())
}
