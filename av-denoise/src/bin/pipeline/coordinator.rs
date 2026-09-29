use std::collections::BTreeMap;
use std::thread;
use std::time::Duration;

use av_denoise::Planes;
use indicatif::ProgressBar;
use y4m::Frame as Y4mFrame;

use super::source::SourceInfo;
use crate::progress::{self, denoise_progress_bar};
use crate::y4m_format::subsampling_to_y4m;

pub struct OutputMsg {
    pub global_idx: u64,
    pub planes: Planes,
}

pub fn spawn_coordinator<W: std::io::Write + Send + 'static>(
    info: SourceInfo,
    rx: crossbeam_channel::Receiver<OutputMsg>,
    staged: crossbeam_channel::Receiver<u64>,
    visible: bool,
    permits: crossbeam_channel::Sender<()>,
    output: W,
) -> thread::JoinHandle<Result<(), anyhow::Error>> {
    thread::spawn(move || run_coordinator(info, rx, staged, visible, permits, output))
}

pub fn run_coordinator<W: std::io::Write>(
    info: SourceInfo,
    rx: crossbeam_channel::Receiver<OutputMsg>,
    staged: crossbeam_channel::Receiver<u64>,
    visible: bool,
    permits: crossbeam_channel::Sender<()>,
    output: W,
) -> Result<(), anyhow::Error> {
    let framerate = info.details.frame_rate;
    let ratio = y4m::Ratio::new(*framerate.numer() as usize, *framerate.denom() as usize);
    let colorspace = subsampling_to_y4m(info.layout.subsampling, info.layout.depth);

    let mut builder = y4m::encode(info.layout.width as usize, info.layout.height as usize, ratio)
        .with_colorspace(colorspace);

    if let Some(pixel_aspect) = info.pixel_aspect {
        builder = builder.with_pixel_aspect(pixel_aspect);
    }

    // Forwards the source's `X` params, `XCOLORRANGE=` being the common one.
    for extension in info.vendor_extensions {
        builder = builder.append_vendor_extension(extension);
    }

    let mut encoder = builder.write_header(output)?;

    // Counts frames written to the output, which lags the frames read by
    // the depth of the worker pipelines. Emitted frames are the honest
    // measure of progress, because the count stalls whenever whatever
    // consumes our stdout stops reading.
    let pb = denoise_progress_bar(info.estimated_frames, visible);

    // The first frame only lands once a worker has compiled its
    // kernels, which takes seconds. A steady tick draws the bar right
    // away and keeps its elapsed time moving until then.
    pb.enable_steady_tick(Duration::from_millis(250));

    let result = emit_frames(&mut encoder, &rx, &staged, &pb, &permits);

    progress::finish(&pb);

    result
}

/// Reorders worker output by frame index and writes it out, updating `pb` as frames land.
///
/// Runs until every worker has hung up, then checks the frames written against the count the
/// dispatcher staged. No count means the dispatcher failed and reports its own error.
pub fn emit_frames<W: std::io::Write>(
    encoder: &mut y4m::Encoder<W>,
    rx: &crossbeam_channel::Receiver<OutputMsg>,
    staged: &crossbeam_channel::Receiver<u64>,
    pb: &ProgressBar,
    permits: &crossbeam_channel::Sender<()>,
) -> Result<(), anyhow::Error> {
    let mut pending: BTreeMap<u64, Planes> = BTreeMap::new();
    let mut next_emit: u64 = 0;

    while let Ok(msg) = rx.recv() {
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

    let Ok(total) = staged.recv() else {
        return Ok(());
    };

    pb.set_length(total);

    if next_emit != total {
        anyhow::bail!(
            "wrote {next_emit} frames but expected {total}. Every worker disconnected \
             before the stream finished, so a frame index was likely lost"
        );
    }

    Ok(())
}
