use av_denoise::ChromaSiting;
#[cfg(feature = "vulkan")]
use avtbx_bestsource::IndexCache;
use indicatif::ProgressBar;

#[cfg(feature = "vulkan")]
use super::{ffmpeg_clip, y4m_reader};
use super::{tiny_layout, tiny_planes};
#[cfg(feature = "vulkan")]
use crate::pipeline::coordinator::run_coordinator;
use crate::pipeline::coordinator::{OutputMsg, emit_frames};
#[cfg(feature = "vulkan")]
use crate::pipeline::source::{SourceInfo, open_file, open_y4m};
use crate::pipeline::stage::frame_permit_channel;
use crate::y4m_format::subsampling_to_y4m;

fn encoder(buffer: &mut Vec<u8>) -> y4m::Encoder<&mut Vec<u8>> {
    let layout = tiny_layout();
    let frame_rate = y4m::Ratio::new(30, 1);
    let colorspace = subsampling_to_y4m(layout.subsampling, layout.depth, ChromaSiting::Unknown);

    y4m::encode(layout.width as usize, layout.height as usize, frame_rate)
        .with_colorspace(colorspace)
        .write_header(buffer)
        .expect("header write failed")
}

fn send_frames(indices: &[u64]) -> crossbeam_channel::Receiver<OutputMsg> {
    let (output_tx, output_rx) = crossbeam_channel::unbounded::<OutputMsg>();
    let layout = tiny_layout();
    let planes = tiny_planes(layout);

    for &global_idx in indices {
        let message = OutputMsg {
            global_idx,
            planes: planes.clone(),
        };
        output_tx.send(message).expect("the receiver is alive");
    }

    output_rx
}

fn staged_count(count: Option<u64>) -> crossbeam_channel::Receiver<u64> {
    let (staged_tx, staged_rx) = crossbeam_channel::bounded(1);

    if let Some(count) = count {
        staged_tx.send(count).expect("the channel has room");
    }

    staged_rx
}

#[test]
fn emit_frames_errors_when_a_frame_index_is_lost() {
    let mut buffer = Vec::new();
    let mut encoder = encoder(&mut buffer);
    let outputs = send_frames(&[0, 2]);
    let staged = staged_count(Some(3));
    let (give, take) = frame_permit_channel(4);
    take.recv().expect("a permit is available");
    take.recv().expect("a permit is available");

    let progress = ProgressBar::hidden();

    let err = emit_frames(&mut encoder, &outputs, &staged, &progress, &give)
        .expect_err("expected a lost-frame error");
    let message = err.to_string();

    assert!(
        message.contains('1') && message.contains('3'),
        "error should name 1 written vs 3 staged: {message}"
    );
}

#[test]
fn emit_frames_finishes_when_every_staged_frame_is_written() {
    let mut buffer = Vec::new();
    let mut encoder = encoder(&mut buffer);
    let outputs = send_frames(&[1, 0, 2]);
    let staged = staged_count(Some(3));
    let (give, take) = frame_permit_channel(4);

    for _ in 0..3 {
        take.recv().expect("a permit is available");
    }

    let progress = ProgressBar::hidden();

    emit_frames(&mut encoder, &outputs, &staged, &progress, &give).expect("three of three frames written");

    assert_eq!(take.len(), 4, "every permit came back");
}

#[test]
fn emit_frames_leaves_the_error_to_a_dispatcher_that_sent_no_count() {
    let mut buffer = Vec::new();
    let mut encoder = encoder(&mut buffer);
    let outputs = send_frames(&[0]);
    let staged = staged_count(None);
    let (give, take) = frame_permit_channel(2);
    take.recv().expect("a permit is available");

    let progress = ProgressBar::hidden();

    emit_frames(&mut encoder, &outputs, &staged, &progress, &give)
        .expect("the dispatcher reports its own failure");
}

/// The header the coordinator writes for `info` when no frames arrive.
#[cfg(feature = "vulkan")]
fn written_header(info: SourceInfo) -> String {
    let (output_tx, output_rx) = crossbeam_channel::unbounded::<OutputMsg>();
    drop(output_tx);
    let staged = staged_count(Some(0));
    let (give, _take) = frame_permit_channel(2);
    let buffer = super::SharedBuffer::default();

    run_coordinator(info, output_rx, staged, false, give, buffer.clone()).expect("zero frames written");

    let written = buffer.0.lock().expect("buffer lock").clone();

    String::from_utf8_lossy(&written).into_owned()
}

#[cfg(feature = "vulkan")]
fn plain_c420_pipe() -> SourceInfo {
    let mut header = Vec::new();
    let frame_rate = y4m::Ratio::new(30, 1);
    y4m::encode(8, 8, frame_rate)
        .with_colorspace(y4m::Colorspace::C420)
        .write_header(&mut header)
        .expect("header write failed");

    let reader = y4m_reader(header);
    let opened = open_y4m(reader).expect("a 4:2:0 pipe opens");

    opened.info
}

#[cfg(feature = "vulkan")]
#[test]
fn the_coordinator_writes_the_source_siting_into_the_header() {
    let mut info = plain_c420_pipe();
    info.layout.chroma_siting = ChromaSiting::Left;

    let header_line = written_header(info);

    assert!(header_line.contains("C420mpeg2"), "got {header_line}");
}

#[cfg(feature = "vulkan")]
#[test]
fn a_file_header_carries_the_declared_frame_rate() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let extra = ["-r", "24000/1001"];
    let Some(path) = ffmpeg_clip(directory.path(), "ntsc.mkv", "64x48", "yuv420p", &extra) else {
        return;
    };

    let opened = open_file(&path, IndexCache::Disabled).expect("the clip opens");

    let header_line = written_header(opened.info);

    assert!(header_line.contains(" F24000:1001 "), "got {header_line}");
}
