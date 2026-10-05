use indicatif::ProgressBar;

use super::{tiny_layout, tiny_planes};
use crate::pipeline::coordinator::{OutputMsg, emit_frames};
use crate::pipeline::stage::frame_permit_channel;
use crate::y4m_format::subsampling_to_y4m;

fn encoder(buffer: &mut Vec<u8>) -> y4m::Encoder<&mut Vec<u8>> {
    let layout = tiny_layout();
    let frame_rate = y4m::Ratio::new(30, 1);
    let colorspace = subsampling_to_y4m(layout.subsampling, layout.depth);

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
