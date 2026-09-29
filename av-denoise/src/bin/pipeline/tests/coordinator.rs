use indicatif::ProgressBar;

use super::{tiny_layout, tiny_planes};
use crate::pipeline::coordinator::{OutputMsg, emit_frames};
use crate::pipeline::stage::frame_permit_channel;
use crate::y4m_format::subsampling_to_y4m;

#[test]
fn emit_frames_errors_when_a_frame_index_is_lost() {
    let layout = tiny_layout();
    let (tx, rx) = crossbeam_channel::unbounded::<OutputMsg>();
    let planes = tiny_planes(layout);

    // Frame 1 is never sent, as if its index was lost somewhere
    // upstream, and every worker then disconnects. This used to make
    // `emit_frames` fall through to `Ok(())` with a truncated y4m.
    tx.send(OutputMsg {
        global_idx: 0,
        planes: planes.clone(),
    })
    .unwrap();
    tx.send(OutputMsg {
        global_idx: 2,
        planes,
    })
    .unwrap();
    drop(tx);

    let mut buf: Vec<u8> = Vec::new();
    let mut encoder = y4m::encode(
        layout.width as usize,
        layout.height as usize,
        y4m::Ratio::new(30, 1),
    )
    .with_colorspace(subsampling_to_y4m(layout.subsampling, layout.depth))
    .write_header(&mut buf)
    .expect("header write failed");

    let pb = ProgressBar::hidden();
    // Two permits stand in for the two staged frames, so returning
    // one has somewhere to go. A full permit channel would block the
    // return, which cannot happen in a real run.
    let (give, take) = frame_permit_channel(4);
    take.recv().expect("a permit is available");
    take.recv().expect("a permit is available");

    let err = emit_frames(&mut encoder, &rx, 3, &pb, &give).expect_err("expected a lost-frame error");

    let msg = err.to_string();
    assert!(
        msg.contains('1') && msg.contains('3'),
        "error should name frames written (1) vs expected (3): {msg}"
    );
}
