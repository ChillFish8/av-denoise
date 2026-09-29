use std::collections::BTreeSet;
use std::io::{Cursor, Read};
use std::num::{NonZeroU8, NonZeroUsize};

use v_frame::chroma::ChromaSubsampling;
use v_frame::frame::{Frame, FrameBuilder};

use super::y4m_clip;
use crate::pipeline::convert::SourcePixel;
use crate::pipeline::decode::{DecodeThread, FrameMsg, pump_frames};
use crate::pipeline::source::open_y4m;
use crate::pipeline::stage::frame_permit_channel;

fn tiny_frame() -> Frame<u8> {
    FrameBuilder::new(
        NonZeroUsize::new(2).expect("width is non-zero"),
        NonZeroUsize::new(2).expect("height is non-zero"),
        ChromaSubsampling::Yuv420,
        NonZeroU8::new(8).expect("depth is non-zero"),
    )
    .build()
    .expect("a 2x2 8-bit frame builds")
}

fn frames(count: usize) -> impl Iterator<Item = Result<Frame<u8>, anyhow::Error>> {
    (0..count).map(|_| Ok(tiny_frame()))
}

#[test]
fn phantom_frames_are_skipped_and_take_no_permit() {
    let (_give, take) = frame_permit_channel(8);
    let (out_tx, out_rx) = crossbeam_channel::unbounded::<FrameMsg>();
    let phantom = BTreeSet::from([1, 3]);

    let decoded = frames(6);

    pump_frames(decoded, &phantom, &take, &out_tx).expect("pumping should succeed");
    drop(out_tx);

    assert_eq!(out_rx.iter().count(), 4);
    assert_eq!(take.len(), 4, "only the four sent frames hold a permit");
}

#[test]
fn a_read_error_stops_pumping_after_earlier_frames() {
    let (_give, take) = frame_permit_channel(8);
    let (out_tx, out_rx) = crossbeam_channel::unbounded::<FrameMsg>();
    let corrupt = Err(anyhow::anyhow!("corrupt packet"));
    let corrupt_frame = std::iter::once(corrupt);
    let leading = frames(2);
    let failing = leading.chain(corrupt_frame);

    let phantom = BTreeSet::new();

    let result = pump_frames(failing, &phantom, &take, &out_tx);

    assert!(result.is_err());
    assert_eq!(out_rx.len(), 2, "frames before the error are still sent");
}

#[test]
fn pumping_stops_quietly_when_the_splitter_hangs_up() {
    let (_give, take) = frame_permit_channel(8);
    let (out_tx, out_rx) = crossbeam_channel::bounded::<FrameMsg>(1);
    drop(out_rx);

    let decoded = frames(4);
    let phantom = BTreeSet::new();

    pump_frames(decoded, &phantom, &take, &out_tx).expect("a closed channel is not an error");
}

#[test]
fn pumping_fails_when_the_coordinator_drops_every_permit() {
    let (give, take) = frame_permit_channel(1);
    drop(give);
    let (out_tx, _out_rx) = crossbeam_channel::unbounded::<FrameMsg>();

    let decoded = frames(4);
    let phantom = BTreeSet::new();

    let result = pump_frames(decoded, &phantom, &take, &out_tx);

    assert!(result.is_err(), "a second frame cannot get a permit");
}

#[test]
fn the_thread_reports_the_source_then_streams_every_frame() {
    let bytes = y4m_clip(3);
    let (thread, info) = DecodeThread::spawn(move || {
        let reader: Box<dyn Read> = Box::new(Cursor::new(bytes));
        open_y4m(reader)
    })
    .expect("the clip opens");

    assert_eq!(info.layout.width, 4);

    let (_give, take) = frame_permit_channel(8);
    let frames = thread.start(take);
    let received: Vec<_> = frames.iter().collect();

    thread.join().expect("the thread exits cleanly");

    assert_eq!(received.len(), 3);

    for message in received {
        let decoded = message.expect("every frame decodes");
        assert!(u8::from_decoded(decoded).is_some());
    }
}

#[test]
fn an_open_failure_is_returned_from_spawn() {
    let result = DecodeThread::spawn(|| Err(anyhow::anyhow!("no such file")));

    let err = match result {
        Ok(_) => panic!("the open failed, so spawn must fail"),
        Err(err) => err,
    };

    assert!(err.to_string().contains("no such file"));
}

#[test]
fn dropping_an_unstarted_thread_lets_it_exit() {
    let bytes = y4m_clip(1);
    let (thread, _info) = DecodeThread::spawn(move || {
        let reader: Box<dyn Read> = Box::new(Cursor::new(bytes));
        open_y4m(reader)
    })
    .expect("the clip opens");

    thread
        .join()
        .expect("an unstarted thread exits once its start channel closes");
}

#[test]
fn a_panicking_opener_is_reported_from_spawn() {
    let result = DecodeThread::spawn(|| panic!("opener blew up"));

    let err = match result {
        Ok(_) => panic!("the opener panicked, so spawn must fail"),
        Err(err) => err,
    };

    assert!(err.to_string().contains("panicked"));
}
