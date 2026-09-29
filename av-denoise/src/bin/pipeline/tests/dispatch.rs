use std::io::{Cursor, Read};
use std::sync::Arc;
use std::thread;

use super::y4m_clip;
use crate::pipeline::convert::SourcePixel;
use crate::pipeline::decode::FrameMsg;
use crate::pipeline::dispatch;
use crate::pipeline::source::open_y4m;
use crate::pipeline::stage::SceneJob;

#[test]
fn dispatch_fails_on_a_forwarded_decode_error() {
    let bytes = y4m_clip(3);
    let reader: Box<dyn Read> = Box::new(Cursor::new(bytes));
    let mut opened = open_y4m(reader).expect("the clip opens");
    let (frames_tx, frames_rx) = crossbeam_channel::unbounded::<FrameMsg>();

    for _ in 0..3 {
        let frame = opened
            .decoder
            .read_video_frame::<u8>()
            .expect("the clip has 3 frames");
        let shared = Arc::new(frame);
        let decoded = u8::into_decoded(shared);
        frames_tx.send(Ok(decoded)).expect("the receiver is alive");
    }

    frames_tx
        .send(Err(anyhow::anyhow!("corrupt packet")))
        .expect("the receiver is alive");
    drop(frames_tx);

    let (job_tx, job_rx) = crossbeam_channel::bounded::<SceneJob>(0);
    let drain = thread::spawn(move || {
        while let Ok(job) = job_rx.recv() {
            for _ in job.frames.iter() {}
        }
    });

    let err = dispatch::<u8>(&frames_rx, &opened.info, &job_tx).expect_err("the error must surface");
    drop(job_tx);
    drain.join().expect("drain panicked");

    assert!(err.to_string().contains("corrupt packet"), "got {err}");
}
