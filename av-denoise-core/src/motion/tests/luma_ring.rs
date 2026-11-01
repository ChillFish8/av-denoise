use crate::engine::align::StorageAlign;
use crate::motion::{LumaMotionRing, MotionCompensationMode, MotionCtx, Subsampling};
use crate::tune::tests::runtime::test_client;

const RADIUS: u32 = 2;
const FRAME: u32 = 3;
const SLOTS: u32 = 2 * RADIUS + 1;

fn ring() -> LumaMotionRing {
    let client = test_client();
    let align = StorageAlign::from_client(&client);
    let mode = MotionCompensationMode::mvtools_default();
    let chroma = MotionCtx::new(mode, 64, 48, align).expect("a searching mode");
    let subsampling = Subsampling {
        shift_x: 1,
        shift_y: 1,
    };

    LumaMotionRing::new(&client, &chroma, RADIUS, subsampling)
}

#[test]
fn a_written_frame_reads_back() {
    let ring = ring();
    ring.window_slot(FRAME);

    assert!(ring.read_window(FRAME).is_some());
}

#[test]
fn a_frame_sharing_the_slot_reads_nothing() {
    let ring = ring();
    ring.window_slot(FRAME);

    assert!(ring.read_window(FRAME + SLOTS).is_none());
}

#[test]
fn clear_empties_every_slot() {
    let ring = ring();
    ring.window_slot(FRAME);
    ring.clear();

    assert!(ring.read_window(FRAME).is_none());
}
