use std::io::{Cursor, Read};
use std::sync::Arc;

use av_decoders::{Decoder, DecoderImpl};
use av_scenechange::{DetectionOptions, detect_scene_changes};

use crate::pipeline::scenes::{Decided, LOOKAHEAD_DISTANCE, SceneSplitter};

const WIDTH: usize = 128;
const HEIGHT: usize = 128;

/// A tiny xorshift so the clip is the same on every run.
fn pattern(seed: u32, len: usize) -> Vec<u8> {
    let mut state = seed.max(1);
    let mut out = Vec::with_capacity(len);

    for _ in 0..len {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        out.push((state & 0xff) as u8);
    }

    out
}

/// Builds an 8-bit 4:2:0 y4m clip where each entry of `scene_lengths` is one scene.
fn clip(scene_lengths: &[usize]) -> Vec<u8> {
    let mut bytes = Vec::new();
    let mut encoder = y4m::encode(WIDTH, HEIGHT, y4m::Ratio::new(24, 1))
        .with_colorspace(y4m::Colorspace::C420)
        .write_header(&mut bytes)
        .expect("header should write");

    let chroma = vec![128u8; (WIDTH / 2) * (HEIGHT / 2)];

    for (scene, &length) in scene_lengths.iter().enumerate() {
        let base = pattern(scene as u32 + 1, WIDTH * HEIGHT);

        for offset in 0..length {
            let luma: Vec<u8> = base
                .iter()
                .map(|&sample| sample.saturating_add(offset as u8))
                .collect();
            let frame = y4m::Frame::new([&luma, &chroma, &chroma], None);

            encoder.write_frame(&frame).expect("frame should write");
        }
    }

    bytes
}

fn decoder_over(bytes: &[u8]) -> Decoder {
    let reader: Box<dyn Read> = Box::new(Cursor::new(bytes.to_vec()));
    let y4m_decoder = y4m::decode(reader).expect("y4m header should parse");

    Decoder::from_decoder_impl(DecoderImpl::Y4m(y4m_decoder)).expect("decoder should open")
}

/// Scene starts and frame count from the whole-clip reference pass.
fn reference(bytes: &[u8]) -> (Vec<usize>, usize) {
    let mut decoder = decoder_over(bytes);
    let options = DetectionOptions::default();
    let results = detect_scene_changes::<u8>(&mut decoder, options, None, None)
        .expect("reference detection should run");

    (results.scene_changes, results.frame_count)
}

/// Every decided frame, in release order.
fn split(bytes: &[u8]) -> Vec<Decided<u8>> {
    let mut decoder = decoder_over(bytes);
    let details = *decoder.get_video_details();
    let mut splitter = SceneSplitter::<u8>::new(&details);
    let mut decided = Vec::new();

    while let Ok(frame) = decoder.read_video_frame::<u8>() {
        let released = splitter.push(Arc::new(frame));
        decided.extend(released);
    }

    let tail = splitter.finish();
    decided.extend(tail);
    decided
}

fn scene_starts(decided: &[Decided<u8>]) -> Vec<usize> {
    decided
        .iter()
        .enumerate()
        .filter(|(_, frame)| frame.starts_scene)
        .map(|(index, _)| index)
        .collect()
}

#[test]
fn cuts_match_the_whole_clip_pass_over_several_scenes() {
    let bytes = clip(&[30, 24, 40, 12]);
    let (expected_starts, expected_count) = reference(&bytes);

    assert!(
        expected_starts.len() > 1,
        "the synthetic clip must produce at least one cut, got {expected_starts:?}",
    );

    let decided = split(&bytes);

    assert_eq!(decided.len(), expected_count);
    assert_eq!(scene_starts(&decided), expected_starts);
}

#[test]
fn cuts_match_the_whole_clip_pass_at_the_look_ahead_length() {
    let bytes = clip(&[LOOKAHEAD_DISTANCE + 1]);
    let (expected_starts, expected_count) = reference(&bytes);
    let decided = split(&bytes);

    assert_eq!(decided.len(), expected_count);
    assert_eq!(scene_starts(&decided), expected_starts);
}

#[test]
fn cuts_match_the_whole_clip_pass_on_two_frames() {
    let bytes = clip(&[1, 1]);
    let (expected_starts, expected_count) = reference(&bytes);
    let decided = split(&bytes);

    assert_eq!(decided.len(), expected_count);
    assert_eq!(scene_starts(&decided), expected_starts);
}

#[test]
fn a_single_frame_is_released_as_a_scene_start() {
    let bytes = clip(&[1]);
    let decided = split(&bytes);

    assert_eq!(decided.len(), 1);
    assert!(decided[0].starts_scene);
}

#[test]
fn every_frame_is_released_once_in_order() {
    let bytes = clip(&[20, 20]);
    let mut decoder = decoder_over(&bytes);
    let details = *decoder.get_video_details();
    let mut splitter = SceneSplitter::<u8>::new(&details);
    let mut pushed = Vec::new();
    let mut released = Vec::new();

    while let Ok(frame) = decoder.read_video_frame::<u8>() {
        let frame = Arc::new(frame);
        pushed.push(Arc::clone(&frame));

        let decided = splitter.push(frame);
        released.extend(decided);
    }

    let tail = splitter.finish();
    released.extend(tail);

    assert_eq!(released.len(), pushed.len());

    for (original, decided) in pushed.iter().zip(&released) {
        assert!(Arc::ptr_eq(original, &decided.frame));
    }
}

#[test]
fn the_first_frame_always_starts_a_scene() {
    let bytes = clip(&[10]);
    let decided = split(&bytes);

    assert!(decided[0].starts_scene);
}

#[test]
fn nothing_is_released_before_the_look_ahead_fills() {
    let bytes = clip(&[20]);
    let mut decoder = decoder_over(&bytes);
    let details = *decoder.get_video_details();
    let mut splitter = SceneSplitter::<u8>::new(&details);

    for pushed in 0..LOOKAHEAD_DISTANCE {
        let frame = decoder.read_video_frame::<u8>().expect("the clip has 20 frames");
        let released = splitter.push(Arc::new(frame));

        assert!(
            released.is_empty(),
            "frame {pushed} released before the window filled"
        );
    }

    let frame = decoder.read_video_frame::<u8>().expect("the clip has 20 frames");
    let released = splitter.push(Arc::new(frame));

    assert_eq!(
        released.len(),
        1,
        "frame 0 is released once frame {LOOKAHEAD_DISTANCE} arrives"
    );
}

#[test]
fn the_look_ahead_matches_the_detector_default() {
    let options = DetectionOptions::default();

    assert_eq!(LOOKAHEAD_DISTANCE, options.lookahead_distance);
}
