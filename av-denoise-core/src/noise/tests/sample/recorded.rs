use super::{Plane, moved_blocks, moving_block, read_back, state_for};
use crate::noise::NOISE_CHANNELS;
use crate::noise::reading::READING_WORDS;
use crate::noise::state::ReadingSettings;
use crate::noise::temporal::RHO_SIGMA_GATE;
use crate::noise::tests::records::{BlockSpec, QuarterSpec, hashed_partials};
use crate::tune::tests::runtime::test_client;

/// A luma frame with moved texture, outliers, quiet blocks, moving blocks and curve quarters.
fn recorded_luma_blocks() -> Vec<BlockSpec> {
    let plane = Plane::luma(256, 192);
    let mut blocks = moved_blocks(plane, 101);

    for (index, block) in blocks.iter_mut().enumerate() {
        let luma = 0.2 + 0.2 * (index % 4) as f32;
        let quarter = QuarterSpec {
            mean: 0.0,
            sigma: block.sigma[0],
            luma,
            flatness: 0.0,
            luma_min: luma - 0.01,
            luma_max: luma + 0.01,
        };
        block.quarters = [quarter; 4];
    }

    for index in [7, 33, 120] {
        blocks[index].sigma = [40.0 / 255.0; NOISE_CHANNELS as usize];
    }

    for index in [2, 58] {
        blocks[index].sigma = [0.5 * RHO_SIGMA_GATE; NOISE_CHANNELS as usize];
    }

    for index in [11, 90, 150, 151] {
        blocks[index] = moving_block(3.0 / 255.0);
    }

    blocks
}

/// The reading words of [recorded_luma_blocks], recorded before each channel had its own checks.
///
/// The fourth word is channel 1's sample flag, which came after the recording and reads 0.
const RECORDED_LUMA_WORDS: [u32; READING_WORDS as usize] = [
    0x3adaa02c, 0x00000000, 0x3f800000, 0x00000000, 0x3c3c3903, 0x00000000, 0x3c1cd525, 0x00000000,
    0x3e8cd626, 0x3f73ffff, 0x43370000, 0x3f800000, 0x3c47f521, 0x40800000, 0x3f6b3254, 0x3f6b3254,
    0x3f6b3254, 0x3f6b3254, 0x3f72067f, 0x3f78daab, 0x3f7faed7, 0x3f7b0aa1, 0x3f76666a, 0x3f71c234,
    0x3f808b01, 0x3f8834e8, 0x3f8fdecf, 0x3f8fdecf, 0x3f8fdecf, 0x3f8fdecf,
];

#[test]
fn a_single_channel_reading_matches_the_channel_0_rules_word_for_word() {
    let client = test_client();
    let plane = Plane::luma(256, 192);
    let partials = hashed_partials(plane.width, plane.height, 101);
    let blocks = recorded_luma_blocks();
    let records = plane.records(&blocks);
    let settings = ReadingSettings { luma_fields: true };
    let state = state_for(&client, plane, true, settings, &partials, &records);
    state.queue_reading(0).expect("the reading queues");

    let (gpu, _) = read_back(&state, 0);
    let words = gpu.to_words();
    let bits: Vec<u32> = words.iter().map(|word| word.to_bits()).collect();

    assert_eq!(bits, RECORDED_LUMA_WORDS);
}
