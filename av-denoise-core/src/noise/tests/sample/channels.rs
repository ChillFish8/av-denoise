use super::{Plane, check_plane, gpu_reading, state_for, static_blocks};
use crate::noise::reading::TAG_ACCEPTED;
use crate::noise::state::{BLOCK_ACCEPTED, ReadingSettings, SAMPLE_RUN_ACCEPTED, channel_flag};
use crate::noise::stats::{lower_quartile, median};
use crate::noise::temporal::{RHO_SIGMA_GATE, STATIC_GATE, TemporalNoiseSample, accepted_channel_indices};
use crate::noise::tests::records::{BlockSpec, hashed_partials};
use crate::tune::tests::runtime::test_client;

/// The median and lower quartile of channel `channel`'s sigma over every block not in `left_out`.
fn expected_channel_sigma(blocks: &[BlockSpec], channel: usize, left_out: &[usize]) -> (f32, f32) {
    let mut sigmas: Vec<f32> = blocks
        .iter()
        .enumerate()
        .filter(|(index, _)| !left_out.contains(index))
        .map(|(_, block)| block.sigma[channel])
        .collect();
    sigmas.sort_by(f32::total_cmp);

    let sigma = median(&sigmas);
    let sigma_low = lower_quartile(&sigmas);

    (sigma, sigma_low)
}

fn assert_within(actual: f32, expected: f32, what: &str) {
    let gap = (actual - expected).abs();
    assert!(gap <= 1e-6, "{what}: {actual} expected {expected}");
}

/// Asserts channel `channel` of `sample` reads the sigma of every block not in `left_out`.
fn assert_channel_excludes(
    sample: TemporalNoiseSample,
    blocks: &[BlockSpec],
    channel: usize,
    left_out: &[usize],
) {
    let (sigma, sigma_low) = expected_channel_sigma(blocks, channel, left_out);
    let sigma_what = format!("sigma[{channel}]");
    let low_what = format!("sigma_low[{channel}]");

    assert_within(sample.sigma[channel], sigma, &sigma_what);
    assert_within(sample.sigma_low[channel], sigma_low, &low_what);
}

/// Static grain where every fourth block carries texture in channel 1 alone.
///
/// Half the textured blocks pan past the static gate and half sit far above the outlier ceiling.
fn channel_1_textured_blocks(plane: Plane, seed: u32) -> (Vec<BlockSpec>, Vec<usize>) {
    let mut blocks = static_blocks(plane, seed);
    let mut textured = Vec::new();

    for (index, block) in blocks.iter_mut().enumerate() {
        match index % 8 {
            0 => {
                block.mean[1] = 4.0 * STATIC_GATE;
                block.sigma[1] = 20.0 / 255.0;
            },
            4 => block.sigma[1] = 30.0 / 255.0,
            _ => continue,
        }

        textured.push(index);
    }

    (blocks, textured)
}

/// Asserts each channel's accepted flags and accepted lanes name the blocks the host accepts, and
/// returns how many blocks each channel accepted.
fn assert_channel_accepted(plane: Plane, partials: &[f32], records: &[f32]) -> Vec<usize> {
    let client = test_client();
    let settings = ReadingSettings::default();
    let state = state_for(&client, plane, true, settings, partials, records);
    state.queue_reading(0).expect("the reading queues");

    let flags = state.read_block_flags();
    let sample_tags = state.read_sample_tags();
    let block_count = plane.blocks();
    let mut counts = Vec::new();

    for channel in 0..plane.channels {
        let host = accepted_channel_indices(records, plane.channels, plane.width, plane.height, channel);
        let host_indices = host.expect("the channel's ceiling is trusted");
        let flag = channel_flag(BLOCK_ACCEPTED, channel);
        let flagged: Vec<usize> = (0..block_count)
            .filter(|&index| flags[index] & flag != 0)
            .collect();

        let run_start = (SAMPLE_RUN_ACCEPTED + channel) as usize * block_count;
        let accepted_tag = TAG_ACCEPTED + channel;
        let tagged: Vec<usize> = (0..block_count)
            .filter(|&index| sample_tags[run_start + index] == accepted_tag)
            .collect();

        assert_eq!(flagged, host_indices, "channel {channel} accepted flags");
        assert_eq!(tagged, host_indices, "channel {channel} accepted lanes");
        counts.push(host_indices.len());
    }

    counts
}

#[test]
fn a_channel_1_moving_texture_block_is_excluded_from_channel_1_only() {
    let client = test_client();
    let plane = Plane::chroma(256, 192);
    let luma = Plane::luma(plane.width, plane.height);
    let partials = hashed_partials(plane.width, plane.height, 103);
    let (blocks, textured) = channel_1_textured_blocks(plane, 103);
    let records = plane.records(&blocks);
    let (gpu, host) = check_plane(plane, &partials, &records);

    let sample = gpu.sample.expect("the chroma frame has a sample");
    let host_sample = host.sample.expect("the host has a sample");
    assert_channel_excludes(sample, &blocks, 1, &textured);
    assert_channel_excludes(host_sample, &blocks, 1, &textured);
    assert_channel_excludes(sample, &blocks, 0, &[]);

    let luma_records = luma.records(&blocks);
    let (luma_gpu, _) = gpu_reading(&client, luma, true, &partials, &luma_records);
    let luma_sample = luma_gpu.sample.expect("the channel 0 frame has a sample");

    assert_eq!(sample.sigma[0].to_bits(), luma_sample.sigma[0].to_bits());
    assert_eq!(sample.sigma_low[0].to_bits(), luma_sample.sigma_low[0].to_bits());
    assert_eq!(sample.rho.to_bits(), luma_sample.rho.to_bits());
    assert_eq!(
        sample.static_fraction.to_bits(),
        luma_sample.static_fraction.to_bits()
    );
    assert_eq!(gpu.accepted, luma_gpu.accepted);

    let block_count = plane.blocks();
    let counts = assert_channel_accepted(plane, &partials, &records);
    assert_eq!(counts, [block_count, block_count - textured.len()]);
}

/// Static grain where every block moved and every fourth carries a raised sigma and texture in
/// channel 1 alone.
fn channel_1_moved_texture_blocks(plane: Plane, seed: u32) -> (Vec<BlockSpec>, Vec<usize>) {
    let mut blocks = static_blocks(plane, seed);
    let mut textured = Vec::new();

    for (index, block) in blocks.iter_mut().enumerate() {
        block.moved = true;

        if index % 4 != 0 {
            continue;
        }

        block.texture[1] = 1000.0;
        block.sigma[1] *= 2.0;
        textured.push(index);
    }

    (blocks, textured)
}

#[test]
fn a_channel_1_only_moved_texture_block_is_excluded_from_channel_1() {
    let plane = Plane::chroma(256, 192);
    let partials = hashed_partials(plane.width, plane.height, 109);
    let (blocks, textured) = channel_1_moved_texture_blocks(plane, 109);
    let records = plane.records(&blocks);
    let (gpu, host) = check_plane(plane, &partials, &records);

    let sample = gpu.sample.expect("the chroma frame has a sample");
    let host_sample = host.sample.expect("the host has a sample");
    assert_channel_excludes(sample, &blocks, 1, &textured);
    assert_channel_excludes(host_sample, &blocks, 1, &textured);
    assert_channel_excludes(sample, &blocks, 0, &[]);

    let block_count = plane.blocks();
    let counts = assert_channel_accepted(plane, &partials, &records);
    assert_eq!(counts, [block_count, block_count]);
}

#[test]
fn each_channel_rejects_its_own_outliers() {
    let plane = Plane::chroma(256, 192);
    let partials = hashed_partials(plane.width, plane.height, 107);
    let mut blocks = static_blocks(plane, 107);
    let channel_0_outliers = [5, 40, 77, 130];
    let channel_1_outliers = [9, 60, 101, 170];

    for index in channel_0_outliers {
        blocks[index].sigma[0] = 30.0 / 255.0;
    }

    for index in channel_1_outliers {
        blocks[index].sigma[1] = 30.0 / 255.0;
    }

    let records = plane.records(&blocks);
    let (gpu, host) = check_plane(plane, &partials, &records);

    let sample = gpu.sample.expect("the chroma frame has a sample");
    let host_sample = host.sample.expect("the host has a sample");
    assert_channel_excludes(sample, &blocks, 0, &channel_0_outliers);
    assert_channel_excludes(sample, &blocks, 1, &channel_1_outliers);
    assert_channel_excludes(host_sample, &blocks, 0, &channel_0_outliers);
    assert_channel_excludes(host_sample, &blocks, 1, &channel_1_outliers);

    let block_count = plane.blocks();
    let counts = assert_channel_accepted(plane, &partials, &records);
    assert_eq!(counts, [block_count - 4, block_count - 4]);
}

/// Static grain whose channel `channel` reads one sigma with no spread, beside a few quiet blocks
/// that leave its ceiling untrusted.
fn untrusted_channel_blocks(plane: Plane, seed: u32, channel: usize) -> Vec<BlockSpec> {
    let mut blocks = static_blocks(plane, seed);

    for (index, block) in blocks.iter_mut().enumerate() {
        let quiet = index % 9 == 0;
        block.mean[channel] = 0.0;
        block.sigma[channel] = if quiet { 0.5 * RHO_SIGMA_GATE } else { 5.0 / 255.0 };
    }

    blocks
}

/// Asserts the GPU flags no block as accepted by channel `channel`.
fn assert_channel_accepts_nothing(plane: Plane, partials: &[f32], records: &[f32], channel: u32) {
    let client = test_client();
    let settings = ReadingSettings::default();
    let state = state_for(&client, plane, true, settings, partials, records);
    state.queue_reading(0).expect("the reading queues");

    let flags = state.read_block_flags();
    let flag = channel_flag(BLOCK_ACCEPTED, channel);
    let flagged = flags.iter().filter(|&&flags| flags & flag != 0).count();

    assert_eq!(flagged, 0, "channel {channel} accepted flags");
}

#[test]
fn a_starved_channel_1_leaves_channel_0_valid() {
    let client = test_client();
    let plane = Plane::chroma(256, 192);
    let luma = Plane::luma(plane.width, plane.height);
    let partials = hashed_partials(plane.width, plane.height, 109);
    let mut blocks = static_blocks(plane, 109);
    for block in &mut blocks {
        block.mean[1] = 0.0;
        block.sigma[1] = 0.0;
    }

    let records = plane.records(&blocks);
    let (gpu, host) = check_plane(plane, &partials, &records);
    let sample = gpu.sample.expect("channel 0 keeps its sample");
    let host_sample = host.sample.expect("the host keeps channel 0's sample");

    let luma_records = luma.records(&blocks);
    let (luma_gpu, _) = gpu_reading(&client, luma, true, &partials, &luma_records);
    let luma_sample = luma_gpu.sample.expect("the channel 0 frame has a sample");

    assert_eq!(sample.valid, [true, false]);
    assert_eq!(host_sample.valid, [true, false]);
    assert_eq!(sample.sigma[0].to_bits(), luma_sample.sigma[0].to_bits());
    assert_eq!(sample.sigma_low[0].to_bits(), luma_sample.sigma_low[0].to_bits());
    assert_eq!(sample.rho.to_bits(), luma_sample.rho.to_bits());
    assert_eq!((sample.sigma[1], sample.sigma_low[1]), (0.0, 0.0));
}

#[test]
fn an_untrusted_channel_1_reference_matches_the_host() {
    let plane = Plane::chroma(256, 192);
    let partials = hashed_partials(plane.width, plane.height, 113);
    let blocks = untrusted_channel_blocks(plane, 113, 1);
    let records = plane.records(&blocks);
    let host_indices = accepted_channel_indices(&records, plane.channels, plane.width, plane.height, 1);
    assert_eq!(host_indices, None, "channel 1's ceiling is not trusted");

    let (gpu, _) = check_plane(plane, &partials, &records);
    let sample = gpu.sample.expect("channel 0 keeps its sample");

    assert_eq!(sample.valid, [true, false]);
    assert_channel_accepts_nothing(plane, &partials, &records, 1);
}

#[test]
fn an_untrusted_channel_0_leaves_channel_1_valid() {
    let plane = Plane::chroma(256, 192);
    let partials = hashed_partials(plane.width, plane.height, 127);
    let blocks = untrusted_channel_blocks(plane, 127, 0);
    let records = plane.records(&blocks);

    let (gpu, _) = check_plane(plane, &partials, &records);
    let sample = gpu.sample.expect("channel 1 keeps its sample");

    assert_eq!(sample.valid, [false, true]);
    assert_eq!((sample.sigma[0], sample.sigma_low[0]), (0.0, 0.0));
    assert_eq!((sample.rho, sample.static_fraction), (0.0, 0.0));
    assert_eq!(gpu.accepted, 0);
    assert_channel_excludes(sample, &blocks, 1, &[]);
    assert_channel_accepts_nothing(plane, &partials, &records, 0);
}
