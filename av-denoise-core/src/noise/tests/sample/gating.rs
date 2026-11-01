use super::{
    CHROMA_1080P,
    HostReading,
    LUMA_1080P,
    MAX_ULPS,
    Plane,
    RESULTS_ROW,
    assert_immerkaer_matches,
    assert_reading_matches,
    assert_sample_matches,
    assert_sampled,
    check_plane,
    gpu_reading,
    host_reading,
    moving_block,
    read_back,
    shape_for,
    state_for,
    static_blocks,
};
use crate::noise::spatial::sigma_block_p25_from_partials;
use crate::noise::state::{NoiseState, ReadingSettings, SlotStats};
use crate::noise::stats::lower_quartile;
use crate::noise::temporal::{RHO_SIGMA_GATE, min_static_blocks, temporal_stats_blocks};
use crate::noise::tests::records::{BlockSpec, QuarterSpec, hashed_partials, static_block};
use crate::tune::gpu::sync;
use crate::tune::tests::runtime::test_client;

#[test]
fn the_immerkaer_quartile_matches_the_host_within_two_ulp() {
    let client = test_client();

    for (seed, plane) in [(3, LUMA_1080P), (5, CHROMA_1080P)] {
        let partials = hashed_partials(plane.width, plane.height, seed);
        let blocks = static_blocks(plane, seed);
        let records = plane.records(&blocks);
        let (gpu, _) = gpu_reading(&client, plane, true, &partials, &records);
        let host = host_reading(plane, &partials, &records);

        assert_immerkaer_matches(&gpu, &host);
    }
}

#[test]
fn a_static_frame_reading_matches_the_host_within_two_ulp() {
    for (seed, plane) in [(11, Plane::luma(256, 192)), (13, Plane::chroma(256, 192))] {
        let partials = hashed_partials(plane.width, plane.height, seed);
        let blocks = static_blocks(plane, seed);
        let records = plane.records(&blocks);
        let (gpu, host) = check_plane(plane, &partials, &records);

        let block_count = plane.blocks();
        let host_count = host.accepted.len();
        let sampled = gpu.sample.is_some();
        assert!(sampled);
        assert_eq!(host_count, block_count);
    }
}

#[test]
fn power_of_two_shapes_give_the_host_fraction_and_counts_exactly() {
    let plane = Plane::luma(64, 64);
    let partials = hashed_partials(plane.width, plane.height, 17);
    let mut blocks = static_blocks(plane, 17);
    for index in [1, 6, 9, 14, 15] {
        blocks[index] = moving_block(3.0 / 255.0);
    }

    let records = plane.records(&blocks);
    let (gpu, host) = check_plane(plane, &partials, &records);

    let gpu_sample = gpu.sample.expect("the gpu has a sample");
    let host_sample = host.sample.expect("the host has a sample");
    let gpu_fraction = gpu_sample.static_fraction.to_bits();
    let host_fraction = host_sample.static_fraction.to_bits();
    assert_eq!(gpu_fraction, host_fraction);
    assert_eq!(gpu.accepted, 11);
}

#[test]
fn a_zeroed_slot_has_no_sample() {
    let plane = Plane::luma(256, 192);
    let partials = hashed_partials(plane.width, plane.height, 19);
    let quiet = vec![BlockSpec::default(); plane.blocks()];
    let records = plane.records(&quiet);
    let (gpu, host) = check_plane(plane, &partials, &records);

    assert_sampled(&gpu, &host, false);
}

#[test]
fn letterboxed_frames_take_the_same_none_path_as_the_host() {
    let plane = Plane::luma(256, 192);
    let (blocks_x, blocks_y) = temporal_stats_blocks(plane.width, plane.height);
    let partials = hashed_partials(plane.width, plane.height, 23);

    let mut barred = static_blocks(plane, 23);
    for block_x in 0..blocks_x {
        let top = block_x as usize;
        let bottom = ((blocks_y - 1) * blocks_x + block_x) as usize;
        barred[top] = BlockSpec::default();
        barred[bottom] = BlockSpec::default();
    }

    let records = plane.records(&barred);
    let (gpu, host) = check_plane(plane, &partials, &records);
    assert_sampled(&gpu, &host, true);

    let mut textured = vec![BlockSpec::default(); plane.blocks()];
    for index in [17, 40, 77, 130] {
        textured[index] = BlockSpec::uniform(0.0, 6.0 / 255.0, 0.3);
    }

    let records = plane.records(&textured);
    let (gpu, host) = check_plane(plane, &partials, &records);
    assert_sampled(&gpu, &host, false);
}

#[test]
fn a_textured_remainder_with_no_spread_has_no_sample() {
    let plane = Plane::luma(128, 128);
    let partials = hashed_partials(plane.width, plane.height, 29);

    let mut blocks = vec![BlockSpec::uniform(0.0, 5.0 / 255.0, 0.3); plane.blocks()];
    let records = plane.records(&blocks);
    let (gpu, host) = check_plane(plane, &partials, &records);
    assert_sampled(&gpu, &host, true);

    blocks[21] = BlockSpec::uniform(0.0, 0.5 * RHO_SIGMA_GATE, 0.3);
    let records = plane.records(&blocks);
    let (gpu, host) = check_plane(plane, &partials, &records);
    assert_sampled(&gpu, &host, false);
}

/// A frame whose blocks all move except `accepted` static blocks spread over it.
fn fraction_frame(plane: Plane, accepted: usize, seed: u32) -> Vec<f32> {
    let block_count = plane.blocks();
    let mut blocks = vec![moving_block(3.0 / 255.0); block_count];

    for position in 0..accepted {
        let index = position * block_count / accepted;
        blocks[index] = static_block(seed, position as u32);
    }

    plane.records(&blocks)
}

#[test]
fn the_static_fraction_gate_matches_the_host_at_its_boundary() {
    let cases = [
        (Plane::luma(160, 160), 5),
        (Plane::luma(320, 320), 20),
        (Plane::luma(640, 360), 46),
    ];

    for (plane, boundary) in cases {
        let block_count = plane.blocks() as u32;
        let min_accepted = min_static_blocks(block_count);
        assert_eq!(min_accepted, boundary, "{block_count} blocks");

        let partials = hashed_partials(plane.width, plane.height, 31);

        for (accepted, expected) in [(boundary - 1, false), (boundary, true)] {
            let records = fraction_frame(plane, accepted as usize, 31);
            let (gpu, host) = check_plane(plane, &partials, &records);

            assert_sampled(&gpu, &host, expected);
            assert_eq!(gpu.accepted, accepted);
        }
    }
}

#[test]
fn an_outlier_above_five_times_the_quartile_is_rejected() {
    let plane = Plane::luma(128, 128);
    let partials = hashed_partials(plane.width, plane.height, 37);
    let mut blocks = static_blocks(plane, 37);
    let rejected = 12;
    let survivor = 50;

    // Both outliers sort above every background block, so placeholders at the top give the
    // quartile the full set has.
    blocks[rejected] = BlockSpec::uniform(0.0, 1.0, 0.3);
    blocks[survivor] = BlockSpec::uniform(0.0, 1.0, 0.3);
    let mut reference: Vec<f32> = blocks.iter().map(|block| block.sigma[0]).collect();
    reference.sort_by(f32::total_cmp);
    let quartile = lower_quartile(&reference);
    blocks[rejected] = BlockSpec::uniform(0.0, 6.0 * quartile, 0.3);
    blocks[survivor] = BlockSpec::uniform(0.0, 4.0 * quartile, 0.3);

    let records = plane.records(&blocks);
    let (_, host) = check_plane(plane, &partials, &records);

    let rejected_kept = host.accepted.contains(&rejected);
    let survivor_kept = host.accepted.contains(&survivor);
    assert!(!rejected_kept);
    assert!(survivor_kept);
}

#[test]
fn ragged_frames_match_the_host_reading() {
    for (seed, plane) in [(41, Plane::luma(70, 54)), (43, Plane::chroma(35, 27))] {
        let partials = hashed_partials(plane.width, plane.height, seed);
        let blocks = static_blocks(plane, seed);
        let records = plane.records(&blocks);
        let (gpu, _) = check_plane(plane, &partials, &records);

        let sampled = gpu.sample.is_some();
        assert!(sampled);
    }
}

#[test]
fn stale_tags_never_join_a_set() {
    let client = test_client();
    let luma = Plane::luma(256, 192);
    let partials = hashed_partials(luma.width, luma.height, 53);
    let blocks = static_blocks(luma, 53);
    let records = luma.records(&blocks);
    let settings = ReadingSettings::default();
    let mut state = state_for(&client, luma, true, settings, &partials, &records);

    state.poison_for_test();
    state.queue_reading(0).expect("the reading queues");

    let (gpu, gpu_accepted) = read_back(&state, 0);
    let host = host_reading(luma, &partials, &records);
    assert_reading_matches(&gpu, &gpu_accepted, &host);

    let sample = gpu.sample.expect("the luma state has a sample");
    let immerkaer_second = gpu.immerkaer_low[1];
    let sigma_second = sample.sigma[1];
    let sigma_low_second = sample.sigma_low[1];
    assert_eq!(immerkaer_second, 0.0);
    assert_eq!(sigma_second, 0.0);
    assert_eq!(sigma_low_second, 0.0);
}

#[test]
fn a_second_slot_reads_only_its_own_stats() {
    let client = test_client();
    let plane = Plane::chroma(256, 192);
    let first_partials = hashed_partials(plane.width, plane.height, 71);
    let first_blocks = static_blocks(plane, 71);
    let first_records = plane.records(&first_blocks);

    let second_partials = hashed_partials(plane.width, plane.height, 73);
    let mut second_blocks = static_blocks(plane, 73);
    for index in (0..second_blocks.len()).step_by(3) {
        second_blocks[index] = moving_block(5.0 / 255.0);
    }

    let second_records = plane.records(&second_blocks);

    let slots = [
        SlotStats {
            partials: &first_partials,
            results_row: RESULTS_ROW,
            records: &first_records,
        },
        SlotStats {
            partials: &second_partials,
            results_row: RESULTS_ROW,
            records: &second_records,
        },
    ];
    let shape = shape_for(&client, plane, true);
    let settings = ReadingSettings::default();
    let state = NoiseState::from_slots_for_test(&client, shape, settings, &slots);

    state.queue_reading(0).expect("the first reading queues");
    state.queue_reading(1).expect("the second reading queues");

    let first = state.read_reading(0);
    let (second, second_accepted) = read_back(&state, 1);
    let first_host = host_reading(plane, &first_partials, &first_records);
    let second_host = host_reading(plane, &second_partials, &second_records);
    let first_count = first_host.accepted.len() as u32;

    assert_immerkaer_matches(&first, &first_host);
    assert_sample_matches(first.sample, first_host.sample, MAX_ULPS);
    assert_eq!(first.accepted, first_count);
    assert_reading_matches(&second, &second_accepted, &second_host);
}

#[test]
fn quarter_lanes_never_join_the_channel_or_correlation_sets() {
    let client = test_client();
    let plane = Plane::luma(256, 192);
    let partials = hashed_partials(plane.width, plane.height, 79);
    let quarter = QuarterSpec {
        mean: 0.0,
        sigma: 9.0 / 255.0,
        luma: 0.5,
        flatness: 0.0,
        luma_min: 0.4,
        luma_max: 0.6,
    };
    let mut blocks = static_blocks(plane, 79);
    for block in &mut blocks {
        block.quarters = [quarter; 4];
    }

    let records = plane.records(&blocks);
    let settings = ReadingSettings { luma_fields: true };
    let state = state_for(&client, plane, true, settings, &partials, &records);
    state.queue_reading(0).expect("the reading queues");

    let (gpu, gpu_accepted) = read_back(&state, 0);
    let host = host_reading(plane, &partials, &records);

    assert_reading_matches(&gpu, &gpu_accepted, &host);
    assert_sampled(&gpu, &host, true);
}

#[test]
fn a_radius_zero_state_reads_immerkaer_only() {
    let client = test_client();
    let plane = Plane::chroma(256, 192);
    let partials = hashed_partials(plane.width, plane.height, 59);
    let (gpu, _) = gpu_reading(&client, plane, false, &partials, &[]);
    let host_low = sigma_block_p25_from_partials(&partials, plane.channels, plane.width, plane.height);
    let host = HostReading {
        immerkaer_low: host_low,
        sample: None,
        accepted: Vec::new(),
    };

    assert_immerkaer_matches(&gpu, &host);
    let sampled = gpu.sample.is_some();
    assert!(!sampled);
    assert_eq!(gpu.accepted, 0);

    sync(&client).expect("the device syncs");
}

#[test]
fn the_reading_is_identical_across_runs() {
    let client = test_client();
    let plane = Plane::chroma(256, 192);
    let partials = hashed_partials(plane.width, plane.height, 61);
    let blocks = static_blocks(plane, 61);
    let records = plane.records(&blocks);

    let (first, _) = gpu_reading(&client, plane, true, &partials, &records);
    let (second, _) = gpu_reading(&client, plane, true, &partials, &records);
    let first_words = first.to_words();
    let second_words = second.to_words();
    let first_bits: Vec<u32> = first_words.iter().map(|word| word.to_bits()).collect();
    let second_bits: Vec<u32> = second_words.iter().map(|word| word.to_bits()).collect();

    assert_eq!(first_bits, second_bits);
}
