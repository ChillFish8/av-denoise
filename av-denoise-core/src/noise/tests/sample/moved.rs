use super::{MAX_ULPS, Plane, assert_close, assert_sampled, check_plane, host_reading, moved_blocks};
use crate::noise::NOISE_CHANNELS;
use crate::noise::reading::ReadingRecord;
use crate::noise::tests::records::{BlockSpec, hashed_partials};

#[test]
fn gpu_sample_matches_host_reference_with_moved_blocks() {
    for (seed, plane) in [(83, Plane::luma(256, 192)), (89, Plane::chroma(256, 192))] {
        let partials = hashed_partials(plane.width, plane.height, seed);
        let blocks = moved_blocks(plane, seed);
        let records = plane.records(&blocks);
        let (gpu, host) = check_plane(plane, &partials, &records);

        let host_record = ReadingRecord {
            immerkaer_low: host.immerkaer_low,
            sample: host.sample,
            accepted: host.accepted.len() as u32,
            curve: None,
            populated_bins: 0,
        };
        let gpu_words = gpu.to_words();
        let host_words = host_record.to_words();
        for (word, (gpu_word, host_word)) in gpu_words.iter().zip(host_words).enumerate() {
            let what = format!("reading word {word}");
            assert_close(*gpu_word, host_word, MAX_ULPS, &what);
        }

        let unmoved: Vec<BlockSpec> = blocks
            .iter()
            .map(|block| BlockSpec {
                moved: false,
                ..*block
            })
            .collect();
        let unmoved_records = plane.records(&unmoved);
        let unmoved_host = host_reading(plane, &partials, &unmoved_records);
        let sample = host.sample.expect("the moved frame has a sample");
        let unmoved_sample = unmoved_host.sample.expect("the unmoved frame has a sample");
        let block_count = plane.blocks() as u32;

        assert!(
            sample.sigma[0] < unmoved_sample.sigma[0],
            "the filter drops the textured blocks"
        );
        assert_eq!(sample.static_fraction, unmoved_sample.static_fraction);
        assert_eq!(gpu.accepted, block_count);
    }
}

#[test]
fn a_reading_with_every_sigma_left_out_has_no_sample() {
    let plane = Plane::luma(320, 16);
    let partials = hashed_partials(plane.width, plane.height, 97);
    let mut blocks = Vec::new();

    for index in 0..plane.blocks() {
        let low_texture = index < 8;
        let sigma = if low_texture { 20.0 / 255.0 } else { 1.0 / 255.0 };
        let texture = if low_texture { 1.0 } else { 100.0 };
        let base = BlockSpec::uniform(0.0, sigma, 0.1);
        let block = BlockSpec {
            texture: [texture; NOISE_CHANNELS as usize],
            moved: true,
            ..base
        };
        blocks.push(block);
    }

    let records = plane.records(&blocks);
    let (gpu, host) = check_plane(plane, &partials, &records);

    assert_sampled(&gpu, &host, false);
    assert_eq!(gpu.accepted, 12);
}
