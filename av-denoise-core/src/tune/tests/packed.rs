use cubecl::prelude::*;

use super::fixtures::{CollabSpec, collab_site, packed_twin};
use super::runtime::{packed_supported, test_client};
use crate::collab::kernels::aggregate::{
    ACCUM_CLAMP,
    cross_frame_accum_scale,
    to_fixed_host,
    to_fixed_weight_host,
};
use crate::collab::kernels::packed::{pack_accum, unpack_accum_host, unpack_weight_host};
use crate::collab::{MAX_K, PATCH_SIZE, STEP};
use crate::nlmeans::ChannelMode;
use crate::tune::VariantSite;
use crate::tune::gpu::{bind, read_f32, read_i32, zero_words};

#[cube(launch_unchecked)]
fn packed_add_stress(words: &mut [Atomic<i64>], accum_fixed: i32, weight_fixed: i32, #[comptime] count: u32) {
    if ABSOLUTE_POS_X < count {
        let word = pack_accum::<i64>(accum_fixed, weight_fixed);
        Atomic::fetch_add(&words[0usize], word);
    }
}

/// The most contributions one pixel can receive at the given radii.
fn worst_case_contributions(spatial_radius: u32, temporal_radius: u32) -> u32 {
    let refs_per_axis = ((PATCH_SIZE - 1) + 2 * spatial_radius) / STEP + 1;
    let passes = 4 * temporal_radius + 1;

    refs_per_axis * refs_per_axis * MAX_K * passes
}

/// The `i64` word at `index` of a buffer read back as 32-bit halves, low half first.
fn word_at(halves: &[i32], index: usize) -> i64 {
    let low = i64::from(halves[2 * index] as u32);
    let high = i64::from(halves[2 * index + 1]) << 32;

    high | low
}

/// Every contribution sits at the clamp, positive and negative, at spatial radius 16 and temporal
/// radius 8.
#[test]
fn packed_words_do_not_overflow_at_the_largest_radii() {
    if !packed_supported() {
        eprintln!("skipped, this device has no 64-bit atomic add");
        return;
    }

    let client = test_client();
    let count = worst_case_contributions(16, 8);
    let scale = cross_frame_accum_scale(16, 8);

    for value in [ACCUM_CLAMP, -ACCUM_CLAMP] {
        let accum_fixed = to_fixed_host(value, scale);
        let weight_fixed = to_fixed_weight_host(1.0, scale);
        let words = client.empty(size_of::<i64>());
        zero_words(&client, &words, 2);

        let threads = 256u32;
        let cubes = count.div_ceil(threads);
        let cube_count = CubeCount::new_1d(cubes);
        let cube_dim = CubeDim::new_1d(threads);
        let words_arg = bind(&words, 1);

        unsafe {
            packed_add_stress::launch_unchecked(
                &client,
                cube_count,
                cube_dim,
                words_arg,
                accum_fixed,
                weight_fixed,
                count,
            );
        }

        let halves = read_i32(&client, &words, 2).expect("the word reads back");
        let word = word_at(&halves, 0);
        let expected_accum = i64::from(accum_fixed) * i64::from(count);
        let expected_weight = i64::from(weight_fixed) * i64::from(count);
        let unpacked_accum = unpack_accum_host(word);
        let unpacked_weight = unpack_weight_host(word);

        assert!(
            expected_accum.abs() <= i64::from(i32::MAX),
            "the scale must keep accum inside i32"
        );
        assert!(
            expected_weight < i64::from(i32::MAX),
            "the scale must keep wsum inside i32"
        );
        assert_eq!(i64::from(unpacked_accum), expected_accum, "value {value}");
        assert_eq!(i64::from(unpacked_weight), expected_weight, "value {value}");
    }
}

/// The packed site's outputs, unpacked, must equal the split layout's word for word.
#[test]
fn the_packed_layout_unpacks_to_the_split_layout() {
    if !packed_supported() {
        eprintln!("skipped, this device has no 64-bit atomic add");
        return;
    }

    let client = test_client();
    let shapes = [
        (72u32, 40u32, ChannelMode::Luma, 2u32, 9u32),
        (72, 40, ChannelMode::Chroma, 2, 9),
        (72, 40, ChannelMode::Yuv, 1, 9),
        (96, 96, ChannelMode::Luma, 8, 16),
    ];

    for (width, height, channels, radius, spatial_radius) in shapes {
        let mut spec = CollabSpec::new(width, height, channels, radius);
        spec.spatial_radius = spatial_radius;
        let mut split = collab_site(&client, &spec);
        split.values.accum_scale = cross_frame_accum_scale(spatial_radius, spec.radius);
        let packed = packed_twin(&split);
        let label = format!(
            "{:?} radius={} spatial={spatial_radius}",
            spec.channels, spec.radius
        );

        let want = split.scratch_outputs();
        let split_baseline = split.baseline();
        split.launch(split_baseline, &want);
        let got = packed.scratch_outputs();
        let packed_baseline = packed.baseline();
        packed.launch(packed_baseline, &got);

        let stored = split.shape.stored_ch as usize;
        let pixels = want.wsum_len;
        let want_accum = read_i32(&client, &want.accum, want.accum_len).expect("accum");
        let want_wsum = read_i32(&client, &want.wsum, want.wsum_len).expect("wsum");
        let want_weights = read_f32(&client, &want.group_weight, want.refs).expect("weights");
        let halves = read_i32(&client, &got.wsum, 2 * pixels).expect("packed words");
        let rest = read_i32(&client, &got.accum, got.accum_len).expect("rest");
        let got_weights = read_f32(&client, &got.group_weight, got.refs).expect("weights");

        for pixel in 0..pixels {
            let word = word_at(&halves, pixel);
            let first_accum = want_accum[pixel * stored];
            let unpacked_weight = unpack_weight_host(word);
            let unpacked_accum = unpack_accum_host(word);

            assert_eq!(unpacked_weight, want_wsum[pixel], "{label}: wsum at {pixel}");
            assert_eq!(unpacked_accum, first_accum, "{label}: channel 0 at {pixel}");

            for channel in 1..stored {
                let rest_index = pixel * (stored - 1) + channel - 1;
                let want_channel = want_accum[pixel * stored + channel];

                assert_eq!(
                    rest[rest_index], want_channel,
                    "{label}: channel {channel} at {pixel}"
                );
            }
        }

        assert_eq!(got_weights, want_weights, "{label}: group weights");
    }
}
