use cubecl::prelude::*;

use super::noise_curve::{assert_columns_identical, assert_identical, columns_differ};
use super::{Aggregated, Setup, cross_frame_setup, run_fused, run_fused_walk};
use crate::collab::geometry::{ref_pos, refs_along, strength_map_dims};
use crate::collab::kernels::fused::strength_map::strength_map_scale;
use crate::collab::kernels::fused::{STRENGTH_MAP_ALL, STRENGTH_MAP_LUMA};
use crate::collab::tests::helpers::{R, make_client, noisy_flat_field};
use crate::nlmeans::{ChannelMode, NOISE_CURVE_BINS};

const FRAME_SIDE: u32 = 64;
const LAMBDA: f32 = 1.0;

#[cube(launch_unchecked)]
fn map_scale_kernel(
    map: &Array<f32>,
    positions: &Array<u32>,
    out: &mut Array<f32>,
    #[comptime] map_cols: u32,
    #[comptime] map_rows: u32,
) {
    let index = ABSOLUTE_POS_X;
    let rx = positions[(2u32 * index) as usize];
    let ry = positions[(2u32 * index + 1u32) as usize];
    out[index as usize] = strength_map_scale(map, rx, ry, map_cols, map_rows);
}

/// Every reference position of a `width` by `height` frame, as `(left, top)` pairs.
fn reference_positions(width: u32, height: u32) -> Vec<u32> {
    let mut positions = Vec::new();
    for ref_y in 0..refs_along(height) {
        for ref_x in 0..refs_along(width) {
            let left = ref_pos(ref_x, width);
            let top = ref_pos(ref_y, height);
            positions.push(left);
            positions.push(top);
        }
    }

    positions
}

#[test]
fn the_map_scale_is_the_mean_of_the_overlapped_quarters_on_a_ragged_frame() {
    let (width, height) = (70u32, 54u32);
    let (map_cols, map_rows) = strength_map_dims(width, height);
    let map: Vec<f32> = (0..map_cols * map_rows)
        .map(|index| 0.5 + index as f32 * 0.01)
        .collect();
    let positions = reference_positions(width, height);
    let count = positions.len() / 2;

    let client = make_client();
    let map_bytes = f32::as_bytes(&map);
    let positions_bytes = u32::as_bytes(&positions);
    let map_buf = client.create_from_slice(map_bytes);
    let positions_buf = client.create_from_slice(positions_bytes);
    let out_buf = client.empty(count * size_of::<f32>());

    unsafe {
        map_scale_kernel::launch_unchecked::<R>(
            &client,
            CubeCount::new_1d(1),
            CubeDim::new_1d(count as u32),
            ArrayArg::from_raw_parts(map_buf, map.len()),
            ArrayArg::from_raw_parts(positions_buf, positions.len()),
            ArrayArg::from_raw_parts(out_buf.clone(), count),
            map_cols,
            map_rows,
        );
    }

    let out_bytes = client.read_one(out_buf).expect("map scale readback failed");
    let got = f32::from_bytes(&out_bytes)[..count].to_vec();

    let cols = map_cols as usize;

    for (index, &scale) in got.iter().enumerate() {
        let left = positions[2 * index] as usize;
        let top = positions[2 * index + 1] as usize;
        let first_col = left / 8;
        let last_col = left.div_ceil(8).min(cols - 1);
        let first_row = top / 8;
        let last_row = top.div_ceil(8).min(map_rows as usize - 1);
        let sum = map[first_row * cols + first_col]
            + map[first_row * cols + last_col]
            + map[last_row * cols + first_col]
            + map[last_row * cols + last_col];
        let want = sum / 4.0;
        assert_eq!(scale, want, "reference at ({left}, {top})");
    }
}

fn uniform_map(width: u32, height: u32, value: f32) -> Vec<f32> {
    let (cols, rows) = strength_map_dims(width, height);
    vec![value; (cols * rows) as usize]
}

#[test]
fn a_unit_map_changes_nothing_in_either_mode() {
    let mut plain = cross_frame_setup(FRAME_SIDE, FRAME_SIDE, 2);
    plain.noise_curve = Some([1.5; NOISE_CURVE_BINS]);
    let want = run_fused(&plain);

    let mut luma = cross_frame_setup(FRAME_SIDE, FRAME_SIDE, 2);
    luma.noise_curve = Some([1.5; NOISE_CURVE_BINS]);
    let luma_map = uniform_map(FRAME_SIDE, FRAME_SIDE, 1.0);
    luma.strength_map = Some((luma_map, STRENGTH_MAP_LUMA));
    let got_luma = run_fused(&luma);
    assert_identical("unit luma map", &got_luma, &want);

    let plain_no_curve = cross_frame_setup(FRAME_SIDE, FRAME_SIDE, 2);
    let want_no_curve = run_fused(&plain_no_curve);
    let mut all = cross_frame_setup(FRAME_SIDE, FRAME_SIDE, 2);
    let all_map = uniform_map(FRAME_SIDE, FRAME_SIDE, 1.0);
    all.strength_map = Some((all_map, STRENGTH_MAP_ALL));
    let got_all = run_fused(&all);
    assert_identical("unit all-channel map", &got_all, &want_no_curve);
}

#[test]
fn a_uniform_luma_map_equals_scaling_lambda() {
    let mut mapped = cross_frame_setup(FRAME_SIDE, FRAME_SIDE, 2);
    mapped.noise_curve = Some([1.0; NOISE_CURVE_BINS]);
    let map = uniform_map(FRAME_SIDE, FRAME_SIDE, 1.5);
    mapped.strength_map = Some((map, STRENGTH_MAP_LUMA));
    let got = run_fused(&mapped);

    let mut scaled = cross_frame_setup(FRAME_SIDE, FRAME_SIDE, 2);
    scaled.noise_curve = Some([1.0; NOISE_CURVE_BINS]);
    scaled.lambda_ht *= 1.5;
    let want = run_fused(&scaled);

    assert_identical("uniform luma map", &got, &want);
}

#[test]
fn a_uniform_all_channel_map_equals_scaling_lambda() {
    let mut mapped = cross_frame_setup(FRAME_SIDE, FRAME_SIDE, 2);
    let map = uniform_map(FRAME_SIDE, FRAME_SIDE, 0.5);
    mapped.strength_map = Some((map, STRENGTH_MAP_ALL));
    let got = run_fused(&mapped);

    let mut scaled = cross_frame_setup(FRAME_SIDE, FRAME_SIDE, 2);
    scaled.lambda_ht *= 0.5;
    let want = run_fused(&scaled);

    assert_identical("uniform all-channel map", &got, &want);
}

#[test]
fn the_luma_map_is_clamped_with_the_curve() {
    let mut mapped = cross_frame_setup(FRAME_SIDE, FRAME_SIDE, 2);
    mapped.noise_curve = Some([2.0; NOISE_CURVE_BINS]);
    let map = uniform_map(FRAME_SIDE, FRAME_SIDE, 3.0);
    mapped.strength_map = Some((map, STRENGTH_MAP_LUMA));
    let got = run_fused(&mapped);

    let mut ceiling = cross_frame_setup(FRAME_SIDE, FRAME_SIDE, 2);
    ceiling.noise_curve = Some([3.0; NOISE_CURVE_BINS]);
    let want = run_fused(&ceiling);

    assert_identical("clamped luma map", &got, &want);
}

#[test]
fn a_two_region_map_thresholds_each_region_by_its_own_multiplier() {
    let side = FRAME_SIDE;
    let frame = noisy_flat_field(side, side, 0.5, 0.02);
    let (cols, rows) = strength_map_dims(side, side);
    let mut map = Vec::with_capacity((cols * rows) as usize);
    for _ in 0..rows {
        for col in 0..cols {
            let value = if col < cols / 2 { 2.0 } else { 0.5 };
            map.push(value);
        }
    }

    let mut mapped_setup = Setup::spatial_only(frame.clone(), side, side);
    mapped_setup.lambda_ht = LAMBDA;
    mapped_setup.strength_map = Some((map, STRENGTH_MAP_ALL));
    let mapped = run_fused(&mapped_setup);

    let run_at = |lambda: f32| {
        let mut setup = Setup::spatial_only(frame.clone(), side, side);
        setup.lambda_ht = lambda;
        run_fused(&setup)
    };
    let plain = run_at(LAMBDA);
    let left_scaled = run_at(LAMBDA * 2.0);
    let right_scaled = run_at(LAMBDA * 0.5);

    // A pixel left of 24 is only reached by references whose quarters all read 2.0, and a pixel
    // from 40 on only by references reading 0.5, given the spatial radius of 4.
    assert_columns_identical("left region", &mapped, &left_scaled, side, 0, 24);
    assert_columns_identical("right region", &mapped, &right_scaled, side, 40, side);

    let left_changed = columns_differ(&left_scaled, &plain, side, 0, 24);
    let right_changed = columns_differ(&right_scaled, &plain, side, 40, side);
    assert!(left_changed);
    assert!(right_changed);
}

#[test]
fn both_walks_agree_with_a_map_active() {
    let mut setup = cross_frame_setup(FRAME_SIDE, FRAME_SIDE, 2);
    setup.noise_curve = Some([1.2; NOISE_CURVE_BINS]);
    let (cols, rows) = strength_map_dims(FRAME_SIDE, FRAME_SIDE);
    let map: Vec<f32> = (0..cols * rows)
        .map(|index| if index % 3 == 0 { 1.5 } else { 0.65 })
        .collect();
    setup.strength_map = Some((map, STRENGTH_MAP_LUMA));

    let uniform = run_fused_walk(&setup, Some(true));
    let branching = run_fused_walk(&setup, Some(false));

    assert_identical("walks with a map", &uniform, &branching);
}

/// A single-frame 3-channel ring, each channel carrying its own noise.
fn three_channel_setup() -> Setup {
    let pixels = (FRAME_SIDE * FRAME_SIDE) as usize;
    let noise = noisy_flat_field(FRAME_SIDE, FRAME_SIDE * 3, 0.5, 0.02);
    let stored_channels = ChannelMode::Yuv.storage_count() as usize;

    let mut ring = vec![0.0f32; pixels * stored_channels];
    for pixel in 0..pixels {
        for channel in 0..3 {
            ring[pixel * stored_channels + channel] = noise[channel * pixels + pixel];
        }
    }

    let luma_placeholder = vec![0.0f32; pixels];
    let mut setup = Setup::spatial_only(luma_placeholder, FRAME_SIDE, FRAME_SIDE);
    setup.ring = ring;
    setup.channel_mode = ChannelMode::Yuv;
    setup.lambda_ht = LAMBDA;

    setup
}

/// Whether any accumulator of `channel` differs between two single-frame runs.
fn channel_differs(first: &Aggregated, second: &Aggregated, channel: usize) -> bool {
    let stored_channels = ChannelMode::Yuv.storage_count() as usize;
    let first_values = first.accum.iter().skip(channel).step_by(stored_channels);
    let second_values = second.accum.iter().skip(channel).step_by(stored_channels);
    first_values
        .zip(second_values)
        .any(|(first_value, second_value)| first_value != second_value)
}

#[test]
fn a_luma_map_scales_only_channel_zero() {
    let mut luma_mapped = three_channel_setup();
    luma_mapped.noise_curve = Some([1.0; NOISE_CURVE_BINS]);
    let luma_map = uniform_map(FRAME_SIDE, FRAME_SIDE, 1.5);
    luma_mapped.strength_map = Some((luma_map, STRENGTH_MAP_LUMA));
    let got = run_fused(&luma_mapped);

    let mut curve_scaled = three_channel_setup();
    curve_scaled.noise_curve = Some([1.5; NOISE_CURVE_BINS]);
    let want = run_fused(&curve_scaled);

    assert_identical("luma map on three channels", &got, &want);

    let mut all_mapped = three_channel_setup();
    let all_map = uniform_map(FRAME_SIDE, FRAME_SIDE, 1.5);
    all_mapped.strength_map = Some((all_map, STRENGTH_MAP_ALL));
    let all_channels = run_fused(&all_mapped);

    let first_chroma_differs = channel_differs(&all_channels, &got, 1);
    let second_chroma_differs = channel_differs(&all_channels, &got, 2);
    assert!(first_chroma_differs);
    assert!(second_chroma_differs);
}
