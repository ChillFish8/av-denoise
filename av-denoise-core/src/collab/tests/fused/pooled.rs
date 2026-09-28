use cubecl::prelude::*;

use super::{Setup, cross_frame_setup, run_fused, run_fused_walk, unique_frame};
use crate::collab::kernels::fused::pooled::pooled_threshold;
use crate::collab::tests::helpers::{R, make_client};
use crate::collab::{PATCH_AREA, PATCH_SIZE};

const SIDE: usize = PATCH_SIZE as usize;
const AREA: usize = PATCH_AREA as usize;
const FLOOR: f32 = 1.0e-20;

/// One 8-lane group's coefficients, indexed `[sub][j][i]`.
type Group = [[[f32; SIDE]; SIDE]; SIDE];

#[cube(launch_unchecked)]
fn pooled_kernel(
    input: &Array<f32>,
    variances: &Array<f32>,
    profile: &Array<f32>,
    output: &mut Array<f32>,
    retained: &mut Array<f32>,
    k_use: u32,
    threshold: f32,
    dc_lambda: f32,
) {
    let sub = UNIT_POS_X;

    let mut stack = Array::<f32>::new(PATCH_AREA as usize);
    #[unroll]
    for slot in 0..PATCH_AREA {
        stack[slot as usize] = input[(sub * PATCH_AREA + slot) as usize];
    }

    let prof_sub = profile[sub as usize];
    let kept = pooled_threshold(
        &mut stack, variances, profile, prof_sub, sub, k_use, threshold, dc_lambda,
    );

    #[unroll]
    for slot in 0..PATCH_AREA {
        output[(sub * PATCH_AREA + slot) as usize] = stack[slot as usize];
    }

    retained[sub as usize] = kept;
}

/// Runs [pooled_threshold] over one group on the GPU.
fn run_kernel(
    group: &Group,
    variances: &[f32; SIDE],
    profile: &[f32; SIDE],
    k_use: u32,
    threshold: f32,
    dc_lambda: f32,
) -> (Group, f32) {
    let flat: Vec<f32> = group.iter().flatten().flatten().copied().collect();

    let client = make_client();
    let input_bytes = f32::as_bytes(&flat);
    let input_buf = client.create_from_slice(input_bytes);
    let variance_bytes = f32::as_bytes(variances);
    let variance_buf = client.create_from_slice(variance_bytes);
    let profile_bytes = f32::as_bytes(profile);
    let profile_buf = client.create_from_slice(profile_bytes);
    let output_buf = client.empty(SIDE * AREA * size_of::<f32>());
    let retained_buf = client.empty(SIDE * size_of::<f32>());

    unsafe {
        pooled_kernel::launch_unchecked::<R>(
            &client,
            CubeCount::new_1d(1),
            CubeDim::new_1d(SIDE as u32),
            ArrayArg::from_raw_parts(input_buf, flat.len()),
            ArrayArg::from_raw_parts(variance_buf, SIDE),
            ArrayArg::from_raw_parts(profile_buf, SIDE),
            ArrayArg::from_raw_parts(output_buf.clone(), SIDE * AREA),
            ArrayArg::from_raw_parts(retained_buf.clone(), SIDE),
            k_use,
            threshold,
            dc_lambda,
        );
    }

    let output_bytes = client.read_one(output_buf).expect("output readback failed");
    let output = f32::from_bytes(&output_bytes);
    let retained_bytes = client.read_one(retained_buf).expect("retained readback failed");
    let retained: f32 = f32::from_bytes(&retained_bytes)[..SIDE].iter().sum();

    let mut result = [[[0.0f32; SIDE]; SIDE]; SIDE];
    for sub in 0..SIDE {
        for j in 0..SIDE {
            for i in 0..SIDE {
                result[sub][j][i] = output[sub * AREA + j * SIDE + i];
            }
        }
    }
    (result, retained)
}

/// The host copy of the pooled keep rule.
fn reference(
    group: &Group,
    variances: &[f32; SIDE],
    profile: &[f32; SIDE],
    k_use: usize,
    threshold: f32,
    dc_lambda: f32,
) -> (Group, f32) {
    let variance = |sub: usize, j: usize, i: usize| variances[j] * profile[i] * profile[sub];
    // The row's variance and the column's profile are floored and inverted separately.
    let energy = |sub: usize, j: usize, i: usize| {
        let coeff = group[sub][j][i];
        let row_inverse = 1.0 / (variances[j] * profile[sub]).max(FLOOR);
        let column_inverse = 1.0 / profile[i].max(FLOOR);
        coeff * coeff * row_inverse * column_inverse
    };

    let mut result = *group;
    let mut retained = 0.0f32;
    for sub in 0..SIDE {
        for j in 0..k_use {
            for i in 0..SIDE {
                let is_dc = sub == 0 && i == 0;
                let keep = if is_dc {
                    let own = group[sub][j][i].abs() >= dc_lambda * variance(sub, j, i).sqrt();
                    own || j == 0
                } else {
                    let mut sum = energy(sub, j, i);
                    let mut count = 1.0f32;
                    let neighbours = [
                        (sub as i32, i as i32 - 1),
                        (sub as i32, i as i32 + 1),
                        (sub as i32 - 1, i as i32),
                        (sub as i32 + 1, i as i32),
                    ];
                    for (row, col) in neighbours {
                        let in_range = (0..SIDE as i32).contains(&row) && (0..SIDE as i32).contains(&col);
                        let is_neighbour_dc = row == 0 && col == 0;
                        if in_range && !is_neighbour_dc {
                            sum += energy(row as usize, j, col as usize);
                            count += 1.0;
                        }
                    }
                    sum >= threshold * threshold * count
                };

                if keep {
                    retained += variance(sub, j, i);
                } else {
                    result[sub][j][i] = 0.0;
                }
            }
        }
    }
    (result, retained)
}

/// A deterministic group of coefficients spread around the threshold.
#[expect(
    clippy::needless_range_loop,
    reason = "the RNG state advances once per element in a fixed sub, j, i order, not just per index"
)]
fn seeded_group(seed: u32) -> Group {
    let mut group = [[[0.0f32; SIDE]; SIDE]; SIDE];
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    for sub in 0..SIDE {
        for j in 0..SIDE {
            for i in 0..SIDE {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                let unit = (state % 10_000) as f32 / 10_000.0;
                group[sub][j][i] = (unit - 0.5) * 6.0;
            }
        }
    }
    group
}

fn assert_groups_equal(got: &Group, want: &Group) {
    for sub in 0..SIDE {
        for j in 0..SIDE {
            for i in 0..SIDE {
                assert_eq!(got[sub][j][i], want[sub][j][i], "sub {sub} j {j} i {i}");
            }
        }
    }
}

const UNIT: [f32; SIDE] = [1.0; SIDE];

#[test]
fn the_kernel_matches_the_host_reference() {
    let profile = [1.3, 1.1, 1.0, 0.95, 0.9, 0.9, 0.9, 0.95];
    let variances = [1.0, 1.0, 1.0, 1.0, 1.2, 1.2, 1.2, 1.2];
    for (seed, k_use) in [(1u32, 8u32), (2, 8), (3, 4), (4, 2)] {
        let group = seeded_group(seed);
        let (got, got_retained) = run_kernel(&group, &variances, &profile, k_use, 1.6, 2.7);
        let (want, want_retained) = reference(&group, &variances, &profile, k_use as usize, 1.6, 2.7);

        assert_groups_equal(&got, &want);
        assert!(
            (got_retained - want_retained).abs() <= want_retained * 1.0e-5,
            "{got_retained} vs {want_retained}"
        );
    }
}

#[test]
fn a_weak_coefficient_with_strong_neighbours_is_kept() {
    let mut group = [[[0.0f32; SIDE]; SIDE]; SIDE];
    group[3][1][3] = 0.5;
    group[3][1][2] = 2.0;
    group[3][1][4] = 2.0;
    group[2][1][3] = 2.0;
    group[4][1][3] = 2.0;

    let (got, _) = run_kernel(&group, &UNIT, &UNIT, 8, 1.0, 1.0);

    assert_eq!(got[3][1][3], 0.5);
}

#[test]
fn a_lone_strong_coefficient_is_zeroed() {
    let mut group = [[[0.0f32; SIDE]; SIDE]; SIDE];
    group[3][1][3] = 1.5;

    let (got, _) = run_kernel(&group, &UNIT, &UNIT, 8, 1.0, 1.0);

    assert_eq!(got[3][1][3], 0.0);
}

#[test]
fn a_corner_averages_over_its_two_in_range_neighbours() {
    let mut group = [[[0.0f32; SIDE]; SIDE]; SIDE];
    group[7][1][7] = 1.0;
    group[7][1][6] = 1.0;
    group[6][1][7] = 1.0;

    let (got, _) = run_kernel(&group, &UNIT, &UNIT, 8, 1.0, 1.0);

    assert_eq!(
        got[7][1][7], 1.0,
        "three unit energies over three counted positions reach the bar"
    );
}

#[test]
fn the_spatial_dc_keeps_its_own_test_and_is_never_a_neighbour() {
    let mut group = [[[0.0f32; SIDE]; SIDE]; SIDE];
    group[0][1][0] = 0.5;
    group[0][1][1] = 0.9;
    group[0][2][1] = 0.9;
    group[0][2][0] = 100.0;

    let (got, _) = run_kernel(&group, &UNIT, &UNIT, 8, 1.0, 1.0);

    assert_eq!(got[0][1][0], 0.0, "a weak spatial DC fails its own test");
    assert_eq!(got[0][2][0], 100.0, "a strong spatial DC passes its own test");
    assert_eq!(got[0][2][1], 0.0, "a huge spatial DC does not lift its neighbour");
}

#[test]
fn the_group_dc_is_always_kept() {
    let mut group = [[[0.0f32; SIDE]; SIDE]; SIDE];
    group[0][0][0] = 1.0e-6;

    let (got, _) = run_kernel(&group, &UNIT, &UNIT, 8, 1.0, 1.0);

    assert_eq!(got[0][0][0], 1.0e-6);
}

#[test]
fn planes_past_k_use_are_left_untouched() {
    let group = seeded_group(7);

    let (got, _) = run_kernel(&group, &UNIT, &UNIT, 2, 5.0, 5.0);

    for sub in 0..SIDE {
        for j in 2..SIDE {
            assert_eq!(got[sub][j], group[sub][j], "sub {sub} j {j}");
        }
    }
}

#[test]
fn zero_noise_variance_stays_finite() {
    let group = seeded_group(9);
    let zeros = [0.0f32; SIDE];

    let (got, retained) = run_kernel(&group, &zeros, &UNIT, 8, 1.0, 1.0);

    assert!(got.iter().flatten().flatten().all(|value| value.is_finite()));
    assert!(retained.is_finite());
}

/// The luma ratio at the default lambda, pinned here so the kernel tests don't move with the default.
const RATIO: f32 = 2.2 / 3.78;

#[test]
fn pooling_changes_the_output() {
    let mut setup = cross_frame_setup(64, 48, 2);
    let plain = run_fused(&setup);
    setup.pooled = Some(RATIO);
    let pooled = run_fused(&setup);

    assert_ne!(plain.accum, pooled.accum);
}

#[test]
fn both_walks_agree_with_pooling_on() {
    let mut setup = cross_frame_setup(64, 48, 2);
    setup.pooled = Some(RATIO);

    let divergent = run_fused_walk(&setup, Some(false));
    let uniform = run_fused_walk(&setup, Some(true));

    assert_eq!(divergent.accum, uniform.accum);
    assert_eq!(divergent.wsum, uniform.wsum);
    assert_eq!(divergent.group_weight, uniform.group_weight);
}

#[test]
fn a_ragged_frame_with_pooling_on_completes_and_both_walks_agree() {
    let mut setup = Setup::spatial_only(unique_frame(70, 54), 70, 54);
    setup.pooled = Some(RATIO);

    let divergent = run_fused_walk(&setup, Some(false));
    let uniform = run_fused_walk(&setup, Some(true));

    assert_eq!(divergent.accum, uniform.accum);
    assert!(
        divergent.frame_weight_sum(0) > 0,
        "the frame should receive weight"
    );
    let covered = (0..setup.pixels()).all(|idx| divergent.wsum[idx] > 0);
    assert!(covered, "every pixel of a ragged frame should be covered");
}

#[test]
fn a_small_fallback_group_with_pooling_on_stays_finite() {
    let mut setup = Setup::spatial_only(unique_frame(64, 48), 64, 48);
    setup.k_max = 4;
    setup.pooled = Some(RATIO);

    let pooled = run_fused(&setup);

    assert!(
        pooled
            .group_weight
            .iter()
            .all(|weight| weight.is_finite() && *weight > 0.0)
    );
}

#[test]
fn a_noiseless_flat_frame_passes_through_with_pooling_on() {
    let mut setup = Setup::spatial_only(vec![0.4f32; 64 * 48], 64, 48);
    setup.sigma = 1.0e-6;
    setup.pooled = Some(RATIO);

    let pooled = run_fused(&setup);

    assert!(pooled.group_weight.iter().all(|weight| weight.is_finite()));
    for idx in 0..setup.pixels() {
        let value = pooled.pixel(idx);
        assert!((value - 0.4).abs() < 1.0e-3, "pixel {idx} is {value}");
    }
}
