use cubecl::prelude::*;

use super::grouping::{BLKSIZE, THSAD, run_group_temporal};
use super::helpers::{
    R,
    deterministic_texture,
    make_client,
    noisy_copy_of,
    noisy_ring,
    planted_ring,
    textured_base,
};
use crate::collab::STEP;
use crate::collab::geometry::refs_along;
use crate::collab::kernels::transforms::haar_variance_ladder;
use crate::nl4d::{Nl4dDenoiser, Nl4dParams};
use crate::nlmeans::{ChannelMode, HqParams, MotionCompensationMode, MotionEstimation, NlmParams};

const REFINE: u32 = 2;
const K_MAX: u32 = 8;
const SPATIAL_RADIUS: u32 = 4;

/// The formula [`crate::collab::kernels::group_temporal::collab_group_temporal`]'s
/// `mismatch_sigma2` runs on the GPU, mirrored on the host for these
/// tests, with the same argument order and the same operations, so
/// floating-point rounding matches to well within the tolerances below.
fn expected_mismatch_sigma2(confidence: f32, thsad: f32, blksize: u32, mismatch_scale: f32) -> f32 {
    let blksize_area = (blksize * blksize) as f32;
    let ratio = (1.0 - confidence) / (1.0 + confidence);
    let e2 = thsad * thsad * ratio;
    let eps = e2.sqrt() / blksize_area;
    std::f32::consts::FRAC_PI_2 * eps * eps * mismatch_scale * mismatch_scale
}

/// `c = 1.0`, a perfect motion match, must give `sigma_m2 = 0.0` for
/// every temporal member, whatever `thsad`/`mismatch_scale` are.
#[test]
fn confidence_one_gives_zero_mismatch_variance_for_every_temporal_member() {
    let (w, h) = (96u32, 96u32);
    let radius = 2u32;
    let ref_pos = (64u32, 64u32);
    let patch = deterministic_texture(7);
    let fx = planted_ring(w, h, radius, ref_pos, 3, &patch, 0.2, |_| 1.0);

    let (_pos, member_frame, member_count, member_sig2) =
        run_group_temporal(&fx, 0.0, 0.05, THSAD, 1.0, REFINE, K_MAX, SPATIAL_RADIUS);

    let refs_x = refs_along(w);
    let ref_idx = ((ref_pos.1 / STEP) * refs_x + (ref_pos.0 / STEP)) as usize;
    let count = member_count[ref_idx] as usize;
    assert_eq!(count, 8, "expected the planted reference to fill to k_max");

    let mut checked_a_temporal_member = false;
    for j in 0..count {
        let idx = ref_idx * K_MAX as usize + j;
        if member_frame[idx] == fx.centre_slot {
            continue;
        }
        checked_a_temporal_member = true;
        assert!(
            member_sig2[idx].abs() < 1e-9,
            "member {j} (frame {}) at confidence 1.0 must carry sigma_m2 ~ 0, got {}",
            member_frame[idx],
            member_sig2[idx]
        );
    }
    assert!(
        checked_a_temporal_member,
        "expected at least one temporal member in this group to exercise the assertion above"
    );
}

/// A known confidence must produce exactly the variance
/// `expected_mismatch_sigma2` derives, for the one neighbour that
/// carries it.
#[test]
fn low_confidence_produces_the_derived_mismatch_variance() {
    let (w, h) = (96u32, 96u32);
    let radius = 2u32;
    let ref_pos = (64u32, 64u32);
    let patch = deterministic_texture(11);
    let low_c = 0.2f32;
    // k = +1 carries the confidence under test, every other neighbour
    // stays at 1.0 so this group still fills without the low-confidence
    // neighbour crowding anything else out of contention.
    let fx = planted_ring(w, h, radius, ref_pos, 3, &patch, 0.2, |k| {
        if k == 1 { low_c } else { 1.0 }
    });
    let mismatch_scale = 1.0f32;

    let (_pos, member_frame, member_count, member_sig2) = run_group_temporal(
        &fx,
        0.0,
        0.05,
        THSAD,
        mismatch_scale,
        REFINE,
        K_MAX,
        SPATIAL_RADIUS,
    );

    let refs_x = refs_along(w);
    let ref_idx = ((ref_pos.1 / STEP) * refs_x + (ref_pos.0 / STEP)) as usize;
    let count = member_count[ref_idx] as usize;
    let expected_slot = 1u32 + radius;

    let member_idx = (0..count)
        .find(|&j| member_frame[ref_idx * K_MAX as usize + j] == expected_slot)
        .expect("the k=+1 neighbour must have joined the group");
    let idx = ref_idx * K_MAX as usize + member_idx;

    let expected = expected_mismatch_sigma2(low_c, THSAD, BLKSIZE, mismatch_scale);
    assert!(
        (member_sig2[idx] - expected).abs() < 1e-6,
        "expected sigma_m2 {expected} for confidence {low_c}, got {}",
        member_sig2[idx]
    );
    // Sanity: the derived value must actually be far from zero, or the
    // assertion above would pass trivially against a broken formula
    // that always returns ~0.
    assert!(
        expected > 1e-4,
        "expected a non-trivial mismatch variance, got {expected}"
    );
}

/// A centre-frame member carries `sigma_m2 = 0.0` exactly, whatever the
/// confidence buffer holds, because centre-frame candidates never read
/// it. `noisy_ring` sets a uniform confidence of `0.0`, the worst value
/// the formula can see, with `c_min = 0.0` so nothing is gated and the
/// self-match's neighbours really do get scored against it.
#[test]
fn centre_frame_members_always_carry_zero_regardless_of_confidence() {
    let (w, h) = (64u32, 64u32);
    let radius = 2u32;
    let fx = noisy_ring(w, h, radius, 0.0);

    let (_pos, member_frame, member_count, member_sig2) =
        run_group_temporal(&fx, 0.0, 0.0, THSAD, 1.0, REFINE, K_MAX, SPATIAL_RADIUS);

    let refs_x = refs_along(w);
    let refs_y = refs_along(h);
    let mut checked_a_centre_member = false;
    for (ref_idx, &count) in member_count.iter().enumerate().take((refs_x * refs_y) as usize) {
        let count = count as usize;
        for j in 0..count {
            let idx = ref_idx * K_MAX as usize + j;
            if member_frame[idx] != fx.centre_slot {
                continue;
            }
            checked_a_centre_member = true;
            assert_eq!(
                member_sig2[idx], 0.0,
                "ref {ref_idx} member {j}: centre-frame member must carry sigma_m2 = 0.0 \
                 exactly, got {}",
                member_sig2[idx]
            );
        }
    }
    assert!(
        checked_a_centre_member,
        "expected at least one centre-frame member (the self-match is always one) to exercise \
         the assertion above"
    );
}

/// Inflating exactly one member's variance must raise exactly the stack
/// rows that member participates in and leave every other row
/// unchanged, checked against the host-mirror ladder
/// [`haar_variance_ladder`] (already pinned against the GPU
/// `variance_ladder` it mirrors by
/// `collab::tests::filter_ht::gpu_variance_ladder_matches_the_host_mirror`).
///
/// For `k_use = 8`, member `m` feeds into exactly the rows the Haar
/// butterfly pairs it into: the DC rows `{0, 1}`, its quad row (`2` for
/// members `0..4`, `3` for `4..8`), and its own pair row (`4 + m / 2`).
/// Inflating member `3` therefore must move rows `{0, 1, 2, 5}` and
/// leave rows `{3, 4, 6, 7}` exactly where the uniform baseline put
/// them.
#[test]
fn inflated_member_variance_raises_exactly_the_rows_it_touches() {
    let base_sig2 = 0.0004f32; // a plausible sigma^2, e.g. sigma = 0.02
    let inflated_member = 3usize;
    let inflated_sig2 = base_sig2 + 0.05;

    let baseline = [base_sig2; 8];
    let mut mixed = [base_sig2; 8];
    mixed[inflated_member] = inflated_sig2;

    let baseline_ladder = haar_variance_ladder(&baseline, 8);
    let mixed_ladder = haar_variance_ladder(&mixed, 8);

    let touched: [usize; 4] = [0, 1, 2, 5];
    let untouched: [usize; 4] = [3, 4, 6, 7];

    for &row in &touched {
        assert!(
            mixed_ladder[row] > baseline_ladder[row] + 1e-9,
            "row {row} should rise when member {inflated_member} is inflated, baseline={} \
             mixed={}",
            baseline_ladder[row],
            mixed_ladder[row]
        );
    }
    for &row in &untouched {
        assert!(
            (mixed_ladder[row] - baseline_ladder[row]).abs() < 1e-9,
            "row {row} should hold steady when member {inflated_member} is inflated, \
             baseline={} mixed={}",
            baseline_ladder[row],
            mixed_ladder[row]
        );
    }
}

fn confidence_variance_test_params(
    temporal_radius: u32,
    mismatch_scale: f32,
    confidence_variance: bool,
) -> Nl4dParams {
    const SIGMA: f32 = 6.0 / 255.0;
    Nl4dParams {
        nlm: NlmParams {
            temporal_radius,
            search_radius: 2,
            patch_radius: 2,
            strength: 1.2,
            self_weight: 1.0,
            channels: ChannelMode::Luma,
            prefilter: crate::nlmeans::PrefilterMode::None,
            motion_compensation: MotionCompensationMode::Mvtools {
                blksize: 16,
                overlap: 8,
                search_radius: 4,
                pyramid_levels: 2,
                estimation: MotionEstimation::Auto,
            },
            hq: Some(HqParams::with_sigma(SIGMA)),
        },
        temporal_radius,
        refine: REFINE,
        spatial_radius: 9,
        lambda_ht: 2.7,
        c_min: 0.05,
        mismatch_scale,
        confidence_variance,
    }
}

/// Runs [`Nl4dDenoiser`] over a short static-content clip and returns
/// every frame it emits.
fn run_denoiser(
    client: &ComputeClient<R>,
    params: Nl4dParams,
    w: u32,
    h: u32,
    frames: &[Vec<f32>],
) -> Vec<Vec<f32>> {
    let mut d = Nl4dDenoiser::<R>::new(client, params, w, h).expect("construction failed");
    let mut outputs = Vec::new();
    for frame in frames {
        d.push_frame(frame);
        if let Some(pending) = d.denoise_submit().expect("denoise_submit failed") {
            outputs.push(pending.wait().expect("readback failed"));
        }
    }
    outputs
}

/// `mismatch_scale = 0.0` and `confidence_variance = false` both reach
/// the same output, bit-for-bit, at otherwise identical settings. This
/// is the mechanism's own regression test, the design's claim that
/// "off" really means off, not just "usually close to off".
#[test]
fn mismatch_scale_zero_matches_confidence_variance_off_bit_for_bit() {
    let client = make_client();
    let (w, h) = (64u32, 64u32);
    let radius = 2u32;
    let base = textured_base(w, h);
    let n = 9usize;
    let frames: Vec<Vec<f32>> = (0..n as u32)
        .map(|seed| noisy_copy_of(&base, w, h, 6.0 / 255.0, seed))
        .collect();

    let off_by_flag = run_denoiser(
        &client,
        confidence_variance_test_params(radius, 1.0, false),
        w,
        h,
        &frames,
    );
    let off_by_scale = run_denoiser(
        &client,
        confidence_variance_test_params(radius, 0.0, true),
        w,
        h,
        &frames,
    );

    assert_eq!(
        off_by_flag.len(),
        off_by_scale.len(),
        "both arms must emit the same frame count"
    );
    assert!(
        !off_by_flag.is_empty(),
        "expected at least one emitted frame from this clip length"
    );
    for (i, (a, b)) in off_by_flag.iter().zip(off_by_scale.iter()).enumerate() {
        assert_eq!(
            a, b,
            "frame {i}: confidence_variance=false and mismatch_scale=0.0 must produce bit-\
             identical output"
        );
    }
}
