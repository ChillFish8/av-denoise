use cubecl::prelude::*;

use super::helpers::{R, make_client, noisy_copy_of, textured_base};
use crate::collab::kernels::transforms::haar_variance_ladder;
use crate::nl4d::{Nl4dDenoiser, Nl4dParams};
use crate::nlmeans::{ChannelMode, HqParams, MotionCompensationMode, MotionEstimation, NlmParams};

const REFINE: u32 = 2;

/// Inflating exactly one member's variance must raise exactly the stack
/// rows that member participates in and leave every other row
/// unchanged, checked against the host-mirror ladder
/// [`haar_variance_ladder`] (already pinned against the GPU
/// `variance_reg_level` it mirrors by
/// `collab::tests::transforms::gpu_variance_ladder_matches_the_host_mirror`).
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

fn confidence_variance_test_params(temporal_radius: u32, confidence_variance: bool) -> Nl4dParams {
    mismatch_scale_test_params(temporal_radius, confidence_variance, 1.0)
}

fn mismatch_scale_test_params(
    temporal_radius: u32,
    confidence_variance: bool,
    mismatch_scale: f32,
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
        // The shipped default, so these run the aggregation a real
        // caller gets.
        kaiser_beta: 2.0,
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
            let frame = pending.wait().expect("readback failed");
            outputs.push(frame.into_f32().expect("f32 output"));
        }
    }
    outputs
}

/// The `confidence_variance` toggle actually reaches the output.
///
/// An ablation arm built on a flag that silently did nothing would
/// measure the same filter twice. The two arms have to differ on content
/// whose members carry imperfect confidence.
///
/// Note this does not assert a direction or a magnitude. It asserts the
/// switch is live, which is the part a refactor can break silently.
#[test]
fn confidence_variance_toggle_changes_the_output() {
    let client = make_client();
    let (w, h) = (64u32, 64u32);
    let radius = 2u32;
    let base = textured_base(w, h);
    let n = 9usize;
    let frames: Vec<Vec<f32>> = (0..n as u32)
        .map(|seed| noisy_copy_of(&base, w, h, 6.0 / 255.0, seed))
        .collect();

    let on = run_denoiser(
        &client,
        confidence_variance_test_params(radius, true),
        w,
        h,
        &frames,
    );
    let off = run_denoiser(
        &client,
        confidence_variance_test_params(radius, false),
        w,
        h,
        &frames,
    );

    assert_eq!(on.len(), off.len(), "both arms must emit the same frame count");
    assert!(
        !on.is_empty(),
        "expected at least one emitted frame from this clip length"
    );
    assert!(
        on.iter().zip(off.iter()).any(|(a, b)| a != b),
        "confidence_variance=true and false produced identical output on every frame, so the \
         toggle is not reaching the filter"
    );
}

/// A scale of zero leaves every member with the plain channel variance,
/// which is exactly what turning the mechanism off does.
///
/// The two arms have to agree bit for bit. They reach the same place by
/// different routes, one zeroing the variance and one skipping the
/// branch that computes it, so this pins the scale's plumbing against
/// the switch that is known to work.
#[test]
fn a_mismatch_scale_of_zero_reproduces_the_mechanism_off_arm() {
    let client = make_client();
    let (w, h) = (64u32, 64u32);
    let radius = 2u32;
    let base = textured_base(w, h);
    let frames: Vec<Vec<f32>> = (0..9u32)
        .map(|seed| noisy_copy_of(&base, w, h, 6.0 / 255.0, seed))
        .collect();

    let scaled_to_zero = run_denoiser(
        &client,
        mismatch_scale_test_params(radius, true, 0.0),
        w,
        h,
        &frames,
    );
    let mechanism_off = run_denoiser(
        &client,
        mismatch_scale_test_params(radius, false, 1.0),
        w,
        h,
        &frames,
    );

    assert_eq!(scaled_to_zero, mechanism_off);
}

/// Raising the scale moves the filter further from the mechanism-off
/// arm, monotonically.
///
/// This is what makes a ladder over the scale readable. A dial that
/// saturated early, or that moved the output without ordering it, would
/// leave every rung of such a sweep uninterpretable.
///
/// The rungs stop short of the saturation point
/// [`crate::nl4d::MAX_MISMATCH_SCALE`] documents, since the whole claim
/// there is that ordering stops past it.
#[test]
fn a_larger_mismatch_scale_moves_further_from_the_mechanism_off_arm() {
    let client = make_client();
    let (w, h) = (64u32, 64u32);
    let radius = 2u32;
    let base = textured_base(w, h);
    let frames: Vec<Vec<f32>> = (0..9u32)
        .map(|seed| noisy_copy_of(&base, w, h, 6.0 / 255.0, seed))
        .collect();

    let off = run_denoiser(
        &client,
        mismatch_scale_test_params(radius, false, 1.0),
        w,
        h,
        &frames,
    );

    let distance_from_off = |scale: f32| {
        let arm = run_denoiser(
            &client,
            mismatch_scale_test_params(radius, true, scale),
            w,
            h,
            &frames,
        );
        assert_eq!(arm.len(), off.len(), "both arms must emit the same frame count");
        let mut sum = 0.0f64;
        let mut n = 0usize;
        for (a, b) in arm.iter().zip(off.iter()) {
            for (x, y) in a.iter().zip(b.iter()) {
                sum += (*x as f64 - *y as f64).powi(2);
                n += 1;
            }
        }
        (sum / n as f64).sqrt()
    };

    let mut previous = 0.0f64;
    for scale in [1.0f32, 2.0, 4.0] {
        let d = distance_from_off(scale);
        assert!(
            d > previous,
            "scale {scale} sits {d} from the mechanism-off arm, no further than the {previous} \
             the rung below reached"
        );
        previous = d;
    }
}
