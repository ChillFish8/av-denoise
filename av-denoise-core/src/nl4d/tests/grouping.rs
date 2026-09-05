use cubecl::prelude::*;

use super::helpers::{
    BLK_STEP,
    R,
    RingFixture,
    deterministic_texture,
    make_client,
    noisy_ring,
    planted_ring,
};
use crate::collab::geometry::{fused_cubes_x, ref_count, refs_along};
use crate::collab::kernels::aggregate::{cross_frame_accum_scale, kaiser_window, weight_scale};
use crate::collab::kernels::fused::collab_fused;
use crate::collab::kernels::transforms::dct_noise_profile;
use crate::collab::{PATCH_SIZE, STEP};

/// The motion block side length these fixtures score confidence and
/// mismatch variance against, distinct from [`BLK_STEP`], which stays
/// at `PATCH_SIZE` so a block boundary lines up with a patch boundary.
pub(super) const BLKSIZE: u32 = 16;

/// `thsad(BLKSIZE, 1.0)` in normalised SAD units, the same value real
/// callers get from `NlmDenoiser::thsad_value` at this block size and
/// the default `thsad_scale`. Duplicated here rather than imported,
/// since `motion::thsad` is crate-private to `nlmeans`.
pub(super) const THSAD: f32 = (BLKSIZE * BLKSIZE) as f32 * 0.02;

const REFINE: u32 = 2;
const K_MAX: u32 = 8;
const SPATIAL_RADIUS: u32 = 4;

/// The knobs a run varies. Everything else follows the fixture.
struct Knobs {
    c_min: f32,
    k_max: u32,
    sigma: f32,
    lambda_ht: f32,
}

impl Default for Knobs {
    fn default() -> Self {
        Knobs {
            c_min: 0.05,
            k_max: K_MAX,
            sigma: 0.02,
            lambda_ht: 2.7,
        }
    }
}

/// What one launch of [`collab_fused`] left behind.
struct FusedRun {
    wsum: Vec<i32>,
    group_weight: Vec<f32>,
    pixels: usize,
}

impl FusedRun {
    /// The total weight one ring slot's region received. A slot no
    /// member scattered into reads exactly zero.
    fn frame_weight_sum(&self, slot: u32) -> i64 {
        let start = slot as usize * self.pixels;
        self.wsum[start..start + self.pixels]
            .iter()
            .map(|&v| v as i64)
            .sum()
    }

    /// The total weight the whole ring received. Every group contributes
    /// one patch of 64 pixels per member, so at a fixed per-group weight
    /// this counts members.
    fn total_weight(&self) -> i64 {
        self.wsum.iter().map(|&v| v as i64).sum()
    }
}

/// Launches [`collab_fused`] over a fixture, on the same one-cube-per-
/// eight-references grid `Nl4dDenoiser` uses, and reads back the
/// accumulator weights and the per-reference group weight.
///
/// Luma always stores one channel per line, so the kernel's `Size`
/// selector is fixed at 1 here rather than threaded through as an
/// argument.
fn run_fused_over(fx: &RingFixture, k: Knobs) -> FusedRun {
    let client = make_client();
    let w = fx.width;
    let h = fx.height;
    let pixels = (w * h) as usize;
    let frames = fx.ring.len() / pixels;
    let refs = ref_count(w, h);
    let refs_x = refs_along(w);
    let profile = dct_noise_profile(0.0);

    let ring_buf = client.create_from_slice(f32::as_bytes(&fx.ring));
    let mv_buf = client.create_from_slice(i32::as_bytes(&fx.mv_field));
    let conf_buf = client.create_from_slice(f32::as_bytes(&fx.confidence));
    let slots_buf = client.create_from_slice(u32::as_bytes(&fx.neighbour_slots));
    let sigma_buf = client.create_from_slice(f32::as_bytes(&[k.sigma]));
    let profile_buf = client.create_from_slice(f32::as_bytes(&profile));
    let kaiser_buf = client.create_from_slice(f32::as_bytes(&kaiser_window(0.0)));
    let accum = client.create_from_slice(i32::as_bytes(&vec![0i32; pixels * frames]));
    let wsum = client.create_from_slice(i32::as_bytes(&vec![0i32; pixels * frames]));
    let group_weight = client.empty(refs * size_of::<f32>());

    unsafe {
        collab_fused::launch_unchecked::<R>(
            &client,
            CubeCount::new_2d(fused_cubes_x(w), refs_along(h)),
            CubeDim::new_1d(64),
            1usize,
            ArrayArg::from_raw_parts(ring_buf, fx.ring.len()),
            ArrayArg::from_raw_parts(mv_buf, fx.mv_field.len()),
            ArrayArg::from_raw_parts(conf_buf, fx.confidence.len()),
            ArrayArg::from_raw_parts(slots_buf, fx.neighbour_slots.len()),
            ArrayArg::from_raw_parts(sigma_buf, 1),
            ArrayArg::from_raw_parts(profile_buf, 8),
            ArrayArg::from_raw_parts(kaiser_buf, PATCH_SIZE as usize),
            ArrayArg::from_raw_parts(accum, pixels * frames),
            ArrayArg::from_raw_parts(wsum.clone(), pixels * frames),
            ArrayArg::from_raw_parts(group_weight.clone(), refs),
            fx.centre_slot,
            0.0f32,
            k.c_min,
            THSAD,
            k.lambda_ht,
            weight_scale(k.sigma, &profile),
            cross_frame_accum_scale(SPATIAL_RADIUS, fx.radius),
            false,
            fx.radius,
            REFINE,
            fx.mv_stride,
            fx.conf_stride,
            BLK_STEP,
            BLKSIZE,
            fx.blocks_x,
            fx.blocks_y,
            w,
            h,
            1u32,
            k.k_max,
            1u32,
            SPATIAL_RADIUS,
            refs_x,
        );
    }

    let wsum_bytes = client.read_one(wsum).expect("wsum readback failed");
    let weight_bytes = client
        .read_one(group_weight)
        .expect("group_weight readback failed");

    FusedRun {
        wsum: i32::from_bytes(&wsum_bytes)[..pixels * frames].to_vec(),
        group_weight: f32::from_bytes(&weight_bytes)[..refs].to_vec(),
        pixels,
    }
}

/// The temporal search looks where the motion field points.
///
/// `planted_ring` puts an exact copy of the reference patch in every
/// neighbour, shifted by `3 * k`, and seeds the motion field to predict
/// exactly that shift. A search that follows the prediction finds four
/// pixel-for-pixel copies of the reference patch, the whole group agrees,
/// and the Haar detail levels collapse to nothing, so the threshold keeps
/// very little and the group weight is high.
///
/// The control zeroes the motion field, leaving every neighbour's refine
/// window over flat background instead. The copies still exist in the
/// ring, so this is a test of the prediction and not of whether the
/// content is reachable at all.
#[test]
fn temporal_members_are_found_at_the_mv_prediction() {
    let (w, h) = (96u32, 96u32);
    let radius = 2u32;
    let ref_pos = (64u32, 64u32);
    let patch = deterministic_texture(7);

    let predicted = planted_ring(w, h, radius, ref_pos, 3, &patch, 0.2, |_| 1.0);
    let mut blind = planted_ring(w, h, radius, ref_pos, 3, &patch, 0.2, |_| 1.0);
    blind.mv_field.fill(0);

    let refs_x = refs_along(w);
    let ref_idx = ((ref_pos.1 / STEP) * refs_x + (ref_pos.0 / STEP)) as usize;

    let with_prediction = run_fused_over(&predicted, Knobs::default()).group_weight[ref_idx];
    let without = run_fused_over(&blind, Knobs::default()).group_weight[ref_idx];

    assert!(
        with_prediction > without * 1.5,
        "expected the group at {ref_pos:?} to agree far better when the motion field points at \
         the planted copies, got weight {with_prediction} with the prediction and {without} \
         with a zeroed field"
    );
}

/// A neighbour whose motion-block confidence sits below `c_min` is
/// skipped outright, so no member ever comes from it.
///
/// The confidence is uniform across every block of a neighbour's plane
/// here, so the skip is the same decision for every group in the frame
/// and that neighbour's whole region of the accumulator ring has to stay
/// exactly zero. A single admitted member anywhere would show up as a
/// non-zero weight sum.
#[test]
fn low_confidence_neighbours_contribute_no_candidates() {
    let (w, h) = (96u32, 96u32);
    let radius = 2u32;
    let ref_pos = (64u32, 64u32);
    let patch = deterministic_texture(11);
    // Confidence 0.0 for k = +1 and +2, 1.0 for k = -1 and -2.
    let fx = planted_ring(w, h, radius, ref_pos, 3, &patch, 0.2, |k| {
        if k > 0 { 0.0 } else { 1.0 }
    });

    let run = run_fused_over(&fx, Knobs::default());

    for k in -(radius as i32)..=(radius as i32) {
        let slot = (k + radius as i32) as u32;
        let weight = run.frame_weight_sum(slot);
        if k > 0 {
            assert_eq!(
                weight, 0,
                "slot {slot} (k={k}) is gated by c_min, so it must receive no scatter at all"
            );
        } else {
            assert!(
                weight > 0,
                "slot {slot} (k={k}) is ungated, so it must receive members"
            );
        }
    }
}

/// Every group fills to `k_max` however poor its candidates are, because
/// there is no admission gate.
///
/// `noisy_ring` is built so no 8x8 window resembles any other, on any
/// frame, so every candidate is a bad match. `lambda_ht` is set high
/// enough that only the forced group DC survives the threshold, which
/// pins every group's retained variance at `sigma^2` and so every
/// group's weight at the same constant. The weight one member's patch
/// deposits is then the same fixed-point value everywhere, and the total
/// weight in the ring counts members outright.
///
/// A run capped at `k_max = 1` holds every group to its self-match, so
/// the eight-member run has to deposit exactly eight times as much. An
/// admission gate anywhere would leave some group short and break the
/// ratio.
#[test]
fn no_admission_gate_means_the_group_always_fills() {
    let (w, h) = (64u32, 64u32);
    let radius = 2u32;
    let fx = noisy_ring(w, h, radius, 1.0);

    // The smallest search space any reference here sees is the 5x5
    // rectangle a corner clips to, so every group has at least eight
    // positions to choose from and rounds up to a full stack.
    let full = run_fused_over(
        &fx,
        Knobs {
            lambda_ht: 1.0e6,
            ..Knobs::default()
        },
    );
    let single = run_fused_over(
        &fx,
        Knobs {
            k_max: 1,
            lambda_ht: 1.0e6,
            ..Knobs::default()
        },
    );

    let one = single.total_weight();
    assert!(one > 0, "the k_max = 1 run deposited no weight at all");
    assert_eq!(
        full.total_weight(),
        one * K_MAX as i64,
        "expected every group to carry {K_MAX} members, so {K_MAX}x the weight the \
         one-member run deposited"
    );
}
