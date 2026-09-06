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
use crate::collab::{PATCH_AREA, PATCH_SIZE, STEP, needs_warp_uniform_search};

/// The motion block side length these fixtures score confidence
/// against, distinct from [`BLK_STEP`], which stays at `PATCH_SIZE` so
/// a block boundary lines up with a patch boundary. The mismatch
/// variance is scored against a member's own match distance instead.
pub(super) const BLKSIZE: u32 = 16;

const REFINE: u32 = 2;
const K_MAX: u32 = 8;
const SPATIAL_RADIUS: u32 = 4;

/// The knobs a run varies. Everything else follows the fixture.
struct Knobs {
    c_min: f32,
    k_max: u32,
    sigma: f32,
    lambda_ht: f32,
    mismatch_scale: f32,
    /// Whether a temporal member's own match distance inflates its
    /// noise variance. Every other test in this file relies on a
    /// uniform `sigma^2` across the whole group, so this defaults off
    /// and only the mismatch-variance test itself turns it on.
    use_member_sigma: bool,
    /// Half-width of each neighbour's refine window, defaulting to the
    /// module's [`REFINE`].
    refine: u32,
    /// The expected distance two noisy copies of the same content show
    /// by chance, subtracted from a member's raw match distance before
    /// it becomes mismatch variance. Every other test in this file
    /// leaves this at `0.0`, so a member's raw distance passes through
    /// unchanged.
    noise_floor: f32,
    /// The motion block side length, defaulting to the module's
    /// [`BLKSIZE`]. At [`BLK_STEP`] exactly one block covers a patch.
    blksize: u32,
}

impl Default for Knobs {
    fn default() -> Self {
        Knobs {
            c_min: 0.05,
            k_max: K_MAX,
            sigma: 0.02,
            lambda_ht: 2.7,
            mismatch_scale: 1.0,
            use_member_sigma: false,
            refine: REFINE,
            noise_floor: 0.0,
            blksize: BLKSIZE,
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
            k.noise_floor,
            k.c_min,
            k.mismatch_scale * k.mismatch_scale,
            k.lambda_ht,
            weight_scale(k.sigma, &profile),
            cross_frame_accum_scale(SPATIAL_RADIUS, fx.radius),
            k.use_member_sigma,
            needs_warp_uniform_search(&client),
            fx.radius,
            k.refine,
            fx.mv_stride,
            fx.conf_stride,
            BLK_STEP,
            k.blksize,
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

/// A temporal member's extra variance is its own match distance, per
/// channel and per pixel, times the scale squared.
///
/// `planted_ring` puts exact copies of the reference patch in every
/// neighbour. Adding a uniform offset `d` to each copy gives every
/// temporal member the distance `3 * 64 * d^2` and so the variance
/// `d^2 * scale^2`. With `lambda_ht` huge only the group DC survives,
/// whose variance is the ladder's level 0, and the group weight is its
/// reciprocal. `haar_variance_ladder` is the host mirror the GPU ladder
/// is already pinned against.
///
/// The run uses `refine: 0`, which collapses each neighbour's window to
/// its single motion-predicted position, exactly where `planted_ring`
/// puts the copy. That makes the group composition exact — self, the
/// four planted copies, and three centre-frame spatial members with no
/// mismatch variance of their own — so the expected variance below can
/// be written down at all. A wider window admits near-miss candidates
/// that tie with genuine spatial ones and leak mismatch variance into
/// what should be a clean baseline.
#[test]
fn a_temporal_member_carries_its_own_match_distance_as_variance() {
    use crate::collab::kernels::transforms::haar_variance_ladder;

    let (w, h) = (96u32, 96u32);
    let radius = 2u32;
    let ref_pos = (64u32, 64u32);
    let patch = deterministic_texture(5);
    let sigma = 0.02f32;
    let refs_x = refs_along(w);
    let ref_idx = ((ref_pos.1 / STEP) * refs_x + (ref_pos.0 / STEP)) as usize;

    for (d, scale) in [(0.0f32, 1.0f32), (0.05, 1.0), (0.05, 2.0), (0.1, 1.0)] {
        let mut fx = planted_ring(w, h, radius, ref_pos, 3, &patch, 0.2, |_| 1.0);
        // Offset every neighbour copy by d. The neighbour slots are
        // every slot but the centre.
        let pixels = (w * h) as usize;
        for slot in 0..(2 * radius + 1) {
            if slot == fx.centre_slot {
                continue;
            }
            let frame = &mut fx.ring[slot as usize * pixels..(slot as usize + 1) * pixels];
            for v in frame.iter_mut() {
                if *v > 0.5 {
                    *v += d;
                }
            }
        }

        let run = run_fused_over(
            &fx,
            Knobs {
                sigma,
                lambda_ht: 1.0e6,
                mismatch_scale: scale,
                use_member_sigma: true,
                refine: 0,
                ..Knobs::default()
            },
        );

        // Members sort by distance: self, then the four temporal
        // copies at 3 * 64 * d^2 each, then three flat spatial patches.
        let base = sigma * sigma;
        let mut v = [base; 8];
        for m in v.iter_mut().take(5).skip(1) {
            *m = base + d * d * scale * scale;
        }
        let expected = 1.0 / haar_variance_ladder(&v, 8)[0];
        let got = run.group_weight[ref_idx];
        assert!(
            (got - expected).abs() <= expected * 1e-3,
            "d={d} scale={scale}: expected group weight {expected}, got {got}"
        );
    }
}

/// A non-zero `noise_floor` subtracts from a temporal member's raw
/// match distance before it becomes variance, so a larger floor lowers
/// the member's variance.
///
/// Same fixture and offset as
/// [`a_temporal_member_carries_its_own_match_distance_as_variance`], at
/// `d = 0.1` and `scale = 1.0`, so each temporal member's raw distance
/// is `3 * 64 * d^2 = 1.92`. A floor of `0.96`, half that distance,
/// leaves excess `0.96` and so variance `0.96 / (3 * 64) = 0.005`, half
/// of the `d^2 = 0.01` a zero floor would give.
#[test]
fn a_noise_floor_lowers_a_temporal_members_variance_by_the_expected_amount() {
    use crate::collab::kernels::transforms::haar_variance_ladder;

    let (w, h) = (96u32, 96u32);
    let radius = 2u32;
    let ref_pos = (64u32, 64u32);
    let patch = deterministic_texture(5);
    let sigma = 0.02f32;
    let d = 0.1f32;
    let refs_x = refs_along(w);
    let ref_idx = ((ref_pos.1 / STEP) * refs_x + (ref_pos.0 / STEP)) as usize;

    let mut fx = planted_ring(w, h, radius, ref_pos, 3, &patch, 0.2, |_| 1.0);
    let pixels = (w * h) as usize;
    for slot in 0..(2 * radius + 1) {
        if slot == fx.centre_slot {
            continue;
        }
        let frame = &mut fx.ring[slot as usize * pixels..(slot as usize + 1) * pixels];
        for v in frame.iter_mut() {
            if *v > 0.5 {
                *v += d;
            }
        }
    }

    let raw_distance = 3.0 * PATCH_AREA as f32 * d * d;
    let noise_floor = raw_distance / 2.0;

    let run = run_fused_over(
        &fx,
        Knobs {
            sigma,
            lambda_ht: 1.0e6,
            use_member_sigma: true,
            refine: 0,
            noise_floor,
            ..Knobs::default()
        },
    );

    let base = sigma * sigma;
    let excess = (raw_distance - noise_floor).max(0.0);
    let member_variance = excess / (3.0 * PATCH_AREA as f32);
    let mut v = [base; 8];
    for m in v.iter_mut().take(5).skip(1) {
        *m = base + member_variance;
    }
    let expected = 1.0 / haar_variance_ladder(&v, 8)[0];
    let got = run.group_weight[ref_idx];
    assert!(
        (got - expected).abs() <= expected * 1e-3,
        "noise_floor={noise_floor}: expected group weight {expected} (member variance \
         {member_variance}), got {got}"
    );

    // The floor must actually have lowered the variance, not left it at
    // the zero-floor value the previous test measured at this same d.
    assert!(
        member_variance < d * d,
        "expected the floor to lower the member variance below the zero-floor value {}, got {}",
        d * d,
        member_variance
    );
}

/// Sets one block's vector toward neighbour `t`.
fn set_block_mv(fx: &mut RingFixture, t: u32, bx: u32, by: u32, mv: [i32; 2]) {
    let block = by * fx.blocks_x + bx;
    let base = (t * fx.mv_stride + block * 2) as usize;
    fx.mv_field[base] = mv[0];
    fx.mv_field[base + 1] = mv[1];
}

/// Writes an 8x8 patch into ring slot `slot` at `(px, py)`.
fn plant_in_slot(fx: &mut RingFixture, slot: u32, px: u32, py: u32, patch: &[f32; 64]) {
    let pixels = (fx.width * fx.height) as usize;
    let frame = &mut fx.ring[slot as usize * pixels..(slot as usize + 1) * pixels];
    for row in 0..8u32 {
        for col in 0..8u32 {
            frame[((py + row) * fx.width + px + col) as usize] = patch[(row * 8 + col) as usize];
        }
    }
}

/// Moves each neighbour's copy of the reference patch 20 pixels right,
/// leaving flat background where the reference sits, and points one
/// block's vector at the copy.
///
/// `planted_ring` at a zero shift puts a copy at the reference position
/// in every frame, so the copy there is erased first. Every block but
/// `(bx, by)` then holds the zeroed vector `planted_ring` left, which
/// points at flat background, so the copy is reachable only through
/// `(bx, by)`.
fn only_reachable_through(
    fx: &mut RingFixture,
    ref_pos: (u32, u32),
    patch: &[f32; 64],
    (bx, by): (u32, u32),
) {
    let flat = [0.2f32; 64];
    for t in 0..fx.neighbour_slots.len() as u32 {
        let slot = fx.neighbour_slots[t as usize];
        plant_in_slot(fx, slot, ref_pos.0, ref_pos.1, &flat);
        plant_in_slot(fx, slot, ref_pos.0 + 20, ref_pos.1, patch);
        set_block_mv(fx, t, 8, 8, [0, 0]);
        set_block_mv(fx, t, bx, by, [20, 0]);
    }
}

/// The corner block's vector points at flat background, and only a
/// neighbouring covering block's vector points at the planted copy.
///
/// The reference at (64, 64) sits on the corner of block (8, 8) and is
/// also covered by blocks (7, 7), (8, 7) and (7, 8), since a 16-pixel
/// block at an 8-pixel step covers two patches per axis. A search that
/// reads only the corner block never sees the copy.
///
/// Each of the three non-corner covering blocks is tried on its own,
/// `(7, 7)` diagonally, `(8, 7)` above and `(7, 8)` to the left, so a
/// kernel that read only the corner and the diagonal fails on two of
/// the three.
///
/// The ring runs at radius 2, so four neighbours each hold a copy the
/// covering block reaches. Half the group is then an exact copy of the
/// reference, against a group of near-background patches when only the
/// corner block is read, and the group weight separates the two by a
/// wide margin. The control leaves every block on the corner's zeroed
/// vector, so no rectangle reaches the copy however many blocks are
/// read.
#[test]
fn a_covering_block_other_than_the_corner_finds_the_match() {
    let (w, h) = (96u32, 96u32);
    let radius = 2u32;
    let ref_pos = (64u32, 64u32);
    let patch = deterministic_texture(13);
    let refs_x = refs_along(w);
    let ref_idx = ((ref_pos.1 / STEP) * refs_x + (ref_pos.0 / STEP)) as usize;

    // The same ring with every vector zeroed, so no block's rectangle
    // reaches the copy however many blocks are read.
    let mut corner_only = planted_ring(w, h, radius, ref_pos, 0, &patch, 0.2, |_| 1.0);
    only_reachable_through(&mut corner_only, ref_pos, &patch, (8, 8));
    corner_only.mv_field.fill(0);
    let without = run_fused_over(&corner_only, Knobs::default()).group_weight[ref_idx];

    for block in [(7u32, 7u32), (8, 7), (7, 8)] {
        let mut fx = planted_ring(w, h, radius, ref_pos, 0, &patch, 0.2, |_| 1.0);
        only_reachable_through(&mut fx, ref_pos, &patch, block);
        let with_covering = run_fused_over(&fx, Knobs::default()).group_weight[ref_idx];

        assert!(
            with_covering > without * 1.5,
            "the copies are only reachable through block {block:?}'s vector, expected a far \
             better group with it, got {with_covering} against {without}"
        );
    }
}

/// Two covering blocks whose vectors differ by one pixel give
/// overlapping rectangles, and a position inside both is scored once.
///
/// A copy planted where both rectangles reach it would otherwise enter
/// the group twice. With `lambda_ht` huge every member deposits the
/// same weight, so the planted patch's pixels receive exactly the
/// weight they receive when only one block points at it.
#[test]
fn overlapping_covering_rectangles_score_each_position_once() {
    let (w, h) = (96u32, 96u32);
    let radius = 1u32;
    let ref_pos = (64u32, 64u32);
    let patch = deterministic_texture(17);
    let pixels = (w * h) as usize;
    let knobs = || Knobs {
        lambda_ht: 1.0e6,
        ..Knobs::default()
    };

    let build = |second_vector: Option<[i32; 2]>| {
        let mut fx = planted_ring(w, h, radius, ref_pos, 0, &patch, 0.2, |_| 1.0);
        for t in 0..2u32 {
            let slot = fx.neighbour_slots[t as usize];
            plant_in_slot(&mut fx, slot, ref_pos.0 + 20, ref_pos.1, &patch);
            set_block_mv(&mut fx, t, 8, 8, [20, 0]);
            if let Some(v) = second_vector {
                set_block_mv(&mut fx, t, 7, 7, v);
            }
        }
        fx
    };

    let one = run_fused_over(&build(None), knobs());
    let two = run_fused_over(&build(Some([21, 0])), knobs());

    let frames = one.wsum.len() / pixels;
    let planted_centre =
        |run: &FusedRun, s: usize| run.wsum[s * pixels + ((ref_pos.1 + 4) * w + ref_pos.0 + 20 + 4) as usize];
    for s in 0..frames {
        if s as u32 == 1 {
            continue;
        }
        assert_eq!(
            planted_centre(&two, s),
            planted_centre(&one, s),
            "slot {s}: the planted copy must carry the same weight whether one or two covering \
             blocks reach it"
        );
    }
    assert!(
        planted_centre(&one, 0) > 0,
        "the copy must be a member in the first place"
    );
}

/// With `blksize == step` exactly one block covers a patch, so a
/// neighbouring block's vector is never consulted.
///
/// The copy is reachable only through block `(7, 7)`, which covers the
/// patch at `blksize = 16` and does not at `blksize = 8`.
#[test]
fn a_block_size_equal_to_the_step_reads_only_the_corner_block() {
    let (w, h) = (96u32, 96u32);
    let radius = 2u32;
    let ref_pos = (64u32, 64u32);
    let patch = deterministic_texture(19);
    let refs_x = refs_along(w);
    let ref_idx = ((ref_pos.1 / STEP) * refs_x + (ref_pos.0 / STEP)) as usize;

    let mut fx = planted_ring(w, h, radius, ref_pos, 0, &patch, 0.2, |_| 1.0);
    only_reachable_through(&mut fx, ref_pos, &patch, (7, 7));

    let covering = run_fused_over(&fx, Knobs::default()).group_weight[ref_idx];
    let single = run_fused_over(
        &fx,
        Knobs {
            blksize: BLK_STEP,
            ..Knobs::default()
        },
    )
    .group_weight[ref_idx];

    assert!(
        covering > single * 1.5,
        "at blksize == step the copy is unreachable, got {single} against {covering} with \
         covering blocks"
    );
}
