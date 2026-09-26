use cubecl::prelude::*;

use super::helpers::{
    BLK_STEP,
    R,
    RingFixture,
    deterministic_texture,
    fractional_pan_ring,
    make_client,
    noisy_ring,
    planted_ring,
};
use crate::collab::geometry::{fused_cubes_x, ref_count, refs_along};
use crate::collab::kernels::aggregate::{cross_frame_accum_scale, kaiser_window, weight_scale};
use crate::collab::kernels::fused::collab_fused;
use crate::collab::kernels::transforms::dct_noise_profile;
use crate::collab::{PATCH_SIZE, STEP, grid_frames, needs_warp_uniform_search};
use crate::nl4d::subpel::{phase_gains, phase_planes_host};
use crate::nlmeans::motion::neighbour_idx_for_k;

/// The motion block side length these fixtures score confidence
/// against, distinct from [`BLK_STEP`], which stays at `PATCH_SIZE` so
/// a block boundary lines up with a patch boundary.
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
    /// Half-width of each neighbour's refine window, defaulting to the
    /// module's [`REFINE`].
    refine: u32,
    /// Half-width of the centre frame's search window, defaulting to the
    /// module's [SPATIAL_RADIUS].
    spatial_radius: u32,
    /// The motion block side length, defaulting to the module's
    /// [`BLKSIZE`]. At [`BLK_STEP`] exactly one block covers a patch.
    blksize: u32,
    /// Pins the search walk rather than taking it from the runtime.
    /// `None`, the default, follows [`needs_warp_uniform_search`].
    warp_uniform: Option<bool>,
    /// The kernel's sub-pixel mode, where 0 is off, 1 half-pel and 2
    /// quarter-pel.
    subpel: u32,
}

impl Default for Knobs {
    fn default() -> Self {
        Knobs {
            c_min: 0.05,
            k_max: K_MAX,
            sigma: 0.02,
            lambda_ht: 2.7,
            refine: REFINE,
            spatial_radius: SPATIAL_RADIUS,
            blksize: BLKSIZE,
            warp_uniform: None,
            subpel: 0,
        }
    }
}

/// What one launch of [`collab_fused`] left behind.
struct FusedRun {
    wsum: Vec<i32>,
    group_weight: Vec<f32>,
    pixels: usize,
    /// The factor the launch multiplied `weight_scale` by.
    weight_floor: f32,
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

    /// [frame_weight_sum](FusedRun::frame_weight_sum) with the weight
    /// floor divided back out, so runs with subpel on and off compare
    /// directly.
    fn unscaled_frame_weight(&self, slot: u32) -> f64 {
        let weight = self.frame_weight_sum(slot) as f64;
        weight / self.weight_floor as f64
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
    let floor = if k.subpel > 0 {
        phase_gains().into_iter().fold(f32::MAX, f32::min)
    } else {
        1.0
    };
    let scaled_weight = weight_scale(k.sigma, &profile) * floor;

    let ring_buf = client.create_from_slice(f32::as_bytes(&fx.ring));
    let phase_ring = phase_ring_for(fx, k.subpel);
    let phase_bytes = f32::as_bytes(&phase_ring);
    let phase_buf = client.create_from_slice(phase_bytes);
    let gain_buf = client.create_from_slice(f32::as_bytes(&phase_gains()));
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
            ArrayArg::from_raw_parts(phase_buf, phase_ring.len()),
            ArrayArg::from_raw_parts(gain_buf, 16),
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
            k.c_min,
            k.lambda_ht,
            scaled_weight,
            cross_frame_accum_scale(k.spatial_radius, fx.radius),
            k.warp_uniform
                .unwrap_or_else(|| needs_warp_uniform_search(&client)),
            k.subpel,
            fx.radius,
            grid_frames(fx.radius),
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
            k.spatial_radius,
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
        weight_floor: floor,
    }
}

/// The phase ring the kernel reads, four host-built planes per frame when
/// `subpel` is on and the plain ring otherwise.
fn phase_ring_for(fx: &RingFixture, subpel: u32) -> Vec<f32> {
    if subpel == 0 {
        return fx.ring.clone();
    }

    let pixels = (fx.width * fx.height) as usize;
    let mut phase_ring = Vec::with_capacity(fx.ring.len() * 4);
    for frame in fx.ring.chunks(pixels) {
        let planes = phase_planes_host(frame, fx.width, fx.height, 1);
        for plane in planes {
            phase_ring.extend(plane);
        }
    }
    phase_ring
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
/// exactly zero. The gated neighbour is k = -2, the first one searched,
/// which wins every tie on this fixture. Gating one of four neighbours
/// leaves every volume its three frames, so every other neighbour still
/// receives members.
#[test]
fn low_confidence_neighbours_contribute_no_candidates() {
    let (w, h) = (96u32, 96u32);
    let radius = 2u32;
    let ref_pos = (64u32, 64u32);
    let patch = deterministic_texture(11);
    // Confidence 0.0 for k = -2, 1.0 for every other neighbour.
    let fx = planted_ring(w, h, radius, ref_pos, 3, &patch, 0.2, |k| {
        if k == -2 { 0.0 } else { 1.0 }
    });

    let run = run_fused_over(&fx, Knobs::default());

    for k in -(radius as i32)..=(radius as i32) {
        let slot = (k + radius as i32) as u32;
        let weight = run.frame_weight_sum(slot);
        if k == -2 {
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

/// Gating two of four neighbours leaves every volume short of three frames, so every group falls
/// back to a single frame and no neighbour slot receives any scatter.
#[test]
fn a_volume_short_of_frames_sends_the_group_to_the_fallback() {
    let radius = 2u32;
    let patch = deterministic_texture(11);
    let fx = planted_ring(96, 96, radius, (64, 64), 3, &patch, 0.2, |k| {
        if k > 0 { 0.0 } else { 1.0 }
    });

    let run = run_fused_over(&fx, Knobs::default());

    assert!(
        run.frame_weight_sum(fx.centre_slot) > 0,
        "the centre slot received nothing"
    );
    for slot in 0..(2 * radius + 1) {
        if slot == fx.centre_slot {
            continue;
        }
        assert_eq!(
            run.frame_weight_sum(slot),
            0,
            "slot {slot} must receive nothing once every group falls back"
        );
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
/// The ring runs at radius 2, so the reference's volume keeps three of
/// the four copies the covering block reaches. A static twin at
/// [TWIN_POS] anchors the second volume on the same patch in every
/// frame, so that volume is identical in every run and the group weight
/// only moves with the reference's own volume. The control leaves every
/// block on the corner's zeroed vector, so no rectangle reaches the copy
/// however many blocks are read.
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
    plant_static_twins(&mut corner_only, &[TWIN_POS], &patch);
    corner_only.mv_field.fill(0);
    let without = run_fused_over(&corner_only, twin_knobs()).group_weight[ref_idx];

    for block in [(7u32, 7u32), (8, 7), (7, 8)] {
        let mut fx = planted_ring(w, h, radius, ref_pos, 0, &patch, 0.2, |_| 1.0);
        only_reachable_through(&mut fx, ref_pos, &patch, block);
        plant_static_twins(&mut fx, &[TWIN_POS], &patch);
        let with_covering = run_fused_over(&fx, twin_knobs()).group_weight[ref_idx];

        assert!(
            with_covering > without * 1.5,
            "the copies are only reachable through block {block:?}'s vector, expected a far \
             better group with it, got {with_covering} against {without}"
        );
    }
}

/// Two covering blocks whose vectors differ by one pixel give
/// overlapping rectangles, and the reference's volume still finds the
/// copy they both reach.
///
/// Block `(7, 7)` is visited first, so with a second vector its
/// rectangle reaches the copy and block `(8, 8)` then skips the overlap.
/// The reference's volume must hold the same copy either way. Three
/// static twins anchor the other three volumes on the same patch in
/// every frame, so the group weight only moves with the reference's own
/// volume, and two runs holding the same patches carry the same weight.
/// The control points no block at the copy, which shows the weight can
/// see the copy go missing.
#[test]
fn overlapping_covering_rectangles_still_find_the_match() {
    let (w, h) = (96u32, 96u32);
    let radius = 1u32;
    let ref_pos = (64u32, 64u32);
    let patch = deterministic_texture(17);
    let refs_x = refs_along(w);
    let ref_idx = ((ref_pos.1 / STEP) * refs_x + (ref_pos.0 / STEP)) as usize;
    let flat = [0.2f32; 64];

    let build = |second_vector: Option<[i32; 2]>| {
        let mut fx = planted_ring(w, h, radius, ref_pos, 0, &patch, 0.2, |_| 1.0);
        plant_static_twins(&mut fx, &[(64, 48), (48, 64), (48, 48)], &patch);
        for t in 0..2u32 {
            let slot = fx.neighbour_slots[t as usize];
            plant_in_slot(&mut fx, slot, ref_pos.0, ref_pos.1, &flat);
            plant_in_slot(&mut fx, slot, ref_pos.0 + 20, ref_pos.1, &patch);
            set_block_mv(&mut fx, t, 8, 8, [20, 0]);
            if let Some(v) = second_vector {
                set_block_mv(&mut fx, t, 7, 7, v);
            }
        }
        fx
    };

    let mut unreachable = build(None);
    unreachable.mv_field.fill(0);

    let one_covering_block = build(None);
    let two_covering_blocks = build(Some([21, 0]));

    let one = run_fused_over(&one_covering_block, twin_knobs()).group_weight[ref_idx];
    let two = run_fused_over(&two_covering_blocks, twin_knobs()).group_weight[ref_idx];
    let none = run_fused_over(&unreachable, twin_knobs()).group_weight[ref_idx];

    assert_eq!(
        two, one,
        "the reference's volume must hold the same copy whether one or two covering blocks \
         reach it"
    );
    assert!(
        one > none * 1.5,
        "the copy must be a member in the first place, got {one} against {none} with no block \
         reaching it"
    );
}

/// Plants `patch` at each of `positions` in every slot of the ring.
///
/// Each becomes an exact spatial twin of a reference carrying the same patch, and its volume holds
/// that patch in every frame as long as its covering blocks carry a zero vector.
fn plant_static_twins(fx: &mut RingFixture, positions: &[(u32, u32)], patch: &[f32; 64]) {
    let frames = 2 * fx.radius + 1;
    for slot in 0..frames {
        for &(px, py) in positions {
            plant_in_slot(fx, slot, px, py, patch);
        }
    }
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

/// Where [twin_ring] plants the reference's spatial twin in the centre frame.
const TWIN_POS: (u32, u32) = (64, 48);

/// A radius-2 ring whose centre frame holds the reference texture at (64, 64) and an exact twin
/// at [TWIN_POS], so the spatial search anchors the second volume on the twin.
///
/// Every block moves the reference's copy by `(3k, 0)`, as `planted_ring` places it. The four
/// blocks covering the twin, `(7..=8, 5..=6)`, carry `twin_mv(k)` instead, and a copy of the twin
/// sits at `TWIN_POS + twin_copy(k)` in each neighbour. The two can differ, so a test can point
/// the twin's motion away from its copies.
fn twin_ring(
    patch: &[f32; 64],
    twin_mv: impl Fn(i32) -> [i32; 2],
    twin_copy: impl Fn(i32) -> [i32; 2],
) -> RingFixture {
    let radius = 2u32;
    let mut fx = planted_ring(96, 96, radius, (64, 64), 3, patch, 0.2, |_| 1.0);
    let centre_slot = fx.centre_slot;
    plant_in_slot(&mut fx, centre_slot, TWIN_POS.0, TWIN_POS.1, patch);

    for k in [-2i32, -1, 1, 2] {
        let t = neighbour_idx_for_k(radius, k);
        let slot = fx.neighbour_slots[t as usize];

        for by in 0..fx.blocks_y {
            for bx in 0..fx.blocks_x {
                set_block_mv(&mut fx, t, bx, by, [3 * k, 0]);
            }
        }

        for by in 5..=6u32 {
            for bx in 7..=8u32 {
                set_block_mv(&mut fx, t, bx, by, twin_mv(k));
            }
        }

        let [copy_dx, copy_dy] = twin_copy(k);
        let copy_x = (TWIN_POS.0 as i32 + copy_dx) as u32;
        let copy_y = (TWIN_POS.1 as i32 + copy_dy) as u32;
        plant_in_slot(&mut fx, slot, copy_x, copy_y, patch);
    }

    fx
}

/// The knobs every twin-ring run shares, a spatial window wide enough to reach the twin.
fn twin_knobs() -> Knobs {
    Knobs {
        spatial_radius: 16,
        ..Knobs::default()
    }
}

/// The second volume follows its own anchor's vector, not the reference's.
///
/// The twin's copies sit at `(0, 2k)` from it. With its blocks carrying that vector, its volume
/// holds three exact copies and the group agrees strongly. With the reference's `(3k, 0)` instead
/// its volume lands on flat background and the group weight collapses.
#[test]
fn each_volume_follows_its_own_anchors_motion() {
    let patch = deterministic_texture(23);
    let refs_x = refs_along(96);
    let ref_idx = ((64 / STEP) * refs_x + (64 / STEP)) as usize;

    let own = twin_ring(&patch, |k| [0, 2 * k], |k| [0, 2 * k]);
    let borrowed = twin_ring(&patch, |k| [3 * k, 0], |k| [0, 2 * k]);

    let with_own = run_fused_over(&own, twin_knobs()).group_weight[ref_idx];
    let with_borrowed = run_fused_over(&borrowed, twin_knobs()).group_weight[ref_idx];

    assert!(
        with_own > with_borrowed * 1.5,
        "the twin's volume should find its copies through its own vector, got {with_own} against \
         {with_borrowed}"
    );
}

/// A volume keeps its three best frames out of four.
///
/// Offsetting the copies of k = -2, the first neighbour searched and so the one that wins every
/// tie, must leave the group untouched, because both volumes skip that frame for the three exact
/// ones. A volume that kept its first three frames would hold the offset one instead. Two groups
/// holding the same patches carry the same weight, so the weight equals the run with no offset.
/// Offsetting k = -1 as well forces an offset frame into each volume, which moves the weight and
/// shows the comparison can see a change.
#[test]
fn a_volume_keeps_its_best_frames() {
    let patch = deterministic_texture(29);
    let refs_x = refs_along(96);
    let ref_idx = ((64 / STEP) * refs_x + (64 / STEP)) as usize;

    let clean = twin_ring(&patch, |k| [0, 2 * k], |k| [0, 2 * k]);
    let mut one_offset = twin_ring(&patch, |k| [0, 2 * k], |k| [0, 2 * k]);
    offset_copies(&mut one_offset, -2);
    let mut two_offset = twin_ring(&patch, |k| [0, 2 * k], |k| [0, 2 * k]);
    offset_copies(&mut two_offset, -2);
    offset_copies(&mut two_offset, -1);

    let clean_weight = run_fused_over(&clean, twin_knobs()).group_weight[ref_idx];
    let one_weight = run_fused_over(&one_offset, twin_knobs()).group_weight[ref_idx];
    let two_weight = run_fused_over(&two_offset, twin_knobs()).group_weight[ref_idx];

    assert_eq!(
        one_weight, clean_weight,
        "one offset neighbour must be skipped, leaving the group identical to the clean run"
    );
    assert!(
        (two_weight - clean_weight).abs() > clean_weight * 1e-3,
        "two offset neighbours force one into each volume, so the weight must move, got \
         {two_weight} against {clean_weight}"
    );
}

/// Raises every texture pixel of neighbour `k`'s frame by 0.1, leaving the background alone.
fn offset_copies(fx: &mut RingFixture, k: i32) {
    let pixels = (fx.width * fx.height) as usize;
    let t = neighbour_idx_for_k(fx.radius, k);
    let slot = fx.neighbour_slots[t as usize] as usize;
    let frame = &mut fx.ring[slot * pixels..(slot + 1) * pixels];
    for value in frame.iter_mut() {
        if *value > 0.5 {
            *value += 0.1;
        }
    }
}

/// A neighbour patch the first volume took is not reused by the second.
///
/// The twin's blocks point at the reference's copies, which match the twin exactly. Reusing them
/// would make the twin's volume hold the same pixels as the control's, where the twin follows its
/// own exact copies, and the two weights would be equal. Skipping them leaves the twin's volume one
/// unclaimed copy and two near misses, so its weight drops below the control's.
#[test]
fn a_position_claimed_by_an_earlier_volume_is_not_reused() {
    let patch = deterministic_texture(31);
    let refs_x = refs_along(96);
    let ref_idx = ((64 / STEP) * refs_x + (64 / STEP)) as usize;

    // The twin's vector lands on the reference's copy at (64 + 3k, 64).
    let bait = twin_ring(&patch, |k| [3 * k, 16], |k| [3 * k, 16]);
    let control = twin_ring(&patch, |k| [0, 2 * k], |k| [0, 2 * k]);

    let with_bait = run_fused_over(&bait, twin_knobs()).group_weight[ref_idx];
    let with_control = run_fused_over(&control, twin_knobs()).group_weight[ref_idx];

    assert!(
        with_bait < with_control * 0.99,
        "the twin's volume must not reuse the reference's copies, got {with_bait} against \
         {with_control}"
    );
}

/// The bait fixture above under both search walks, so the position skip
/// it relies on is not an artefact of whichever walk the runtime happens
/// to pick.
///
/// The bait's claimed positions are exactly where the two walks take
/// different turns to reach the same candidates, which is where a walk
/// that skipped the claim check differently would show up.
#[test]
fn a_position_claimed_by_an_earlier_volume_agrees_across_search_walks() {
    let patch = deterministic_texture(31);
    let bait = twin_ring(&patch, |k| [3 * k, 16], |k| [3 * k, 16]);

    let clipped_knobs = Knobs {
        warp_uniform: Some(false),
        ..twin_knobs()
    };
    let uniform_knobs = Knobs {
        warp_uniform: Some(true),
        ..twin_knobs()
    };

    let clipped = run_fused_over(&bait, clipped_knobs);
    let uniform = run_fused_over(&bait, uniform_knobs);

    assert_eq!(
        clipped.group_weight, uniform.group_weight,
        "the two search walks retired different groups"
    );
    assert_eq!(
        clipped.wsum, uniform.wsum,
        "the two search walks scattered different weights"
    );
    assert!(
        uniform.group_weight.iter().any(|&w| w > 0.0),
        "neither walk aggregated anything, so agreeing proves nothing"
    );
}

/// At 2.5 px per frame the neighbours one frame away sit exactly half a
/// pixel off the whole-pixel grid. With subpel on, their matches land at
/// the half phase, so those frames stop receiving scatter while the group
/// agrees better.
#[test]
fn a_half_pixel_pan_is_matched_at_the_half_phase() {
    let sigma = 0.004;
    let fixture = fractional_pan_ring(96, 96, 2, 2.5, sigma);
    let off_knobs = Knobs {
        sigma,
        ..Knobs::default()
    };
    let half_knobs = Knobs {
        sigma,
        subpel: 1,
        ..Knobs::default()
    };
    let off = run_fused_over(&fixture, off_knobs);
    let half = run_fused_over(&fixture, half_knobs);

    for k in [-1i32, 1] {
        let t = neighbour_idx_for_k(fixture.radius, k);
        let slot = fixture.neighbour_slots[t as usize];
        let off_weight = off.unscaled_frame_weight(slot);
        let half_weight = half.unscaled_frame_weight(slot);
        assert!(
            half_weight * 5.0 <= off_weight,
            "k={k}: half {half_weight} should be under a fifth of off {off_weight}"
        );
    }

    // The frames two away move a whole 5 px, so they keep their scatter.
    for k in [-2i32, 2] {
        let t = neighbour_idx_for_k(fixture.radius, k);
        let slot = fixture.neighbour_slots[t as usize];
        let off_weight = off.unscaled_frame_weight(slot);
        let half_weight = half.unscaled_frame_weight(slot);
        assert!(
            half_weight >= 0.9 * off_weight,
            "k={k}: half {half_weight} fell below 90% of off {off_weight}"
        );
    }

    let off_mean = mean_group_weight(&off);
    let half_mean = mean_group_weight(&half);
    assert!(
        half_mean > off_mean,
        "half {half_mean} should beat off {off_mean}"
    );
}

/// Whole-pixel motion over noisy content keeps its members on the whole
/// grid, so neighbour frames keep nearly all their scatter.
#[test]
fn whole_pixel_motion_keeps_whole_pixel_members() {
    let sigma = 0.02;
    let fixture = fractional_pan_ring(96, 96, 2, 3.0, sigma);
    assert_quarter_keeps_neighbour_scatter(&fixture, sigma);
}

/// Flat noisy content has no alignment to gain, so sub-pixel phases must
/// not win on noise alone.
///
/// `noisy_ring` is uniform noise around a flat 0.5, so the kernel is told
/// that noise's standard deviation, `1 / sqrt(12)`.
#[test]
fn flat_noise_keeps_whole_pixel_members() {
    let fixture = noisy_ring(96, 96, 2, 1.0);
    let sigma = (1.0f32 / 12.0).sqrt();
    assert_quarter_keeps_neighbour_scatter(&fixture, sigma);
}

#[test]
fn subpel_search_is_identical_under_the_warp_uniform_walk() {
    let sigma = 0.01;
    let fixture = fractional_pan_ring(96, 96, 2, 2.5, sigma);
    let plain_knobs = Knobs {
        sigma,
        subpel: 2,
        warp_uniform: Some(false),
        ..Knobs::default()
    };
    let uniform_knobs = Knobs {
        sigma,
        subpel: 2,
        warp_uniform: Some(true),
        ..Knobs::default()
    };
    let off_knobs = Knobs {
        sigma,
        ..Knobs::default()
    };
    let plain = run_fused_over(&fixture, plain_knobs);
    let uniform = run_fused_over(&fixture, uniform_knobs);
    let off = run_fused_over(&fixture, off_knobs);

    assert_eq!(plain.wsum, uniform.wsum);
    assert_eq!(plain.group_weight, uniform.group_weight);

    let any_aggregated = plain.group_weight.iter().any(|&weight| weight > 0.0);
    assert!(any_aggregated, "neither walk aggregated anything");

    // A neighbour frame losing most of its scatter shows fractional
    // members were chosen, so the walks agreed on a real subpel group.
    let fractional_slots = fixture
        .neighbour_slots
        .iter()
        .filter(|&&slot| {
            let plain_weight = plain.unscaled_frame_weight(slot);
            let off_weight = off.unscaled_frame_weight(slot);
            plain_weight < 0.5 * off_weight
        })
        .count();
    assert!(
        fractional_slots > 0,
        "no neighbour frame lost scatter, so no fractional member was chosen"
    );
}

/// Motion that drives every refine rectangle into the right and bottom
/// edges still reads inside the frame.
#[test]
fn subpel_search_at_the_frame_edges_stays_finite() {
    let sigma = 0.01;
    let mut fixture = fractional_pan_ring(64, 48, 2, 2.5, sigma);
    for vector in fixture.mv_field.chunks_mut(2) {
        vector[0] = 40;
        vector[1] = 40;
    }

    let knobs = Knobs {
        sigma,
        subpel: 2,
        ..Knobs::default()
    };

    let run = run_fused_over(&fixture, knobs);

    let all_finite = run.group_weight.iter().all(|weight| weight.is_finite());
    let all_non_negative = run.wsum.iter().all(|&weight| weight >= 0);
    let any_aggregated = run.group_weight.iter().any(|&weight| weight > 0.0);
    assert!(all_finite);
    assert!(all_non_negative);
    assert!(
        any_aggregated,
        "nothing aggregated, so the launch may have written nothing"
    );
}

fn mean_group_weight(run: &FusedRun) -> f32 {
    let total: f32 = run.group_weight.iter().sum();
    total / run.group_weight.len() as f32
}

/// Runs the fixture with subpel off and at quarter-pel, and checks every
/// neighbour frame keeps at least 90% of the scatter it had with it off.
fn assert_quarter_keeps_neighbour_scatter(fixture: &RingFixture, sigma: f32) {
    let off_knobs = Knobs {
        sigma,
        ..Knobs::default()
    };
    let quarter_knobs = Knobs {
        sigma,
        subpel: 2,
        ..Knobs::default()
    };
    let off = run_fused_over(fixture, off_knobs);
    let quarter = run_fused_over(fixture, quarter_knobs);

    for &slot in &fixture.neighbour_slots {
        let off_weight = off.unscaled_frame_weight(slot);
        let quarter_weight = quarter.unscaled_frame_weight(slot);
        assert!(
            quarter_weight >= 0.9 * off_weight,
            "slot {slot}: quarter {quarter_weight} fell below 90% of off {off_weight}"
        );
    }
}
