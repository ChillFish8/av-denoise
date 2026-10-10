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
use crate::collab::geometry::{fused_cubes_x, ref_count, refs_along, strength_map_dims};
use crate::collab::kernels::aggregate::{cross_frame_accum_scale, kaiser_window, weight_scale};
use crate::collab::kernels::fused::{STRENGTH_MAP_OFF, collab_fused};
use crate::collab::kernels::transforms::dct_noise_profile;
use crate::collab::{COLLAB_GROUPS, PATCH_SIZE, STEP, grid_frames, needs_warp_uniform_search};
use crate::nlmeans::NOISE_CURVE_BINS;
use crate::nlmeans::motion::neighbour_idx_for_k;

/// The motion block side length these fixtures score confidence against.
///
/// It differs from [BLK_STEP], which stays at `PATCH_SIZE` so a block boundary lines up with a
/// patch boundary.
pub(super) const BLKSIZE: u32 = 16;

const REFINE: u32 = 2;
const K_MAX: u32 = 8;
const SPATIAL_RADIUS: u32 = 4;

/// Where [twin_ring] plants the reference's spatial twin in the centre frame.
const TWIN_POS: (u32, u32) = (64, 48);

/// The knobs a run varies. Everything else follows the fixture.
struct Knobs {
    c_min: f32,
    k_max: u32,
    sigma: f32,
    lambda_ht: f32,
    /// Half-width of each neighbour's refine window.
    refine: u32,
    /// Half-width of the centre frame's search window.
    spatial_radius: u32,
    /// The motion block side length. At [BLK_STEP] exactly one block covers a patch.
    blksize: u32,
    /// Pins the search walk. `None` follows [needs_warp_uniform_search].
    warp_uniform: Option<bool>,
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
        }
    }
}

/// What one launch of [collab_fused] left behind.
struct FusedRun {
    wsum: Vec<i32>,
    group_weight: Vec<f32>,
    pixels: usize,
}

impl FusedRun {
    /// The total weight one ring slot's region received.
    fn frame_weight_sum(&self, slot: u32) -> i64 {
        let start = slot as usize * self.pixels;
        self.wsum[start..start + self.pixels]
            .iter()
            .map(|&weight| weight as i64)
            .sum()
    }

    /// The total weight the whole ring received.
    ///
    /// Every group contributes one patch of 64 pixels per member, so at a fixed per-group weight
    /// this counts members.
    fn total_weight(&self) -> i64 {
        self.wsum.iter().map(|&weight| weight as i64).sum()
    }
}

/// Launches [collab_fused] over a fixture on the denoiser's eight-references-per-cube grid.
///
/// Luma always stores one channel per line, so the kernel's `Size` selector is fixed at 1.
fn run_fused_over(fixture: &RingFixture, knobs: &Knobs) -> FusedRun {
    let client = make_client();
    let width = fixture.width;
    let height = fixture.height;
    let pixels = (width * height) as usize;
    let frames = fixture.ring.len() / pixels;
    let refs = ref_count(width, height);
    let refs_x = refs_along(width);
    let profile = dct_noise_profile(0.0);
    let kaiser = kaiser_window(0.0);
    let zeroed_curve = [0.0f32; NOISE_CURVE_BINS];
    let zeroed_accum = vec![0i32; pixels * frames];
    let zeroed_wsum = vec![0i32; pixels * frames];

    let ring_bytes = f32::as_bytes(&fixture.ring);
    let mv_bytes = i32::as_bytes(&fixture.mv_field);
    let conf_bytes = f32::as_bytes(&fixture.confidence);
    let slots_bytes = u32::as_bytes(&fixture.neighbour_slots);
    let sigma = [knobs.sigma];
    let sigma_bytes = f32::as_bytes(&sigma);
    let profile_bytes = f32::as_bytes(&profile);
    let kaiser_bytes = f32::as_bytes(&kaiser);
    let curve_bytes = f32::as_bytes(&zeroed_curve);
    let ring_buf = client.create_from_slice(ring_bytes);
    let mv_buf = client.create_from_slice(mv_bytes);
    let conf_buf = client.create_from_slice(conf_bytes);
    let slots_buf = client.create_from_slice(slots_bytes);
    let sigma_buf = client.create_from_slice(sigma_bytes);
    let profile_buf = client.create_from_slice(profile_bytes);
    let kaiser_buf = client.create_from_slice(kaiser_bytes);
    let zero_curve = client.create_from_slice(curve_bytes);

    let (map_cols, map_rows) = strength_map_dims(width, height);
    let map_len = (map_cols * map_rows) as usize;
    let unit_map = vec![1.0f32; map_len];
    let unit_map_bytes = f32::as_bytes(&unit_map);
    let unit_map_buf = client.create_from_slice(unit_map_bytes);

    let accum_bytes = i32::as_bytes(&zeroed_accum);
    let wsum_bytes = i32::as_bytes(&zeroed_wsum);
    let accum = client.create_from_slice(accum_bytes);
    let wsum = client.create_from_slice(wsum_bytes);
    let group_weight = client.empty(refs * size_of::<f32>());

    let cubes_x = fused_cubes_x(width);
    let refs_y = refs_along(height);
    let grid = CubeCount::new_2d(cubes_x, refs_y);
    let dim = CubeDim::new_1d(64);
    let scale = weight_scale(knobs.sigma, &profile);
    let accum_scale = cross_frame_accum_scale(knobs.spatial_radius, fixture.radius);
    let warp_uniform = knobs
        .warp_uniform
        .unwrap_or_else(|| needs_warp_uniform_search(&client));
    let grid_frame_count = grid_frames(fixture.radius);

    let stored_ch = 1usize;

    unsafe {
        collab_fused::launch_unchecked::<f32, R>(
            &client,
            grid,
            dim,
            stored_ch,
            ArrayArg::from_raw_parts(ring_buf.clone(), fixture.ring.len()),
            ArrayArg::from_raw_parts(ring_buf, stored_ch),
            ArrayArg::from_raw_parts(mv_buf, fixture.mv_field.len()),
            ArrayArg::from_raw_parts(conf_buf, fixture.confidence.len()),
            ArrayArg::from_raw_parts(slots_buf, fixture.neighbour_slots.len()),
            ArrayArg::from_raw_parts(sigma_buf, 1),
            ArrayArg::from_raw_parts(zero_curve, NOISE_CURVE_BINS),
            ArrayArg::from_raw_parts(unit_map_buf, map_len),
            ArrayArg::from_raw_parts(profile_buf, 8),
            ArrayArg::from_raw_parts(kaiser_buf, PATCH_SIZE as usize),
            ArrayArg::from_raw_parts(accum, pixels * frames),
            ArrayArg::from_raw_parts(wsum.clone(), pixels * frames),
            ArrayArg::from_raw_parts(group_weight.clone(), refs),
            fixture.centre_slot,
            knobs.c_min,
            knobs.lambda_ht,
            0u32,
            STRENGTH_MAP_OFF,
            scale,
            accum_scale,
            warp_uniform,
            false,
            fixture.radius,
            grid_frame_count,
            knobs.refine,
            fixture.mv_stride,
            fixture.conf_stride,
            BLK_STEP,
            knobs.blksize,
            fixture.blocks_x,
            fixture.blocks_y,
            width,
            height,
            1u32,
            knobs.k_max,
            1u32,
            knobs.spatial_radius,
            refs_x,
            map_cols,
            map_rows,
            0.0f32,
            false,
            COLLAB_GROUPS,
            false,
            false,
            0,
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
/// Every neighbour holds an exact copy of the reference patch shifted by `3 * k`, and the motion
/// field predicts that shift. Following it gives a group of exact copies whose Haar detail
/// collapses, so the group weight is high. The control zeroes the motion field while the copies
/// stay in the ring, so this tests the prediction and not whether the content is reachable.
#[test]
fn temporal_members_are_found_at_the_mv_prediction() {
    let (width, height) = (96u32, 96u32);
    let radius = 2u32;
    let ref_pos = (64u32, 64u32);
    let patch = deterministic_texture(7);

    let predicted = planted_ring(width, height, radius, ref_pos, 3, &patch, 0.2, |_| 1.0);
    let mut blind = planted_ring(width, height, radius, ref_pos, 3, &patch, 0.2, |_| 1.0);
    blind.mv_field.fill(0);

    let refs_x = refs_along(width);
    let ref_idx = ((ref_pos.1 / STEP) * refs_x + (ref_pos.0 / STEP)) as usize;

    let knobs = Knobs::default();
    let with_prediction = run_fused_over(&predicted, &knobs).group_weight[ref_idx];
    let without = run_fused_over(&blind, &knobs).group_weight[ref_idx];

    assert!(
        with_prediction > without * 1.5,
        "expected the group at {ref_pos:?} to agree far better when the motion field points at \
         the planted copies, got weight {with_prediction} with the prediction and {without} \
         with a zeroed field"
    );
}

/// A neighbour whose motion-block confidence sits below `c_min` is skipped outright.
///
/// The confidence is uniform per neighbour, so the gated neighbour's whole region of the ring
/// must stay exactly zero. The gated neighbour is k = -2, the first one searched, which wins every
/// tie on this fixture. Gating one of four neighbours leaves every volume its three frames, so
/// every other neighbour still receives members.
#[test]
fn low_confidence_neighbours_contribute_no_candidates() {
    let (width, height) = (96u32, 96u32);
    let radius = 2u32;
    let ref_pos = (64u32, 64u32);
    let patch = deterministic_texture(11);
    let fixture = planted_ring(width, height, radius, ref_pos, 3, &patch, 0.2, |k| {
        if k == -2 { 0.0 } else { 1.0 }
    });

    let knobs = Knobs::default();
    let run = run_fused_over(&fixture, &knobs);

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
    let fixture = planted_ring(96, 96, radius, (64, 64), 3, &patch, 0.2, |k| {
        if k > 0 { 0.0 } else { 1.0 }
    });

    let knobs = Knobs::default();
    let run = run_fused_over(&fixture, &knobs);
    let centre_weight = run.frame_weight_sum(fixture.centre_slot);

    assert!(centre_weight > 0, "the centre slot received nothing");

    for slot in 0..(2 * radius + 1) {
        if slot == fixture.centre_slot {
            continue;
        }

        let weight = run.frame_weight_sum(slot);
        assert_eq!(
            weight, 0,
            "slot {slot} must receive nothing once every group falls back"
        );
    }
}

/// Every group fills to `k_max` however poor its candidates are, because there is no admission
/// gate.
///
/// No 8x8 window of `noisy_ring` resembles any other, so every candidate is a bad match. A huge
/// `lambda_ht` keeps only the forced group DC, which pins every group's weight at the same
/// constant, so the total weight in the ring counts members. A run capped at `k_max = 1` holds
/// every group to its self-match, so the full run must deposit exactly eight times as much.
#[test]
fn no_admission_gate_means_the_group_always_fills() {
    let (width, height) = (64u32, 64u32);
    let radius = 2u32;
    let fixture = noisy_ring(width, height, radius, 1.0);

    // The smallest search space here is the 5x5 rectangle a corner clips to, so every group has
    // at least eight positions to choose from.
    let full_knobs = Knobs {
        lambda_ht: 1.0e6,
        ..Knobs::default()
    };
    let single_knobs = Knobs {
        k_max: 1,
        lambda_ht: 1.0e6,
        ..Knobs::default()
    };
    let full = run_fused_over(&fixture, &full_knobs);
    let single = run_fused_over(&fixture, &single_knobs);

    let one = single.total_weight();
    assert!(one > 0, "the k_max = 1 run deposited no weight at all");

    let full_weight = full.total_weight();
    assert_eq!(
        full_weight,
        one * K_MAX as i64,
        "expected every group to carry {K_MAX} members, so {K_MAX}x the weight the \
         one-member run deposited"
    );
}

/// Sets the vector of block `(block_x, block_y)` toward neighbour index `neighbour`.
fn set_block_mv(fixture: &mut RingFixture, neighbour: u32, block_x: u32, block_y: u32, vector: [i32; 2]) {
    let block = block_y * fixture.blocks_x + block_x;
    let base = (neighbour * fixture.mv_stride + block * 2) as usize;
    fixture.mv_field[base] = vector[0];
    fixture.mv_field[base + 1] = vector[1];
}

/// Writes an 8x8 patch into ring slot `slot` with its top-left corner at `(x, y)`.
fn plant_in_slot(fixture: &mut RingFixture, slot: u32, x: u32, y: u32, patch: &[f32; 64]) {
    let pixels = (fixture.width * fixture.height) as usize;
    let frame = &mut fixture.ring[slot as usize * pixels..(slot as usize + 1) * pixels];
    for row in 0..8u32 {
        for col in 0..8u32 {
            frame[((y + row) * fixture.width + x + col) as usize] = patch[(row * 8 + col) as usize];
        }
    }
}

/// Moves each neighbour's copy of the reference patch 20 pixels right and points one block's
/// vector at it.
///
/// The copy at the reference position is erased first. Every other block keeps the zero vector,
/// which points at flat background, so the copy is reachable only through `block`.
fn only_reachable_through(
    fixture: &mut RingFixture,
    ref_pos: (u32, u32),
    patch: &[f32; 64],
    (block_x, block_y): (u32, u32),
) {
    let flat = [0.2f32; 64];
    for neighbour in 0..fixture.neighbour_slots.len() as u32 {
        let slot = fixture.neighbour_slots[neighbour as usize];
        plant_in_slot(fixture, slot, ref_pos.0, ref_pos.1, &flat);
        plant_in_slot(fixture, slot, ref_pos.0 + 20, ref_pos.1, patch);
        set_block_mv(fixture, neighbour, 8, 8, [0, 0]);
        set_block_mv(fixture, neighbour, block_x, block_y, [20, 0]);
    }
}

/// The corner block's vector points at flat background, and only a neighbouring covering block's
/// vector points at the planted copy.
///
/// A 16-pixel block at an 8-pixel step covers two patches per axis, so the reference at (64, 64)
/// is covered by blocks (8, 8), (7, 7), (8, 7) and (7, 8). Each non-corner block is tried on its
/// own, so a kernel that read only the corner and the diagonal fails on two of the three.
///
/// The ring runs at radius 2, so the reference's volume keeps three of the four copies the
/// covering block reaches. A static twin at [TWIN_POS] keeps the second volume identical in every
/// run, so the group weight only moves with the reference's own volume. The control leaves every
/// block on the zero vector, so no rectangle reaches the copy.
#[test]
fn a_covering_block_other_than_the_corner_finds_the_match() {
    let (width, height) = (96u32, 96u32);
    let radius = 2u32;
    let ref_pos = (64u32, 64u32);
    let patch = deterministic_texture(13);
    let refs_x = refs_along(width);
    let ref_idx = ((ref_pos.1 / STEP) * refs_x + (ref_pos.0 / STEP)) as usize;

    let knobs = twin_knobs();

    let mut corner_only = planted_ring(width, height, radius, ref_pos, 0, &patch, 0.2, |_| 1.0);
    only_reachable_through(&mut corner_only, ref_pos, &patch, (8, 8));
    plant_static_twins(&mut corner_only, &[TWIN_POS], &patch);
    corner_only.mv_field.fill(0);
    let without = run_fused_over(&corner_only, &knobs).group_weight[ref_idx];

    for block in [(7u32, 7u32), (8, 7), (7, 8)] {
        let mut fixture = planted_ring(width, height, radius, ref_pos, 0, &patch, 0.2, |_| 1.0);
        only_reachable_through(&mut fixture, ref_pos, &patch, block);
        plant_static_twins(&mut fixture, &[TWIN_POS], &patch);
        let with_covering = run_fused_over(&fixture, &knobs).group_weight[ref_idx];

        assert!(
            with_covering > without * 1.5,
            "the copies are only reachable through block {block:?}'s vector, expected a far \
             better group with it, got {with_covering} against {without}"
        );
    }
}

/// Two covering blocks whose vectors differ by one pixel give overlapping rectangles, and the
/// reference's volume still finds the copy they both reach.
///
/// Block `(7, 7)` is visited first, so with a second vector its rectangle reaches the copy and
/// block `(8, 8)` then skips the overlap. Three static twins keep the other volumes identical in
/// every run, so the group weight only moves with the reference's own volume. The control points
/// no block at the copy, which shows the weight can see the copy go missing.
#[test]
fn overlapping_covering_rectangles_still_find_the_match() {
    let (width, height) = (96u32, 96u32);
    let radius = 1u32;
    let ref_pos = (64u32, 64u32);
    let patch = deterministic_texture(17);
    let refs_x = refs_along(width);
    let ref_idx = ((ref_pos.1 / STEP) * refs_x + (ref_pos.0 / STEP)) as usize;
    let flat = [0.2f32; 64];

    let build = |second_vector: Option<[i32; 2]>| {
        let mut fixture = planted_ring(width, height, radius, ref_pos, 0, &patch, 0.2, |_| 1.0);
        plant_static_twins(&mut fixture, &[(64, 48), (48, 64), (48, 48)], &patch);

        for neighbour in 0..2u32 {
            let slot = fixture.neighbour_slots[neighbour as usize];
            plant_in_slot(&mut fixture, slot, ref_pos.0, ref_pos.1, &flat);
            plant_in_slot(&mut fixture, slot, ref_pos.0 + 20, ref_pos.1, &patch);
            set_block_mv(&mut fixture, neighbour, 8, 8, [20, 0]);
            if let Some(vector) = second_vector {
                set_block_mv(&mut fixture, neighbour, 7, 7, vector);
            }
        }

        fixture
    };

    let mut unreachable = build(None);
    unreachable.mv_field.fill(0);

    let one_covering_block = build(None);
    let two_covering_blocks = build(Some([21, 0]));

    let knobs = twin_knobs();
    let one = run_fused_over(&one_covering_block, &knobs).group_weight[ref_idx];
    let two = run_fused_over(&two_covering_blocks, &knobs).group_weight[ref_idx];
    let none = run_fused_over(&unreachable, &knobs).group_weight[ref_idx];

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
fn plant_static_twins(fixture: &mut RingFixture, positions: &[(u32, u32)], patch: &[f32; 64]) {
    let frames = 2 * fixture.radius + 1;
    for slot in 0..frames {
        for &(x, y) in positions {
            plant_in_slot(fixture, slot, x, y, patch);
        }
    }
}

/// With `blksize == step` exactly one block covers a patch, so a neighbouring block's vector is
/// never consulted.
///
/// The copy is reachable only through block `(7, 7)`, which covers the patch at `blksize = 16`
/// and does not at `blksize = 8`.
#[test]
fn a_block_size_equal_to_the_step_reads_only_the_corner_block() {
    let (width, height) = (96u32, 96u32);
    let radius = 2u32;
    let ref_pos = (64u32, 64u32);
    let patch = deterministic_texture(19);
    let refs_x = refs_along(width);
    let ref_idx = ((ref_pos.1 / STEP) * refs_x + (ref_pos.0 / STEP)) as usize;

    let mut fixture = planted_ring(width, height, radius, ref_pos, 0, &patch, 0.2, |_| 1.0);
    only_reachable_through(&mut fixture, ref_pos, &patch, (7, 7));

    let covering_knobs = Knobs::default();
    let single_block_knobs = Knobs {
        blksize: BLK_STEP,
        ..Knobs::default()
    };
    let covering = run_fused_over(&fixture, &covering_knobs).group_weight[ref_idx];
    let single = run_fused_over(&fixture, &single_block_knobs).group_weight[ref_idx];

    assert!(
        covering > single * 1.5,
        "at blksize == step the copy is unreachable, got {single} against {covering} with \
         covering blocks"
    );
}

/// A radius-2 ring whose centre frame holds the reference texture at (64, 64) and an exact twin
/// at [TWIN_POS], so the spatial search anchors the second volume on the twin.
///
/// Every block moves the reference's copy by `(3k, 0)`. The four blocks covering the twin,
/// `(7..=8, 5..=6)`, carry `twin_mv(k)` instead, and a copy of the twin sits at
/// `TWIN_POS + twin_copy(k)` in each neighbour. The two can differ, so a test can point the
/// twin's motion away from its copies.
fn twin_ring(
    patch: &[f32; 64],
    twin_mv: impl Fn(i32) -> [i32; 2],
    twin_copy: impl Fn(i32) -> [i32; 2],
) -> RingFixture {
    let radius = 2u32;
    let mut fixture = planted_ring(96, 96, radius, (64, 64), 3, patch, 0.2, |_| 1.0);
    let centre_slot = fixture.centre_slot;
    plant_in_slot(&mut fixture, centre_slot, TWIN_POS.0, TWIN_POS.1, patch);

    for k in [-2i32, -1, 1, 2] {
        let neighbour = neighbour_idx_for_k(radius, k);
        let slot = fixture.neighbour_slots[neighbour as usize];

        for block_y in 0..fixture.blocks_y {
            for block_x in 0..fixture.blocks_x {
                set_block_mv(&mut fixture, neighbour, block_x, block_y, [3 * k, 0]);
            }
        }

        let twin_vector = twin_mv(k);
        for block_y in 5..=6u32 {
            for block_x in 7..=8u32 {
                set_block_mv(&mut fixture, neighbour, block_x, block_y, twin_vector);
            }
        }

        let [copy_dx, copy_dy] = twin_copy(k);
        let copy_x = (TWIN_POS.0 as i32 + copy_dx) as u32;
        let copy_y = (TWIN_POS.1 as i32 + copy_dy) as u32;
        plant_in_slot(&mut fixture, slot, copy_x, copy_y, patch);
    }

    fixture
}

/// A spatial window wide enough to reach the twin.
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

    let knobs = twin_knobs();
    let with_own = run_fused_over(&own, &knobs).group_weight[ref_idx];
    let with_borrowed = run_fused_over(&borrowed, &knobs).group_weight[ref_idx];

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
/// holding the same patches carry the same weight. Offsetting k = -1 as well forces an offset
/// frame into each volume, which shows the comparison can see a change.
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

    let knobs = twin_knobs();
    let clean_weight = run_fused_over(&clean, &knobs).group_weight[ref_idx];
    let one_weight = run_fused_over(&one_offset, &knobs).group_weight[ref_idx];
    let two_weight = run_fused_over(&two_offset, &knobs).group_weight[ref_idx];

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
fn offset_copies(fixture: &mut RingFixture, k: i32) {
    let pixels = (fixture.width * fixture.height) as usize;
    let neighbour = neighbour_idx_for_k(fixture.radius, k);
    let slot = fixture.neighbour_slots[neighbour as usize] as usize;
    let frame = &mut fixture.ring[slot * pixels..(slot + 1) * pixels];
    for value in frame.iter_mut() {
        if *value > 0.5 {
            *value += 0.1;
        }
    }
}

/// A neighbour patch the first volume took is not reused by the second.
///
/// The twin's blocks point at the reference's copies, which match the twin exactly. Reusing them
/// would give the twin's volume the same pixels as the control's, where the twin follows its own
/// exact copies, and the two weights would be equal. Skipping them leaves the twin's volume one
/// unclaimed copy and two near misses, so its weight drops below the control's.
#[test]
fn a_position_claimed_by_an_earlier_volume_is_not_reused() {
    let patch = deterministic_texture(31);
    let refs_x = refs_along(96);
    let ref_idx = ((64 / STEP) * refs_x + (64 / STEP)) as usize;

    // The twin's vector lands on the reference's copy at (64 + 3k, 64).
    let bait = twin_ring(&patch, |k| [3 * k, 16], |k| [3 * k, 16]);
    let control = twin_ring(&patch, |k| [0, 2 * k], |k| [0, 2 * k]);

    let knobs = twin_knobs();
    let with_bait = run_fused_over(&bait, &knobs).group_weight[ref_idx];
    let with_control = run_fused_over(&control, &knobs).group_weight[ref_idx];

    assert!(
        with_bait < with_control * 0.99,
        "the twin's volume must not reuse the reference's copies, got {with_bait} against \
         {with_control}"
    );
}

/// The claimed-position bait fixture under both search walks.
///
/// The claimed positions are exactly where the two walks take different turns to reach the same
/// candidates, so a walk that skipped the claim check differently would show up here.
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

    let clipped = run_fused_over(&bait, &clipped_knobs);
    let uniform = run_fused_over(&bait, &uniform_knobs);

    assert_eq!(
        clipped.group_weight, uniform.group_weight,
        "the two search walks retired different groups"
    );
    assert_eq!(
        clipped.wsum, uniform.wsum,
        "the two search walks scattered different weights"
    );
    assert!(
        uniform.group_weight.iter().any(|&weight| weight > 0.0),
        "neither walk aggregated anything, so agreeing proves nothing"
    );
}
