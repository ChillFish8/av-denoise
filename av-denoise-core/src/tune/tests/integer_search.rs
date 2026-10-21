use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use cubecl::prelude::*;

use super::corpus::{integer_corpus, is_flat_case};
use super::fixtures::{CollabSpec, collab_site, collab_site_with_ring, integer_split_sites, split_sites};
use super::runtime::{TestRuntime, test_client};
use crate::collab::geometry::{ref_pos, refs_along};
use crate::collab::integer::{CodeView, host_centre_members, pack_position, patch_ssd};
use crate::collab::kernels::fused::{GROUP_SIZE_MASK, GROUP_USES_GRID, MEMBER_WORDS, Walk};
use crate::collab::{MAX_K, PATCH_SIZE, STEP};
use crate::nlmeans::ChannelMode;
use crate::tune::gpu::{read_f32, read_i32, read_u16};
use crate::tune::sites::{
    CollabSearchSite,
    IntCentre,
    IntNeighbour,
    IntegerVariant,
    SearchOutputs,
    SearchVariant,
};
use crate::tune::{
    CompareResult,
    DeviceLimits,
    EngineKernels,
    Fidelity,
    IntegerCodes,
    KernelSelector,
    LaunchShape,
    Legality,
    SiteCandidate,
    SiteKey,
    TuneLedger,
    TunePhase,
    VariantSite,
    legal_candidates,
};

const EIGHT_BIT: IntegerCodes = IntegerCodes { depth: 8 };
pub(super) const TEN_BIT: IntegerCodes = IntegerCodes { depth: 10 };

fn integer_variant(walk: Walk) -> SearchVariant {
    let variant = IntegerVariant {
        walk,
        ..IntegerVariant::reference_for(false)
    };

    SearchVariant::Integer(variant)
}

fn tiled_variant() -> SearchVariant {
    let variant = IntegerVariant {
        centre: IntCentre::Tiled,
        ..IntegerVariant::reference_for(false)
    };

    SearchVariant::Integer(variant)
}

/// The integer reference this device runs, the uniform walk where only it may run.
pub(super) fn reference_variant(search: &CollabSearchSite<TestRuntime, f32>) -> SearchVariant {
    search
        .integer_reference()
        .expect("a search with codes has an integer reference")
}

/// Launches `candidate` into fresh outputs and returns them with their words.
pub(super) fn run_search(
    search: &CollabSearchSite<TestRuntime, f32>,
    candidate: SearchVariant,
) -> (SearchOutputs, Vec<u32>) {
    let outputs = search.scratch_outputs();
    search.launch(candidate, &outputs);

    let word_count = outputs.words();
    let words = read_i32(&search.client, &outputs.members, word_count).expect("the members read back");
    let unsigned = words.iter().map(|&word| word as u32).collect();

    (outputs, unsigned)
}

/// The packed top-left of reference `index` in a frame of `spec`.
pub(super) fn reference_position(spec: &CollabSpec, index: usize) -> u32 {
    let refs_x = refs_along(spec.width) as usize;
    let column = (index % refs_x) as u32;
    let row = (index / refs_x) as u32;
    let ref_x = ref_pos(column, spec.width);
    let ref_y = ref_pos(row, spec.height);

    pack_position(ref_x, ref_y, 0)
}

/// Checks the centre members in `words` against the host search of every reference, for a
/// single-frame search of `spec`.
pub(super) fn assert_matches_host_centre(
    search: &CollabSearchSite<TestRuntime, f32>,
    spec: &CollabSpec,
    words: &[u32],
    label: &str,
) {
    let inputs = &search.inputs;
    let code_values = read_u16(&search.client, &inputs.codes, inputs.codes_len).expect("the codes read back");
    let widened: Vec<u32> = code_values.iter().map(|&code| u32::from(code)).collect();
    let view = CodeView {
        codes: &widened,
        width: spec.width,
        height: spec.height,
        stored_ch: spec.channels.storage_count(),
        channels: spec.channels.count(),
        centre_slot: 0,
        neighbour_slots: &[],
    };

    let groups = words.chunks(MEMBER_WORDS as usize).enumerate();
    for (index, group) in groups {
        let reference_word = reference_position(spec, index);
        let ref_x = reference_word & 0x1FFF;
        let ref_y = reference_word >> 13;
        let expected = host_centre_members(&view, ref_x, ref_y, spec.spatial_radius);

        assert_eq!(
            &group[..expected.len()],
            expected.as_slice(),
            "{label}: reference {index}"
        );
    }
}

#[test]
fn scalar_int_matches_the_host_reference_on_small_frames() {
    let client = test_client();
    let channel_modes = [ChannelMode::Luma, ChannelMode::Chroma, ChannelMode::Yuv];

    // Flat frames tie every candidate, so the members follow the position order alone.
    let cases = channel_modes
        .into_iter()
        .flat_map(|channels| [(channels, false), (channels, true)]);

    for (channels, flat) in cases {
        for codes in [EIGHT_BIT, TEN_BIT] {
            let label = format!("{channels:?} flat={flat} {}-bit", codes.depth);
            let mut spec = CollabSpec::new(40, 24, channels, 0);
            spec.flat = flat;
            let fused = collab_site(&client, &spec);
            let (search, _) = integer_split_sites(&fused, codes);
            let reference = reference_variant(&search);
            let (_, words) = run_search(&search, reference);

            assert_matches_host_centre(&search, &spec, &words, &label);
        }
    }
}

/// The frame widths and bright patch columns of the brightest 10-bit frames.
///
/// The 40-pixel frame tiles every reference into sliding blocks. In the 42-pixel frame the patch
/// sits under a reference whose blocks do not tile, so the sliding centre scores it directly.
pub(super) const BRIGHTEST_LAYOUTS: [(u32, u32); 2] = [(40, 16), (42, 32)];

/// A yuv frame `width` by 24 of code 0 around an all-1023 patch at `(left, 8)`, where the columns
/// and rows beside the patch sit at the f32 partial sums' bound.
///
/// The diagonal neighbours at `(left - 1, 7)` and `(left + 1, 9)` tie exactly. One code of 1022
/// in each, at opposite ends of their dark column and row, would round them apart if a column or
/// row summed its channels in f32, which the position order then exposes.
pub(super) fn brightest_ten_bit_frame(width: u32, left: u32) -> (CollabSpec, Vec<f32>) {
    let spec = CollabSpec::new(width, 24, ChannelMode::Yuv, 0);
    let stored_ch = spec.channels.storage_count();
    let summed_ch = spec.channels.count();
    let pixels = spec.width * spec.height;
    let mut frame = vec![0.0f32; (pixels * stored_ch) as usize];

    for row in 8..16 {
        for column in left..left + 8 {
            let first_lane = (row * spec.width + column) * stored_ch;
            let lanes = first_lane as usize..(first_lane + summed_ch) as usize;
            frame[lanes].fill(1.0);
        }
    }

    let below_top = 1022.0 / 1023.0;
    for (column, row) in [(left - 1, 7u32), (left + 8, 16)] {
        let first_lane = (row * spec.width + column) * stored_ch;
        frame[first_lane as usize] = below_top;
    }

    (spec, frame)
}

#[test]
fn the_f32_partial_sums_hold_the_brightest_10_bit_patch() {
    let client = test_client();

    for (width, left) in BRIGHTEST_LAYOUTS {
        let (spec, frame) = brightest_ten_bit_frame(width, left);
        let fused = collab_site_with_ring(&client, &spec, &frame);
        let (search, _) = integer_split_sites(&fused, TEN_BIT);
        let reference = reference_variant(&search);
        let (_, words) = run_search(&search, reference);

        let label = format!("brightest 10-bit yuv {width} wide");
        assert_matches_host_centre(&search, &spec, &words, &label);

        // The tied diagonals follow the four single steps, in position order.
        let ref_y = 8u32;
        let refs_x = refs_along(spec.width);
        let reference_index = (ref_y / STEP * refs_x + left / STEP) as usize;
        let first_word = reference_index * MEMBER_WORDS as usize;
        let tied = [pack_position(left - 1, 7, 0), pack_position(left + 1, 9, 0)];
        assert_eq!(&words[first_word + 5..first_word + 7], &tied, "{label}");

        // The other walk and the tile sum the same columns, so they hold the bound too.
        for variant in [integer_variant(Walk::Uniform), tiled_variant()] {
            let variant_label = format!("{label} {}", variant.name());
            let (_, variant_words) = run_search(&search, variant);

            assert_matches_host_centre(&search, &spec, &variant_words, &variant_label);
        }
    }
}

#[test]
fn scalar_int_stays_within_the_tie_bound_of_the_f32_baseline() {
    let client = test_client();

    for depth in [8u32, 10] {
        let codes = IntegerCodes { depth };

        for case in integer_corpus(depth) {
            let label = format!("{} {depth}-bit", case.label);
            let fused = collab_site_with_ring(&client, &case.spec, &case.ring);
            let (search, _) = integer_split_sites(&fused, codes);
            let reference = reference_variant(&search);
            let baseline = search.baseline();
            let (baseline_outputs, baseline_words) = run_search(&search, baseline);
            let (reference_outputs, _) = run_search(&search, reference);

            let fidelity = reference.fidelity();
            let verdict = search
                .compare(&baseline_outputs, &reference_outputs, fidelity)
                .expect("the outputs read back");
            assert!(verdict.passed(), "{label}: {verdict:?}");

            // A cross-frame pass that searches its neighbours must build volume grids, or the
            // trajectory search went unchecked.
            let searches_neighbours = case.spec.radius > 0 && case.spec.c_min <= 1.0;
            let group_words = baseline_words.chunks(MEMBER_WORDS as usize);
            let any_grid = group_words
                .map(|group| group[MEMBER_WORDS as usize - 1])
                .any(|word| word & GROUP_USES_GRID != 0);
            assert_eq!(any_grid, searches_neighbours, "{label}: grid groups");
        }
    }
}

#[test]
fn the_uniform_scalar_walk_matches_the_clipped_one_bitwise() {
    let client = test_client();

    for depth in [8u32, 10] {
        let codes = IntegerCodes { depth };

        for case in integer_corpus(depth) {
            let label = format!("{} {depth}-bit", case.label);
            let fused = collab_site_with_ring(&client, &case.spec, &case.ring);
            let (search, _) = integer_split_sites(&fused, codes);
            let clipped = integer_variant(Walk::Clipped);
            let uniform = integer_variant(Walk::Uniform);
            let (clipped_outputs, _) = run_search(&search, clipped);
            let (uniform_outputs, _) = run_search(&search, uniform);

            let verdict = search
                .compare(&clipped_outputs, &uniform_outputs, Fidelity::Exact)
                .expect("the outputs read back");
            assert!(verdict.passed(), "{label}: {verdict:?}");
        }
    }
}

#[test]
fn the_integer_tile_matches_scalar_int_bitwise() {
    let client = test_client();
    let scalar = integer_variant(Walk::Clipped);
    let tiled = tiled_variant();

    for depth in [8u32, 10] {
        let codes = IntegerCodes { depth };

        for case in integer_corpus(depth) {
            let label = format!("{} {depth}-bit", case.label);
            let fused = collab_site_with_ring(&client, &case.spec, &case.ring);
            let (search, _) = integer_split_sites(&fused, codes);
            let (scalar_outputs, scalar_words) = run_search(&search, scalar);
            let (tiled_outputs, tiled_words) = run_search(&search, tiled);

            let verdict = search
                .compare(&scalar_outputs, &tiled_outputs, Fidelity::Exact)
                .expect("the outputs read back");
            assert!(verdict.passed(), "{label}: {verdict:?}");
            assert_eq!(scalar_words, tiled_words, "{label}: member words");
        }
    }
}

#[test]
fn scalar_int_keeps_the_self_match_first_on_flat_content() {
    let client = test_client();

    for depth in [8u32, 10] {
        let codes = IntegerCodes { depth };
        let corpus = integer_corpus(depth);
        let flat_cases = corpus.iter().filter(|case| is_flat_case(case.label));

        for case in flat_cases {
            let label = format!("{} {depth}-bit", case.label);
            let fused = collab_site_with_ring(&client, &case.spec, &case.ring);
            let (search, _) = integer_split_sites(&fused, codes);
            let reference = reference_variant(&search);
            let (_, words) = run_search(&search, reference);

            let groups = words.chunks(MEMBER_WORDS as usize).enumerate();
            for (index, group) in groups {
                let expected = reference_position(&case.spec, index);
                assert_eq!(group[0], expected, "{label}: reference {index}");
            }
        }
    }
}

/// The site's code ring and motion inputs, read back for a host search.
struct HostInputs {
    codes: Vec<u32>,
    mv_field: Vec<i32>,
    confidence: Vec<f32>,
    neighbour_slots: Vec<u32>,
}

impl HostInputs {
    fn read(search: &CollabSearchSite<TestRuntime, f32>) -> HostInputs {
        let client = &search.client;
        let inputs = &search.inputs;
        let neighbours = (2 * search.shape.radius) as usize;

        let code_values = read_u16(client, &inputs.codes, inputs.codes_len).expect("the codes read back");
        let codes = code_values.iter().map(|&code| u32::from(code)).collect();
        let mv_field =
            read_i32(client, &inputs.mv_field, inputs.mv_len).expect("the motion field reads back");
        let confidence =
            read_f32(client, &inputs.confidence, inputs.conf_len).expect("the confidence reads back");
        let slot_words = read_i32(client, &inputs.neighbour_slots, inputs.neighbour_slots_len)
            .expect("the slots read back");
        let neighbour_slots = slot_words
            .iter()
            .take(neighbours)
            .map(|&slot| slot as u32)
            .collect();

        HostInputs {
            codes,
            mv_field,
            confidence,
            neighbour_slots,
        }
    }
}

/// The lowest block whose span holds the patch at `patch_start`, before the clamp to the highest.
fn covering_low(patch_start: u32, blksize: u32, step: u32) -> u32 {
    let overhang = u32::max(patch_start + PATCH_SIZE, blksize) - blksize;
    overhang.div_ceil(step)
}

/// The best `(ssd, position)` in neighbour frame `t` for the anchor, as the integer walks rank it.
///
/// Every scored rectangle joins one union, held positions are left out, and ties go to the lowest
/// position whatever rectangle holds it.
fn host_frame_match(
    search: &CollabSearchSite<TestRuntime, f32>,
    host: &HostInputs,
    view: &CodeView<'_>,
    anchor: u32,
    held: &[u32],
    t: u32,
) -> Option<(u64, u32)> {
    let shape = &search.shape;
    let anchor_x = anchor & 0x1FFF;
    let anchor_y = (anchor >> 13) & 0x1FFF;
    let max_x = (shape.width - PATCH_SIZE) as i32;
    let max_y = (shape.height - PATCH_SIZE) as i32;
    let covers = shape.blksize.div_ceil(shape.blk_step);
    let refine = shape.refine as i32;

    let bx_hi = u32::min(anchor_x / shape.blk_step, shape.blocks_x - 1);
    let by_hi = u32::min(anchor_y / shape.blk_step, shape.blocks_y - 1);
    let bx_covering = covering_low(anchor_x, shape.blksize, shape.blk_step);
    let by_covering = covering_low(anchor_y, shape.blksize, shape.blk_step);
    let bx_lo = u32::min(bx_covering, bx_hi);
    let by_lo = u32::min(by_covering, by_hi);
    let mut best: Option<(u64, u32)> = None;

    for iy in 0..covers {
        for ix in 0..covers {
            let block_x = bx_lo + ix;
            let block_y = by_lo + iy;
            if block_x > bx_hi || block_y > by_hi {
                continue;
            }

            let block = block_y * shape.blocks_x + block_x;
            let confidence = host.confidence[(t * shape.conf_stride + block) as usize];
            if confidence < search.values.c_min {
                continue;
            }

            let mv_index = (t * shape.mv_stride + 2 * block) as usize;
            let predicted_x = anchor_x as i32 + host.mv_field[mv_index];
            let predicted_y = anchor_y as i32 + host.mv_field[mv_index + 1];
            let left = (predicted_x - refine).clamp(0, max_x) as u32;
            let right = (predicted_x + refine).clamp(0, max_x) as u32;
            let top = (predicted_y - refine).clamp(0, max_y) as u32;
            let bottom = (predicted_y + refine).clamp(0, max_y) as u32;

            for y in top..=bottom {
                for x in left..=right {
                    let position = pack_position(x, y, t + 1);
                    if held.contains(&position) {
                        continue;
                    }

                    let ssd = patch_ssd(view, anchor, position);
                    let scored = (ssd, position);
                    let improves = best.is_none_or(|held_best| scored < held_best);
                    if improves {
                        best = Some(scored);
                    }
                }
            }
        }
    }

    best
}

/// The frame members of one volume as the integer walks pick them.
///
/// Each frame's best joins the ascending list in frame order, and a tie keeps the earlier frame.
fn host_volume_frames(
    search: &CollabSearchSite<TestRuntime, f32>,
    host: &HostInputs,
    view: &CodeView<'_>,
    anchor: u32,
    held: &[u32],
) -> Vec<u32> {
    let tail = (search.shape.grid_frames - 1) as usize;
    let neighbours = 2 * search.shape.radius;
    let mut slots = vec![(u64::MAX, 0u32); tail];

    for t in 0..neighbours {
        let Some(mut carry) = host_frame_match(search, host, view, anchor, held, t) else {
            continue;
        };

        for slot in slots.iter_mut() {
            if carry.0 < slot.0 {
                std::mem::swap(slot, &mut carry);
            }
        }
    }

    slots.into_iter().map(|(_, position)| position).collect()
}

#[test]
fn the_scalar_walks_and_the_tile_keep_the_host_trajectory_frames() {
    let client = test_client();

    for depth in [8u32, 10] {
        let codes = IntegerCodes { depth };
        let corpus = integer_corpus(depth);
        let cross_frame_cases = corpus.iter().filter(|case| case.spec.radius > 0);

        for case in cross_frame_cases {
            let fused = collab_site_with_ring(&client, &case.spec, &case.ring);
            let (search, _) = integer_split_sites(&fused, codes);
            let host = HostInputs::read(&search);
            let shape = &search.shape;
            let grid_frames = shape.grid_frames as usize;
            let view = CodeView {
                codes: &host.codes,
                width: shape.width,
                height: shape.height,
                stored_ch: shape.stored_ch,
                channels: shape.channels,
                centre_slot: search.values.centre_slot,
                neighbour_slots: &host.neighbour_slots,
            };

            let variants = [
                integer_variant(Walk::Clipped),
                integer_variant(Walk::Uniform),
                tiled_variant(),
            ];
            for variant in variants {
                let name = variant.name();
                let label = format!("{} {depth}-bit {name}", case.label);
                let (_, words) = run_search(&search, variant);
                let groups = words.chunks(MEMBER_WORDS as usize).enumerate();
                let grid_groups = groups.filter(|(_, group)| group[MAX_K as usize] & GROUP_USES_GRID != 0);

                for (index, group) in grid_groups {
                    for first in (0..MAX_K as usize).step_by(grid_frames) {
                        let anchor = group[first];
                        let held = &group[..=first];
                        let expected = host_volume_frames(&search, &host, &view, anchor, held);
                        let frames = &group[first + 1..first + grid_frames];

                        assert_eq!(
                            frames,
                            expected.as_slice(),
                            "{label}: reference {index} member {first}"
                        );
                    }
                }
            }
        }
    }
}

/// Writes an 8x8 patch of `code` into a single-channel frame of normalised codes at `(left, top)`.
fn plant_codes(frame: &mut [f32], width: u32, left: u32, top: u32, code: f32) {
    for row in 0..8 {
        for column in 0..8 {
            let index = (top + row) * width + left + column;
            frame[index as usize] = code;
        }
    }
}

#[test]
fn ten_bit_codes_one_square_apart_rank_in_order() {
    let client = test_client();
    let width = 40u32;
    let height = 24u32;
    let top_code = 1.0f32;
    let below_top = 1022.0 / 1023.0;
    let (ref_x, ref_y) = (16u32, 8u32);
    let (two_apart_x, one_apart_x) = (7u32, 25u32);

    // Code 0 around patches of the top code, which every partial overlap scores far from.
    let mut frame = vec![0.0f32; (width * height) as usize];
    plant_codes(&mut frame, width, ref_x, ref_y, top_code);
    plant_codes(&mut frame, width, two_apart_x, 0, top_code);
    plant_codes(&mut frame, width, one_apart_x, 0, top_code);
    frame[two_apart_x as usize] = below_top;
    frame[two_apart_x as usize + 1] = below_top;
    frame[one_apart_x as usize] = below_top;

    let spec = CollabSpec::new(width, height, ChannelMode::Luma, 0);
    let fused = collab_site_with_ring(&client, &spec, &frame);
    let (search, _) = integer_split_sites(&fused, TEN_BIT);
    let reference = reference_variant(&search);
    let (_, words) = run_search(&search, reference);

    let refs_x = refs_along(width);
    let reference_index = (ref_y / 4 * refs_x + ref_x / 4) as usize;
    let first_word = reference_index * MEMBER_WORDS as usize;
    let group = &words[first_word..first_word + MEMBER_WORDS as usize];

    let self_match = pack_position(ref_x, ref_y, 0);
    let one_apart = pack_position(one_apart_x, 0, 0);
    let two_apart = pack_position(two_apart_x, 0, 0);
    assert_eq!(&group[..3], &[self_match, one_apart, two_apart]);
}

#[test]
fn every_reference_gets_a_group_past_the_dead_groups_with_integer_codes() {
    let client = test_client();
    let spec = CollabSpec::new(104, 32, ChannelMode::Luma, 1);
    let fused = collab_site(&client, &spec);
    let (search, _) = integer_split_sites(&fused, EIGHT_BIT);
    let reference = reference_variant(&search);
    let (outputs, words) = run_search(&search, reference);

    let group_words = words.chunks(MEMBER_WORDS as usize);
    let filled = group_words
        .clone()
        .all(|group| group[MEMBER_WORDS as usize - 1] & GROUP_SIZE_MASK != 0);
    assert_eq!(group_words.len(), outputs.refs);
    assert!(filled, "every live reference holds a group");
}

#[test]
fn the_integer_reference_follows_the_walk_the_device_allows() {
    let client = test_client();
    let spec = CollabSpec::new(64, 48, ChannelMode::Luma, 1);
    let fused = collab_site(&client, &spec);
    let (float_search, _) = split_sites(&fused);
    let (mut search, _) = integer_split_sites(&fused, EIGHT_BIT);

    assert_eq!(float_search.integer_reference(), None);

    for (uniform_only, name) in [(false, "int_groups8_clipped"), (true, "int_groups8_uniform")] {
        search.shape.uniform_walk_only = uniform_only;
        let reference = search
            .integer_reference()
            .expect("codes give an integer reference");
        let expected = IntegerVariant::reference_for(uniform_only);

        assert_eq!(reference, SearchVariant::Integer(expected));
        assert_eq!(reference.name(), name);
        assert_eq!(expected.centre, IntCentre::Scalar);
        assert_eq!(expected.neighbour, IntNeighbour::Scalar);
        assert_eq!((expected.cube_refs_x, expected.ref_rows), (8, 1));
    }
}

/// One output check a tuning pass ran, with the candidate launched just before it.
#[derive(Debug, Clone, Copy, PartialEq)]
struct SearchCheck {
    candidate: SearchVariant,
    fidelity: Fidelity,
    passed: bool,
}

/// The integer search site with its pre-screen off, logging every output check a tune runs.
///
/// Each check clears, launches and then compares, so the candidate launched last is the one checked.
/// With no pre-screen every compiled candidate is checked, however slow it is under load.
#[derive(Clone)]
struct CheckedSearchSite {
    inner: CollabSearchSite<TestRuntime, f32>,
    last_launch: Arc<Mutex<Option<SearchVariant>>>,
    checks: Arc<Mutex<Vec<SearchCheck>>>,
}

impl CheckedSearchSite {
    fn new(inner: CollabSearchSite<TestRuntime, f32>) -> Self {
        Self {
            inner,
            last_launch: Arc::default(),
            checks: Arc::default(),
        }
    }

    fn checks(&self) -> Vec<SearchCheck> {
        let checks = self.checks.lock().expect("the check log is not poisoned");
        checks.clone()
    }
}

impl VariantSite<TestRuntime> for CheckedSearchSite {
    type Candidate = SearchVariant;
    type Outputs = SearchOutputs;

    const COPIES: bool = true;

    fn client(&self) -> &ComputeClient<TestRuntime> {
        self.inner.client()
    }

    fn key(&self) -> SiteKey {
        self.inner.key()
    }

    fn shape(&self, candidate: SearchVariant) -> LaunchShape {
        self.inner.shape(candidate)
    }

    fn shared_bytes(&self, candidate: SearchVariant) -> usize {
        self.inner.shared_bytes(candidate)
    }

    fn min_plane_size(&self, candidate: SearchVariant) -> u32 {
        self.inner.min_plane_size(candidate)
    }

    fn launch(&self, candidate: SearchVariant, outputs: &SearchOutputs) {
        {
            let mut last_launch = self.last_launch.lock().expect("the launch log is not poisoned");
            *last_launch = Some(candidate);
        }

        self.inner.launch(candidate, outputs);
    }

    fn real_outputs(&self) -> SearchOutputs {
        self.inner.real_outputs()
    }

    fn scratch_outputs(&self) -> SearchOutputs {
        self.inner.scratch_outputs()
    }

    fn clear_outputs(&self, outputs: &SearchOutputs) {
        self.inner.clear_outputs(outputs);
    }

    fn compile_identity(&self) -> String {
        self.inner.compile_identity()
    }

    fn on_client(&self, client: &ComputeClient<TestRuntime>) -> Option<Self> {
        let inner = self.inner.on_client(client)?;

        Some(Self {
            inner,
            ..self.clone()
        })
    }

    fn compile_only(&self, candidate: SearchVariant, outputs: &SearchOutputs) {
        self.inner.compile_only(candidate, outputs);
    }

    fn compare(
        &self,
        baseline: &SearchOutputs,
        candidate: &SearchOutputs,
        fidelity: Fidelity,
    ) -> Result<CompareResult, anyhow::Error> {
        let verdict = self.inner.compare(baseline, candidate, fidelity)?;
        let last_launch = *self.last_launch.lock().expect("the launch log is not poisoned");
        let checked = last_launch.expect("a check follows a launch");
        let check = SearchCheck {
            candidate: checked,
            fidelity,
            passed: verdict.passed(),
        };

        {
            let mut checks = self.checks.lock().expect("the check log is not poisoned");
            checks.push(check);
        }

        Ok(verdict)
    }

    fn baseline(&self) -> SearchVariant {
        self.inner.baseline()
    }

    fn integer_reference(&self) -> Option<SearchVariant> {
        self.inner.integer_reference()
    }

    fn legal(&self, candidate: SearchVariant, limits: &DeviceLimits, legality: &Legality) -> bool {
        self.inner.legal(candidate, limits, legality)
    }

    fn fits(&self, candidate: SearchVariant, limits: &DeviceLimits, legality: &Legality) -> bool {
        self.inner.fits(candidate, limits, legality)
    }
}

/// Tunes the key of `site` once and returns the candidates the selector allowed it.
fn tune_search(site: &CheckedSearchSite) -> Vec<SearchVariant> {
    let client = test_client();
    let limits = DeviceLimits::from_client(&client);
    let ledger = TuneLedger::new(false);
    ledger.set_phase(TunePhase::Tune);

    let key = site.key();
    let deadline = Instant::now() + Duration::from_secs(20);
    ledger.set_target(key, deadline);

    let kernels = EngineKernels {
        ledger: Some(ledger),
        ..EngineKernels::default()
    };
    let selector = KernelSelector::with_limits(&client, kernels, limits);
    let site_limits = selector.limits();
    let legality = selector.legality();
    let allowed = legal_candidates(site, site_limits, legality).expect("the baseline fits");
    selector.run(site).expect("the tuned run succeeds");

    allowed
}

#[test]
fn tuning_the_integer_search_checks_the_reference_then_every_follower() {
    let client = test_client();
    let spec = CollabSpec::new(96, 64, ChannelMode::Yuv, 1);
    let fused = collab_site(&client, &spec);

    for codes in [EIGHT_BIT, TEN_BIT] {
        let depth = codes.depth;
        let (search, _) = integer_split_sites(&fused, codes);
        let reference = reference_variant(&search);
        let site = CheckedSearchSite::new(search);

        let allowed = tune_search(&site);
        let checks = site.checks();
        let followers: Vec<SearchVariant> = allowed
            .iter()
            .copied()
            .filter(|&candidate| matches!(candidate, SearchVariant::Integer(_)) && candidate != reference)
            .collect();
        assert!(allowed.contains(&reference), "{depth}-bit: {allowed:?}");
        assert!(!followers.is_empty(), "{depth}-bit: {allowed:?}");

        let reference_check = SearchCheck {
            candidate: reference,
            fidelity: reference.fidelity(),
            passed: true,
        };
        let reference_position = checks.iter().position(|check| *check == reference_check);
        let reference_index = reference_position.expect("the reference passes its tie-bound check");

        for follower in followers {
            let follower_check = SearchCheck {
                candidate: follower,
                fidelity: Fidelity::Exact,
                passed: true,
            };
            let follower_position = checks.iter().position(|check| *check == follower_check);
            let Some(follower_index) = follower_position else {
                panic!(
                    "{depth}-bit {} never passed against the reference: {checks:?}",
                    follower.name()
                );
            };

            assert!(
                follower_index > reference_index,
                "{depth}-bit {}: {checks:?}",
                follower.name()
            );
        }

        let failed: Vec<&SearchCheck> = checks.iter().filter(|check| !check.passed).collect();
        assert!(failed.is_empty(), "{depth}-bit: {failed:?}");
    }
}
