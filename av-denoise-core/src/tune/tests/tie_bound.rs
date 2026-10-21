use crate::collab::MAX_K;
use crate::collab::integer::{CodeView, host_centre_members, pack_position, patch_ssd, tie_bound_verdict};
use crate::collab::kernels::fused::GROUP_USES_GRID;
use crate::tune::CompareResult;

const BLOCK: u32 = 8;

/// Centre blocks for the single-frame groups. Blocks 1 and 2 tie, and block 9 is past block 8.
const CENTRE_BLOCKS: [u32; 10] = [0, 1, 1, 2, 3, 4, 5, 6, 7, 9];

/// Centre blocks for the grid groups, block `v` anchoring volume `v`.
const GRID_CENTRE_BLOCKS: [u32; 8] = [0, 1, 2, 3, 4, 5, 6, 7];

/// Neighbour blocks for the grid groups. Blocks 0 and 1 tie for the anchor of volume 0.
const GRID_NEIGHBOUR_BLOCKS: [u32; 8] = [0, 0, 1, 2, 3, 4, 5, 6];

/// One 8-row single-channel frame of 8x8 blocks, block `b` holding `values[b]` everywhere.
fn block_frame(values: &[u32]) -> Vec<u32> {
    let width = values.len() as u32 * BLOCK;
    let pixels = width * BLOCK;

    (0..pixels)
        .map(|pixel| {
            let x = pixel % width;
            values[(x / BLOCK) as usize]
        })
        .collect()
}

fn block_view<'a>(codes: &'a [u32], blocks: usize, neighbour_slots: &'a [u32]) -> CodeView<'a> {
    CodeView {
        codes,
        width: blocks as u32 * BLOCK,
        height: BLOCK,
        stored_ch: 1,
        channels: 1,
        centre_slot: 0,
        neighbour_slots,
    }
}

fn centre_block(block: u32) -> u32 {
    pack_position(block * BLOCK, 0, 0)
}

fn neighbour_block(block: u32) -> u32 {
    pack_position(block * BLOCK, 0, 1)
}

/// The member words of one reference, its members then the group word.
fn group_words(members: [u32; MAX_K as usize], group_word: u32) -> Vec<u32> {
    let mut words = members.to_vec();
    words.push(group_word);
    words
}

fn single_frame_group(blocks: [u32; MAX_K as usize]) -> Vec<u32> {
    let members = blocks.map(centre_block);
    group_words(members, MAX_K)
}

/// A grid group of four two-frame volumes, anchored on centre blocks 0..4.
fn grid_group(frame_blocks: [u32; 4]) -> Vec<u32> {
    let mut members = [0u32; MAX_K as usize];
    for (volume, frame_block) in frame_blocks.into_iter().enumerate() {
        members[2 * volume] = centre_block(volume as u32);
        members[2 * volume + 1] = neighbour_block(frame_block);
    }

    group_words(members, MAX_K | GROUP_USES_GRID)
}

fn single_frame_verdict(baseline: &[u32], candidate: &[u32]) -> CompareResult {
    let codes = block_frame(&CENTRE_BLOCKS);
    let view = block_view(&codes, CENTRE_BLOCKS.len(), &[]);
    tie_bound_verdict(&view, 1, 1, 1, baseline, candidate)
}

/// The verdict over a two-slot view, the grid's centre frame then its neighbour frame.
fn two_slot_verdict(grid_frames: u32, baseline: &[u32], candidate: &[u32]) -> CompareResult {
    let mut codes = block_frame(&GRID_CENTRE_BLOCKS);
    let neighbour = block_frame(&GRID_NEIGHBOUR_BLOCKS);
    codes.extend(neighbour);

    let neighbour_slots = [1];
    let view = block_view(&codes, GRID_CENTRE_BLOCKS.len(), &neighbour_slots);
    tie_bound_verdict(&view, 1, 1, grid_frames, baseline, candidate)
}

fn grid_verdict(baseline: &[u32], candidate: &[u32]) -> CompareResult {
    two_slot_verdict(2, baseline, candidate)
}

fn failure_message(verdict: CompareResult) -> String {
    match verdict {
        CompareResult::Pass => panic!("the tie bound passed"),
        CompareResult::Fail(message) => message,
    }
}

fn assert_fails_with(verdict: CompareResult, needle: &str) {
    let message = failure_message(verdict);
    assert!(message.contains(needle), "{message}");
}

fn hashed_code(index: u32) -> u32 {
    let mut hash = index.wrapping_mul(0x9E37_79B9);
    hash ^= hash >> 15;
    hash = hash.wrapping_mul(0x2C1B_3C6D);
    hash ^= hash >> 12;

    hash % 1024
}

#[test]
fn patch_ssd_sums_every_channel_of_the_codes() {
    let width = 16u32;
    let height = 8u32;
    let codes: Vec<u32> = (0..width * height)
        .flat_map(|pixel| {
            let shifted = u32::from(pixel % width >= BLOCK);
            [10 + shifted, 20 + 2 * shifted]
        })
        .collect();
    let view = CodeView {
        codes: &codes,
        width,
        height,
        stored_ch: 2,
        channels: 2,
        centre_slot: 0,
        neighbour_slots: &[],
    };

    let first = pack_position(0, 0, 0);
    let second = pack_position(BLOCK, 0, 0);

    let forward = patch_ssd(&view, first, second);
    let backward = patch_ssd(&view, second, first);
    let itself = patch_ssd(&view, first, first);

    assert_eq!(forward, 64 + 64 * 4);
    assert_eq!(backward, 64 + 64 * 4);
    assert_eq!(itself, 0);
}

#[test]
fn patch_ssd_reads_a_neighbour_from_its_slot() {
    let mut codes = vec![0u32; 64];
    let neighbour_frame = vec![3u32; 64];
    codes.extend(neighbour_frame);

    let neighbour_slots = [1];
    let view = block_view(&codes, 1, &neighbour_slots);
    let centre = centre_block(0);
    let neighbour = neighbour_block(0);

    let ssd = patch_ssd(&view, centre, neighbour);

    assert_eq!(ssd, 64 * 9);
}

#[test]
fn host_centre_members_rank_ties_by_position_and_pin_the_self_match() {
    let codes = vec![5u32; 24 * 24];
    let view = CodeView {
        codes: &codes,
        width: 24,
        height: 24,
        stored_ch: 1,
        channels: 1,
        centre_slot: 0,
        neighbour_slots: &[],
    };

    let members = host_centre_members(&view, 8, 8, 4);

    let mut expected = vec![pack_position(8, 8, 0)];
    let first_row = (4..11).map(|x| pack_position(x, 4, 0));
    expected.extend(first_row);
    assert_eq!(members, expected);
}

#[test]
fn host_centre_members_rank_by_distance_before_position() {
    // Columns 0..8 hold 1 except column 6, and the reference patch at x 8 is all 0, so the
    // distance falls as x rises and x 6 ties x 7.
    let width = 16u32;
    let codes: Vec<u32> = (0..width * BLOCK)
        .map(|pixel| {
            let x = pixel % width;
            u32::from(x < BLOCK && x != 6)
        })
        .collect();
    let view = CodeView {
        codes: &codes,
        width,
        height: BLOCK,
        stored_ch: 1,
        channels: 1,
        centre_slot: 0,
        neighbour_slots: &[],
    };

    let members = host_centre_members(&view, 8, 0, 8);

    let expected = [8, 6, 7, 5, 4, 3, 2, 1].map(|x| pack_position(x, 0, 0));
    assert_eq!(members, expected);
}

#[test]
fn host_centre_members_return_every_position_of_a_small_rectangle() {
    let codes = vec![0u32; 10 * 8];
    let view = CodeView {
        codes: &codes,
        width: 10,
        height: 8,
        stored_ch: 1,
        channels: 1,
        centre_slot: 0,
        neighbour_slots: &[],
    };

    let members = host_centre_members(&view, 1, 0, 4);

    let expected = [(1, 0), (0, 0), (2, 0)].map(|(x, y)| pack_position(x, y, 0));
    assert_eq!(members, expected);
}

#[test]
fn the_tie_bound_accepts_a_swap_of_equal_distances() {
    let baseline = single_frame_group([0, 1, 3, 4, 5, 6, 7, 8]);
    let candidate = single_frame_group([0, 2, 3, 4, 5, 6, 7, 8]);

    let verdict = single_frame_verdict(&baseline, &candidate);

    assert_eq!(verdict, CompareResult::Pass);
}

#[test]
fn the_tie_bound_rejects_a_member_worse_than_the_eighth_best() {
    let baseline = single_frame_group([0, 1, 2, 3, 4, 5, 6, 7]);
    let candidate = single_frame_group([0, 1, 2, 3, 4, 5, 6, 9]);

    let verdict = single_frame_verdict(&baseline, &candidate);

    let message = failure_message(verdict);
    let candidate_ssd = (64 * 9 * 9).to_string();
    let baseline_ssd = (64 * 6 * 6).to_string();
    assert!(message.contains("reference 0 member 7"), "{message}");
    assert!(message.contains(&candidate_ssd), "{message}");
    assert!(message.contains(&baseline_ssd), "{message}");
}

#[test]
fn the_tie_bound_rejects_a_group_without_the_self_match_first() {
    let baseline = single_frame_group([0, 1, 2, 3, 4, 5, 6, 7]);
    let candidate = single_frame_group([1, 0, 2, 3, 4, 5, 6, 7]);

    let verdict = single_frame_verdict(&baseline, &candidate);

    let message = failure_message(verdict);
    assert!(message.contains("member 0"), "{message}");
}

#[test]
fn the_tie_bound_rejects_a_changed_group_word() {
    let baseline = single_frame_group([0, 1, 2, 3, 4, 5, 6, 7]);
    let mut candidate = baseline.clone();
    candidate[MAX_K as usize] = 4;

    let verdict = single_frame_verdict(&baseline, &candidate);

    let message = failure_message(verdict);
    assert!(message.contains("reference 0 group word"), "{message}");
}

#[test]
fn the_tie_bound_ignores_members_past_the_group_count() {
    let mut baseline = single_frame_group([0, 1, 2, 3, 4, 5, 6, 7]);
    baseline[MAX_K as usize] = 4;
    let mut candidate = baseline.clone();
    candidate[7] = centre_block(9);

    let verdict = single_frame_verdict(&baseline, &candidate);

    assert_eq!(verdict, CompareResult::Pass);
}

#[test]
fn the_tie_bound_compares_frame_members_of_matching_volumes() {
    let baseline = grid_group([0, 2, 3, 4]);
    let swapped = grid_group([1, 2, 3, 4]);
    let worse = grid_group([0, 7, 3, 4]);

    let swap_verdict = grid_verdict(&baseline, &swapped);
    let worse_verdict = grid_verdict(&baseline, &worse);

    assert_eq!(swap_verdict, CompareResult::Pass);
    let message = failure_message(worse_verdict);
    assert!(message.contains("reference 0 member 3"), "{message}");
}

#[test]
fn the_tie_bound_skips_frame_members_after_an_earlier_volume_differs() {
    let baseline = grid_group([0, 2, 3, 4]);
    let candidate = grid_group([1, 7, 3, 4]);

    let verdict = grid_verdict(&baseline, &candidate);

    assert_eq!(verdict, CompareResult::Pass);
}

#[test]
fn the_tie_bound_rejects_an_anchor_past_the_baseline_anchors() {
    let baseline = grid_group([0, 2, 3, 4]);
    let mut candidate = baseline.clone();
    candidate[6] = centre_block(7);

    let verdict = grid_verdict(&baseline, &candidate);

    let message = failure_message(verdict);
    assert!(message.contains("reference 0 member 6"), "{message}");
}

#[test]
fn ten_bit_max_codes_sum_without_overflow() {
    let stored_ch = 4u32;
    let width = 16u32;
    let height = 8u32;
    let codes: Vec<u32> = (0..width * height)
        .flat_map(|pixel| {
            let code = if pixel % width < BLOCK { 1023 } else { 0 };
            // The fourth lane is padding the SSD must skip.
            [code, code, code, 1023 - code]
        })
        .collect();
    let view = CodeView {
        codes: &codes,
        width,
        height,
        stored_ch,
        channels: 3,
        centre_slot: 0,
        neighbour_slots: &[],
    };

    let first = pack_position(0, 0, 0);
    let second = pack_position(BLOCK, 0, 0);

    let ssd = patch_ssd(&view, first, second);

    assert_eq!(ssd, 200_933_568);
}

#[test]
fn the_tie_bound_rejects_a_mismatched_member_buffer_length() {
    let baseline = single_frame_group([0, 1, 2, 3, 4, 5, 6, 7]);
    let candidate = &baseline[..baseline.len() - 1];

    let verdict = single_frame_verdict(&baseline, candidate);

    assert_fails_with(verdict, "member words");
}

#[test]
fn the_tie_bound_rejects_an_x_past_the_frame_that_stays_inside_the_codes() {
    // x 57 is past max_x 56, but its reads run on into the next row and slot, so it scores 2744,
    // inside the baseline's worst of 3136.
    let baseline = single_frame_group([0, 1, 2, 3, 4, 5, 6, 7]);
    let mut candidate = baseline.clone();
    candidate[7] = pack_position(57, 0, 0);

    let verdict = two_slot_verdict(1, &baseline, &candidate);

    assert_fails_with(verdict, "reference 0 member 7");
}

#[test]
fn the_tie_bound_rejects_a_y_past_the_frame() {
    let baseline = single_frame_group([0, 1, 2, 3, 4, 5, 6, 7]);
    let mut candidate = baseline.clone();
    candidate[3] = pack_position(0, 1, 0);

    let verdict = single_frame_verdict(&baseline, &candidate);

    assert_fails_with(verdict, "patch leaves the frame");
}

#[test]
fn the_tie_bound_rejects_a_frame_past_the_neighbour_slots() {
    let baseline = grid_group([0, 2, 3, 4]);
    let mut candidate = baseline.clone();
    candidate[5] = pack_position(3 * BLOCK, 0, 2);

    let verdict = grid_verdict(&baseline, &candidate);

    assert_fails_with(verdict, "past the neighbour slots");
}

#[test]
fn the_tie_bound_rejects_a_centre_pick_outside_the_centre_frame() {
    let baseline = single_frame_group([0, 1, 2, 3, 4, 5, 6, 7]);
    let mut candidate = baseline.clone();
    candidate[3] = neighbour_block(3);

    let verdict = two_slot_verdict(1, &baseline, &candidate);

    assert_fails_with(verdict, "reference 0 member 3");
}

#[test]
fn the_tie_bound_rejects_an_anchor_outside_the_centre_frame() {
    let baseline = grid_group([0, 2, 3, 4]);
    let mut candidate = baseline.clone();
    candidate[2] = neighbour_block(5);

    let verdict = grid_verdict(&baseline, &candidate);

    assert_fails_with(verdict, "centre pick is outside the centre frame");
}

#[test]
fn the_tie_bound_rejects_grid_frames_that_do_not_divide_the_group() {
    let baseline = grid_group([0, 2, 3, 4]);

    let no_frames = two_slot_verdict(0, &baseline, &baseline);
    let three_frames = two_slot_verdict(3, &baseline, &baseline);

    assert_fails_with(no_frames, "grid frames");
    assert_fails_with(three_frames, "grid frames");
}

#[test]
fn the_tie_bound_rejects_a_slot_past_the_codes() {
    let codes = block_frame(&CENTRE_BLOCKS);
    let neighbour_slots = [1];
    let view = block_view(&codes, CENTRE_BLOCKS.len(), &neighbour_slots);
    let group = single_frame_group([0, 1, 2, 3, 4, 5, 6, 7]);

    let verdict = tie_bound_verdict(&view, 1, 1, 1, &group, &group);

    assert_fails_with(verdict, "slot 1");
}

#[test]
fn the_tie_bound_rejects_a_duplicated_best_position() {
    let baseline = single_frame_group([0, 1, 2, 3, 4, 5, 6, 7]);
    let candidate = single_frame_group([0, 1, 1, 3, 4, 5, 6, 7]);

    let verdict = single_frame_verdict(&baseline, &candidate);

    assert_fails_with(verdict, "reference 0 member 2");
}

#[test]
fn the_tie_bound_maps_each_reference_to_its_clamped_position() {
    // An 18x12 frame clamps the last column to x 10, so the eight references sit here.
    let reference_positions = [(0, 0), (4, 0), (8, 0), (10, 0), (0, 4), (4, 4), (8, 4), (10, 4)];
    let width = 18u32;
    let height = 12u32;
    let spatial_radius = 4;
    let codes: Vec<u32> = (0..width * height).map(hashed_code).collect();
    let view = CodeView {
        codes: &codes,
        width,
        height,
        stored_ch: 1,
        channels: 1,
        centre_slot: 0,
        neighbour_slots: &[],
    };

    let mut baseline = Vec::new();
    for (ref_x, ref_y) in reference_positions {
        let members = host_centre_members(&view, ref_x, ref_y, spatial_radius);
        baseline.extend(members);
        baseline.push(MAX_K);
    }

    let matching = tie_bound_verdict(&view, 4, 2, 1, &baseline, &baseline);

    // Reference 7's furthest position in its window is past its eighth best.
    let reference = pack_position(10, 4, 0);
    let window = (0..=4).flat_map(|y| (6..=10).map(move |x| pack_position(x, y, 0)));
    let furthest = window
        .max_by_key(|&position| patch_ssd(&view, reference, position))
        .expect("the window holds positions");
    let last_word = 7 * (MAX_K as usize + 1) + 7;
    let eighth_best = patch_ssd(&view, reference, baseline[last_word]);
    let furthest_ssd = patch_ssd(&view, reference, furthest);
    assert!(furthest_ssd > eighth_best, "{furthest_ssd} <= {eighth_best}");

    let mut candidate = baseline.clone();
    candidate[last_word] = furthest;
    let wrong = tie_bound_verdict(&view, 4, 2, 1, &baseline, &candidate);

    assert_eq!(matching, CompareResult::Pass);
    assert_fails_with(wrong, "reference 7 member 7");
}

/// Centre blocks for the grid-choice groups. Blocks 1 and 2 tie as the second anchor.
const TIED_ANCHOR_BLOCKS: [u32; 9] = [0, 1, 1, 2, 3, 4, 5, 6, 9];

/// The verdict over a centre frame and three neighbour frames, with two four-frame volumes.
fn tied_anchor_verdict(baseline: &[u32], candidate: &[u32]) -> CompareResult {
    let frame = block_frame(&TIED_ANCHOR_BLOCKS);
    let codes = frame.repeat(4);

    let neighbour_slots = [1, 2, 3];
    let view = block_view(&codes, TIED_ANCHOR_BLOCKS.len(), &neighbour_slots);
    tie_bound_verdict(&view, 1, 1, 4, baseline, candidate)
}

/// A grid group of two four-frame volumes, each frame member at its anchor's block.
fn four_frame_grid_group(anchor_blocks: [u32; 2]) -> Vec<u32> {
    let mut members = [0u32; MAX_K as usize];
    for (volume, anchor_block) in anchor_blocks.into_iter().enumerate() {
        let first = 4 * volume;
        members[first] = centre_block(anchor_block);

        for frame in 1..4 {
            let frame_member = pack_position(anchor_block * BLOCK, 0, frame);
            members[first + frame as usize] = frame_member;
        }
    }

    group_words(members, MAX_K | GROUP_USES_GRID)
}

#[test]
fn the_tie_bound_accepts_a_grid_choice_that_follows_a_tied_anchor() {
    let baseline = four_frame_grid_group([0, 1]);
    let candidate = single_frame_group([0, 2, 1, 3, 4, 5, 6, 7]);

    let verdict = tied_anchor_verdict(&baseline, &candidate);

    assert_eq!(verdict, CompareResult::Pass);
}

#[test]
fn the_tie_bound_accepts_a_grid_built_on_a_tied_anchor() {
    let baseline = single_frame_group([0, 1, 2, 3, 4, 5, 6, 7]);
    let candidate = four_frame_grid_group([0, 2]);

    let verdict = tied_anchor_verdict(&baseline, &candidate);

    assert_eq!(verdict, CompareResult::Pass);
}

#[test]
fn the_tie_bound_rejects_a_grid_choice_with_the_same_anchors() {
    let baseline = four_frame_grid_group([0, 1]);
    let candidate = single_frame_group([0, 1, 2, 3, 4, 5, 6, 7]);

    let verdict = tied_anchor_verdict(&baseline, &candidate);

    assert_fails_with(
        verdict,
        "reference 0 group word is 0x8, the baseline's is 0x108 with the same anchors",
    );
}

#[test]
fn the_tie_bound_rejects_a_grid_choice_past_the_baseline_anchors() {
    let baseline = four_frame_grid_group([0, 1]);
    let candidate = single_frame_group([0, 3, 1, 2, 4, 5, 6, 7]);

    let verdict = tied_anchor_verdict(&baseline, &candidate);

    assert_fails_with(verdict, "reference 0 member 1");
}

#[test]
fn the_tie_bound_rejects_a_grid_choice_on_a_partial_group() {
    let mut baseline = four_frame_grid_group([0, 1]);
    baseline[MAX_K as usize] = 4 | GROUP_USES_GRID;
    let mut candidate = single_frame_group([0, 2, 1, 3, 4, 5, 6, 7]);
    candidate[MAX_K as usize] = 4;

    let verdict = tied_anchor_verdict(&baseline, &candidate);

    assert_fails_with(verdict, "reference 0 group word is 0x4, the baseline's is 0x104");
}

#[test]
fn the_tie_bound_rejects_a_grid_choice_without_the_self_match_first() {
    let baseline = four_frame_grid_group([0, 1]);
    let candidate = single_frame_group([2, 0, 1, 3, 4, 5, 6, 7]);

    let verdict = tied_anchor_verdict(&baseline, &candidate);

    assert_fails_with(verdict, "reference 0 member 0");
}

#[test]
fn the_tie_bound_rejects_a_grid_choice_with_a_repeated_frame_member() {
    let baseline = single_frame_group([0, 1, 2, 3, 4, 5, 6, 7]);
    let mut candidate = four_frame_grid_group([0, 2]);
    candidate[6] = candidate[5];

    let verdict = tied_anchor_verdict(&baseline, &candidate);

    assert_fails_with(verdict, "candidate reference 0 member 6");
}

#[test]
fn the_tie_bound_rejects_a_grid_choice_with_a_frame_member_past_the_slots() {
    let baseline = single_frame_group([0, 1, 2, 3, 4, 5, 6, 7]);
    let mut candidate = four_frame_grid_group([0, 2]);
    candidate[7] = pack_position(2 * BLOCK, 0, 4);

    let verdict = tied_anchor_verdict(&baseline, &candidate);

    assert_fails_with(verdict, "past the neighbour slots");
}

#[test]
fn the_tie_bound_rejects_a_grid_choice_whose_later_picks_descend() {
    let baseline = four_frame_grid_group([0, 1]);
    let candidate = single_frame_group([0, 2, 1, 3, 4, 6, 5, 7]);

    let verdict = tied_anchor_verdict(&baseline, &candidate);

    assert_fails_with(verdict, "reference 0 member 6");
}
