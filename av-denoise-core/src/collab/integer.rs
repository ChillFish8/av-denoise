use crate::collab::kernels::fused::{GROUP_SIZE_MASK, GROUP_USES_GRID, MEMBER_WORDS};
use crate::collab::{MAX_K, PATCH_SIZE, STEP};
use crate::tune::CompareResult;

/// The integer code ring as the host reads it back, one `u32` per stored lane.
pub(crate) struct CodeView<'a> {
    pub(crate) codes: &'a [u32],
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) stored_ch: u32,
    pub(crate) channels: u32,
    pub(crate) centre_slot: u32,
    pub(crate) neighbour_slots: &'a [u32],
}

impl CodeView<'_> {
    /// The ring slot a position's `t` field names.
    ///
    /// `t == 0` is the centre frame and `t` is `neighbour_index + 1` otherwise.
    fn slot_of(&self, t: u32) -> u32 {
        match t {
            0 => self.centre_slot,
            _ => self.neighbour_slots[(t - 1) as usize],
        }
    }

    fn code_at(&self, slot: u32, x: u32, y: u32, channel: u32) -> u32 {
        let row = slot as usize * self.height as usize + y as usize;
        let pixel = row * self.width as usize + x as usize;
        let index = pixel * self.stored_ch as usize + channel as usize;
        self.codes[index]
    }

    /// Why the view cannot be read safely, if it cannot.
    fn problem(&self) -> Option<String> {
        if self.width < PATCH_SIZE || self.height < PATCH_SIZE {
            let (width, height) = (self.width, self.height);
            return Some(format!("a {width}x{height} frame holds no patch"));
        }

        if self.stored_ch == 0 || self.channels > self.stored_ch {
            let (channels, stored_ch) = (self.channels, self.stored_ch);
            return Some(format!("{channels} channels do not fit {stored_ch} stored lanes"));
        }

        let frame_lanes = self.width as usize * self.height as usize * self.stored_ch as usize;
        let slots = self.codes.len() / frame_lanes;
        let mut ring_slots = self.neighbour_slots.iter().chain([&self.centre_slot]);
        let outside = ring_slots.find(|&&slot| slot as usize >= slots);
        if let Some(slot) = outside {
            return Some(format!("slot {slot} is past the {slots} slots the codes hold"));
        }

        None
    }

    /// Why a member word cannot be scored, if it cannot.
    ///
    /// The patch must lie inside the frame and `t` must name a slot. A centre pick must also be in
    /// the centre frame.
    fn position_problem(&self, word: u32, centre_pick: bool) -> Option<&'static str> {
        let (x, y, t) = unpack_position(word);

        if x > self.width - PATCH_SIZE || y > self.height - PATCH_SIZE {
            return Some("its patch leaves the frame");
        }

        if t as usize > self.neighbour_slots.len() {
            return Some("its frame is past the neighbour slots");
        }

        if centre_pick && t != 0 {
            return Some("a centre pick is outside the centre frame");
        }

        None
    }
}

/// Packs a position the way the search kernels' `pack_pos_t` does.
///
/// x takes bits 0-12, y bits 13-25 and `t` bits 26-31.
pub(crate) fn pack_position(x: u32, y: u32, t: u32) -> u32 {
    (t << 26) | (y << 13) | x
}

/// Splits a `pack_pos_t` word into its x, y and `t` fields.
fn unpack_position(word: u32) -> (u32, u32, u32) {
    let x = word & 0x1FFF;
    let y = (word >> 13) & 0x1FFF;
    let t = word >> 26;
    (x, y, t)
}

/// The exact SSD between the 8x8 patches at two packed positions, over `channels` lanes.
pub(crate) fn patch_ssd(view: &CodeView<'_>, first: u32, second: u32) -> u64 {
    let (first_x, first_y, first_t) = unpack_position(first);
    let (second_x, second_y, second_t) = unpack_position(second);
    let first_slot = view.slot_of(first_t);
    let second_slot = view.slot_of(second_t);

    let mut ssd = 0u64;
    for row in 0..PATCH_SIZE {
        for column in 0..PATCH_SIZE {
            for channel in 0..view.channels {
                let first_code = view.code_at(first_slot, first_x + column, first_y + row, channel);
                let second_code = view.code_at(second_slot, second_x + column, second_y + row, channel);
                let difference = first_code.abs_diff(second_code) as u64;
                ssd += difference * difference;
            }
        }
    }

    ssd
}

/// The centre search's members of the reference at `(ref_x, ref_y)`, ranked by `(ssd, position)`
/// with the self-match first.
///
/// The candidates are the `spatial_radius` rectangle around the reference, clipped to the frame.
/// Returns at most [MAX_K](crate::collab::MAX_K) words.
#[cfg(test)]
pub(crate) fn host_centre_members(
    view: &CodeView<'_>,
    ref_x: u32,
    ref_y: u32,
    spatial_radius: u32,
) -> Vec<u32> {
    let max_x = view.width - PATCH_SIZE;
    let max_y = view.height - PATCH_SIZE;
    let left = ref_x.saturating_sub(spatial_radius);
    let right = u32::min(ref_x + spatial_radius, max_x);
    let top = ref_y.saturating_sub(spatial_radius);
    let bottom = u32::min(ref_y + spatial_radius, max_y);
    let reference = pack_position(ref_x, ref_y, 0);

    let mut ranked = Vec::new();
    for y in top..=bottom {
        for x in left..=right {
            let position = pack_position(x, y, 0);
            if position == reference {
                continue;
            }

            let ssd = patch_ssd(view, reference, position);
            ranked.push((ssd, position));
        }
    }

    ranked.sort_unstable();

    let mut members = vec![reference];
    let others = ranked
        .into_iter()
        .map(|(_, position)| position)
        .take(MAX_K as usize - 1);
    members.extend(others);
    members
}

/// Checks a candidate's member words against the baseline's under the eighth-best tie bound.
///
/// Both buffers hold [MEMBER_WORDS](crate::collab::kernels::fused::MEMBER_WORDS) words per
/// reference. The group words must match unless a tied anchor changes the grid choice. Every
/// checked member must be a distinct position inside the view, and member 0 must be the reference
/// itself. Each centre pick may be no further from the reference than the baseline's furthest. In a
/// volume grid the anchors are the centre picks. A volume's frame members are held to the same
/// bound against the anchor, but only when the anchor and every earlier volume match the baseline
/// word for word, because earlier volumes change what a later one may pick. When a tied anchor
/// changes the grid choice, only the anchors are bounded against the baseline, because the two
/// groups hold different members after them.
pub(crate) fn tie_bound_verdict(
    view: &CodeView<'_>,
    refs_x: u32,
    refs_y: u32,
    grid_frames: u32,
    baseline: &[u32],
    candidate: &[u32],
) -> CompareResult {
    let expected_words = (refs_x * refs_y * MEMBER_WORDS) as usize;
    if baseline.len() != expected_words || candidate.len() != expected_words {
        let baseline_words = baseline.len();
        let candidate_words = candidate.len();
        let message = format!(
            "expected {expected_words} member words, got {baseline_words} baseline and \
             {candidate_words} candidate"
        );
        return CompareResult::Fail(message);
    }

    let view_problem = view.problem();
    if let Some(problem) = view_problem {
        return CompareResult::Fail(problem);
    }

    if grid_frames == 0 || !MAX_K.is_multiple_of(grid_frames) {
        let message = format!("{grid_frames} grid frames do not divide a group of {MAX_K}");
        return CompareResult::Fail(message);
    }

    let max_x = view.width - PATCH_SIZE;
    let max_y = view.height - PATCH_SIZE;
    let baseline_groups = baseline.chunks(MEMBER_WORDS as usize);
    let candidate_groups = candidate.chunks(MEMBER_WORDS as usize);

    let group_pairs = baseline_groups.zip(candidate_groups).enumerate();
    for (reference_index, (baseline_group, candidate_group)) in group_pairs {
        let ref_column = reference_index as u32 % refs_x;
        let ref_row = reference_index as u32 / refs_x;
        let ref_x = u32::min(ref_column * STEP, max_x);
        let ref_y = u32::min(ref_row * STEP, max_y);
        let reference = pack_position(ref_x, ref_y, 0);

        let group = GroupPair {
            view,
            reference_index,
            reference,
            baseline: baseline_group,
            candidate: candidate_group,
        };
        let verdict = group.verdict(grid_frames);
        if !verdict.passed() {
            return verdict;
        }
    }

    CompareResult::Pass
}

/// One reference's baseline and candidate groups.
struct GroupPair<'a> {
    view: &'a CodeView<'a>,
    reference_index: usize,
    reference: u32,
    baseline: &'a [u32],
    candidate: &'a [u32],
}

impl GroupPair<'_> {
    fn verdict(&self, grid_frames: u32) -> CompareResult {
        let group_word = self.baseline[MAX_K as usize];
        let candidate_word = self.candidate[MAX_K as usize];
        if group_word != candidate_word {
            return self.grid_choice_verdict(grid_frames);
        }

        let count = u32::min(group_word & GROUP_SIZE_MASK, MAX_K) as usize;
        let use_grid = group_word & GROUP_USES_GRID != 0;
        let grid_frames = grid_frames as usize;
        let checked = if use_grid { MAX_K as usize } else { count };
        let layout = GroupLayout { checked, use_grid };

        let layout_verdict = self.layout_verdict(layout, layout, grid_frames);
        if !layout_verdict.passed() {
            return layout_verdict;
        }

        if !use_grid {
            return self.bounded(&MEMBER_INDICES[..count], self.reference);
        }

        let volumes = MAX_K as usize / grid_frames;
        let anchor_slots = anchor_indices(true, grid_frames);
        let anchors = &anchor_slots[..volumes];
        let anchor_verdict = self.bounded(anchors, self.reference);
        if !anchor_verdict.passed() {
            return anchor_verdict;
        }

        for &first in anchors {
            // Earlier volumes' members are excluded from this volume's search, so a volume is only
            // comparable while everything before its frame members is the same.
            let same_prefix = self.baseline[..=first] == self.candidate[..=first];
            if !same_prefix {
                break;
            }

            let anchor = self.candidate[first];
            let frame_members = &MEMBER_INDICES[first + 1..first + grid_frames];
            let frame_verdict = self.bounded(frame_members, anchor);
            if !frame_verdict.passed() {
                return frame_verdict;
            }
        }

        CompareResult::Pass
    }

    /// Checks a pair whose group words differ.
    ///
    /// It passes only when the words differ in
    /// [GROUP_USES_GRID](crate::collab::kernels::fused::GROUP_USES_GRID) alone, both groups are
    /// full and well formed, member 0 is the reference, and the anchors differ. Each candidate
    /// anchor must also be no further from the reference than the baseline's furthest anchor. When
    /// the candidate is the single-frame group, each centre pick after its anchors must be no
    /// closer to the reference than the pick before it or any of its anchors.
    fn grid_choice_verdict(&self, grid_frames: u32) -> CompareResult {
        let reference_index = self.reference_index;
        let group_word = self.baseline[MAX_K as usize];
        let candidate_word = self.candidate[MAX_K as usize];
        let message = format!(
            "reference {reference_index} group word is {candidate_word:#x}, the baseline's is \
             {group_word:#x}"
        );

        let only_grid_differs = group_word ^ candidate_word == GROUP_USES_GRID;
        let both_full = group_word & GROUP_SIZE_MASK == MAX_K && candidate_word & GROUP_SIZE_MASK == MAX_K;
        if !only_grid_differs || !both_full {
            return CompareResult::Fail(message);
        }

        let grid_frames = grid_frames as usize;
        let baseline_grid = group_word & GROUP_USES_GRID != 0;
        let candidate_grid = !baseline_grid;
        let baseline_layout = GroupLayout {
            checked: MAX_K as usize,
            use_grid: baseline_grid,
        };
        let candidate_layout = GroupLayout {
            checked: MAX_K as usize,
            use_grid: candidate_grid,
        };

        let layout_verdict = self.layout_verdict(baseline_layout, candidate_layout, grid_frames);
        if !layout_verdict.passed() {
            return layout_verdict;
        }

        let volumes = MAX_K as usize / grid_frames;
        let baseline_slots = anchor_indices(baseline_grid, grid_frames);
        let candidate_slots = anchor_indices(candidate_grid, grid_frames);
        let baseline_anchors = &baseline_slots[..volumes];
        let candidate_anchors = &candidate_slots[..volumes];

        let baseline_words = baseline_anchors.iter().map(|&index| self.baseline[index]);
        let candidate_words = candidate_anchors.iter().map(|&index| self.candidate[index]);
        if baseline_words.eq(candidate_words) {
            return CompareResult::Fail(format!("{message} with the same anchors"));
        }

        let anchor_verdict = self.bounded_between(baseline_anchors, candidate_anchors, self.reference);
        if !anchor_verdict.passed() || candidate_grid {
            return anchor_verdict;
        }

        self.ascending_after_anchors(volumes)
    }

    /// Whether each candidate centre pick after the first `volumes` is no closer to the reference
    /// than the pick before it or any of those first `volumes`.
    fn ascending_after_anchors(&self, volumes: usize) -> CompareResult {
        let anchor_distances = self.candidate[..volumes]
            .iter()
            .map(|&word| patch_ssd(self.view, self.reference, word));
        let mut previous = anchor_distances.max().unwrap_or(0);

        for index in volumes..MAX_K as usize {
            let distance = patch_ssd(self.view, self.reference, self.candidate[index]);
            if distance < previous {
                let reference_index = self.reference_index;
                let message = format!(
                    "reference {reference_index} member {index} has ssd {distance}, below the {previous} \
                     before it"
                );
                return CompareResult::Fail(message);
            }

            previous = distance;
        }

        CompareResult::Pass
    }

    /// Whether both groups are well formed under their layouts and the candidate's member 0 is the
    /// reference.
    fn layout_verdict(
        &self,
        baseline: GroupLayout,
        candidate: GroupLayout,
        grid_frames: usize,
    ) -> CompareResult {
        let baseline_problem = self.malformed(self.baseline, baseline, grid_frames);
        if let Some(problem) = baseline_problem {
            return CompareResult::Fail(format!("baseline {problem}"));
        }

        let candidate_problem = self.malformed(self.candidate, candidate, grid_frames);
        if let Some(problem) = candidate_problem {
            return CompareResult::Fail(format!("candidate {problem}"));
        }

        let first_member = self.candidate[0];
        if first_member != self.reference {
            let reference_index = self.reference_index;
            let expected = self.reference;
            let message = format!(
                "reference {reference_index} member 0 is {first_member:#x}, not the reference {expected:#x}"
            );
            return CompareResult::Fail(message);
        }

        CompareResult::Pass
    }

    /// Whether each candidate member at `candidate_indices` is no further from `anchor` than the
    /// baseline's furthest member at `baseline_indices`.
    fn bounded_between(
        &self,
        baseline_indices: &[usize],
        candidate_indices: &[usize],
        anchor: u32,
    ) -> CompareResult {
        let worst_baseline = baseline_indices
            .iter()
            .map(|&index| patch_ssd(self.view, anchor, self.baseline[index]))
            .max()
            .unwrap_or(0);

        for &index in candidate_indices {
            let candidate_ssd = patch_ssd(self.view, anchor, self.candidate[index]);
            if candidate_ssd > worst_baseline {
                let reference_index = self.reference_index;
                let message = format!(
                    "reference {reference_index} member {index} has ssd {candidate_ssd}, past the baseline's \
                     worst {worst_baseline}"
                );
                return CompareResult::Fail(message);
            }
        }

        CompareResult::Pass
    }

    /// Why the first `layout.checked` words of `group` cannot be compared, if they cannot.
    ///
    /// Each word must be a position [CodeView] can score, and no word may repeat. Without the grid
    /// every word is a centre pick, and in a grid the anchors are.
    fn malformed(&self, group: &[u32], layout: GroupLayout, grid_frames: usize) -> Option<String> {
        let reference_index = self.reference_index;
        let words = &group[..layout.checked];

        for (index, &word) in words.iter().enumerate() {
            let centre_pick = !layout.use_grid || index % grid_frames == 0;
            let problem = self.view.position_problem(word, centre_pick);
            if let Some(problem) = problem {
                let message =
                    format!("reference {reference_index} member {index} ({word:#x}) is invalid, {problem}");
                return Some(message);
            }

            let repeated = words[..index].contains(&word);
            if repeated {
                let message =
                    format!("reference {reference_index} member {index} ({word:#x}) repeats a member");
                return Some(message);
            }
        }

        None
    }

    /// Whether each candidate member at `indices` is no further from `anchor` than the baseline's
    /// furthest member at the same indices.
    fn bounded(&self, indices: &[usize], anchor: u32) -> CompareResult {
        self.bounded_between(indices, indices, anchor)
    }
}

/// How many leading words of a group are checked, and whether they form a volume grid.
#[derive(Clone, Copy)]
struct GroupLayout {
    checked: usize,
    use_grid: bool,
}

/// The member indices of a group, in order.
const MEMBER_INDICES: [usize; MAX_K as usize] = [0, 1, 2, 3, 4, 5, 6, 7];

/// The member indices of a group's volume anchors, valid for the first `MAX_K / grid_frames`.
///
/// A volume grid holds each anchor at the start of its volume. The single-frame group holds them
/// first, since its leading centre picks are the anchors the grid would have used.
fn anchor_indices(use_grid: bool, grid_frames: usize) -> [usize; MAX_K as usize] {
    let stride = if use_grid { grid_frames } else { 1 };
    std::array::from_fn(|volume| volume * stride)
}
