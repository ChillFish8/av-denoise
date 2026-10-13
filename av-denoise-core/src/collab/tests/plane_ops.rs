use cubecl::prelude::*;

use super::helpers::{R, make_client};
use crate::collab::MAX_K;
use crate::collab::kernels::plane_ops::{
    group_any,
    group_base,
    plane_ssd_reduce8,
    scatter_sums8,
    shift_insert8,
    shift_insert8_gated,
    transpose8,
};

#[cube(launch_unchecked)]
fn ssd_reduce_kernel(input: &Array<f32>, out: &mut Array<f32>) {
    let tid = ABSOLUTE_POS_X;
    out[tid as usize] = plane_ssd_reduce8(input[tid as usize]);
}

/// Runs [plane_ssd_reduce8] on the GPU, one lane per input.
fn run_ssd_reduce(input: &[f32]) -> Vec<f32> {
    let count = input.len();
    let client = make_client();
    let input_bytes = f32::as_bytes(input);
    let input_buf = client.create_from_slice(input_bytes);
    #[expect(
        clippy::manual_slice_size_calculation,
        reason = "count is the element count this output holds, not the input's byte length"
    )]
    let out_buf = client.empty(count * size_of::<f32>());

    unsafe {
        ssd_reduce_kernel::launch_unchecked::<R>(
            &client,
            CubeCount::new_1d(1),
            CubeDim::new_1d(count as u32),
            ArrayArg::from_raw_parts(input_buf, count),
            ArrayArg::from_raw_parts(out_buf.clone(), count),
        );
    }

    let out_bytes = client.read_one(out_buf).expect("ssd reduce readback failed");

    f32::from_bytes(&out_bytes)[..count].to_vec()
}

#[test]
fn ssd_reduce8_sums_within_its_own_group() {
    // Group 0 holds 1..=8 summing to 36, group 1 holds 100..=800 summing to 3600. The magnitudes
    // differ so a leak across the boundary cannot coincidentally match.
    let input: Vec<f32> = (1..=8)
        .map(|value| value as f32)
        .chain((1..=8).map(|value| (value * 100) as f32))
        .collect();
    let sums = run_ssd_reduce(&input);

    for (lane, &sum) in sums.iter().take(8).enumerate() {
        assert_eq!(sum, 36.0, "lane {lane} of group 0");
    }

    for (lane, &sum) in sums.iter().enumerate().take(16).skip(8) {
        assert_eq!(sum, 3600.0, "lane {lane} of group 1");
    }
}

#[cube(launch_unchecked)]
fn shift_insert_kernel(
    dists: &Array<f32>,
    posns: &Array<u32>,
    n: u32,
    out_d: &mut Array<f32>,
    out_p: &mut Array<u32>,
) {
    let tid = ABSOLUTE_POS_X;
    let sub = UNIT_POS_PLANE % 8;
    let mut best_d = 3.0e38f32;
    let mut best_p = 0u32;
    let mut i = 0u32;
    while i < n {
        shift_insert8(
            &mut best_d,
            &mut best_p,
            dists[i as usize],
            posns[i as usize],
            sub,
        );
        i += 1u32;
    }
    out_d[tid as usize] = best_d;
    out_p[tid as usize] = best_p;
}

#[cube(launch_unchecked)]
fn shift_insert_gated_kernel(
    dists: &Array<f32>,
    posns: &Array<u32>,
    n: u32,
    out_d: &mut Array<f32>,
    out_p: &mut Array<u32>,
) {
    let tid = ABSOLUTE_POS_X;
    let sub = UNIT_POS_PLANE % 8;
    let base = group_base();
    let mut best_d = 3.0e38f32;
    let mut best_p = 0u32;
    let mut i = 0u32;
    while i < n {
        shift_insert8_gated(
            &mut best_d,
            &mut best_p,
            dists[i as usize],
            posns[i as usize],
            sub,
            base,
        );
        i += 1u32;
    }
    out_d[tid as usize] = best_d;
    out_p[tid as usize] = best_p;
}

/// Runs [shift_insert8] over `dists`/`posns` on a single 8-lane group, feeding every candidate
/// through in order.
fn run_shift_insert_with(dists: &[f32], posns: &[u32]) -> (Vec<f32>, Vec<u32>) {
    let count = dists.len();
    assert_eq!(posns.len(), count);
    let client = make_client();
    let dists_bytes = f32::as_bytes(dists);
    let posns_bytes = u32::as_bytes(posns);
    let dists_buf = client.create_from_slice(dists_bytes);
    let posns_buf = client.create_from_slice(posns_bytes);
    let out_d_buf = client.empty(MAX_K as usize * size_of::<f32>());
    let out_p_buf = client.empty(MAX_K as usize * size_of::<u32>());

    unsafe {
        shift_insert_kernel::launch_unchecked::<R>(
            &client,
            CubeCount::new_1d(1),
            CubeDim::new_1d(8),
            ArrayArg::from_raw_parts(dists_buf, count),
            ArrayArg::from_raw_parts(posns_buf, count),
            count as u32,
            ArrayArg::from_raw_parts(out_d_buf.clone(), 8),
            ArrayArg::from_raw_parts(out_p_buf.clone(), 8),
        );
    }

    let dist_bytes = client
        .read_one(out_d_buf)
        .expect("shift_insert dist readback failed");
    let pos_bytes = client
        .read_one(out_p_buf)
        .expect("shift_insert pos readback failed");
    let best_dists = f32::from_bytes(&dist_bytes)[..8].to_vec();
    let best_posns = u32::from_bytes(&pos_bytes)[..8].to_vec();

    (best_dists, best_posns)
}

/// [run_shift_insert_with] with `posns` set to `0..dists.len()`.
fn run_shift_insert(dists: &[f32]) -> (Vec<f32>, Vec<u32>) {
    let posns: Vec<u32> = (0..dists.len() as u32).collect();
    run_shift_insert_with(dists, &posns)
}

/// Runs [shift_insert8_gated] over `dists`/`posns` on a single 8-lane group, feeding every
/// candidate through in order.
fn run_shift_insert_gated_with(dists: &[f32], posns: &[u32]) -> (Vec<f32>, Vec<u32>) {
    let count = dists.len();
    assert_eq!(posns.len(), count);
    let client = make_client();
    let dists_bytes = f32::as_bytes(dists);
    let posns_bytes = u32::as_bytes(posns);
    let dists_buf = client.create_from_slice(dists_bytes);
    let posns_buf = client.create_from_slice(posns_bytes);
    let out_d_buf = client.empty(MAX_K as usize * size_of::<f32>());
    let out_p_buf = client.empty(MAX_K as usize * size_of::<u32>());

    unsafe {
        shift_insert_gated_kernel::launch_unchecked::<R>(
            &client,
            CubeCount::new_1d(1),
            CubeDim::new_1d(8),
            ArrayArg::from_raw_parts(dists_buf, count),
            ArrayArg::from_raw_parts(posns_buf, count),
            count as u32,
            ArrayArg::from_raw_parts(out_d_buf.clone(), 8),
            ArrayArg::from_raw_parts(out_p_buf.clone(), 8),
        );
    }

    let dist_bytes = client
        .read_one(out_d_buf)
        .expect("shift_insert_gated dist readback failed");
    let pos_bytes = client
        .read_one(out_p_buf)
        .expect("shift_insert_gated pos readback failed");
    let best_dists = f32::from_bytes(&dist_bytes)[..8].to_vec();
    let best_posns = u32::from_bytes(&pos_bytes)[..8].to_vec();

    (best_dists, best_posns)
}

/// One [shift_insert8] test input, named so the gated test can report which case failed.
struct InsertCase {
    name: &'static str,
    dists: Vec<f32>,
    posns: Vec<u32>,
}

/// The inputs of the four `shift_insert8` tests, shared so the gated test runs the identical inputs.
fn insert_test_cases() -> Vec<InsertCase> {
    vec![
        InsertCase {
            name: "matches_a_host_sort",
            dists: vec![5.0, 3.0, 9.0, 1.0, 7.0, 2.0, 8.0, 4.0, 6.0, 0.5, 11.0, 10.0],
            posns: (0..12).collect(),
        },
        InsertCase {
            name: "breaks_ties_toward_the_first_seen",
            dists: vec![1.0f32; 12],
            posns: (0..12).collect(),
        },
        InsertCase {
            name: "leaves_unfilled_slots_at_the_sentinel",
            dists: vec![3.0f32, 1.0, 2.0],
            posns: (0..3).collect(),
        },
        InsertCase {
            name: "handles_a_descending_feed",
            dists: (0..12).rev().map(|value| value as f32).collect(),
            posns: (0..12).collect(),
        },
    ]
}

fn find_case(name: &str) -> InsertCase {
    insert_test_cases()
        .into_iter()
        .find(|case| case.name == name)
        .unwrap_or_else(|| panic!("no insert_test_cases entry named {name}"))
}

#[test]
fn shift_insert8_matches_a_host_sort() {
    let case = find_case("matches_a_host_sort");
    let (got_d, _) = run_shift_insert(&case.dists);
    let mut want = case.dists.clone();
    want.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(&got_d[..8], &want[..8]);
}

/// Ties resolve to the candidate seen first, which fixes which member a group keeps on flat content.
#[test]
fn shift_insert8_breaks_ties_toward_the_first_seen() {
    let case = find_case("breaks_ties_toward_the_first_seen");
    let (_, got_p) = run_shift_insert_with(&case.dists, &case.posns);
    assert_eq!(&got_p[..8], &[0u32, 1, 2, 3, 4, 5, 6, 7]);
}

#[test]
fn shift_insert8_leaves_unfilled_slots_at_the_sentinel() {
    let case = find_case("leaves_unfilled_slots_at_the_sentinel");
    let (got_d, _) = run_shift_insert(&case.dists);
    assert_eq!(&got_d[..3], &[1.0, 2.0, 3.0]);

    for (slot, &dist) in got_d.iter().enumerate().take(8).skip(3) {
        assert!(dist > 1.0e38, "slot {slot} should still be the sentinel");
    }
}

/// A strictly descending feed takes the insert path on every candidate, the case the shift logic
/// is easiest to get wrong on.
#[test]
fn shift_insert8_handles_a_descending_feed() {
    let case = find_case("handles_a_descending_feed");
    let (got_d, _) = run_shift_insert(&case.dists);
    assert_eq!(&got_d[..8], &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
}

/// The gate is a compute saving, never an admission decision, so it must keep the same eight slots
/// as the ungated insert.
#[test]
fn shift_insert8_gated_matches_the_ungated_insert() {
    for case in insert_test_cases() {
        let want = run_shift_insert_with(&case.dists, &case.posns);
        let got = run_shift_insert_gated_with(&case.dists, &case.posns);
        assert_eq!(got, want, "case {}", case.name);
    }
}

#[cube(launch_unchecked)]
fn transpose_kernel(input: &Array<f32>, out: &mut Array<f32>) {
    let tid = UNIT_POS_X;
    let grp = tid / 8u32;
    let sub = tid % 8u32;
    let mut buf = SharedMemory::<f32>::new(8usize * 65usize);
    let mut values = Array::<f32>::new(8usize);
    #[unroll]
    for i in 0..8u32 {
        values[i as usize] = input[(tid * 8u32 + i) as usize];
    }
    transpose8(&mut buf, &mut values, sub, grp);
    #[unroll]
    for i in 0..8u32 {
        out[(tid * 8u32 + i) as usize] = values[i as usize];
    }
}

/// Runs [transpose8] over one cube of two 8-lane groups, so cross-group bleed through `buf` shows.
fn run_transpose(input: &[f32]) -> Vec<f32> {
    let count = input.len();
    assert_eq!(count, 128);
    let client = make_client();
    let input_bytes = f32::as_bytes(input);
    let input_buf = client.create_from_slice(input_bytes);
    #[expect(
        clippy::manual_slice_size_calculation,
        reason = "count is the element count this output holds, not the input's byte length"
    )]
    let out_buf = client.empty(count * size_of::<f32>());

    unsafe {
        transpose_kernel::launch_unchecked::<R>(
            &client,
            CubeCount::new_1d(1),
            CubeDim::new_1d(16),
            ArrayArg::from_raw_parts(input_buf, count),
            ArrayArg::from_raw_parts(out_buf.clone(), count),
        );
    }

    let out_bytes = client.read_one(out_buf).expect("transpose readback failed");

    f32::from_bytes(&out_bytes)[..count].to_vec()
}

/// Lane `sub` holds column `sub` on entry, so `input[sub * 8 + r]` is element `(r, sub)`. After the
/// transpose the same slot is element `(sub, r)`. Group 1 is offset by 1000 so a cross-group leak
/// cannot coincidentally match.
#[test]
fn transpose8_swaps_rows_and_columns_within_each_group() {
    let mut input = vec![0.0f32; 128];
    for grp in 0..2u32 {
        for sub in 0..8u32 {
            for r in 0..8u32 {
                let value = (r * 8 + sub) as f32 + grp as f32 * 1000.0;
                input[(grp * 64 + sub * 8 + r) as usize] = value;
            }
        }
    }

    let transposed = run_transpose(&input);

    for grp in 0..2u32 {
        for sub in 0..8u32 {
            for r in 0..8u32 {
                let want = (sub * 8 + r) as f32 + grp as f32 * 1000.0;
                let got = transposed[(grp * 64 + sub * 8 + r) as usize];
                assert_eq!(got, want, "group {grp} lane {sub} slot {r}");
            }
        }
    }
}

#[cube(launch_unchecked)]
fn scatter_sums_kernel(input: &Array<f32>, scattered: &mut Array<f32>, reduced: &mut Array<f32>) {
    let lane = ABSOLUTE_POS_X;
    let sub = UNIT_POS_PLANE % 8u32;
    let mut partials = Array::<f32>::new(8usize);

    #[unroll]
    for k in 0..8u32 {
        partials[k as usize] = input[(lane * 8u32 + k) as usize];
    }

    scattered[lane as usize] = scatter_sums8(&partials, sub);

    let mut own_total = 0.0f32;

    #[unroll]
    for k in 0..8u32 {
        let total = plane_ssd_reduce8(partials[k as usize]);
        own_total = select(sub == k, total, own_total);
    }

    reduced[lane as usize] = own_total;
}

#[test]
fn scatter_sums8_matches_the_group_reduction_bit_for_bit() {
    let lanes = 16usize;
    let input: Vec<f32> = (0..lanes * 8)
        .map(|index| {
            let magnitude = 10f32.powi((index % 7) as i32 - 3);
            let fraction = ((index * 2_654_435_761) % 1000) as f32 / 1000.0;
            magnitude * (0.5 + fraction)
        })
        .collect();
    let client = make_client();
    let input_bytes = f32::as_bytes(&input);
    let input_buf = client.create_from_slice(input_bytes);
    let scattered_buf = client.empty(lanes * size_of::<f32>());
    let reduced_buf = client.empty(lanes * size_of::<f32>());

    unsafe {
        scatter_sums_kernel::launch_unchecked::<R>(
            &client,
            CubeCount::new_1d(1),
            CubeDim::new_1d(lanes as u32),
            ArrayArg::from_raw_parts(input_buf, input.len()),
            ArrayArg::from_raw_parts(scattered_buf.clone(), lanes),
            ArrayArg::from_raw_parts(reduced_buf.clone(), lanes),
        );
    }

    let scattered_bytes = client.read_one(scattered_buf).expect("scattered readback failed");
    let reduced_bytes = client.read_one(reduced_buf).expect("reduced readback failed");
    let scattered = f32::from_bytes(&scattered_bytes);
    let reduced = f32::from_bytes(&reduced_bytes);

    let totals = scattered.iter().zip(reduced.iter()).take(lanes);
    for (lane, (scattered_total, reduced_total)) in totals.enumerate() {
        assert_eq!(scattered_total.to_bits(), reduced_total.to_bits(), "lane {lane}");
    }
}

#[cube(launch_unchecked)]
fn group_any_kernel(flags: &Array<u32>, out: &mut Array<u32>) {
    let lane = ABSOLUTE_POS_X;
    let flag = flags[lane as usize] != 0u32;
    let voted = group_any(flag);
    out[lane as usize] = select(voted, 1u32, 0u32);
}

#[test]
fn group_any_votes_within_its_own_group() {
    // Group 0 has one raised flag in its last lane, groups 1 and 3 have none, group 2 has every
    // flag raised.
    let lanes = 32usize;
    let mut flags = vec![0u32; lanes];
    flags[7] = 1;
    flags[16..24].fill(1);
    let client = make_client();
    let flag_bytes = u32::as_bytes(&flags);
    let flag_buf = client.create_from_slice(flag_bytes);
    let out_buf = client.empty(lanes * size_of::<u32>());

    unsafe {
        group_any_kernel::launch_unchecked::<R>(
            &client,
            CubeCount::new_1d(1),
            CubeDim::new_1d(lanes as u32),
            ArrayArg::from_raw_parts(flag_buf, lanes),
            ArrayArg::from_raw_parts(out_buf.clone(), lanes),
        );
    }

    let out_bytes = client.read_one(out_buf).expect("vote readback failed");
    let votes = u32::from_bytes(&out_bytes);

    for (lane, &vote) in votes.iter().take(lanes).enumerate() {
        let raised_group = lane < 8 || (16..24).contains(&lane);
        let expected = u32::from(raised_group);
        assert_eq!(vote, expected, "lane {lane}");
    }
}
