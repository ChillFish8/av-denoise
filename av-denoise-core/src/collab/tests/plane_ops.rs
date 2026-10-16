use cubecl::prelude::*;

use super::helpers::{R, make_client};
use crate::collab::MAX_K;
use crate::collab::kernels::plane_ops::{
    UNFILLED_DISTANCE,
    group_any,
    group_base,
    plane_ssd_reduce8,
    scatter_sums8,
    shift_insert8,
    shift_insert8_cached,
    shift_insert8_ordered,
    shift_insert8_ordered_cached,
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
    let out_launched = out_buf.clone();

    unsafe {
        let input_arg = ArrayArg::from_raw_parts(input_buf, count);
        let out_arg = ArrayArg::from_raw_parts(out_launched, count);

        ssd_reduce_kernel::launch_unchecked::<R>(
            &client,
            CubeCount::new_1d(1),
            CubeDim::new_1d(count as u32),
            input_arg,
            out_arg,
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
    let mut best_d = UNFILLED_DISTANCE.runtime();
    let mut best_p = 0u32;
    let mut i = 0u32;

    while i < n {
        let distance = dists[i as usize];
        let position = posns[i as usize];
        shift_insert8(&mut best_d, &mut best_p, distance, position, sub);
        i += 1u32;
    }

    out_d[tid as usize] = best_d;
    out_p[tid as usize] = best_p;
}

#[cube(launch_unchecked)]
fn shift_insert_cached_kernel(
    dists: &Array<f32>,
    posns: &Array<u32>,
    n: u32,
    out_d: &mut Array<f32>,
    out_p: &mut Array<u32>,
    out_worst: &mut Array<f32>,
) {
    let tid = ABSOLUTE_POS_X;
    let sub = UNIT_POS_PLANE % 8;
    let base = group_base();
    let mut best_d = UNFILLED_DISTANCE.runtime();
    let mut best_p = 0u32;
    let mut worst = UNFILLED_DISTANCE.runtime();
    let mut i = 0u32;

    while i < n {
        let distance = dists[i as usize];
        let position = posns[i as usize];
        shift_insert8_cached(
            &mut best_d,
            &mut best_p,
            &mut worst,
            distance,
            position,
            sub,
            base,
        );
        i += 1u32;
    }

    out_d[tid as usize] = best_d;
    out_p[tid as usize] = best_p;
    out_worst[tid as usize] = worst;
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
    let out_d_launched = out_d_buf.clone();
    let out_p_launched = out_p_buf.clone();

    unsafe {
        let dists_arg = ArrayArg::from_raw_parts(dists_buf, count);
        let posns_arg = ArrayArg::from_raw_parts(posns_buf, count);
        let out_d_arg = ArrayArg::from_raw_parts(out_d_launched, 8);
        let out_p_arg = ArrayArg::from_raw_parts(out_p_launched, 8);

        shift_insert_kernel::launch_unchecked::<R>(
            &client,
            CubeCount::new_1d(1),
            CubeDim::new_1d(8),
            dists_arg,
            posns_arg,
            count as u32,
            out_d_arg,
            out_p_arg,
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

/// Runs [shift_insert8_cached] over `dists`/`posns` on a single 8-lane group, feeding every
/// candidate through in order. Returns the slots and each lane's cached worst.
fn run_shift_insert_cached_with(dists: &[f32], posns: &[u32]) -> ((Vec<f32>, Vec<u32>), Vec<f32>) {
    let count = dists.len();
    assert_eq!(posns.len(), count);
    let client = make_client();
    let dists_bytes = f32::as_bytes(dists);
    let posns_bytes = u32::as_bytes(posns);
    let dists_buf = client.create_from_slice(dists_bytes);
    let posns_buf = client.create_from_slice(posns_bytes);
    let out_d_buf = client.empty(MAX_K as usize * size_of::<f32>());
    let out_p_buf = client.empty(MAX_K as usize * size_of::<u32>());
    let out_worst_buf = client.empty(MAX_K as usize * size_of::<f32>());

    let out_d_launched = out_d_buf.clone();
    let out_p_launched = out_p_buf.clone();
    let out_worst_launched = out_worst_buf.clone();

    unsafe {
        let dists_arg = ArrayArg::from_raw_parts(dists_buf, count);
        let posns_arg = ArrayArg::from_raw_parts(posns_buf, count);
        let out_d_arg = ArrayArg::from_raw_parts(out_d_launched, 8);
        let out_p_arg = ArrayArg::from_raw_parts(out_p_launched, 8);
        let out_worst_arg = ArrayArg::from_raw_parts(out_worst_launched, 8);

        shift_insert_cached_kernel::launch_unchecked::<R>(
            &client,
            CubeCount::new_1d(1),
            CubeDim::new_1d(8),
            dists_arg,
            posns_arg,
            count as u32,
            out_d_arg,
            out_p_arg,
            out_worst_arg,
        );
    }

    let dist_bytes = client
        .read_one(out_d_buf)
        .expect("shift_insert_cached dist readback failed");
    let pos_bytes = client
        .read_one(out_p_buf)
        .expect("shift_insert_cached pos readback failed");
    let worst_bytes = client
        .read_one(out_worst_buf)
        .expect("shift_insert_cached worst readback failed");
    let best_dists = f32::from_bytes(&dist_bytes)[..8].to_vec();
    let best_posns = u32::from_bytes(&pos_bytes)[..8].to_vec();
    let worsts = f32::from_bytes(&worst_bytes)[..8].to_vec();

    ((best_dists, best_posns), worsts)
}

/// One [shift_insert8] test input, named so the cached test can report which case failed.
struct InsertCase {
    name: &'static str,
    dists: Vec<f32>,
    posns: Vec<u32>,
}

/// The inputs of the four `shift_insert8` tests, shared so the cached test runs the identical inputs.
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
        assert_eq!(dist, UNFILLED_DISTANCE, "slot {slot} should be unfilled");
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

/// The cache is a compute saving, never an admission decision, so it must keep the same eight slots
/// as the ungated insert, and its worst must be the eighth slot.
#[test]
fn shift_insert8_cached_matches_the_ungated_insert() {
    for case in insert_test_cases() {
        let want = run_shift_insert_with(&case.dists, &case.posns);
        let (got, worsts) = run_shift_insert_cached_with(&case.dists, &case.posns);
        let eighth = want.0[7];

        assert_eq!(got, want, "case {}", case.name);
        assert!(worsts.iter().all(|&worst| worst == eighth), "case {}", case.name);
    }
}

#[cube(launch_unchecked)]
fn shift_insert_ordered_kernel(
    dists: &Array<f32>,
    posns: &Array<u32>,
    n: u32,
    out_d: &mut Array<f32>,
    out_p: &mut Array<u32>,
) {
    let tid = ABSOLUTE_POS_X;
    let sub = UNIT_POS_PLANE % 8;
    let mut best_d = UNFILLED_DISTANCE.runtime();
    let mut best_p = 0u32;
    let mut i = 0u32;

    while i < n {
        let distance = dists[i as usize];
        let position = posns[i as usize];
        shift_insert8_ordered(&mut best_d, &mut best_p, distance, position, sub);
        i += 1u32;
    }

    out_d[tid as usize] = best_d;
    out_p[tid as usize] = best_p;
}

#[cube(launch_unchecked)]
fn shift_insert_ordered_cached_kernel(
    dists: &Array<f32>,
    posns: &Array<u32>,
    n: u32,
    out_d: &mut Array<f32>,
    out_p: &mut Array<u32>,
    out_worst_d: &mut Array<f32>,
    out_worst_p: &mut Array<u32>,
) {
    let tid = ABSOLUTE_POS_X;
    let sub = UNIT_POS_PLANE % 8;
    let base = group_base();
    let mut best_d = UNFILLED_DISTANCE.runtime();
    let mut best_p = 0u32;
    let mut worst_d = UNFILLED_DISTANCE.runtime();
    let mut worst_p = 0u32;
    let mut i = 0u32;

    while i < n {
        let distance = dists[i as usize];
        let position = posns[i as usize];
        shift_insert8_ordered_cached(
            &mut best_d,
            &mut best_p,
            &mut worst_d,
            &mut worst_p,
            distance,
            position,
            sub,
            base,
        );
        i += 1u32;
    }

    out_d[tid as usize] = best_d;
    out_p[tid as usize] = best_p;
    out_worst_d[tid as usize] = worst_d;
    out_worst_p[tid as usize] = worst_p;
}

/// Runs [shift_insert8_ordered] over `dists`/`posns` on a single 8-lane group, feeding every
/// candidate through in order.
fn run_shift_insert_ordered_with(dists: &[f32], posns: &[u32]) -> (Vec<f32>, Vec<u32>) {
    let count = dists.len();
    assert_eq!(posns.len(), count);
    let client = make_client();
    let dists_bytes = f32::as_bytes(dists);
    let posns_bytes = u32::as_bytes(posns);
    let dists_buf = client.create_from_slice(dists_bytes);
    let posns_buf = client.create_from_slice(posns_bytes);
    let out_d_buf = client.empty(MAX_K as usize * size_of::<f32>());
    let out_p_buf = client.empty(MAX_K as usize * size_of::<u32>());
    let out_d_launched = out_d_buf.clone();
    let out_p_launched = out_p_buf.clone();

    unsafe {
        let dists_arg = ArrayArg::from_raw_parts(dists_buf, count);
        let posns_arg = ArrayArg::from_raw_parts(posns_buf, count);
        let out_d_arg = ArrayArg::from_raw_parts(out_d_launched, 8);
        let out_p_arg = ArrayArg::from_raw_parts(out_p_launched, 8);

        shift_insert_ordered_kernel::launch_unchecked::<R>(
            &client,
            CubeCount::new_1d(1),
            CubeDim::new_1d(8),
            dists_arg,
            posns_arg,
            count as u32,
            out_d_arg,
            out_p_arg,
        );
    }

    let dist_bytes = client
        .read_one(out_d_buf)
        .expect("shift_insert_ordered dist readback failed");
    let pos_bytes = client
        .read_one(out_p_buf)
        .expect("shift_insert_ordered pos readback failed");
    let best_dists = f32::from_bytes(&dist_bytes)[..8].to_vec();
    let best_posns = u32::from_bytes(&pos_bytes)[..8].to_vec();

    (best_dists, best_posns)
}

/// The slots, then each lane's cached worst distance and position.
type OrderedCachedOutput = ((Vec<f32>, Vec<u32>), Vec<f32>, Vec<u32>);

/// Runs [shift_insert8_ordered_cached] over `dists`/`posns` on a single 8-lane group, feeding every
/// candidate through in order.
fn run_shift_insert_ordered_cached_with(dists: &[f32], posns: &[u32]) -> OrderedCachedOutput {
    let count = dists.len();
    assert_eq!(posns.len(), count);
    let client = make_client();
    let dists_bytes = f32::as_bytes(dists);
    let posns_bytes = u32::as_bytes(posns);
    let dists_buf = client.create_from_slice(dists_bytes);
    let posns_buf = client.create_from_slice(posns_bytes);
    let out_d_buf = client.empty(MAX_K as usize * size_of::<f32>());
    let out_p_buf = client.empty(MAX_K as usize * size_of::<u32>());
    let out_worst_d_buf = client.empty(MAX_K as usize * size_of::<f32>());
    let out_worst_p_buf = client.empty(MAX_K as usize * size_of::<u32>());

    let out_d_launched = out_d_buf.clone();
    let out_p_launched = out_p_buf.clone();
    let out_worst_d_launched = out_worst_d_buf.clone();
    let out_worst_p_launched = out_worst_p_buf.clone();

    unsafe {
        let dists_arg = ArrayArg::from_raw_parts(dists_buf, count);
        let posns_arg = ArrayArg::from_raw_parts(posns_buf, count);
        let out_d_arg = ArrayArg::from_raw_parts(out_d_launched, 8);
        let out_p_arg = ArrayArg::from_raw_parts(out_p_launched, 8);
        let out_worst_d_arg = ArrayArg::from_raw_parts(out_worst_d_launched, 8);
        let out_worst_p_arg = ArrayArg::from_raw_parts(out_worst_p_launched, 8);

        shift_insert_ordered_cached_kernel::launch_unchecked::<R>(
            &client,
            CubeCount::new_1d(1),
            CubeDim::new_1d(8),
            dists_arg,
            posns_arg,
            count as u32,
            out_d_arg,
            out_p_arg,
            out_worst_d_arg,
            out_worst_p_arg,
        );
    }

    let dist_bytes = client
        .read_one(out_d_buf)
        .expect("shift_insert_ordered_cached dist readback failed");
    let pos_bytes = client
        .read_one(out_p_buf)
        .expect("shift_insert_ordered_cached pos readback failed");
    let worst_d_bytes = client
        .read_one(out_worst_d_buf)
        .expect("shift_insert_ordered_cached worst dist readback failed");
    let worst_p_bytes = client
        .read_one(out_worst_p_buf)
        .expect("shift_insert_ordered_cached worst pos readback failed");
    let best_dists = f32::from_bytes(&dist_bytes)[..8].to_vec();
    let best_posns = u32::from_bytes(&pos_bytes)[..8].to_vec();
    let worst_dists = f32::from_bytes(&worst_d_bytes)[..8].to_vec();
    let worst_posns = u32::from_bytes(&worst_p_bytes)[..8].to_vec();

    ((best_dists, best_posns), worst_dists, worst_posns)
}

/// A row-major feed, where every tie arrives with its smaller position first.
fn row_major_ties() -> (Vec<f32>, Vec<u32>) {
    let dists = vec![2.0f32, 1.0, 1.0, 3.0, 1.0, 2.0, 1.0, 0.5, 2.0, 1.0, 1.0, 4.0];
    let posns = (0..12u32).map(|index| index * 3).collect();

    (dists, posns)
}

#[test]
fn the_ordered_insert_matches_first_wins_on_a_row_major_feed() {
    let (dists, posns) = row_major_ties();

    let want = run_shift_insert_with(&dists, &posns);
    let got = run_shift_insert_ordered_with(&dists, &posns);

    assert_eq!(got, want);
}

/// The shared insert cases plus [row_major_ties], each fed in row-major order.
fn ordering_cases() -> Vec<InsertCase> {
    let (dists, posns) = row_major_ties();
    let ties = InsertCase {
        name: "row_major_ties",
        dists,
        posns,
    };

    let mut cases = insert_test_cases();
    cases.push(ties);
    cases
}

/// The reversed, interleaved and rotated visit orders of `count` candidates, as index lists.
fn visit_orders(count: usize) -> Vec<(&'static str, Vec<usize>)> {
    let reversed: Vec<usize> = (0..count).rev().collect();

    let evens = (0..count).step_by(2);
    let odds = (1..count).step_by(2);
    let interleaved: Vec<usize> = evens.chain(odds).collect();

    let mut rotated: Vec<usize> = (0..count).collect();
    let shift = count / 3;
    rotated.rotate_left(shift);

    vec![
        ("reversed", reversed),
        ("interleaved", interleaved),
        ("rotated", rotated),
    ]
}

/// Reorders `case`'s candidates into the visit order `order`.
fn reorder(case: &InsertCase, order: &[usize]) -> (Vec<f32>, Vec<u32>) {
    let dists = order.iter().map(|&index| case.dists[index]).collect();
    let posns = order.iter().map(|&index| case.posns[index]).collect();

    (dists, posns)
}

#[test]
fn the_ordered_insert_does_not_depend_on_visit_order() {
    for case in ordering_cases() {
        let want = run_shift_insert_with(&case.dists, &case.posns);

        for (order_name, order) in visit_orders(case.dists.len()) {
            let (dists, posns) = reorder(&case, &order);
            let got = run_shift_insert_ordered_with(&dists, &posns);

            assert_eq!(got, want, "case {} fed {order_name}", case.name);
        }
    }
}

#[test]
fn the_cached_ordered_insert_does_not_depend_on_visit_order() {
    for case in ordering_cases() {
        let want = run_shift_insert_with(&case.dists, &case.posns);
        let eighth_d = want.0[7];
        let eighth_p = want.1[7];

        for (order_name, order) in visit_orders(case.dists.len()) {
            let (dists, posns) = reorder(&case, &order);
            let (got, worst_d, worst_p) = run_shift_insert_ordered_cached_with(&dists, &posns);

            assert_eq!(got, want, "case {} fed {order_name}", case.name);
            assert!(
                worst_d.iter().all(|&distance| distance == eighth_d),
                "case {} fed {order_name}",
                case.name
            );
            assert!(
                worst_p.iter().all(|&position| position == eighth_p),
                "case {} fed {order_name}",
                case.name
            );
        }
    }
}

/// A masked turn carries the unfilled distance, so it ties an unfilled slot and must never place.
#[test]
fn the_ordered_insert_never_places_a_masked_turn() {
    let dists = vec![1.0f32, UNFILLED_DISTANCE, 2.0, UNFILLED_DISTANCE];
    let posns = vec![5u32, 0, 9, 3];

    let (got_d, got_p) = run_shift_insert_ordered_with(&dists, &posns);

    assert_eq!(&got_d[..2], &[1.0, 2.0]);
    assert_eq!(&got_p[..2], &[5, 9]);
    assert!(got_d[2..].iter().all(|&distance| distance == UNFILLED_DISTANCE));
    assert!(got_p[2..].iter().all(|&position| position == 0));
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
    let out_launched = out_buf.clone();

    unsafe {
        let input_arg = ArrayArg::from_raw_parts(input_buf, count);
        let out_arg = ArrayArg::from_raw_parts(out_launched, count);

        transpose_kernel::launch_unchecked::<R>(
            &client,
            CubeCount::new_1d(1),
            CubeDim::new_1d(16),
            input_arg,
            out_arg,
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
    let input_len = input.len();
    let scattered_launched = scattered_buf.clone();
    let reduced_launched = reduced_buf.clone();

    unsafe {
        let input_arg = ArrayArg::from_raw_parts(input_buf, input_len);
        let scattered_arg = ArrayArg::from_raw_parts(scattered_launched, lanes);
        let reduced_arg = ArrayArg::from_raw_parts(reduced_launched, lanes);

        scatter_sums_kernel::launch_unchecked::<R>(
            &client,
            CubeCount::new_1d(1),
            CubeDim::new_1d(lanes as u32),
            input_arg,
            scattered_arg,
            reduced_arg,
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
    let out_launched = out_buf.clone();

    unsafe {
        let flag_arg = ArrayArg::from_raw_parts(flag_buf, lanes);
        let out_arg = ArrayArg::from_raw_parts(out_launched, lanes);

        group_any_kernel::launch_unchecked::<R>(
            &client,
            CubeCount::new_1d(1),
            CubeDim::new_1d(lanes as u32),
            flag_arg,
            out_arg,
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
