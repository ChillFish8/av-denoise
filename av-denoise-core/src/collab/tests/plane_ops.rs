use cubecl::prelude::*;

use super::helpers::{R, make_client};
use crate::collab::MAX_K;
use crate::collab::kernels::plane_ops::{
    group_base,
    plane_ssd_reduce8,
    shift_insert8,
    shift_insert8_gated,
    transpose8,
};

#[cube(launch_unchecked)]
fn ssd_reduce_kernel(input: &Array<f32>, out: &mut Array<f32>) {
    let tid = ABSOLUTE_POS_X;
    out[tid as usize] = plane_ssd_reduce8(input[tid as usize]);
}

/// Runs [`plane_ssd_reduce8`] on the GPU, one lane per input, so the
/// host checks the code the kernels actually consume.
fn run_ssd_reduce(input: &[f32]) -> Vec<f32> {
    let n = input.len();
    let client = make_client();
    let input_buf = client.create_from_slice(f32::as_bytes(input));
    // One output slot per input lane. `size_of_val(input)` reaches the
    // same number but ties the output's size to the input's slice.
    #[expect(
        clippy::manual_slice_size_calculation,
        reason = "n is the element count this output holds, not the input's byte length"
    )]
    let out_buf = client.empty(n * size_of::<f32>());

    unsafe {
        ssd_reduce_kernel::launch_unchecked::<R>(
            &client,
            CubeCount::new_1d(1),
            CubeDim::new_1d(n as u32),
            ArrayArg::from_raw_parts(input_buf, n),
            ArrayArg::from_raw_parts(out_buf.clone(), n),
        );
    }

    let out = client.read_one(out_buf).expect("ssd reduce readback failed");
    f32::from_bytes(&out)[..n].to_vec()
}

/// Every lane of a group must come back holding that group's whole sum,
/// and groups must not bleed into each other.
#[test]
fn ssd_reduce8_sums_within_its_own_group() {
    // Group 0 holds 1..=8 summing to 36, group 1 holds 100..=800
    // summing to 3600. Distinct magnitudes so a leak across the
    // boundary cannot coincidentally match.
    let input: Vec<f32> = (1..=8)
        .map(|v| v as f32)
        .chain((1..=8).map(|v| (v * 100) as f32))
        .collect();
    let out = run_ssd_reduce(&input);
    for (lane, &v) in out.iter().take(8).enumerate() {
        assert_eq!(v, 36.0, "lane {lane} of group 0");
    }
    for (lane, &v) in out.iter().enumerate().take(16).skip(8) {
        assert_eq!(v, 3600.0, "lane {lane} of group 1");
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

/// Runs [`shift_insert8`] over `dists`/`posns` on a single 8-lane group,
/// feeding every candidate through in order.
fn run_shift_insert_with(dists: &[f32], posns: &[u32]) -> (Vec<f32>, Vec<u32>) {
    let n = dists.len();
    assert_eq!(posns.len(), n);
    let client = make_client();
    let dists_buf = client.create_from_slice(f32::as_bytes(dists));
    let posns_buf = client.create_from_slice(u32::as_bytes(posns));
    let out_d_buf = client.empty(MAX_K as usize * size_of::<f32>());
    let out_p_buf = client.empty(MAX_K as usize * size_of::<u32>());

    unsafe {
        shift_insert_kernel::launch_unchecked::<R>(
            &client,
            CubeCount::new_1d(1),
            CubeDim::new_1d(8),
            ArrayArg::from_raw_parts(dists_buf, n),
            ArrayArg::from_raw_parts(posns_buf, n),
            n as u32,
            ArrayArg::from_raw_parts(out_d_buf.clone(), 8),
            ArrayArg::from_raw_parts(out_p_buf.clone(), 8),
        );
    }

    let out_d = client
        .read_one(out_d_buf)
        .expect("shift_insert dist readback failed");
    let out_p = client
        .read_one(out_p_buf)
        .expect("shift_insert pos readback failed");
    (
        f32::from_bytes(&out_d)[..8].to_vec(),
        u32::from_bytes(&out_p)[..8].to_vec(),
    )
}

/// [`run_shift_insert_with`] with `posns` set to `(0..n)`.
fn run_shift_insert(dists: &[f32]) -> (Vec<f32>, Vec<u32>) {
    let posns: Vec<u32> = (0..dists.len() as u32).collect();
    run_shift_insert_with(dists, &posns)
}

/// Runs [`shift_insert8_gated`] over `dists`/`posns` on a single 8-lane
/// group, feeding every candidate through in order.
fn run_shift_insert_gated_with(dists: &[f32], posns: &[u32]) -> (Vec<f32>, Vec<u32>) {
    let n = dists.len();
    assert_eq!(posns.len(), n);
    let client = make_client();
    let dists_buf = client.create_from_slice(f32::as_bytes(dists));
    let posns_buf = client.create_from_slice(u32::as_bytes(posns));
    let out_d_buf = client.empty(MAX_K as usize * size_of::<f32>());
    let out_p_buf = client.empty(MAX_K as usize * size_of::<u32>());

    unsafe {
        shift_insert_gated_kernel::launch_unchecked::<R>(
            &client,
            CubeCount::new_1d(1),
            CubeDim::new_1d(8),
            ArrayArg::from_raw_parts(dists_buf, n),
            ArrayArg::from_raw_parts(posns_buf, n),
            n as u32,
            ArrayArg::from_raw_parts(out_d_buf.clone(), 8),
            ArrayArg::from_raw_parts(out_p_buf.clone(), 8),
        );
    }

    let out_d = client
        .read_one(out_d_buf)
        .expect("shift_insert_gated dist readback failed");
    let out_p = client
        .read_one(out_p_buf)
        .expect("shift_insert_gated pos readback failed");
    (
        f32::from_bytes(&out_d)[..8].to_vec(),
        u32::from_bytes(&out_p)[..8].to_vec(),
    )
}

/// One [`shift_insert8`] test input, named so the gated test can report
/// which case failed.
struct InsertCase {
    name: &'static str,
    dists: Vec<f32>,
    posns: Vec<u32>,
}

/// The inputs the four `shift_insert8` tests below use, so the gated
/// test in this module runs the identical inputs rather than a
/// duplicated copy of the literals.
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
            dists: (0..12).rev().map(|v| v as f32).collect(),
            posns: (0..12).collect(),
        },
    ]
}

/// The case from [`insert_test_cases`] with the given name.
fn find_case(name: &str) -> InsertCase {
    insert_test_cases()
        .into_iter()
        .find(|case| case.name == name)
        .unwrap_or_else(|| panic!("no insert_test_cases entry named {name}"))
}

/// The eight lanes must hold the eight smallest distances in ascending
/// order, slot 0 the smallest.
#[test]
fn shift_insert8_matches_a_host_sort() {
    let case = find_case("matches_a_host_sort");
    let (got_d, _) = run_shift_insert(&case.dists);
    let mut want = case.dists.clone();
    want.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(&got_d[..8], &want[..8]);
}

/// Ties must resolve to the candidate seen first, which is what fixes
/// which member a group keeps on flat content.
#[test]
fn shift_insert8_breaks_ties_toward_the_first_seen() {
    let case = find_case("breaks_ties_toward_the_first_seen");
    let (_, got_p) = run_shift_insert_with(&case.dists, &case.posns);
    assert_eq!(&got_p[..8], &[0u32, 1, 2, 3, 4, 5, 6, 7]);
}

/// Fewer candidates than slots leaves the tail at the sentinel rather
/// than at a stale or duplicated entry.
#[test]
fn shift_insert8_leaves_unfilled_slots_at_the_sentinel() {
    let case = find_case("leaves_unfilled_slots_at_the_sentinel");
    let (got_d, _) = run_shift_insert(&case.dists);
    assert_eq!(&got_d[..3], &[1.0, 2.0, 3.0]);
    for (slot, &d) in got_d.iter().enumerate().take(8).skip(3) {
        assert!(d > 1.0e38, "slot {slot} should still be the sentinel");
    }
}

/// A strictly descending feed exercises the insert path on every single
/// candidate, which is the case the shift logic is easiest to get wrong on.
#[test]
fn shift_insert8_handles_a_descending_feed() {
    let case = find_case("handles_a_descending_feed");
    let (got_d, _) = run_shift_insert(&case.dists);
    assert_eq!(&got_d[..8], &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
}

/// The gate is a compute saving, never an admission decision. It must
/// produce the same eight slots as the ungated insert on every input the
/// ungated one is tested on.
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
    let mut v = Array::<f32>::new(8usize);
    #[unroll]
    for i in 0..8u32 {
        v[i as usize] = input[(tid * 8u32 + i) as usize];
    }
    transpose8(&mut buf, &mut v, sub, grp);
    #[unroll]
    for i in 0..8u32 {
        out[(tid * 8u32 + i) as usize] = v[i as usize];
    }
}

/// Runs [`transpose8`] over one cube of two 8-lane groups and reads back
/// both transposed blocks.
///
/// Two groups is the point, because it catches cross-group bleed
/// through `buf`.
fn run_transpose(input: &[f32]) -> Vec<f32> {
    let n = input.len();
    assert_eq!(n, 128);
    let client = make_client();
    let input_buf = client.create_from_slice(f32::as_bytes(input));
    // One output slot per input value, sized from the count rather than
    // from the input's own slice.
    #[expect(
        clippy::manual_slice_size_calculation,
        reason = "n is the element count this output holds, not the input's byte length"
    )]
    let out_buf = client.empty(n * size_of::<f32>());

    unsafe {
        transpose_kernel::launch_unchecked::<R>(
            &client,
            CubeCount::new_1d(1),
            CubeDim::new_1d(16),
            ArrayArg::from_raw_parts(input_buf, n),
            ArrayArg::from_raw_parts(out_buf.clone(), n),
        );
    }

    let out = client.read_one(out_buf).expect("transpose readback failed");
    f32::from_bytes(&out)[..n].to_vec()
}

/// Element (row, col) of a group's block must come back at (col, row),
/// and two groups sharing a cube must not bleed into each other.
///
/// Lane `sub` holds column `sub` on entry, so `input[sub * 8 + r]` is
/// element `(r, sub)`. After the transpose lane `sub` holds row `sub`,
/// so the same slot is element `(sub, r)`. Group 1's block is offset by
/// 1000 so a cross-group leak cannot coincidentally match.
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
    let out = run_transpose(&input);
    for grp in 0..2u32 {
        for sub in 0..8u32 {
            for r in 0..8u32 {
                let want = (sub * 8 + r) as f32 + grp as f32 * 1000.0;
                let got = out[(grp * 64 + sub * 8 + r) as usize];
                assert_eq!(got, want, "group {grp} lane {sub} slot {r}");
            }
        }
    }
}
