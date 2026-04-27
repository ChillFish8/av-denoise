use cubecl::cube;
use cubecl::prelude::*;

#[cube]
/// Find the blocks most similar to the reference block at (ref_r, ref_c).
///
/// Similarity is the normalised squared Euclidean distance between
/// 2-D DCT representations:
///
/// d(P, Q) = ‖ T_{2D}(P) − T_{2D}(Q) ‖² / k²
///
/// Only blocks whose distance is ≤ tau_match are kept.
/// At most n_max blocks are returned (the n_max−1 most similar plus the
/// reference block, which is always included with d=0).
///
/// The group is then padded to the next power of 2 by repeating the last
/// (least similar) block.  This padding is required because the WHT only
/// accepts power-of-2 lengths.  Padded entries are excluded from the final
/// aggregation step.
///
/// # Parameters
/// - `dcts`: precomputed 2-D DCT for every valid block position,
///   shape `[img_h − k + 1, img_w − k + 1, k, k]`.
/// - `cand_dist`, `cand_r`, `cand_c`: scratch tensors, each with at least
///   `(2·n_s+1)²` elements.
/// - `out_rows`, `out_cols`: output position arrays, each with at least
///   `next_pow2(n_max)` elements.
///
/// Returns the genuine match count `n_actual` (before padding).
pub(crate) fn find_similar_blocks(
    dcts: &Tensor<f32>,
    ref_r: usize,
    ref_c: usize,
    cand_dist: &mut Tensor<f32>,
    cand_r: &mut Tensor<f32>,
    cand_c: &mut Tensor<f32>,
    out_rows: &mut Tensor<f32>,
    out_cols: &mut Tensor<f32>,
    tau_match: f32,
    #[comptime] img_h: usize,
    #[comptime] img_w: usize,
    #[comptime] k: usize,
    #[comptime] n_s: usize,
    #[comptime] n_max: usize,
) -> usize {
    let norm_factor = f32::cast_from(k * k);

    let s0 = dcts.stride(0);
    let s1 = dcts.stride(1);
    let s2 = dcts.stride(2);
    let s3 = dcts.stride(3);

    let ref_base = ref_r * s0 + ref_c * s1;

    // let-mut + if pattern: cubecl can't unify ExpandElementTyped<usize> with
    // a plain usize literal in if-else expressions, so we use assignment instead.
    // `ref_r * 0` forces a comptime limit (img_h - k + 1) into the runtime domain
    // by adding it to an ExpandElementTyped zero; otherwise the variable becomes
    // const and the later conditional assignment fails at kernel tracing time.
    let mut r_min: usize = 0;
    if ref_r >= n_s {
        r_min = ref_r - n_s;
    }
    let r_max_upper = ref_r + n_s + 1;
    let valid_rows = ref_r * 0 + (img_h - k + 1);
    let mut r_max = valid_rows;
    if r_max_upper < valid_rows {
        r_max = r_max_upper;
    }

    let mut c_min: usize = 0;
    if ref_c >= n_s {
        c_min = ref_c - n_s;
    }
    let c_max_upper = ref_c + n_s + 1;
    let valid_cols = ref_c * 0 + (img_w - k + 1);
    let mut c_max = valid_cols;
    if c_max_upper < valid_cols {
        c_max = c_max_upper;
    }

    let mut n_cand: usize = 0;

    for r in r_min..r_max {
        for c in c_min..c_max {
            if r != ref_r || c != ref_c {
                let cand_base = r * s0 + c * s1;
                let mut dist_sq = f32::new(0.0);

                #[unroll]
                for i in 0..k {
                    #[unroll]
                    for j in 0..k {
                        let ref_val = dcts[ref_base + i * s2 + j * s3];
                        let cand_val = dcts[cand_base + i * s2 + j * s3];
                        let diff = ref_val - cand_val;
                        dist_sq += diff * diff;
                    }
                }

                let dist = dist_sq / norm_factor;

                if dist <= tau_match {
                    cand_dist[n_cand] = dist;
                    cand_r[n_cand] = f32::cast_from(r);
                    cand_c[n_cand] = f32::cast_from(c);
                    n_cand += 1;
                }
            }
        }
    }

    for i in 1..n_cand {
        let key_dist = cand_dist[i];
        let key_r = cand_r[i];
        let key_c = cand_c[i];
        let mut j = i;

        while j > 0 && cand_dist[j - 1] > key_dist {
            cand_dist[j] = cand_dist[j - 1];
            cand_r[j] = cand_r[j - 1];
            cand_c[j] = cand_c[j - 1];
            j -= 1;
        }

        cand_dist[j] = key_dist;
        cand_r[j] = key_r;
        cand_c[j] = key_c;
    }

    let mut n_keep = n_cand;
    if n_cand >= n_max - 1 {
        n_keep = n_max - 1;
    }

    out_rows[0] = f32::cast_from(ref_r);
    out_cols[0] = f32::cast_from(ref_c);

    for idx in 0..n_keep {
        out_rows[idx + 1] = cand_r[idx];
        out_cols[idx + 1] = cand_c[idx];
    }

    let n_actual = n_keep + 1;

    let last_r = out_rows[n_actual - 1];
    let last_c = out_cols[n_actual - 1];

    let mut n_padded: usize = 1;
    while n_padded < n_actual {
        n_padded *= 2;
    }

    for idx in n_actual..n_padded {
        out_rows[idx] = last_r;
        out_cols[idx] = last_c;
    }

    n_actual
}

#[cfg(all(test, feature = "cpu"))]
mod tests {
    use std::mem::size_of;

    use cubecl::prelude::*;

    use super::*;
    use crate::kernels::test_util::{
        cpu_client,
        f32_as_bytes,
        read_1d_f32_allocation,
        tensor_arg_1d_f32,
        tensor_arg_f32,
    };

    const K: usize = 2;
    const N_S: usize = 1;
    const N_MAX: usize = 4;
    const SCRATCH_SIZE: usize = (2 * N_S + 1) * (2 * N_S + 1);
    const OUT_SIZE: usize = N_MAX.next_power_of_two();

    #[cube(launch)]
    fn find_similar_blocks_test_kernel(
        dcts: &Tensor<f32>,
        ref_r: usize,
        ref_c: usize,
        cand_dist: &mut Tensor<f32>,
        cand_r: &mut Tensor<f32>,
        cand_c: &mut Tensor<f32>,
        out_rows: &mut Tensor<f32>,
        out_cols: &mut Tensor<f32>,
        out_n_actual: &mut Tensor<f32>,
        tau_match: f32,
        #[comptime] img_h: usize,
        #[comptime] img_w: usize,
        #[comptime] k: usize,
        #[comptime] n_s: usize,
        #[comptime] n_max: usize,
    ) {
        let n_actual = find_similar_blocks(
            dcts, ref_r, ref_c, cand_dist, cand_r, cand_c, out_rows,
            out_cols, tau_match, img_h, img_w, k, n_s, n_max,
        );
        out_n_actual[0] = f32::cast_from(n_actual);
    }

    #[test]
    fn reference_block_is_always_first() {
        let (img_h, img_w) = (4, 4);
        let dcts = make_constant_dcts(img_h, img_w);
        let (out_rows, out_cols, _) = run_kernel(&dcts, 1, 1, img_h, img_w, 100.0);

        assert_eq!(out_rows[0], 1.0, "first row should be ref_r");
        assert_eq!(out_cols[0], 1.0, "first col should be ref_c");
    }

    #[test]
    fn all_candidates_included_when_tau_is_large() {
        let (img_h, img_w) = (4, 4);
        let dcts = make_constant_dcts(img_h, img_w);
        let (_, _, n_actual) = run_kernel(&dcts, 1, 1, img_h, img_w, 100.0);

        assert_eq!(n_actual, N_MAX);
    }

    #[test]
    fn no_candidates_when_tau_is_zero() {
        let (img_h, img_w) = (4, 4);
        let dcts = make_distinct_dcts(img_h, img_w);
        let (_, _, n_actual) = run_kernel(&dcts, 1, 1, img_h, img_w, 0.0);

        assert_eq!(n_actual, 1);
    }

    #[test]
    fn matches_host_reference() {
        let (img_h, img_w) = (5, 5);
        let dcts_flat = make_distinct_dcts(img_h, img_w);
        let tau_match = 5.0f32;

        let (out_rows, out_cols, n_actual) =
            run_kernel(&dcts_flat, 2, 2, img_h, img_w, tau_match);

        let (exp_rows, exp_cols, exp_n_actual) =
            host_find_similar_blocks(&dcts_flat, 2, 2, img_h, img_w, tau_match);

        assert_eq!(n_actual, exp_n_actual);

        let out_pairs: Vec<(usize, usize)> = (0..n_actual)
            .map(|i| (out_rows[i] as usize, out_cols[i] as usize))
            .collect();
        let exp_pairs: Vec<(usize, usize)> = (0..exp_n_actual)
            .map(|i| (exp_rows[i], exp_cols[i]))
            .collect();

        assert_eq!(out_pairs, exp_pairs);
    }

    #[test]
    fn corner_reference_respects_search_window_bounds() {
        let (img_h, img_w) = (4, 4);
        let dcts = make_constant_dcts(img_h, img_w);
        let (out_rows, out_cols, n_actual) = run_kernel(&dcts, 0, 0, img_h, img_w, 100.0);

        assert_eq!(n_actual, N_MAX);

        let out_pairs: Vec<(usize, usize)> = (0..n_actual)
            .map(|i| (out_rows[i] as usize, out_cols[i] as usize))
            .collect();

        assert_eq!(out_pairs, vec![(0, 0), (0, 1), (1, 0), (1, 1)]);
    }

    #[test]
    fn padded_tail_repeats_last_genuine_match() {
        let (img_h, img_w) = (4, 4);
        let dcts = make_ranked_dcts(img_h, img_w);
        let (out_rows, out_cols, n_actual) = run_kernel(&dcts, 0, 0, img_h, img_w, 5.0);

        assert_eq!(n_actual, 3);
        assert_eq!(out_rows.len(), OUT_SIZE);
        assert_eq!(out_cols.len(), OUT_SIZE);

        let actual_pairs: Vec<(usize, usize)> = (0..n_actual)
            .map(|i| (out_rows[i] as usize, out_cols[i] as usize))
            .collect();

        assert_eq!(actual_pairs, vec![(0, 0), (1, 0), (0, 1)]);
        assert_eq!(out_rows[3], out_rows[2]);
        assert_eq!(out_cols[3], out_cols[2]);
    }

    #[test]
    fn keeps_only_closest_matches_when_candidates_exceed_limit() {
        let (img_h, img_w) = (5, 5);
        let dcts = make_ranked_dcts(img_h, img_w);
        let tau_match = 100.0;

        let (out_rows, out_cols, n_actual) = run_kernel(&dcts, 1, 1, img_h, img_w, tau_match);

        assert_eq!(n_actual, N_MAX);

        let out_pairs: Vec<(usize, usize)> = (0..n_actual)
            .map(|i| (out_rows[i] as usize, out_cols[i] as usize))
            .collect();

        assert_eq!(out_pairs, vec![(1, 1), (1, 2), (2, 1), (0, 1)]);
    }

    fn run_kernel(
        dcts_flat: &[f32],
        ref_r: usize,
        ref_c: usize,
        img_h: usize,
        img_w: usize,
        tau_match: f32,
    ) -> (Vec<f32>, Vec<f32>, usize) {
        let client = cpu_client();
        let valid_h = img_h - K + 1;
        let valid_w = img_w - K + 1;
        let dcts_shape = [valid_h, valid_w, K, K];

        let dcts_alloc = client.create_tensor(
            cubecl::bytes::Bytes::from_bytes_vec(f32_as_bytes(dcts_flat)),
            &dcts_shape,
            size_of::<f32>(),
        );
        let cand_dist_alloc = client.empty_tensor(&[SCRATCH_SIZE], size_of::<f32>());
        let cand_r_alloc = client.empty_tensor(&[SCRATCH_SIZE], size_of::<f32>());
        let cand_c_alloc = client.empty_tensor(&[SCRATCH_SIZE], size_of::<f32>());
        let out_rows_alloc = client.empty_tensor(&[OUT_SIZE], size_of::<f32>());
        let out_cols_alloc = client.empty_tensor(&[OUT_SIZE], size_of::<f32>());
        let out_n_actual_alloc = client.empty_tensor(&[1], size_of::<f32>());

        find_similar_blocks_test_kernel::launch(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            tensor_arg_f32(&dcts_alloc, &dcts_shape),
            ScalarArg::new(ref_r),
            ScalarArg::new(ref_c),
            tensor_arg_1d_f32(&cand_dist_alloc, &[SCRATCH_SIZE]),
            tensor_arg_1d_f32(&cand_r_alloc, &[SCRATCH_SIZE]),
            tensor_arg_1d_f32(&cand_c_alloc, &[SCRATCH_SIZE]),
            tensor_arg_1d_f32(&out_rows_alloc, &[OUT_SIZE]),
            tensor_arg_1d_f32(&out_cols_alloc, &[OUT_SIZE]),
            tensor_arg_1d_f32(&out_n_actual_alloc, &[1]),
            ScalarArg::new(tau_match),
            img_h,
            img_w,
            K,
            N_S,
            N_MAX,
        )
        .expect("find_similar_blocks kernel should launch");

        let out_rows = read_1d_f32_allocation(&client, &out_rows_alloc, OUT_SIZE);
        let out_cols = read_1d_f32_allocation(&client, &out_cols_alloc, OUT_SIZE);
        let n_actual_f = read_1d_f32_allocation(&client, &out_n_actual_alloc, 1);
        let n_actual = n_actual_f[0] as usize;

        (out_rows, out_cols, n_actual)
    }

    fn make_constant_dcts(img_h: usize, img_w: usize) -> Vec<f32> {
        let valid_h = img_h - K + 1;
        let valid_w = img_w - K + 1;
        let n = valid_h * valid_w * K * K;
        let mut dcts = vec![0.0f32; n];
        for r in 0..valid_h {
            for c in 0..valid_w {
                dcts[(r * valid_w + c) * K * K] = 1.0;
            }
        }
        dcts
    }

    fn make_distinct_dcts(img_h: usize, img_w: usize) -> Vec<f32> {
        let valid_h = img_h - K + 1;
        let valid_w = img_w - K + 1;
        let n = valid_h * valid_w * K * K;
        let mut dcts = vec![0.0f32; n];
        for r in 0..valid_h {
            for c in 0..valid_w {
                dcts[(r * valid_w + c) * K * K] = (r * valid_w + c) as f32;
            }
        }
        dcts
    }

    fn make_ranked_dcts(img_h: usize, img_w: usize) -> Vec<f32> {
        let valid_h = img_h - K + 1;
        let valid_w = img_w - K + 1;
        let n = valid_h * valid_w * K * K;
        let mut dcts = vec![0.0f32; n];

        for r in 0..valid_h {
            for c in 0..valid_w {
                let base = (r * valid_w + c) * K * K;
                let value = match (r, c) {
                    (1, 1) => 0.0,
                    (1, 2) => 1.0,
                    (2, 1) => 2.0,
                    (0, 1) => 3.0,
                    (1, 0) => 4.0,
                    (2, 2) => 5.0,
                    (0, 0) => 6.0,
                    (0, 2) => 7.0,
                    (2, 0) => 8.0,
                    _ => 50.0 + (r * valid_w + c) as f32,
                };
                dcts[base] = value;
            }
        }

        dcts
    }

    fn host_find_similar_blocks(
        dcts: &[f32],
        ref_r: usize,
        ref_c: usize,
        img_h: usize,
        img_w: usize,
        tau_match: f32,
    ) -> (Vec<usize>, Vec<usize>, usize) {
        let valid_h = img_h - K + 1;
        let valid_w = img_w - K + 1;
        let norm_factor = (K * K) as f32;

        let ref_base = (ref_r * valid_w + ref_c) * K * K;

        let r_min = ref_r.saturating_sub(N_S);
        let r_max = (ref_r + N_S + 1).min(valid_h);
        let c_min = ref_c.saturating_sub(N_S);
        let c_max = (ref_c + N_S + 1).min(valid_w);

        let mut candidates: Vec<(f32, usize, usize)> = Vec::new();

        for r in r_min..r_max {
            for c in c_min..c_max {
                if r == ref_r && c == ref_c {
                    continue;
                }
                let cand_base = (r * valid_w + c) * K * K;
                let dist_sq: f32 = (0..K * K)
                    .map(|idx| {
                        let diff = dcts[ref_base + idx] - dcts[cand_base + idx];
                        diff * diff
                    })
                    .sum();
                let dist = dist_sq / norm_factor;
                if dist <= tau_match {
                    candidates.push((dist, r, c));
                }
            }
        }

        candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        candidates.truncate(N_MAX - 1);

        let mut rows = vec![ref_r];
        let mut cols = vec![ref_c];
        for (_, r, c) in &candidates {
            rows.push(*r);
            cols.push(*c);
        }
        let n_actual = rows.len();

        let mut n_padded = 1;
        while n_padded < n_actual {
            n_padded *= 2;
        }
        let last_r = *rows.last().unwrap();
        let last_c = *cols.last().unwrap();
        while rows.len() < n_padded {
            rows.push(last_r);
            cols.push(last_c);
        }

        (rows, cols, n_actual)
    }
}
