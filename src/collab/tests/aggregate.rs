use cubecl::prelude::*;

use super::helpers::{R, make_client};
use crate::collab::geometry::{filtered_buf_len, ref_count, ref_pos, refs_along};
use crate::collab::kernels::aggregate::collab_aggregate;
use crate::nlmeans::{BLOCK_X, BLOCK_Y};

/// Launches [`collab_aggregate`] over a plain luma buffer and reads back
/// the output plane.
fn run_aggregate(filtered_host: &[f32], weight_host: &[f32], width: u32, height: u32) -> Vec<f32> {
    let refs = ref_count(width, height);
    let filt_len = filtered_buf_len(width, height);
    assert_eq!(filtered_host.len(), filt_len);
    assert_eq!(weight_host.len(), refs);

    let client = make_client();
    let filtered = client.create_from_slice(f32::as_bytes(filtered_host));
    let group_weight = client.create_from_slice(f32::as_bytes(weight_host));
    let out_len = (width * height) as usize;
    let output = client.empty(out_len * size_of::<f32>());

    let refs_x = refs_along(width);
    let refs_y = refs_along(height);
    let grid = CubeCount::new_2d(width.div_ceil(BLOCK_X), height.div_ceil(BLOCK_Y));
    let dim = CubeDim::new_2d(BLOCK_X, BLOCK_Y);

    unsafe {
        collab_aggregate::launch_unchecked::<R>(
            &client,
            grid,
            dim,
            1usize,
            ArrayArg::from_raw_parts(filtered, filt_len),
            ArrayArg::from_raw_parts(group_weight, refs),
            ArrayArg::from_raw_parts(output.clone(), out_len),
            width,
            height,
            refs_x,
            refs_y,
        );
    }

    let bytes = client.read_one(output).expect("aggregate output readback failed");
    f32::from_bytes(&bytes)[..out_len].to_vec()
}

/// A deterministic, non-constant fill for `filtered`, keyed on the flat
/// `(ref, pos)` index so a wrong reference or a wrong in-patch position
/// both produce a value that doesn't match any other slot.
fn hashed_filtered(refs: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; refs * 64];
    for (idx, v) in out.iter_mut().enumerate() {
        let mut hash = (idx as u32).wrapping_mul(2654435761).wrapping_add(0x9E3779B9);
        hash ^= hash >> 15;
        hash = hash.wrapping_mul(0x85EBCA6B);
        hash ^= hash >> 13;
        *v = (hash as f32 / u32::MAX as f32) * 2.0 - 1.0;
    }
    out
}

/// A varied, monotonic-in-`ref` weight, so two different references
/// never carry the same weight by accident.
fn varied_group_weight(refs: usize) -> Vec<f32> {
    (0..refs).map(|r| 1.0 + r as f32 * 0.1).collect()
}

/// The direct, obviously-correct mirror of [`collab_aggregate`]. Every
/// pixel loops over every reference unconditionally and tests real
/// coverage against `ref_pos`, with no shortcut for the contiguous run
/// or the clamped edge the kernel takes.
fn cpu_aggregate(filtered: &[f32], group_weight: &[f32], width: u32, height: u32) -> Vec<f32> {
    let refs_x = refs_along(width);
    let refs_y = refs_along(height);
    let mut output = vec![0.0f32; (width * height) as usize];

    for y in 0..height {
        for x in 0..width {
            let mut acc = 0.0f64;
            let mut wsum = 0.0f64;

            for iy in 0..refs_y {
                let ry = ref_pos(iy, height);
                if !(ry <= y && y < ry + 8) {
                    continue;
                }
                for ix in 0..refs_x {
                    let rx = ref_pos(ix, width);
                    if !(rx <= x && x < rx + 8) {
                        continue;
                    }

                    let ref_idx = (iy * refs_x + ix) as usize;
                    let w = group_weight[ref_idx] as f64;
                    let patch_idx = ref_idx * 64 + ((y - ry) * 8 + (x - rx)) as usize;
                    acc += w * filtered[patch_idx] as f64;
                    wsum += w;
                }
            }

            output[(y * width + x) as usize] = (acc / wsum) as f32;
        }
    }

    output
}

#[test]
fn uniform_patches_pass_through() {
    let (w, h) = (21u32, 16u32);
    let refs = ref_count(w, h);
    let filtered = vec![0.7f32; filtered_buf_len(w, h)];
    let weights = vec![1.0f32; refs];

    let output = run_aggregate(&filtered, &weights, w, h);

    for (idx, &v) in output.iter().enumerate() {
        assert!((v - 0.7).abs() < 1e-5, "idx={idx}: got {v}");
    }
}

/// 21x16 isn't a multiple of `STEP`, so the last reference on each axis
/// genuinely clamps inward rather than landing on a regular grid point,
/// which is exactly the case the "extra" candidate check exists for.
/// Both buffers get varied, non-constant fills, so a wrong weight or a
/// double-counted reference would change the result rather than
/// disappearing into a fixed point.
#[test]
fn gather_matches_a_cpu_mirror() {
    let (w, h) = (21u32, 16u32);
    let refs = ref_count(w, h);
    let filtered = hashed_filtered(refs);
    let weights = varied_group_weight(refs);

    let expected = cpu_aggregate(&filtered, &weights, w, h);
    let got = run_aggregate(&filtered, &weights, w, h);

    for (idx, (&want, &have)) in expected.iter().zip(got.iter()).enumerate() {
        assert!((want - have).abs() < 1e-5, "idx={idx}: want {want} got {have}");
    }
}
