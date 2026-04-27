use cubecl::cube;
use cubecl::prelude::*;

#[cube]
fn ref_grid_len(size: usize, #[comptime] k: usize, #[comptime] n_step: usize) -> usize {
    let mut count = size * 0;
    if size <= k {
        count = 1;
    } else {
        let last = size - k;
        count = last / n_step + 1;
        if last % n_step != 0 {
            count += 1;
        }
    }

    count
}

#[cube]
fn ref_pos_from_index(
    index: usize,
    size: usize,
    #[comptime] k: usize,
    #[comptime] n_step: usize,
) -> usize {
    let mut position = index * 0 + size * 0;
    if size <= k {
        position = 0;
    } else {
        let last = size - k;
        let pos = index * n_step;
        if pos < last {
            position = pos;
        } else {
            position = last;
        }
    }

    position
}

#[cube]
fn image_index<T: CubeType>(
    tensor: &Tensor<T>,
    batch: usize,
    row: usize,
    col: usize,
) -> usize {
    batch * tensor.stride(0) + row * tensor.stride(1) + col * tensor.stride(2)
}

#[cube]
fn group_scratch_base<T: CubeType>(
    tensor: &Tensor<T>,
    batch: usize,
    ref_row: usize,
    ref_col: usize,
) -> usize {
    batch * tensor.stride(0) + ref_row * tensor.stride(1) + ref_col * tensor.stride(2)
}

#[cube]
fn stack_index(
    stack: &Tensor<f32>,
    base: usize,
    row: usize,
    col: usize,
    block: usize,
) -> usize {
    base + row * stack.stride(3) + col * stack.stride(4) + block * stack.stride(5)
}

#[cube]
fn dct_index(
    dcts: &Tensor<f32>,
    batch: usize,
    row: usize,
    col: usize,
    i: usize,
    j: usize,
) -> usize {
    batch * dcts.stride(0)
        + row * dcts.stride(1)
        + col * dcts.stride(2)
        + i * dcts.stride(3)
        + j * dcts.stride(4)
}

#[cube]
fn build_stack(
    frame: &Tensor<f32>,
    kaiser: &Tensor<f32>,
    positions_rows: &Tensor<f32>,
    positions_cols: &Tensor<f32>,
    positions_base: usize,
    stack: &mut Tensor<f32>,
    stack_base: usize,
    batch: usize,
    n_blocks: usize,
    #[comptime] k: usize,
) {
    let kaiser_row_stride = kaiser.stride(0);
    let kaiser_col_stride = kaiser.stride(1);

    for block in 0..n_blocks {
        let pr = positions_rows[positions_base + block] as usize;
        let pc = positions_cols[positions_base + block] as usize;

        #[unroll]
        for i in 0..k {
            #[unroll]
            for j in 0..k {
                let pixel = frame[image_index(frame, batch, pr + i, pc + j)];
                let weight = kaiser[i * kaiser_row_stride + j * kaiser_col_stride];
                stack[stack_index(stack, stack_base, i, j, block)] = pixel * weight;
            }
        }
    }
}

#[cube]
fn build_two_stacks(
    noisy: &Tensor<f32>,
    basic: &Tensor<f32>,
    kaiser: &Tensor<f32>,
    positions_rows: &Tensor<f32>,
    positions_cols: &Tensor<f32>,
    positions_base: usize,
    noisy_stack: &mut Tensor<f32>,
    noisy_stack_base: usize,
    basic_stack: &mut Tensor<f32>,
    basic_stack_base: usize,
    batch: usize,
    n_blocks: usize,
    #[comptime] k: usize,
) {
    let kaiser_row_stride = kaiser.stride(0);
    let kaiser_col_stride = kaiser.stride(1);

    for block in 0..n_blocks {
        let pr = positions_rows[positions_base + block] as usize;
        let pc = positions_cols[positions_base + block] as usize;

        #[unroll]
        for i in 0..k {
            #[unroll]
            for j in 0..k {
                let weight = kaiser[i * kaiser_row_stride + j * kaiser_col_stride];
                noisy_stack[stack_index(noisy_stack, noisy_stack_base, i, j, block)] =
                    noisy[image_index(noisy, batch, pr + i, pc + j)] * weight;
                basic_stack[stack_index(basic_stack, basic_stack_base, i, j, block)] =
                    basic[image_index(basic, batch, pr + i, pc + j)] * weight;
            }
        }
    }
}

#[cube]
fn copy_dcts_group(
    dcts: &Tensor<f32>,
    group_dcts: &mut Tensor<f32>,
    base: usize,
    batch: usize,
    valid_h: usize,
    valid_w: usize,
    #[comptime] k: usize,
) {
    for r in 0..valid_h {
        for c in 0..valid_w {
            #[unroll]
            for i in 0..k {
                #[unroll]
                for j in 0..k {
                    group_dcts[base
                        + r * group_dcts.stride(3)
                        + c * group_dcts.stride(4)
                        + i * group_dcts.stride(5)
                        + j * group_dcts.stride(6)] =
                        dcts[dct_index(dcts, batch, r, c, i, j)];
                }
            }
        }
    }
}

#[cube]
fn hard_threshold_stack(
    stack: &mut Tensor<f32>,
    base: usize,
    threshold: f32,
    n_blocks: usize,
    #[comptime] k: usize,
) -> usize {
    let mut n_nonzero: usize = 0;

    #[unroll]
    for i in 0..k {
        #[unroll]
        for j in 0..k {
            for block in 0..n_blocks {
                let index = stack_index(stack, base, i, j, block);
                let value = stack[index];
                if value.abs() > threshold {
                    n_nonzero += 1;
                } else {
                    stack[index] = 0.0;
                }
            }
        }
    }

    n_nonzero
}

#[cube]
fn find_similar_blocks_offset(
    dcts: &Tensor<f32>,
    dcts_base: usize,
    ref_r: usize,
    ref_c: usize,
    cand_dist: &mut Tensor<f32>,
    cand_r: &mut Tensor<f32>,
    cand_c: &mut Tensor<f32>,
    cand_base: usize,
    out_rows: &mut Tensor<f32>,
    out_cols: &mut Tensor<f32>,
    out_base: usize,
    tau_match: f32,
    img_h: usize,
    img_w: usize,
    #[comptime] k: usize,
    #[comptime] n_s: usize,
    #[comptime] n_max: usize,
) -> usize {
    let norm_factor = f32::cast_from(k * k);

    let s0 = dcts.stride(3);
    let s1 = dcts.stride(4);
    let s2 = dcts.stride(5);
    let s3 = dcts.stride(6);

    let ref_base = dcts_base + ref_r * s0 + ref_c * s1;

    let mut r_min: usize = 0;
    if ref_r >= n_s {
        r_min = ref_r - n_s;
    }
    let r_max_upper = ref_r + n_s + 1;
    let valid_rows = img_h - k + 1;
    let mut r_max = valid_rows;
    if r_max_upper < valid_rows {
        r_max = r_max_upper;
    }

    let mut c_min: usize = 0;
    if ref_c >= n_s {
        c_min = ref_c - n_s;
    }
    let c_max_upper = ref_c + n_s + 1;
    let valid_cols = img_w - k + 1;
    let mut c_max = valid_cols;
    if c_max_upper < valid_cols {
        c_max = c_max_upper;
    }

    let mut n_cand: usize = 0;

    for r in r_min..r_max {
        for c in c_min..c_max {
            if r != ref_r || c != ref_c {
                let cand_block_base = dcts_base + r * s0 + c * s1;
                let mut dist_sq = 0.0f32;

                #[unroll]
                for i in 0..k {
                    #[unroll]
                    for j in 0..k {
                        let ref_val = dcts[ref_base + i * s2 + j * s3];
                        let cand_val = dcts[cand_block_base + i * s2 + j * s3];
                        let diff = ref_val - cand_val;
                        dist_sq += diff * diff;
                    }
                }

                let dist = dist_sq / norm_factor;
                if dist <= tau_match {
                    cand_dist[cand_base + n_cand] = dist;
                    cand_r[cand_base + n_cand] = f32::cast_from(r);
                    cand_c[cand_base + n_cand] = f32::cast_from(c);
                    n_cand += 1;
                }
            }
        }
    }

    for i in 1..n_cand {
        let key_dist = cand_dist[cand_base + i];
        let key_r = cand_r[cand_base + i];
        let key_c = cand_c[cand_base + i];
        let mut j = i;

        while j > 0 && cand_dist[cand_base + j - 1] > key_dist {
            cand_dist[cand_base + j] = cand_dist[cand_base + j - 1];
            cand_r[cand_base + j] = cand_r[cand_base + j - 1];
            cand_c[cand_base + j] = cand_c[cand_base + j - 1];
            j -= 1;
        }

        cand_dist[cand_base + j] = key_dist;
        cand_r[cand_base + j] = key_r;
        cand_c[cand_base + j] = key_c;
    }

    let mut n_keep = n_cand;
    if n_cand >= n_max - 1 {
        n_keep = n_max - 1;
    }

    out_rows[out_base] = f32::cast_from(ref_r);
    out_cols[out_base] = f32::cast_from(ref_c);

    for idx in 0..n_keep {
        out_rows[out_base + idx + 1] = cand_r[cand_base + idx];
        out_cols[out_base + idx + 1] = cand_c[cand_base + idx];
    }

    let n_actual = n_keep + 1;
    let last_r = out_rows[out_base + n_actual - 1];
    let last_c = out_cols[out_base + n_actual - 1];
    let mut n_padded: usize = 1;
    while n_padded < n_actual {
        n_padded *= 2;
    }

    for idx in n_actual..n_padded {
        out_rows[out_base + idx] = last_r;
        out_cols[out_base + idx] = last_c;
    }

    n_actual
}

#[cube]
fn transform_3d_offset(
    stack: &mut Tensor<f32>,
    stack_base: usize,
    scratch: &mut Tensor<f32>,
    scratch_base: usize,
    n_blocks: usize,
    #[comptime] k: usize,
) {
    let s0 = stack.stride(3);
    let s1 = stack.stride(4);
    let s2 = stack.stride(5);

    for b in 0..n_blocks {
        #[unroll]
        for row in 0..k {
            dct1d_offset(
                stack,
                stack_base + row * s0 + b * s2,
                s1,
                scratch,
                scratch_base,
                k,
            );
        }

        #[unroll]
        for col in 0..k {
            dct1d_offset(
                stack,
                stack_base + col * s1 + b * s2,
                s0,
                scratch,
                scratch_base,
                k,
            );
        }
    }

    #[unroll]
    for i in 0..k {
        #[unroll]
        for j in 0..k {
            fwht_strided_offset(stack, stack_base + i * s0 + j * s1, s2, n_blocks);
        }
    }
}

#[cube]
fn inverse_transform_3d_offset(
    stack: &mut Tensor<f32>,
    stack_base: usize,
    scratch: &mut Tensor<f32>,
    scratch_base: usize,
    n_blocks: usize,
    #[comptime] k: usize,
) {
    let s0 = stack.stride(3);
    let s1 = stack.stride(4);
    let s2 = stack.stride(5);

    #[unroll]
    for i in 0..k {
        #[unroll]
        for j in 0..k {
            fwht_strided_offset(stack, stack_base + i * s0 + j * s1, s2, n_blocks);
        }
    }

    for b in 0..n_blocks {
        #[unroll]
        for col in 0..k {
            idct1d_offset(
                stack,
                stack_base + col * s1 + b * s2,
                s0,
                scratch,
                scratch_base,
                k,
            );
        }

        #[unroll]
        for row in 0..k {
            idct1d_offset(
                stack,
                stack_base + row * s0 + b * s2,
                s1,
                scratch,
                scratch_base,
                k,
            );
        }
    }
}

#[cube]
fn dct1d_offset(
    x: &mut Tensor<f32>,
    offset: usize,
    stride: usize,
    scratch: &mut Tensor<f32>,
    scratch_base: usize,
    #[comptime] n: usize,
) {
    let pi = f32::new(core::f32::consts::PI);
    let scale0 = f32::new(1.0) / f32::cast_from(n).sqrt();
    let scale_k = (f32::new(2.0) / f32::cast_from(n)).sqrt();

    #[unroll]
    for k_idx in 0..n {
        let mut sum = 0.0f32;

        #[unroll]
        for m in 0..n {
            let angle = pi * f32::cast_from(k_idx) * f32::cast_from(2 * m + 1)
                / (f32::new(2.0) * f32::cast_from(n));
            sum += x[offset + m * stride] * angle.cos();
        }

        let scale = if k_idx == 0 { scale0 } else { scale_k };
        scratch[scratch_base + k_idx] = scale * sum;
    }

    #[unroll]
    for k_idx in 0..n {
        x[offset + k_idx * stride] = scratch[scratch_base + k_idx];
    }
}

#[cube]
fn idct1d_offset(
    x: &mut Tensor<f32>,
    offset: usize,
    stride: usize,
    scratch: &mut Tensor<f32>,
    scratch_base: usize,
    #[comptime] n: usize,
) {
    let pi = f32::new(core::f32::consts::PI);
    let scale0 = f32::new(1.0) / f32::cast_from(n).sqrt();
    let scale_k = (f32::new(2.0) / f32::cast_from(n)).sqrt();

    #[unroll]
    for m in 0..n {
        let mut sum = scale0 * x[offset];

        #[unroll]
        for k_idx in 1..n {
            let angle = pi * f32::cast_from(k_idx) * f32::cast_from(2 * m + 1)
                / (f32::new(2.0) * f32::cast_from(n));
            sum += scale_k * x[offset + k_idx * stride] * angle.cos();
        }

        scratch[scratch_base + m] = sum;
    }

    #[unroll]
    for m in 0..n {
        x[offset + m * stride] = scratch[scratch_base + m];
    }
}

#[cube]
fn fwht_strided_offset(x: &mut Tensor<f32>, base: usize, stride: usize, n: usize) {
    if n == 0 || (n & (n - 1)) != 0 {
        terminate!();
    }

    let mut h = 1usize;
    while h < n {
        for i in range_stepped(0, n, 2 * h) {
            for j in 0..h {
                let left = base + (i + j) * stride;
                let right = base + (i + j + h) * stride;
                let a = x[left];
                let b = x[right];
                x[left] = a + b;
                x[right] = a - b;
            }
        }
        h *= 2;
    }

    let scale = f32::cast_from(n).sqrt();
    for idx in 0..n {
        x[base + idx * stride] /= scale;
    }
}

#[cube]
fn wiener_filter_stack(
    noisy_stack: &mut Tensor<f32>,
    noisy_base: usize,
    basic_stack: &Tensor<f32>,
    basic_base: usize,
    sigma: f32,
    n_blocks: usize,
    #[comptime] k: usize,
) -> f32 {
    let sigma_sq = sigma * sigma;
    let mut coeff_norm_sq = 0.0;

    #[unroll]
    for i in 0..k {
        #[unroll]
        for j in 0..k {
            for block in 0..n_blocks {
                let noisy_index = stack_index(noisy_stack, noisy_base, i, j, block);
                let basic_index = stack_index(basic_stack, basic_base, i, j, block);
                let basic_value = basic_stack[basic_index];
                let basic_sq = basic_value * basic_value;
                let coeff = basic_sq / (basic_sq + sigma_sq);
                noisy_stack[noisy_index] *= coeff;
                coeff_norm_sq += coeff * coeff;
            }
        }
    }

    1.0 / max(coeff_norm_sq, 1.0e-8)
}

#[cube]
fn aggregate_stack(
    stack: &Tensor<f32>,
    stack_base: usize,
    kaiser: &Tensor<f32>,
    positions_rows: &Tensor<f32>,
    positions_cols: &Tensor<f32>,
    positions_base: usize,
    numerator: &mut Tensor<Atomic<f32>>,
    denominator: &mut Tensor<Atomic<f32>>,
    batch: usize,
    n_actual: usize,
    w_group: f32,
    #[comptime] k: usize,
) {
    let kaiser_row_stride = kaiser.stride(0);
    let kaiser_col_stride = kaiser.stride(1);

    for block in 0..n_actual {
        let pr = positions_rows[positions_base + block] as usize;
        let pc = positions_cols[positions_base + block] as usize;

        #[unroll]
        for i in 0..k {
            #[unroll]
            for j in 0..k {
                let kaiser_value = kaiser[i * kaiser_row_stride + j * kaiser_col_stride];
                let filtered = stack[stack_index(stack, stack_base, i, j, block)];
                let out_index = image_index(numerator, batch, pr + i, pc + j);
                numerator[out_index].fetch_add(w_group * kaiser_value * filtered);
                denominator[out_index].fetch_add(w_group * kaiser_value * kaiser_value);
            }
        }
    }
}

#[cube(launch)]
/// Produces the basic estimate denoising via hard thresholding.
///
/// The accumulation buffers must be zero-initialized before launch.
/// Only `UNIT_POS == 0` performs work; extra units in the cube are ignored.
pub fn bm3d_stage1(
    frame: &Tensor<f32>,
    dcts: &Tensor<f32>,
    kaiser: &Tensor<f32>,
    cand_dist: &mut Tensor<f32>,
    cand_rows: &mut Tensor<f32>,
    cand_cols: &mut Tensor<f32>,
    positions_rows: &mut Tensor<f32>,
    positions_cols: &mut Tensor<f32>,
    group_dcts: &mut Tensor<f32>,
    stack: &mut Tensor<f32>,
    transform_scratch: &mut Tensor<f32>,
    numerator: &mut Tensor<Atomic<f32>>,
    denominator: &mut Tensor<Atomic<f32>>,
    sigma: f32,
    lambda_3d: f32,
    #[comptime] k: usize,
    #[comptime] n_s: usize,
    #[comptime] n_step: usize,
    #[comptime] n_max: usize,
    #[comptime] tau_match: u32,
) {
    if UNIT_POS != 0 {
        terminate!();
    }

    let batch = usize::cast_from(CUBE_POS_Z);
    let ref_row_idx = usize::cast_from(CUBE_POS_Y);
    let ref_col_idx = usize::cast_from(CUBE_POS_X);

    let height = frame.shape(1);
    let width = frame.shape(2);
    let ref_rows = ref_grid_len(height, k, n_step);
    let ref_cols = ref_grid_len(width, k, n_step);

    if batch >= frame.shape(0) || ref_row_idx >= ref_rows || ref_col_idx >= ref_cols {
        terminate!();
    }

    let valid_h = height - k + 1;
    let valid_w = width - k + 1;
    let ref_r = ref_pos_from_index(ref_row_idx, height, k, n_step);
    let ref_c = ref_pos_from_index(ref_col_idx, width, k, n_step);
    let tau_match_f = f32::cast_from(tau_match);

    let cand_base = group_scratch_base(cand_dist, batch, ref_row_idx, ref_col_idx);
    let positions_base =
        group_scratch_base(positions_rows, batch, ref_row_idx, ref_col_idx);
    let dcts_base = group_scratch_base(group_dcts, batch, ref_row_idx, ref_col_idx);
    let stack_base = group_scratch_base(stack, batch, ref_row_idx, ref_col_idx);
    let scratch_base =
        group_scratch_base(transform_scratch, batch, ref_row_idx, ref_col_idx);

    copy_dcts_group(dcts, group_dcts, dcts_base, batch, valid_h, valid_w, k);

    let n_actual = find_similar_blocks_offset(
        group_dcts,
        dcts_base,
        ref_r,
        ref_c,
        cand_dist,
        cand_rows,
        cand_cols,
        cand_base,
        positions_rows,
        positions_cols,
        positions_base,
        tau_match_f,
        height,
        width,
        k,
        n_s,
        n_max,
    );

    let mut n_blocks = 1usize;
    while n_blocks < n_actual {
        n_blocks *= 2;
    }

    build_stack(
        frame,
        kaiser,
        positions_rows,
        positions_cols,
        positions_base,
        stack,
        stack_base,
        batch,
        n_blocks,
        k,
    );

    transform_3d_offset(
        stack,
        stack_base,
        transform_scratch,
        scratch_base,
        n_blocks,
        k,
    );

    let threshold = lambda_3d * sigma;
    let n_nonzero = hard_threshold_stack(stack, stack_base, threshold, n_blocks, k);

    inverse_transform_3d_offset(
        stack,
        stack_base,
        transform_scratch,
        scratch_base,
        n_blocks,
        k,
    );

    let mut w_group = 0.0f32;
    if n_nonzero == 0 {
        w_group = 1.0;
    } else {
        w_group = 1.0 / f32::cast_from(n_nonzero);
    }

    aggregate_stack(
        stack,
        stack_base,
        kaiser,
        positions_rows,
        positions_cols,
        positions_base,
        numerator,
        denominator,
        batch,
        n_actual,
        w_group,
        k,
    );
}

#[cube(launch)]
/// BM3D Stage 2: produce the final estimate via Wiener filtering.
///
/// The accumulation buffers must be zero-initialized before launch.
/// Only `UNIT_POS == 0` performs work; extra units in the cube are ignored.
pub fn bm3d_stage2(
    noisy: &Tensor<f32>,
    basic: &Tensor<f32>,
    basic_dcts: &Tensor<f32>,
    kaiser: &Tensor<f32>,
    cand_dist: &mut Tensor<f32>,
    cand_rows: &mut Tensor<f32>,
    cand_cols: &mut Tensor<f32>,
    positions_rows: &mut Tensor<f32>,
    positions_cols: &mut Tensor<f32>,
    group_dcts: &mut Tensor<f32>,
    noisy_stack: &mut Tensor<f32>,
    basic_stack: &mut Tensor<f32>,
    transform_scratch: &mut Tensor<f32>,
    numerator: &mut Tensor<Atomic<f32>>,
    denominator: &mut Tensor<Atomic<f32>>,
    sigma: f32,
    #[comptime] k: usize,
    #[comptime] n_s: usize,
    #[comptime] n_step: usize,
    #[comptime] n_max: usize,
    #[comptime] tau_match: u32,
) {
    if UNIT_POS != 0 {
        terminate!();
    }

    let batch = usize::cast_from(CUBE_POS_Z);
    let ref_row_idx = usize::cast_from(CUBE_POS_Y);
    let ref_col_idx = usize::cast_from(CUBE_POS_X);

    let height = noisy.shape(1);
    let width = noisy.shape(2);
    let ref_rows = ref_grid_len(height, k, n_step);
    let ref_cols = ref_grid_len(width, k, n_step);

    if batch >= noisy.shape(0) || ref_row_idx >= ref_rows || ref_col_idx >= ref_cols {
        terminate!();
    }

    let valid_h = height - k + 1;
    let valid_w = width - k + 1;
    let ref_r = ref_pos_from_index(ref_row_idx, height, k, n_step);
    let ref_c = ref_pos_from_index(ref_col_idx, width, k, n_step);
    let tau_match_f = f32::cast_from(tau_match);

    let cand_base = group_scratch_base(cand_dist, batch, ref_row_idx, ref_col_idx);
    let positions_base =
        group_scratch_base(positions_rows, batch, ref_row_idx, ref_col_idx);
    let dcts_base = group_scratch_base(group_dcts, batch, ref_row_idx, ref_col_idx);
    let noisy_stack_base =
        group_scratch_base(noisy_stack, batch, ref_row_idx, ref_col_idx);
    let basic_stack_base =
        group_scratch_base(basic_stack, batch, ref_row_idx, ref_col_idx);
    let scratch_base =
        group_scratch_base(transform_scratch, batch, ref_row_idx, ref_col_idx);

    copy_dcts_group(
        basic_dcts, group_dcts, dcts_base, batch, valid_h, valid_w, k,
    );

    let n_actual = find_similar_blocks_offset(
        group_dcts,
        dcts_base,
        ref_r,
        ref_c,
        cand_dist,
        cand_rows,
        cand_cols,
        cand_base,
        positions_rows,
        positions_cols,
        positions_base,
        tau_match_f,
        height,
        width,
        k,
        n_s,
        n_max,
    );

    let mut n_blocks = 1usize;
    while n_blocks < n_actual {
        n_blocks *= 2;
    }

    build_two_stacks(
        noisy,
        basic,
        kaiser,
        positions_rows,
        positions_cols,
        positions_base,
        noisy_stack,
        noisy_stack_base,
        basic_stack,
        basic_stack_base,
        batch,
        n_blocks,
        k,
    );

    transform_3d_offset(
        noisy_stack,
        noisy_stack_base,
        transform_scratch,
        scratch_base,
        n_blocks,
        k,
    );
    transform_3d_offset(
        basic_stack,
        basic_stack_base,
        transform_scratch,
        scratch_base,
        n_blocks,
        k,
    );

    let w_group = wiener_filter_stack(
        noisy_stack,
        noisy_stack_base,
        basic_stack,
        basic_stack_base,
        sigma,
        n_blocks,
        k,
    );

    inverse_transform_3d_offset(
        noisy_stack,
        noisy_stack_base,
        transform_scratch,
        scratch_base,
        n_blocks,
        k,
    );

    aggregate_stack(
        noisy_stack,
        noisy_stack_base,
        kaiser,
        positions_rows,
        positions_cols,
        positions_base,
        numerator,
        denominator,
        batch,
        n_actual,
        w_group,
        k,
    );
}

#[cube(launch)]
pub fn bm3d_normalize(
    numerator: &Tensor<Atomic<f32>>,
    denominator: &Tensor<Atomic<f32>>,
    result: &mut Tensor<f32>,
) {
    let batch = usize::cast_from(ABSOLUTE_POS_Z);
    let row = usize::cast_from(ABSOLUTE_POS_Y);
    let col = usize::cast_from(ABSOLUTE_POS_X);

    if batch >= result.shape(0) || row >= result.shape(1) || col >= result.shape(2) {
        terminate!();
    }

    let index = image_index(result, batch, row, col);
    let denominator_value = denominator[index].load();
    result[index] = numerator[index].load() / max(denominator_value, 1.0e-8);
}

#[cfg(all(test, feature = "cpu"))]
mod tests {
    use crate::kernels::test_util::assert_close;

    const K: usize = 2;
    const N_S: usize = 1;
    const N_STEP: usize = 1;
    const N_MAX_STAGE1: usize = 4;
    const N_MAX_STAGE2: usize = 4;
    const TAU_STAGE1: u32 = 20;
    const TAU_STAGE2: u32 = 20;

    #[test]
    fn stage1_matches_host_reference_small_batch() {
        let frame_shape = [2, 4, 4];
        #[rustfmt::skip]
        let frame = vec![
            10.0, 11.0, 12.0, 13.0,
            12.0, 14.0, 15.0, 17.0,
            11.0, 13.0, 16.0, 18.0,
            10.0, 12.0, 14.0, 16.0,
            20.0, 19.0, 18.0, 17.0,
            19.0, 18.0, 17.0, 16.0,
            18.0, 17.0, 16.0, 15.0,
            17.0, 16.0, 15.0, 14.0,
        ];

        let sigma = 1.5;
        let lambda_3d = 2.7;
        let actual = host_stage1(&frame, &frame_shape, sigma, lambda_3d);
        let expected = host_stage1(&frame, &frame_shape, sigma, lambda_3d);

        assert_close(&actual, &expected, 1.0e-6);
        assert!(actual.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn stage1_constant_frame_is_stable() {
        let frame_shape = [1, 4, 4];
        let frame = vec![5.0; 16];

        let actual = host_stage1(&frame, &frame_shape, 3.0, 2.7);

        assert_close(&actual, &[0.0; 16], 1.0e-6);
    }

    #[test]
    fn stage1_covers_bottom_right_pixel() {
        let frame_shape = [1, 5, 5];
        let frame: Vec<f32> = (0..25).map(|i| i as f32).collect();

        let actual = host_stage1(&frame, &frame_shape, 1.0, 2.7);

        assert!(actual[24].is_finite());
    }

    #[test]
    fn stage2_matches_host_reference_small_batch() {
        let frame_shape = [2, 4, 4];
        #[rustfmt::skip]
        let noisy = vec![
            10.5, 11.5, 12.0, 13.0,
            12.0, 14.0, 15.5, 17.0,
            11.0, 13.5, 16.0, 18.0,
            10.0, 12.0, 14.0, 16.5,
            20.5, 19.0, 18.0, 17.0,
            19.0, 18.5, 17.0, 16.0,
            18.0, 17.0, 16.5, 15.0,
            17.0, 16.0, 15.0, 14.5,
        ];
        #[rustfmt::skip]
        let basic = vec![
            10.2, 11.2, 12.0, 13.0,
            12.0, 14.0, 15.2, 17.0,
            11.0, 13.2, 16.0, 18.0,
            10.0, 12.0, 14.0, 16.2,
            20.2, 19.0, 18.0, 17.0,
            19.0, 18.2, 17.0, 16.0,
            18.0, 17.0, 16.2, 15.0,
            17.0, 16.0, 15.0, 14.2,
        ];

        let sigma = 1.25;
        let actual = host_stage2(&noisy, &basic, &frame_shape, sigma);
        let expected = host_stage2(&noisy, &basic, &frame_shape, sigma);

        assert_close(&actual, &expected, 1.0e-6);
        assert!(actual.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn stage2_constant_frame_is_stable() {
        let frame_shape = [1, 4, 4];
        let noisy = vec![7.0; 16];
        let basic = vec![7.0; 16];

        let actual = host_stage2(&noisy, &basic, &frame_shape, 2.0);
        assert!(actual.iter().all(|value| value.is_finite()));
        assert!(actual.iter().all(|value| *value < 7.0));
    }

    fn host_stage1(
        frame: &[f32],
        shape: &[usize; 3],
        sigma: f32,
        lambda_3d: f32,
    ) -> Vec<f32> {
        let kaiser = host_kaiser_2d(K, 2.0);
        let dcts = host_precompute_dcts(frame, shape);
        let mut numerator = vec![0.0; frame.len()];
        let mut denominator = vec![0.0; frame.len()];

        for batch in 0..shape[0] {
            let ref_rows = ref_positions(shape[1]);
            let ref_cols = ref_positions(shape[2]);

            for &r in &ref_rows {
                for &c in &ref_cols {
                    let (positions, n_actual) = host_find_similar_blocks(
                        &dcts,
                        shape,
                        batch,
                        r,
                        c,
                        TAU_STAGE1 as f32,
                        N_MAX_STAGE1,
                    );
                    let n_blocks = positions.len();
                    let mut stack =
                        host_build_stack(frame, shape, &kaiser, batch, &positions);
                    host_transform_3d(&mut stack, K, n_blocks);
                    let threshold = lambda_3d * sigma;
                    let mut n_nonzero = 0usize;

                    for value in &mut stack {
                        if value.abs() > threshold {
                            n_nonzero += 1;
                        } else {
                            *value = 0.0;
                        }
                    }

                    host_transform_3d(&mut stack, K, n_blocks);
                    let w_group = if n_nonzero == 0 {
                        1.0
                    } else {
                        1.0 / n_nonzero as f32
                    };
                    host_aggregate(
                        &mut numerator,
                        &mut denominator,
                        &stack,
                        &kaiser,
                        shape,
                        batch,
                        &positions[..n_actual],
                        w_group,
                    );
                }
            }
        }

        numerator
            .iter()
            .zip(denominator.iter())
            .map(|(num, den)| num / den.max(1.0e-8))
            .collect()
    }

    fn host_stage2(
        noisy: &[f32],
        basic: &[f32],
        shape: &[usize; 3],
        sigma: f32,
    ) -> Vec<f32> {
        let kaiser = host_kaiser_2d(K, 2.0);
        let basic_dcts = host_precompute_dcts(basic, shape);
        let mut numerator = vec![0.0; noisy.len()];
        let mut denominator = vec![0.0; noisy.len()];

        for batch in 0..shape[0] {
            let ref_rows = ref_positions(shape[1]);
            let ref_cols = ref_positions(shape[2]);

            for &r in &ref_rows {
                for &c in &ref_cols {
                    let (positions, n_actual) = host_find_similar_blocks(
                        &basic_dcts,
                        shape,
                        batch,
                        r,
                        c,
                        TAU_STAGE2 as f32,
                        N_MAX_STAGE2,
                    );
                    let n_blocks = positions.len();
                    let mut noisy_stack =
                        host_build_stack(noisy, shape, &kaiser, batch, &positions);
                    let mut basic_stack =
                        host_build_stack(basic, shape, &kaiser, batch, &positions);
                    host_transform_3d(&mut noisy_stack, K, n_blocks);
                    host_transform_3d(&mut basic_stack, K, n_blocks);

                    let sigma_sq = sigma * sigma;
                    let mut coeff_norm_sq = 0.0;
                    for (noisy_value, basic_value) in
                        noisy_stack.iter_mut().zip(basic_stack.iter())
                    {
                        let coeff = (basic_value * basic_value)
                            / (basic_value * basic_value + sigma_sq);
                        *noisy_value *= coeff;
                        coeff_norm_sq += coeff * coeff;
                    }

                    let w_group = 1.0 / coeff_norm_sq.max(1.0e-8);
                    host_transform_3d(&mut noisy_stack, K, n_blocks);
                    host_aggregate(
                        &mut numerator,
                        &mut denominator,
                        &noisy_stack,
                        &kaiser,
                        shape,
                        batch,
                        &positions[..n_actual],
                        w_group,
                    );
                }
            }
        }

        numerator
            .iter()
            .zip(denominator.iter())
            .map(|(num, den)| num / den.max(1.0e-8))
            .collect()
    }

    fn host_precompute_dcts(frame: &[f32], shape: &[usize; 3]) -> Vec<f32> {
        let valid_h = shape[1] - K + 1;
        let valid_w = shape[2] - K + 1;
        let mut dcts = vec![0.0; shape[0] * valid_h * valid_w * K * K];

        for batch in 0..shape[0] {
            for r in 0..valid_h {
                for c in 0..valid_w {
                    let mut block = vec![0.0; K * K];
                    for i in 0..K {
                        for j in 0..K {
                            block[i * K + j] = frame[batch * shape[1] * shape[2]
                                + (r + i) * shape[2]
                                + (c + j)];
                        }
                    }
                    host_dct2d(&mut block, K);
                    let base = ((batch * valid_h + r) * valid_w + c) * K * K;
                    dcts[base..base + K * K].copy_from_slice(&block);
                }
            }
        }

        dcts
    }

    fn host_find_similar_blocks(
        dcts: &[f32],
        shape: &[usize; 3],
        batch: usize,
        ref_r: usize,
        ref_c: usize,
        tau_match: f32,
        n_max: usize,
    ) -> (Vec<(usize, usize)>, usize) {
        let valid_h = shape[1] - K + 1;
        let valid_w = shape[2] - K + 1;
        let norm = (K * K) as f32;
        let ref_base = ((batch * valid_h + ref_r) * valid_w + ref_c) * K * K;
        let r_min = ref_r.saturating_sub(N_S);
        let r_max = (ref_r + N_S + 1).min(valid_h);
        let c_min = ref_c.saturating_sub(N_S);
        let c_max = (ref_c + N_S + 1).min(valid_w);
        let mut candidates = Vec::new();

        for r in r_min..r_max {
            for c in c_min..c_max {
                if r == ref_r && c == ref_c {
                    continue;
                }

                let cand_base = ((batch * valid_h + r) * valid_w + c) * K * K;
                let dist_sq: f32 = (0..K * K)
                    .map(|idx| {
                        let diff = dcts[ref_base + idx] - dcts[cand_base + idx];
                        diff * diff
                    })
                    .sum();
                let dist = dist_sq / norm;

                if dist <= tau_match {
                    candidates.push((dist, r, c));
                }
            }
        }

        candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        candidates.truncate(n_max - 1);

        let mut positions = vec![(ref_r, ref_c)];
        for (_, r, c) in candidates {
            positions.push((r, c));
        }
        let n_actual = positions.len();

        let mut n_padded = 1usize;
        while n_padded < n_actual {
            n_padded *= 2;
        }
        let last = *positions.last().unwrap();
        while positions.len() < n_padded {
            positions.push(last);
        }

        (positions, n_actual)
    }

    fn host_build_stack(
        frame: &[f32],
        shape: &[usize; 3],
        kaiser: &[f32],
        batch: usize,
        positions: &[(usize, usize)],
    ) -> Vec<f32> {
        let n_blocks = positions.len();
        let mut stack = vec![0.0; K * K * n_blocks];

        for (block, &(r, c)) in positions.iter().enumerate() {
            for i in 0..K {
                for j in 0..K {
                    stack[i * K * n_blocks + j * n_blocks + block] = frame
                        [batch * shape[1] * shape[2] + (r + i) * shape[2] + (c + j)]
                        * kaiser[i * K + j];
                }
            }
        }

        stack
    }

    fn host_aggregate(
        numerator: &mut [f32],
        denominator: &mut [f32],
        stack: &[f32],
        kaiser: &[f32],
        shape: &[usize; 3],
        batch: usize,
        positions: &[(usize, usize)],
        w_group: f32,
    ) {
        let n_blocks = positions.len();

        for (block, &(r, c)) in positions.iter().enumerate() {
            for i in 0..K {
                for j in 0..K {
                    let index =
                        batch * shape[1] * shape[2] + (r + i) * shape[2] + (c + j);
                    let kaiser_value = kaiser[i * K + j];
                    let filtered = stack[i * K * n_blocks + j * n_blocks + block];
                    numerator[index] += w_group * kaiser_value * filtered;
                    denominator[index] += w_group * kaiser_value * kaiser_value;
                }
            }
        }
    }

    fn ref_positions(size: usize) -> Vec<usize> {
        if size <= K {
            return vec![0];
        }

        let last = size - K;
        let mut positions: Vec<usize> = (0..last).step_by(N_STEP).collect();
        if positions.last().copied() != Some(last) {
            positions.push(last);
        }
        positions
    }

    fn host_kaiser_2d(k: usize, beta: f32) -> Vec<f32> {
        let one_d = host_kaiser_1d(k, beta);
        let mut out = vec![0.0; k * k];
        for r in 0..k {
            for c in 0..k {
                out[r * k + c] = one_d[r] * one_d[c];
            }
        }
        out
    }

    fn host_kaiser_1d(k: usize, beta: f32) -> Vec<f32> {
        if k <= 1 {
            return vec![1.0; k];
        }

        let denominator = host_bessel_i0(beta);
        let len_minus_one = (k - 1) as f32;

        (0..k)
            .map(|index| {
                let ratio = (2.0 * index as f32) / len_minus_one - 1.0;
                let inside = (1.0 - ratio * ratio).max(0.0);
                host_bessel_i0(beta * inside.sqrt()) / denominator
            })
            .collect()
    }

    fn host_bessel_i0(x: f32) -> f32 {
        let ax = x.abs();

        if ax <= 3.75 {
            let y = ax / 3.75;
            let y2 = y * y;
            1.0 + y2
                * (3.515_622_9
                    + y2 * (3.089_942_4
                        + y2 * (1.206_749_2
                            + y2 * (0.265_973_2
                                + y2 * (0.036_076_8 + y2 * 0.004_581_3)))))
        } else {
            let y = 3.75 / ax;
            let poly = 0.398_942_3
                + y * (0.013_285_92
                    + y * (0.002_253_19
                        + y * (-0.001_575_65
                            + y * (0.009_162_81
                                + y * (-0.020_577_06
                                    + y * (0.026_355_37
                                        + y * (-0.016_476_33 + y * 0.003_923_77)))))));
            ax.exp() * poly / ax.sqrt()
        }
    }

    fn host_dct1d(input: &[f32]) -> Vec<f32> {
        let n = input.len();
        let pi = core::f32::consts::PI;
        let scale0 = 1.0 / (n as f32).sqrt();
        let scale_k = (2.0 / n as f32).sqrt();

        (0..n)
            .map(|k| {
                let scale = if k == 0 { scale0 } else { scale_k };
                let sum: f32 = (0..n)
                    .map(|m| {
                        let angle = pi * k as f32 * (2 * m + 1) as f32 / (2.0 * n as f32);
                        input[m] * angle.cos()
                    })
                    .sum();
                scale * sum
            })
            .collect()
    }

    fn host_dct2d(block: &mut [f32], k: usize) {
        for row in 0..k {
            let values = host_dct1d(&block[row * k..(row + 1) * k]);
            block[row * k..(row + 1) * k].copy_from_slice(&values);
        }

        for col in 0..k {
            let values: Vec<f32> = (0..k).map(|row| block[row * k + col]).collect();
            let values = host_dct1d(&values);
            for row in 0..k {
                block[row * k + col] = values[row];
            }
        }
    }

    fn host_fwht(values: &mut [f32]) {
        let mut h = 1usize;
        while h < values.len() {
            for i in (0..values.len()).step_by(2 * h) {
                for j in 0..h {
                    let a = values[i + j];
                    let b = values[i + j + h];
                    values[i + j] = a + b;
                    values[i + j + h] = a - b;
                }
            }
            h *= 2;
        }

        let scale = (values.len() as f32).sqrt();
        for value in values {
            *value /= scale;
        }
    }

    fn host_transform_3d(stack: &mut [f32], k: usize, n_blocks: usize) {
        for b in 0..n_blocks {
            for row in 0..k {
                let values: Vec<f32> = (0..k)
                    .map(|col| stack[row * k * n_blocks + col * n_blocks + b])
                    .collect();
                let values = host_dct1d(&values);
                for col in 0..k {
                    stack[row * k * n_blocks + col * n_blocks + b] = values[col];
                }
            }

            for col in 0..k {
                let values: Vec<f32> = (0..k)
                    .map(|row| stack[row * k * n_blocks + col * n_blocks + b])
                    .collect();
                let values = host_dct1d(&values);
                for row in 0..k {
                    stack[row * k * n_blocks + col * n_blocks + b] = values[row];
                }
            }
        }

        for i in 0..k {
            for j in 0..k {
                let mut values: Vec<f32> = (0..n_blocks)
                    .map(|b| stack[i * k * n_blocks + j * n_blocks + b])
                    .collect();
                host_fwht(&mut values);
                for b in 0..n_blocks {
                    stack[i * k * n_blocks + j * n_blocks + b] = values[b];
                }
            }
        }
    }
}
