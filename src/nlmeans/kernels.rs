use cubecl::prelude::*;
use cubecl::terminate;

/// GPU-to-GPU buffer copy. Copies `length` f32s from `src` into `dst`
/// starting at `offset` in `dst`.
#[cube(launch)]
pub fn gpu_copy(
    src: &Array<f32>,
    dst: &mut Array<f32>,
    offset: u32,
    #[comptime] length: u32,
) {
    let idx = ABSOLUTE_POS_X;

    if idx >= length {
        terminate!();
    }

    dst[(offset + idx) as usize] = src[idx as usize];
}

/// Zero-fill a GPU buffer.
#[cube(launch)]
pub fn gpu_zero(dst: &mut Array<f32>, #[comptime] length: u32) {
    let idx = ABSOLUTE_POS_X;

    if idx >= length {
        terminate!();
    }

    dst[idx as usize] = 0.0f32;
}

#[cube]
fn clamp_coord(v: i32, #[comptime] limit: u32) -> u32 {
    let mut result = v as u32;

    if v < 0 {
        result = 0u32;
    } else if v >= limit as i32 {
        result = limit - 1;
    }

    result
}

/// Reads a pixel value from a flat frame buffer with clamp-to-edge
/// boundary handling.
/// Buffer layout: [num_frames * height * width * channels].
#[cube]
fn read_clamped(
    buf: &Array<f32>,
    x: i32,
    y: i32,
    frame: i32,
    channel: u32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] channels: u32,
    #[comptime] num_frames: u32,
) -> f32 {
    let cx = clamp_coord(x, width);
    let cy = clamp_coord(y, height);
    let cf = clamp_coord(frame, num_frames);

    let idx = ((cf * height + cy) * width + cx) * channels + channel;
    buf[idx as usize]
}

/// Fused distance + 2D box filter + Welsch weight.
///
/// Replaces the separate dist_horizontal + vertical pipeline with a
/// single kernel. Computes per-pixel squared channel distance,
/// loads a 2D tile into shared memory, sums the (2p+1)² patch,
/// and applies the exponential weight function.
///
/// Eliminates the intermediate distance buffer and one kernel launch.
#[cube(launch)]
pub fn nlm_dist_2d_weight(
    input: &Array<f32>,
    output: &mut Array<f32>,
    t: i32,
    q_x: i32,
    q_y: i32,
    q_k: i32,
    h2_inv_norm: f32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] channels: u32,
    #[comptime] num_frames: u32,
    #[comptime] patch_radius: u32,
    #[comptime] block_x: u32,
    #[comptime] block_y: u32,
) {
    let p = patch_radius;
    let tile_w = block_x + 2 * p;
    let tile_h = block_y + 2 * p;
    let num_elems = tile_w * tile_h;
    let mut smem = SharedMemory::<f32>::new(num_elems as usize);

    let lx = UNIT_POS_X;
    let ly = UNIT_POS_Y;

    let gx = CUBE_POS_X * block_x + lx;
    let gy = CUBE_POS_Y * block_y + ly;

    let tile_start_x = CUBE_POS_X as i32 * block_x as i32 - p as i32;
    let tile_start_y = CUBE_POS_Y as i32 * block_y as i32 - p as i32;

    let frame_q = t + q_k;
    let scale = if channels == 1 {
        3.0f32
    } else if channels == 2 {
        1.5f32
    } else {
        1.0f32
    };

    // Cooperatively load 2D distance tile into shared memory.
    let threads = block_x * block_y;
    let tid = ly * block_x + lx;
    let mut idx = tid;

    while idx < num_elems {
        let tx = idx % tile_w;
        let ty = idx / tile_w;
        let src_x = tile_start_x + tx as i32;
        let src_y = tile_start_y + ty as i32;

        let mut dist = 0.0f32;

        for c in 0..channels {
            let a = read_clamped(
                input, src_x, src_y, t, c, width, height, channels, num_frames,
            );
            let b = read_clamped(
                input,
                src_x + q_x,
                src_y + q_y,
                frame_q,
                c,
                width,
                height,
                channels,
                num_frames,
            );
            let d = a - b;
            dist += d * d;
        }
        dist *= scale;

        smem[idx as usize] = dist;
        idx += threads;
    }

    sync_cube();

    if gx >= width || gy >= height {
        terminate!();
    }

    // 2D box sum over (2p+1)² patch centered at (lx+p, ly+p) in tile.
    let cx = lx + p;
    let cy = ly + p;
    let ksize = 2 * p + 1;
    let mut sum = 0.0f32;

    for dy in 0..ksize {
        for dx in 0..ksize {
            sum += smem[((cy - p + dy) * tile_w + cx - p + dx) as usize];
        }
    }

    let weight = f32::exp(-sum * h2_inv_norm);
    output[(gy * width + gx) as usize] = weight;
}

/// Accumulate weighted pixel contributions for offset q, processing
/// both +q and -q simultaneously (symmetry exploitation).
///
/// weights_fwd: weight map computed from center frame perspective (for +q direction)
/// weights_bwd: weight map computed from mirror frame perspective (for -q direction)
/// For spatial-only (k==0), both buffers point to the same data.
///
/// accum layout: [pixels * channels] — weighted pixel sums.
/// weight_sum layout: [pixels] — total weight per pixel.
#[cube(launch)]
pub fn nlm_accumulate(
    input: &Array<f32>,
    accum: &mut Array<f32>,
    weight_sum: &mut Array<f32>,
    weights_fwd: &Array<f32>,
    weights_bwd: &Array<f32>,
    max_weight: &mut Array<f32>,
    t: i32,
    q_x: i32,
    q_y: i32,
    q_k: i32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] channels: u32,
    #[comptime] num_frames: u32,
) {
    let x = ABSOLUTE_POS_X;
    let y = ABSOLUTE_POS_Y;

    if x >= width || y >= height {
        terminate!();
    }

    let p_idx = (y * width + x) as usize;
    let acc_base = p_idx * channels as usize;

    // w_pq: weight for pixel p toward p+q (center frame perspective)
    let w_pq = weights_fwd[p_idx];

    // w_mq: weight for pixel p-q toward p (mirror frame perspective)
    let mx = x as i32 - q_x;
    let my = y as i32 - q_y;
    let cmx = clamp_coord(mx, width);
    let cmy = clamp_coord(my, height);
    let mq_idx = (cmy * width + cmx) as usize;
    let w_mq = weights_bwd[mq_idx];

    // Update max weight
    let cur_max = max_weight[p_idx];
    let new_max = f32::max(f32::max(w_pq, w_mq), cur_max);
    max_weight[p_idx] = new_max;

    // Accumulate weighted pixel values for each channel
    for c in 0..channels {
        let pq_val = read_clamped(
            input,
            x as i32 + q_x,
            y as i32 + q_y,
            t + q_k,
            c,
            width,
            height,
            channels,
            num_frames,
        );
        let mq_val = read_clamped(
            input,
            x as i32 - q_x,
            y as i32 - q_y,
            t - q_k,
            c,
            width,
            height,
            channels,
            num_frames,
        );

        accum[acc_base + c as usize] += w_pq * pq_val + w_mq * mq_val;
    }

    weight_sum[p_idx] += w_pq + w_mq;
}

/// Normalize accumulated weighted sums to produce the denoised output.
///
/// output[c] = (original[c] * self_w + weighted_sum[c])
///           / (self_w + total_weight)
/// where self_w = wref * max_neighbor_weight
///
/// When denominator is near zero (no matches), preserves original.
#[cube(launch)]
pub fn nlm_finish(
    input: &Array<f32>,
    output: &mut Array<f32>,
    accum: &Array<f32>,
    weight_sum: &Array<f32>,
    max_weight: &Array<f32>,
    t: i32,
    wref: f32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] channels: u32,
    #[comptime] _num_frames: u32,
) {
    let x = ABSOLUTE_POS_X;
    let y = ABSOLUTE_POS_Y;

    if x >= width || y >= height {
        terminate!();
    }

    let p_idx = (y * width + x) as usize;
    let acc_base = p_idx * channels as usize;

    let m = wref * max_weight[p_idx];
    let den = m + weight_sum[p_idx];

    let frame_base = ((t as u32 * height + y) * width + x) * channels;

    for c in 0..channels {
        let original = input[(frame_base + c) as usize];
        let out_idx = p_idx * channels as usize + c as usize;

        if den > 1e-30f32 {
            output[out_idx] = (original * m + accum[acc_base + c as usize]) / den;
        } else {
            output[out_idx] = original;
        }
    }
}
