use cubecl::prelude::*;
use cubecl::terminate;

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

/// Returns the distance scaling factor for the given channel count.
/// Luma (1ch): 3.0, Chroma (2ch): 1.5, YUV (3ch): 1.0
pub fn distance_scale(channels: u32) -> f32 {
    match channels {
        1 => 3.0,
        2 => 1.5,
        _ => 1.0,
    }
}

/// Compute per-pixel squared color distance between pixel p and
/// pixel p+q.
///
/// Luma (1ch): dist = 3.0 * (a - b)^2
/// Chroma (2ch): dist = 1.5 * ((a0-b0)^2 + (a1-b1)^2)
/// YUV (3ch): dist = (a0-b0)^2 + (a1-b1)^2 + (a2-b2)^2
#[cube(launch)]
pub fn nlm_distance(
    input: &Array<f32>,
    output: &mut Array<f32>,
    t: i32,
    q_x: i32,
    q_y: i32,
    q_k: i32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] channels: u32,
    #[comptime] num_frames: u32,
) {
    let x = ABSOLUTE_POS_X as i32;
    let y = ABSOLUTE_POS_Y as i32;

    if x >= width as i32 || y >= height as i32 {
        terminate!();
    }

    let frame_q = t + q_k;
    let scale = if channels == 1 {
        3.0f32
    } else if channels == 2 {
        1.5f32
    } else {
        1.0f32
    };
    let mut val = 0.0f32;

    for c in 0..channels {
        let a = read_clamped(input, x, y, t, c, width, height, channels, num_frames);
        let b = read_clamped(
            input,
            x + q_x,
            y + q_y,
            frame_q,
            c,
            width,
            height,
            channels,
            num_frames,
        );
        let d = a - b;
        val += d * d;
    }

    val *= scale;

    let idx = y as u32 * width + x as u32;
    output[idx as usize] = val;
}

/// Horizontal box filter over the distance map using shared memory.
///
/// Each work group loads a horizontal tile (with halo) into shared
/// memory, then each thread sums 2*patch_radius+1 consecutive values.
#[cube(launch)]
pub fn nlm_horizontal(
    input: &Array<f32>,
    output: &mut Array<f32>,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] patch_radius: u32,
    #[comptime] block_x: u32,
    #[comptime] block_y: u32,
) {
    let tile_w = block_x + 2 * patch_radius;
    let smem_size = (tile_w * block_y) as usize;
    let mut smem = SharedMemory::<f32>::new(smem_size);

    let lx = UNIT_POS_X;
    let ly = UNIT_POS_Y;

    let gx = CUBE_POS_X * block_x + lx;
    let gy = CUBE_POS_Y * block_y + ly;

    let tile_start_x = CUBE_POS_X as i32 * block_x as i32 - patch_radius as i32;

    let row_offset = ly * tile_w;
    let mut i = lx;

    while i < tile_w {
        let src_x = tile_start_x + i as i32;
        let src_y = gy as i32;

        let mut val = 0.0f32;
        if src_x >= 0 && src_x < width as i32 && src_y >= 0 && src_y < height as i32 {
            val = input[(src_y as u32 * width + src_x as u32) as usize];
        }

        smem[(row_offset + i) as usize] = val;
        i += block_x;
    }

    sync_cube();

    if gx >= width || gy >= height {
        terminate!();
    }

    let center = lx + patch_radius;
    let mut sum = 0.0f32;
    let kernel_size = 2 * patch_radius + 1;

    for j in 0..kernel_size {
        sum += smem[(row_offset + center - patch_radius + j) as usize];
    }

    output[(gy * width + gx) as usize] = sum;
}

/// Vertical box filter + Welsch weight: w = exp(-sum * h2_inv_norm).
///
/// Uses shared memory for the vertical tile.
#[cube(launch)]
pub fn nlm_vertical(
    input: &Array<f32>,
    output: &mut Array<f32>,
    h2_inv_norm: f32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] patch_radius: u32,
    #[comptime] block_x: u32,
    #[comptime] block_y: u32,
) {
    let tile_h = block_y + 2 * patch_radius;
    let smem_size = (block_x * tile_h) as usize;
    let mut smem = SharedMemory::<f32>::new(smem_size);

    let lx = UNIT_POS_X;
    let ly = UNIT_POS_Y;

    let gx = CUBE_POS_X * block_x + lx;
    let gy = CUBE_POS_Y * block_y + ly;

    let tile_start_y = CUBE_POS_Y as i32 * block_y as i32 - patch_radius as i32;

    let col_offset = lx;
    let mut i = ly;

    while i < tile_h {
        let src_x = gx as i32;
        let src_y = tile_start_y + i as i32;

        let mut val = 0.0f32;
        if src_x >= 0 && src_x < width as i32 && src_y >= 0 && src_y < height as i32 {
            val = input[(src_y as u32 * width + src_x as u32) as usize];
        }

        smem[(i * block_x + col_offset) as usize] = val;
        i += block_y;
    }

    sync_cube();

    if gx >= width || gy >= height {
        terminate!();
    }

    let center = ly + patch_radius;
    let mut sum = 0.0f32;
    let kernel_size = 2 * patch_radius + 1;

    for j in 0..kernel_size {
        sum += smem[((center - patch_radius + j) * block_x + col_offset) as usize];
    }

    let weight = f32::exp(-sum * h2_inv_norm);
    output[(gy * width + gx) as usize] = weight;
}

/// Accumulate weighted pixel contributions for offset q, processing
/// both +q and -q simultaneously (symmetry exploitation).
///
/// accum layout: [pixels * channels] — weighted pixel sums.
/// weight_sum layout: [pixels] — total weight per pixel.
#[cube(launch)]
pub fn nlm_accumulate(
    input: &Array<f32>,
    accum: &mut Array<f32>,
    weight_sum: &mut Array<f32>,
    weights: &Array<f32>,
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

    let w_pq = weights[p_idx];

    let mx = x as i32 - q_x;
    let my = y as i32 - q_y;
    let cmx = clamp_coord(mx, width);
    let cmy = clamp_coord(my, height);
    let mq_idx = (cmy * width + cmx) as usize;
    let w_mq = weights[mq_idx];

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
