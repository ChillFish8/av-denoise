use cubecl::prelude::*;
use cubecl::terminate;

// ==================== Utility functions ====================

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

/// Reads a vectorized pixel (Line of `channels` f32s) from a flat frame
/// buffer with clamp-to-edge boundary handling on x and y.
///
/// Frame indices are *not* clamped — by construction the host always
/// schedules denoising with `t = temporal_radius` and `q_k ∈ [-d..d]`,
/// so `t ± q_k` is always in [0, num_frames). Border frames at flush
/// time are handled by duplicating ring buffer entries.
///
/// Buffer layout: [num_frames * height * width] lines, each line holding
/// `channels` f32 values (one full pixel). The launch site sets the
/// vectorization factor equal to the storage channel count so a single
/// line read fetches every channel of a pixel in one coalesced load.
#[cube]
fn read_clamped_line(
    buf: &Array<Line<f32>>,
    x: i32,
    y: i32,
    frame: u32,
    #[comptime] width: u32,
    #[comptime] height: u32,
) -> Line<f32> {
    let cx = clamp_coord(x, width);
    let cy = clamp_coord(y, height);

    let idx = (frame * height + cy) * width + cx;
    buf[idx as usize]
}

/// Unchecked variant — caller guarantees `x ∈ [0, width)` and
/// `y ∈ [0, height)`. Saves the two clamping branches per axis on the
/// (typically dominant) interior fast path.
#[cube]
fn read_line(
    buf: &Array<Line<f32>>,
    x: u32,
    y: u32,
    frame: u32,
    #[comptime] width: u32,
    #[comptime] height: u32,
) -> Line<f32> {
    let idx = (frame * height + y) * width + x;
    buf[idx as usize]
}

/// Sum-of-squares reduction over the lanes of a Line.
/// `channels` is comptime so the loop is fully unrolled.
#[cube]
fn line_sum_sq(diff: Line<f32>, #[comptime] channels: u32) -> f32 {
    let mut d = 0.0f32;

    #[unroll]
    for c in 0..channels {
        d += diff[c as usize] * diff[c as usize];
    }

    d
}

// ==================== Buffer management kernels ====================

/// GPU-to-GPU buffer copy. Copies `length` f32s from `src` into `dst`.
/// To copy at an offset within dst, use Handle::offset_start on the dst
/// handle when constructing the ArrayArg (CubeCL CPU runtime doesn't
/// correctly handle runtime scalar offsets in array indexing).
///
/// Uses strided loop so the grid can be capped below the 65535 limit.
#[cube(launch)]
pub fn gpu_copy(
    src: &Array<f32>,
    dst: &mut Array<f32>,
    #[comptime] length: u32,
    #[comptime] total_threads: u32,
) {
    let mut idx = ABSOLUTE_POS_X;

    while idx < length {
        dst[idx as usize] = src[idx as usize];
        idx += total_threads;
    }
}

/// Zero-fill a GPU buffer.
/// Uses strided loop so the grid can be capped below the 65535 limit.
#[cube(launch)]
pub fn gpu_zero(
    dst: &mut Array<f32>,
    #[comptime] length: u32,
    #[comptime] total_threads: u32,
) {
    let mut idx = ABSOLUTE_POS_X;

    while idx < length {
        dst[idx as usize] = 0.0f32;
        idx += total_threads;
    }
}

/// Fused zero for accum + weight_sum + max_weight in a single dispatch.
/// Uses strided loop so the grid can be capped below the 65535 limit.
#[cube(launch)]
pub fn gpu_zero_buffers(
    accum: &mut Array<f32>,
    weight_sum: &mut Array<f32>,
    max_weight: &mut Array<f32>,
    #[comptime] accum_len: u32,
    #[comptime] weight_len: u32,
    #[comptime] total_threads: u32,
) {
    // accum_len >= weight_len always holds (accum carries `stored_ch >= 1`
    // lanes per pixel). Tight loop zeroes all three up to weight_len;
    // tail loop continues zeroing only `accum` for the channel-padded
    // remainder. Removes the per-iter `idx < weight_len` branch from
    // the hot path.
    let mut idx = ABSOLUTE_POS_X;
    while idx < weight_len {
        accum[idx as usize] = 0.0f32;
        weight_sum[idx as usize] = 0.0f32;
        max_weight[idx as usize] = 0.0f32;
        idx += total_threads;
    }
    while idx < accum_len {
        accum[idx as usize] = 0.0f32;
        idx += total_threads;
    }
}

// ==================== Fused distance + weight kernel ====================
// Used for small patch_radius (<=2) where the (2p+1)^2 loop is cheap.

/// Fused distance + 2D box filter + Welsch weight.
///
/// Computes per-pixel squared channel distance using vectorized line
/// reads, loads a 2D tile of scalar distances into shared memory, sums
/// the (2p+1)^2 patch, and applies the exponential weight function.
///
/// Hot/cold split: a cube-uniform interior check decides whether the
/// entire tile (and its q-shifted twin) lies fully inside the image.
/// Interior cubes use unclamped reads. Border cubes use clamped reads.
/// The branch is uniform per cube → no warp divergence.
#[cube(launch)]
pub fn nlm_dist_2d_weight(
    input: &Array<Line<f32>>,
    output: &mut Array<f32>,
    frame_t: u32,
    frame_q: u32,
    q_x: i32,
    q_y: i32,
    h2_inv_norm: f32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] channels: u32,
    #[comptime] _num_frames: u32,
    #[comptime] patch_radius: u32,
    #[comptime] block_x: u32,
    #[comptime] block_y: u32,
) {
    let p = patch_radius;
    let tile_w = comptime!(block_x + 2 * patch_radius);
    let tile_h = comptime!(block_y + 2 * patch_radius);
    let num_elems = comptime!((block_x + 2 * patch_radius) * (block_y + 2 * patch_radius));
    let mut smem = SharedMemory::<f32>::new(comptime!((block_x + 2 * patch_radius) * (block_y + 2 * patch_radius)) as usize);

    let lx = UNIT_POS_X;
    let ly = UNIT_POS_Y;

    let gx = CUBE_POS_X * block_x + lx;
    let gy = CUBE_POS_Y * block_y + ly;

    let tile_start_x = CUBE_POS_X as i32 * block_x as i32 - p as i32;
    let tile_start_y = CUBE_POS_Y as i32 * block_y as i32 - p as i32;

    let scale = if channels == 1 {
        3.0f32
    } else if channels == 2 {
        1.5f32
    } else {
        1.0f32
    };

    // Cube-uniform interior check: does the union of both tile reads
    // (frame_t at offset 0 and frame_q at offset q) lie fully inside
    // the image?
    let tile_end_x = tile_start_x + tile_w as i32;
    let tile_end_y = tile_start_y + tile_h as i32;
    let interior = tile_start_x >= 0
        && tile_end_x <= width as i32
        && tile_start_y >= 0
        && tile_end_y <= height as i32
        && (tile_start_x + q_x) >= 0
        && (tile_end_x + q_x) <= width as i32
        && (tile_start_y + q_y) >= 0
        && (tile_end_y + q_y) <= height as i32;

    // Cooperatively load 2D distance tile into shared memory.
    let threads = block_x * block_y;
    let tid = ly * block_x + lx;
    let mut idx = tid;

    if interior {
        while idx < num_elems {
            let tx = idx % tile_w;
            let ty = idx / tile_w;
            let src_x = (tile_start_x + tx as i32) as u32;
            let src_y = (tile_start_y + ty as i32) as u32;

            let a = read_line(input, src_x, src_y, frame_t, width, height);
            let b = read_line(
                input,
                (src_x as i32 + q_x) as u32,
                (src_y as i32 + q_y) as u32,
                frame_q,
                width,
                height,
            );
            let diff = a - b;
            let mut dist = line_sum_sq(diff, channels);
            dist *= scale;

            smem[idx as usize] = dist;
            idx += threads;
        }
    } else {
        while idx < num_elems {
            let tx = idx % tile_w;
            let ty = idx / tile_w;
            let src_x = tile_start_x + tx as i32;
            let src_y = tile_start_y + ty as i32;

            let a = read_clamped_line(input, src_x, src_y, frame_t, width, height);
            let b = read_clamped_line(
                input,
                src_x + q_x,
                src_y + q_y,
                frame_q,
                width,
                height,
            );
            let diff = a - b;
            let mut dist = line_sum_sq(diff, channels);
            dist *= scale;

            smem[idx as usize] = dist;
            idx += threads;
        }
    }

    sync_cube();

    if gx >= width || gy >= height {
        terminate!();
    }

    // 2D box sum over (2p+1)^2 patch centered at (lx+p, ly+p) in tile.
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

/// Paired fused distance + weight for the temporal symmetry pair (+q, -q).
///
/// Each thread (gx, gy) writes both:
///   out_fwd[gx, gy] = exp(-h * Σpatch dist((t,   p), (t+k, p+q)))
///   out_bwd[gx, gy] = exp(-h * Σpatch dist((t,   p), (t-k, p-q)))
/// where p ranges over the (2*patch_radius+1)² patch around (gx, gy).
///
/// The center read at (t, p) is shared between fwd and bwd via SMEM,
/// reducing memory traffic from 4 reads/elem (two separate kernels) to
/// 3 reads/elem. Accumulate reads weights_fwd[p] and weights_bwd[p]
/// for displacement q exactly as in the unpaired path.
#[cube(launch)]
pub fn nlm_dist_2d_weight_pair(
    input: &Array<Line<f32>>,
    out_fwd: &mut Array<f32>,
    out_bwd: &mut Array<f32>,
    frame_t: u32,
    frame_fwd: u32,
    frame_bwd: u32,
    q_x: i32,
    q_y: i32,
    h2_inv_norm: f32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] channels: u32,
    #[comptime] _num_frames: u32,
    #[comptime] patch_radius: u32,
    #[comptime] block_x: u32,
    #[comptime] block_y: u32,
) {
    let p = patch_radius;
    let tile_w = comptime!(block_x + 2 * patch_radius);
    let tile_h = comptime!(block_y + 2 * patch_radius);
    let num_elems = comptime!((block_x + 2 * patch_radius) * (block_y + 2 * patch_radius));
    let mut smem_fwd = SharedMemory::<f32>::new(comptime!((block_x + 2 * patch_radius) * (block_y + 2 * patch_radius)) as usize);
    let mut smem_bwd = SharedMemory::<f32>::new(comptime!((block_x + 2 * patch_radius) * (block_y + 2 * patch_radius)) as usize);

    let lx = UNIT_POS_X;
    let ly = UNIT_POS_Y;

    let gx = CUBE_POS_X * block_x + lx;
    let gy = CUBE_POS_Y * block_y + ly;

    let tile_start_x = CUBE_POS_X as i32 * block_x as i32 - p as i32;
    let tile_start_y = CUBE_POS_Y as i32 * block_y as i32 - p as i32;

    let scale = if channels == 1 {
        3.0f32
    } else if channels == 2 {
        1.5f32
    } else {
        1.0f32
    };

    // Cube-uniform interior check: covers union of all three reads
    // — (frame_t, tile), (frame_fwd, tile+q), (frame_bwd, tile-q).
    let tile_end_x = tile_start_x + tile_w as i32;
    let tile_end_y = tile_start_y + tile_h as i32;
    let interior = tile_start_x >= 0
        && tile_end_x <= width as i32
        && tile_start_y >= 0
        && tile_end_y <= height as i32
        && (tile_start_x + q_x) >= 0
        && (tile_end_x + q_x) <= width as i32
        && (tile_start_y + q_y) >= 0
        && (tile_end_y + q_y) <= height as i32
        && (tile_start_x - q_x) >= 0
        && (tile_end_x - q_x) <= width as i32
        && (tile_start_y - q_y) >= 0
        && (tile_end_y - q_y) <= height as i32;

    let threads = block_x * block_y;
    let tid = ly * block_x + lx;
    let mut idx = tid;

    if interior {
        while idx < num_elems {
            let tx = idx % tile_w;
            let ty = idx / tile_w;
            let src_x = (tile_start_x + tx as i32) as u32;
            let src_y = (tile_start_y + ty as i32) as u32;

            let a = read_line(input, src_x, src_y, frame_t, width, height);
            let b_fwd = read_line(
                input,
                (src_x as i32 + q_x) as u32,
                (src_y as i32 + q_y) as u32,
                frame_fwd,
                width,
                height,
            );
            let b_bwd = read_line(
                input,
                (src_x as i32 - q_x) as u32,
                (src_y as i32 - q_y) as u32,
                frame_bwd,
                width,
                height,
            );

            let mut d_fwd = line_sum_sq(a - b_fwd, channels);
            let mut d_bwd = line_sum_sq(a - b_bwd, channels);
            d_fwd *= scale;
            d_bwd *= scale;

            smem_fwd[idx as usize] = d_fwd;
            smem_bwd[idx as usize] = d_bwd;
            idx += threads;
        }
    } else {
        while idx < num_elems {
            let tx = idx % tile_w;
            let ty = idx / tile_w;
            let src_x = tile_start_x + tx as i32;
            let src_y = tile_start_y + ty as i32;

            let a = read_clamped_line(input, src_x, src_y, frame_t, width, height);
            let b_fwd = read_clamped_line(
                input,
                src_x + q_x,
                src_y + q_y,
                frame_fwd,
                width,
                height,
            );
            let b_bwd = read_clamped_line(
                input,
                src_x - q_x,
                src_y - q_y,
                frame_bwd,
                width,
                height,
            );

            let mut d_fwd = line_sum_sq(a - b_fwd, channels);
            let mut d_bwd = line_sum_sq(a - b_bwd, channels);
            d_fwd *= scale;
            d_bwd *= scale;

            smem_fwd[idx as usize] = d_fwd;
            smem_bwd[idx as usize] = d_bwd;
            idx += threads;
        }
    }

    sync_cube();

    if gx >= width || gy >= height {
        terminate!();
    }

    let cx = lx + p;
    let cy = ly + p;
    let ksize = 2 * p + 1;
    let mut sum_fwd = 0.0f32;
    let mut sum_bwd = 0.0f32;

    for dy in 0..ksize {
        for dx in 0..ksize {
            let s_idx = ((cy - p + dy) * tile_w + cx - p + dx) as usize;
            sum_fwd += smem_fwd[s_idx];
            sum_bwd += smem_bwd[s_idx];
        }
    }

    out_fwd[(gy * width + gx) as usize] = f32::exp(-sum_fwd * h2_inv_norm);
    out_bwd[(gy * width + gx) as usize] = f32::exp(-sum_bwd * h2_inv_norm);
}

// ==================== Separable distance + weight kernels ====================
// Used for larger patch_radius (>2) where separable 1D passes are faster
// than the O((2p+1)^2) fused approach.

/// Paired per-pixel distance for the temporal symmetry pair (+q, -q).
///
/// Computes both fwd and bwd raw distances at every pixel:
///   dist_fwd[idx] = dist((t,   idx), (t+k, idx+q))
///   dist_bwd[idx] = dist((t-k, idx), (t,   idx+q))
///
/// Frame indices are passed pre-resolved to physical ring slots by the
/// host: `frame_t` is the center, `frame_fwd` / `frame_bwd` are the
/// +q_k / -q_k temporal neighbours. The kernel does no modulo.
#[cube(launch)]
pub fn nlm_distance_pair(
    input: &Array<Line<f32>>,
    dist_fwd: &mut Array<f32>,
    dist_bwd: &mut Array<f32>,
    frame_t: u32,
    frame_fwd: u32,
    frame_bwd: u32,
    q_x: i32,
    q_y: i32,
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

    let scale = if channels == 1 {
        3.0f32
    } else if channels == 2 {
        1.5f32
    } else {
        1.0f32
    };

    // Centers: (frame_t, x, y) for fwd, (frame_bwd, x, y) for bwd. Both
    // reads are at in-bounds (x, y) so no clamp needed.
    let a_fwd = read_line(input, x, y, frame_t, width, height);
    let a_bwd = read_line(input, x, y, frame_bwd, width, height);

    // Hot/cold split for the offset reads (warp-uniform for most warps).
    let nx = x as i32 + q_x;
    let ny = y as i32 + q_y;
    let interior =
        nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32;

    let b_fwd = if interior {
        read_line(input, nx as u32, ny as u32, frame_fwd, width, height)
    } else {
        read_clamped_line(input, nx, ny, frame_fwd, width, height)
    };
    let b_bwd = if interior {
        read_line(input, nx as u32, ny as u32, frame_t, width, height)
    } else {
        read_clamped_line(input, nx, ny, frame_t, width, height)
    };

    let mut d_fwd = line_sum_sq(a_fwd - b_fwd, channels);
    let mut d_bwd = line_sum_sq(a_bwd - b_bwd, channels);
    d_fwd *= scale;
    d_bwd *= scale;

    let p_idx = (y * width + x) as usize;
    dist_fwd[p_idx] = d_fwd;
    dist_bwd[p_idx] = d_bwd;
}

/// Per-pixel squared channel distance (no box filtering).
#[cube(launch)]
pub fn nlm_distance(
    input: &Array<Line<f32>>,
    dist: &mut Array<f32>,
    frame_t: u32,
    frame_q: u32,
    q_x: i32,
    q_y: i32,
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

    let scale = if channels == 1 {
        3.0f32
    } else if channels == 2 {
        1.5f32
    } else {
        1.0f32
    };

    // Center-pixel read is always in-bounds (x,y already passed bounds check).
    let a = read_line(input, x, y, frame_t, width, height);

    // Hot/cold split: skip clamping when the offset pixel is in-range.
    // The branch is uniform across most warps — only a thin border of
    // warps near the image edge takes the slow path.
    let bx = x as i32 + q_x;
    let by = y as i32 + q_y;
    let interior =
        bx >= 0 && bx < width as i32 && by >= 0 && by < height as i32;

    let b = if interior {
        read_line(input, bx as u32, by as u32, frame_q, width, height)
    } else {
        read_clamped_line(input, bx, by, frame_q, width, height)
    };

    let diff = a - b;
    let mut d = line_sum_sq(diff, channels);
    d *= scale;

    dist[(y * width + x) as usize] = d;
}

/// Horizontal 1D box filter over distance values using shared memory.
#[cube(launch)]
pub fn nlm_horizontal_sum(
    input: &Array<f32>,
    output: &mut Array<f32>,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] patch_radius: u32,
    #[comptime] block_x: u32,
    #[comptime] block_y: u32,
) {
    let p = patch_radius;
    let tile_w = comptime!(block_x + 2 * patch_radius);
    let num_elems = comptime!((block_x + 2 * patch_radius) * block_y);
    let mut smem = SharedMemory::<f32>::new(comptime!((block_x + 2 * patch_radius) * block_y) as usize);

    let lx = UNIT_POS_X;
    let ly = UNIT_POS_Y;
    let gx = CUBE_POS_X * block_x + lx;
    let gy = CUBE_POS_Y * block_y + ly;

    let tile_start_x = CUBE_POS_X as i32 * block_x as i32 - p as i32;

    // Cooperatively load horizontal tile with clamp-to-edge.
    let threads = block_x * block_y;
    let tid = ly * block_x + lx;
    let mut idx = tid;

    while idx < num_elems {
        let tx = idx % tile_w;
        let ty = idx / tile_w;
        let src_x = tile_start_x + tx as i32;
        let src_y = CUBE_POS_Y * block_y + ty;

        let cx = clamp_coord(src_x, width);
        let cy = clamp_coord(src_y as i32, height);

        smem[idx as usize] = input[(cy * width + cx) as usize];
        idx += threads;
    }

    sync_cube();

    if gx >= width || gy >= height {
        terminate!();
    }

    // Horizontal sum of (2p+1) values.
    let ksize = 2 * p + 1;
    let smem_base = ly * tile_w + lx;
    let mut sum = 0.0f32;
    for dx in 0..ksize {
        sum += smem[(smem_base + dx) as usize];
    }

    output[(gy * width + gx) as usize] = sum;
}

/// Vertical 1D box filter + Welsch weight using shared memory.
#[cube(launch)]
pub fn nlm_vertical_weight(
    input: &Array<f32>,
    output: &mut Array<f32>,
    h2_inv_norm: f32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] patch_radius: u32,
    #[comptime] block_x: u32,
    #[comptime] block_y: u32,
) {
    let p = patch_radius;
    let num_elems = comptime!(block_x * (block_y + 2 * patch_radius));
    let mut smem = SharedMemory::<f32>::new(comptime!(block_x * (block_y + 2 * patch_radius)) as usize);

    let lx = UNIT_POS_X;
    let ly = UNIT_POS_Y;
    let gx = CUBE_POS_X * block_x + lx;
    let gy = CUBE_POS_Y * block_y + ly;

    let tile_start_y = CUBE_POS_Y as i32 * block_y as i32 - p as i32;

    // Cooperatively load vertical tile with clamp-to-edge.
    let threads = block_x * block_y;
    let tid = ly * block_x + lx;
    let mut idx = tid;

    while idx < num_elems {
        let tx = idx % block_x;
        let ty = idx / block_x;
        let src_x = CUBE_POS_X * block_x + tx;
        let src_y = tile_start_y + ty as i32;

        let cx = clamp_coord(src_x as i32, width);
        let cy = clamp_coord(src_y, height);

        smem[idx as usize] = input[(cy * width + cx) as usize];
        idx += threads;
    }

    sync_cube();

    if gx >= width || gy >= height {
        terminate!();
    }

    // Vertical sum of (2p+1) values, then apply Welsch weight.
    let ksize = 2 * p + 1;
    let mut sum = 0.0f32;
    for dy in 0..ksize {
        sum += smem[((ly + dy) * block_x + lx) as usize];
    }

    let weight = f32::exp(-sum * h2_inv_norm);
    output[(gy * width + gx) as usize] = weight;
}

// ==================== Accumulate kernel ====================

/// Accumulate weighted pixel contributions for offset q, processing
/// both +q and -q simultaneously (symmetry exploitation).
///
/// weights_fwd: weight map from center frame perspective (for +q)
/// weights_bwd: weight map from mirror frame perspective (for -q)
/// For spatial-only (k==0), both buffers may point to the same data.
///
/// Input and accum are vectorized as Line<f32> with line size matching
/// the storage channel count so each pixel is read/written in a single
/// coalesced operation.
///
/// Hot/cold split: per-thread interior check covers both the +q and -q
/// reads; warps fully inside the safe region use unclamped loads.
#[cube(launch)]
pub fn nlm_accumulate(
    input: &Array<Line<f32>>,
    accum: &mut Array<Line<f32>>,
    weight_sum: &mut Array<f32>,
    weights_fwd: &Array<f32>,
    weights_bwd: &Array<f32>,
    max_weight: &mut Array<f32>,
    frame_fwd: u32,
    frame_bwd: u32,
    q_x: i32,
    q_y: i32,
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

    // w_pq: weight for pixel p toward p+q (center frame perspective)
    let w_pq = weights_fwd[p_idx];

    // w_mq: weight for pixel p-q toward p (mirror frame perspective)
    // Need clamped index for the lookup since (x-q_x, y-q_y) may fall
    // off the image at borders.
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

    // Hot/cold split: are both the +q and -q neighbor reads in-range?
    let pqx = x as i32 + q_x;
    let pqy = y as i32 + q_y;
    let mqx = x as i32 - q_x;
    let mqy = y as i32 - q_y;
    let interior = pqx >= 0
        && pqx < width as i32
        && pqy >= 0
        && pqy < height as i32
        && mqx >= 0
        && mqx < width as i32
        && mqy >= 0
        && mqy < height as i32;

    let pq_val = if interior {
        read_line(input, pqx as u32, pqy as u32, frame_fwd, width, height)
    } else {
        read_clamped_line(input, pqx, pqy, frame_fwd, width, height)
    };
    let mq_val = if interior {
        read_line(input, mqx as u32, mqy as u32, frame_bwd, width, height)
    } else {
        read_clamped_line(input, mqx, mqy, frame_bwd, width, height)
    };

    // Broadcast scalar weights to Line and accumulate via lane-wise
    // ops (no per-lane indexing — works for line_size 1 too).
    let _ = channels;
    let line_w_pq = Line::<f32>::empty(input.line_size()).fill(w_pq);
    let line_w_mq = Line::<f32>::empty(input.line_size()).fill(w_mq);
    let cur = accum[p_idx];
    accum[p_idx] = cur + pq_val * line_w_pq + mq_val * line_w_mq;

    weight_sum[p_idx] += w_pq + w_mq;
}

// ==================== Finish kernel ====================

/// Normalize accumulated weighted sums to produce the denoised output.
///
/// output[c] = (original[c as usize] * self_w + weighted_sum[c])
///           / (self_w + total_weight)
/// where self_w = wref * max_neighbor_weight
///
/// When denominator is near zero (no matches), preserves original.
///
/// Input/output/accum are vectorized as Line<f32> with line size ==
/// channels.
#[cube(launch)]
pub fn nlm_finish(
    input: &Array<Line<f32>>,
    output: &mut Array<Line<f32>>,
    accum: &Array<Line<f32>>,
    weight_sum: &Array<f32>,
    max_weight: &Array<f32>,
    center_frame: u32,
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
    let frame_idx = ((center_frame * height + y) * width + x) as usize;

    let m = wref * max_weight[p_idx];
    let den = m + weight_sum[p_idx];

    let original = input[frame_idx];
    let acc = accum[p_idx];

    // Allocate output line matching the buffer's storage line size
    // (may be larger than `channels` due to padding for vec3 -> vec4).
    // `Line::empty` zero-initializes so any padding lanes stay 0.
    let mut out = Line::empty(input.line_size());

    if den > 1e-30f32 {
        let inv_den = 1.0f32 / den;
        #[unroll]
        for c in 0..channels {
            out[c as usize] = (original[c as usize] * m + acc[c as usize]) * inv_den;
        }
    } else {
        #[unroll]
        for c in 0..channels {
            out[c as usize] = original[c as usize];
        }
    }

    output[p_idx] = out;
}
