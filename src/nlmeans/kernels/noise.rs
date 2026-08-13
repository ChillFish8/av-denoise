use cubecl::prelude::*;

use super::helpers::read_line;

/// Per-cube stage of the Immerkær noise estimate. Every interior thread
/// applies the 3×3 Laplacian-difference mask to its pixel and the cube
/// reduces the absolute responses per channel into one partial sum.
/// Border pixels (and threads that fall outside the image because the
/// grid overshoots on the last row/column of cubes) contribute zero.
/// The result layout is `partials[cube_index * 4 + lane]` with unused
/// lanes left at zero.
#[cube(launch_unchecked)]
pub fn nlm_noise_partial<N: Size>(
    input: &Array<Vector<f32, N>>,
    partials: &mut Array<f32>,
    frame: u32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] channels: u32,
    #[comptime] block_x: u32,
    #[comptime] block_y: u32,
) {
    let threads = comptime!(block_x * block_y);
    let mut scratch = SharedMemory::<f32>::new(comptime!(block_x * block_y * 4) as usize);

    let x = ABSOLUTE_POS_X;
    let y = ABSOLUTE_POS_Y;
    let tid = UNIT_POS_Y * block_x + UNIT_POS_X;

    let interior = x >= 1 && x < width - 1 && y >= 1 && y < height - 1;

    let mut response = Vector::<f32, N>::empty();
    if interior {
        let c = read_line(input, x, y, frame, width, height);
        let l = read_line(input, x - 1, y, frame, width, height);
        let r = read_line(input, x + 1, y, frame, width, height);
        let u = read_line(input, x, y - 1, frame, width, height);
        let d = read_line(input, x, y + 1, frame, width, height);
        let ul = read_line(input, x - 1, y - 1, frame, width, height);
        let ur = read_line(input, x + 1, y - 1, frame, width, height);
        let dl = read_line(input, x - 1, y + 1, frame, width, height);
        let dr = read_line(input, x + 1, y + 1, frame, width, height);
        let four = Vector::<f32, N>::empty().fill(4.0f32);
        let two = Vector::<f32, N>::empty().fill(2.0f32);
        response = c * four - (l + r + u + d) * two + (ul + ur + dl + dr);
    }

    #[unroll]
    for ch in 0..channels {
        scratch[(tid * 4 + ch) as usize] = f32::abs(response[ch as usize]);
    }
    #[unroll]
    for ch in channels..4u32 {
        scratch[(tid * 4 + ch) as usize] = 0.0f32;
    }

    sync_cube();

    if tid == 0 {
        let cube_index = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
        #[unroll]
        for ch in 0..4u32 {
            let mut sum = 0.0f32;
            for t in 0..threads {
                sum += scratch[(t * 4 + ch) as usize];
            }
            partials[(cube_index * 4 + ch) as usize] = sum;
        }
    }
}

/// Final stage of the Immerkær noise estimate. One cube sums every
/// per-cube partial into the per-channel totals for the given ring
/// slot. Each thread accumulates a strided share of the partials and
/// thread zero folds the shares together.
#[cube(launch_unchecked)]
pub fn nlm_noise_reduce(
    partials: &Array<f32>,
    results: &mut Array<f32>,
    slot: u32,
    num_partials: u32,
    #[comptime] block: u32,
) {
    let mut scratch = SharedMemory::<f32>::new(comptime!(block * 4) as usize);
    let tid = UNIT_POS_X;

    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;
    let mut sum3 = 0.0f32;
    let mut i = tid;
    while i < num_partials {
        sum0 += partials[(i * 4) as usize];
        sum1 += partials[(i * 4 + 1) as usize];
        sum2 += partials[(i * 4 + 2) as usize];
        sum3 += partials[(i * 4 + 3) as usize];
        i += block;
    }
    scratch[(tid * 4) as usize] = sum0;
    scratch[(tid * 4 + 1) as usize] = sum1;
    scratch[(tid * 4 + 2) as usize] = sum2;
    scratch[(tid * 4 + 3) as usize] = sum3;

    sync_cube();

    if tid == 0 {
        #[unroll]
        for ch in 0..4u32 {
            let mut total = 0.0f32;
            for t in 0..block {
                total += scratch[(t * 4 + ch) as usize];
            }
            results[(slot * 4 + ch) as usize] = total;
        }
    }
}

/// Per-block temporal residual statistics. One cube per `block × block`
/// spatial block. Computes `d = input[slot_new] − input[slot_prev]` for
/// every pixel in the block and reduces it into one stats record:
/// `sum_d` and `sum_d2` per stored channel, plus the channel-0
/// horizontal lag-1 product `d0[x] · d0[x+1]` summed over in-block
/// adjacent pairs. A block that runs past the frame edge uses its
/// truncated in-frame extent for every sum; pixels outside the frame
/// contribute nothing, and a pair is only formed when its second pixel
/// is still inside that truncated extent (never across a block
/// boundary).
///
/// Records are written row-major, one per block, into `stats`:
/// `stats[block_index * (2 * stored_ch + 1) ..]`, laid out
/// `[sum_d(ch0..stored_ch-1), sum_d2(ch0..stored_ch-1), sum_lag]`.
/// `stats` is expected to already be sliced down to `slot_new`'s own
/// region of a larger ring buffer (see `noise::run_temporal_noise_stats`),
/// the same convention the motion-compensation kernels use for their
/// per-neighbour slices, so the kernel itself never needs to know
/// about the ring's other slots or any stride padding between them.
#[cube(launch_unchecked)]
pub fn nlm_temporal_noise_stats<N: Size>(
    input: &Array<Vector<f32, N>>,
    stats: &mut Array<f32>,
    slot_new: u32,
    slot_prev: u32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] stored_ch: u32,
    #[comptime] block: u32,
) {
    let record_len = comptime!(2 * stored_ch + 1);
    let threads = comptime!(block * block);

    let mut scratch = SharedMemory::<f32>::new(comptime!(threads * record_len) as usize);
    let mut d0_tile = SharedMemory::<f32>::new(threads as usize);

    let local_x = UNIT_POS_X;
    let local_y = UNIT_POS_Y;
    let tid = local_y * block + local_x;

    let block_origin_x = CUBE_POS_X * block;
    let block_origin_y = CUBE_POS_Y * block;
    let gx = block_origin_x + local_x;
    let gy = block_origin_y + local_y;

    let valid = gx < width && gy < height;

    let mut d = Vector::<f32, N>::empty();
    if valid {
        let c = read_line(input, gx, gy, slot_new, width, height);
        let p = read_line(input, gx, gy, slot_prev, width, height);
        d = c - p;
    }

    // The `.into()` calls satisfy cubecl's `if`/`else` type unification
    // (both arms must expand to the same `NativeExpand<f32>`); the
    // clippy lint doesn't see that requirement.
    #[allow(clippy::useless_conversion)]
    let d0 = if valid { d[0] } else { 0.0f32.into() };
    d0_tile[tid as usize] = d0;

    #[unroll]
    for ch in 0..stored_ch {
        #[allow(clippy::useless_conversion)]
        let v = if valid { d[ch as usize] } else { 0.0f32.into() };
        scratch[(tid * record_len + ch) as usize] = v;
        scratch[(tid * record_len + stored_ch + ch) as usize] = v * v;
    }

    sync_cube();

    // Truncated in-block width, so a pair never reaches past this
    // block's own slice of the frame (ragged right/bottom edges use
    // this truncated extent, mirroring how the block matcher's coarse
    // kernel seeds its ragged last block by position rather than by a
    // fixed block size).
    let block_w = u32::min(block, width - block_origin_x);
    let pair_valid = valid && local_x + 1 < block_w;
    #[allow(clippy::useless_conversion)]
    let lag = if pair_valid {
        d0 * d0_tile[(tid + 1) as usize]
    } else {
        0.0f32.into()
    };
    scratch[(tid * record_len + 2 * stored_ch) as usize] = lag;

    sync_cube();

    if tid == 0 {
        let block_index = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
        let out_base = block_index * record_len;

        #[unroll]
        for lane in 0..record_len {
            let mut total = 0.0f32;
            for t in 0..threads {
                total += scratch[(t * record_len + lane) as usize];
            }
            stats[(out_base + lane) as usize] = total;
        }
    }
}

/// Zero-fill a slice of the temporal-stats ring. A duplicated ring
/// slot mirrors its predecessor's pixels exactly, so diffing it for
/// real would just recompute an all-zero record; zeroing it directly
/// is cheaper and gives aggregation the same "nothing measurable
/// here" signal.
#[cube(launch_unchecked)]
pub fn nlm_temporal_stats_zero(
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
