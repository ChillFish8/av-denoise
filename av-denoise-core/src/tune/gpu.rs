use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::nlmeans::kernels::gpu_zero_buffers;
use crate::nlmeans::{BLOCK_1D, MAX_GRID_1D};

/// Binds the first `len` elements of `handle` as a kernel array argument.
pub(crate) fn bind<R: Runtime>(handle: &Handle, len: usize) -> ArrayArg<R> {
    let owned = handle.clone();

    // SAFETY: every caller passes at most the element count the handle was allocated for.
    unsafe { ArrayArg::from_raw_parts(owned, len) }
}

/// Zeroes the first `len` 32-bit words of `handle` with the zero kernel.
///
/// A zero bit pattern is zero for both `f32` and `i32`, so this clears either kind of buffer.
pub(crate) fn zero_words<R: Runtime>(client: &ComputeClient<R>, handle: &Handle, len: usize) {
    // The kernel also clears two per-pixel buffers, and a zero length leaves these untouched.
    let unused_sum = client.empty(size_of::<f32>());
    let unused_max = client.empty(size_of::<f32>());

    zero_accumulators(client, handle, &unused_sum, &unused_max, len, 0);
}

/// Zeroes `accum_len` words of `accum` and `pixels` words of each per-pixel buffer in one launch.
///
/// `accum_len` must be at least `pixels`.
pub(crate) fn zero_accumulators<R: Runtime>(
    client: &ComputeClient<R>,
    accum: &Handle,
    weight_sum: &Handle,
    max_weight: &Handle,
    accum_len: usize,
    pixels: usize,
) {
    let word_count = accum_len as u32;
    let grid = word_count.div_ceil(BLOCK_1D).clamp(1, MAX_GRID_1D);
    let total_threads = grid * BLOCK_1D;
    let cube_count = CubeCount::new_1d(grid);
    let cube_dim = CubeDim::new_1d(BLOCK_1D);

    let accum_arg = bind::<R>(accum, accum_len);
    let weight_sum_arg = bind::<R>(weight_sum, pixels.max(1));
    let max_weight_arg = bind::<R>(max_weight, pixels.max(1));

    // SAFETY: each handle holds at least the element count bound above, and the kernel only
    // writes below `accum_len` and `pixels`.
    unsafe {
        gpu_zero_buffers::launch_unchecked::<R>(
            client,
            cube_count,
            cube_dim,
            accum_arg,
            weight_sum_arg,
            max_weight_arg,
            word_count,
            pixels as u32,
            total_threads,
        );
    }
}

/// Reads the first `len` values of `handle` back to the host.
fn read_values<R: Runtime, T: CubePrimitive + CubeElement>(
    client: &ComputeClient<R>,
    handle: &Handle,
    len: usize,
) -> Result<Vec<T>, anyhow::Error> {
    let owned = handle.clone();
    let bytes = client
        .read_one(owned)
        .map_err(|error| anyhow::anyhow!("readback failed: {error:?}"))?;
    let values = T::from_bytes(&bytes);
    let prefix = values
        .get(..len)
        .ok_or_else(|| anyhow::anyhow!("readback returned {} values, expected {len}", values.len()))?;

    Ok(prefix.to_vec())
}

/// Reads the first `len` `f32` values of `handle` back to the host.
pub(crate) fn read_f32<R: Runtime>(
    client: &ComputeClient<R>,
    handle: &Handle,
    len: usize,
) -> Result<Vec<f32>, anyhow::Error> {
    read_values(client, handle, len)
}

/// Reads the first `len` `i32` values of `handle` back to the host.
pub(crate) fn read_i32<R: Runtime>(
    client: &ComputeClient<R>,
    handle: &Handle,
    len: usize,
) -> Result<Vec<i32>, anyhow::Error> {
    read_values(client, handle, len)
}
