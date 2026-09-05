// The search geometry `nl4d` ships with, in one place. Every value here
// comes from `Nl4dParams::default()`.
//
// `benches/kernels/collab_fused.rs` builds its frame ring and member
// buffers from these, so one set of constants describes what the
// denoiser actually runs.

/// `Nl4dParams::default().temporal_radius`.
pub const RADIUS: u32 = 2;
/// `Nl4dParams::default().refine`.
pub const REFINE: u32 = 2;
/// `Nl4dParams::default().spatial_radius`.
pub const SPATIAL_RADIUS: u32 = 9;
/// `collab::MAX_K`, the group size the filter runs at.
pub const K_MAX: u32 = 8;
/// `Nl4dParams::default().lambda_ht`.
pub const LAMBDA_HT: f32 = 4.24;
/// `Nl4dParams::default().confidence_variance`, the `use_member_sigma`
/// flag `collab_fused` compiles against.
pub const CONFIDENCE_VARIANCE: bool = true;

/// The motion field's block stride. Held at `collab::PATCH_SIZE` so a
/// block boundary lines up with a patch boundary.
pub const BLK_STEP: u32 = 8;
/// The library's own default motion block side length
/// (`MotionCompensationMode::Mvtools`'s `blksize`), distinct from
/// [`BLK_STEP`] above.
pub const BLKSIZE: u32 = 16;
/// `Nl4dParams::default().mismatch_scale` squared, the kernel's
/// `mismatch_scale2` argument.
pub const MISMATCH_SCALE2: f32 = 1.0;

/// Frames in the ring a pass reads.
pub const N_FRAMES: u32 = 2 * RADIUS + 1;
/// The physical ring slot a pass is centred on.
pub const CENTRE_SLOT: u32 = RADIUS;

/// The centre slot is skipped, and physical slots run `0..N_FRAMES`, so
/// `NEIGHBOUR_SLOTS[t]` for the neighbour at temporal offset `k` is
/// `k + RADIUS`, laid out in the same `neighbour_idx_for_k` order
/// `crate::nlmeans::motion::chain` uses: ascending k on the negative
/// side first, then ascending k on the positive side.
pub const NEIGHBOUR_SLOTS: [u32; (2 * RADIUS) as usize] = [0, 1, 3, 4];

/// Sigma the hard-threshold bench filters at.
pub const SIGMA: f32 = 0.02;

/// The motion-field stride one neighbour occupies, in `i32` elements.
///
/// `MotionCtx` pads each neighbour's slice of the motion buffer up to
/// the runtime's buffer-binding alignment, and passes the padded
/// element count to the kernel as a `#[comptime]` stride. A rig that
/// passes the unpadded count compiles the kernel against a stride the
/// pipeline never uses. Pass `client.properties().memory.alignment` as
/// `align`.
pub fn mv_stride(blocks_x: u32, blocks_y: u32, align: u64) -> u32 {
    padded_elems::<i32>(blocks_x as u64 * blocks_y as u64 * 2, align)
}

/// The confidence stride one neighbour occupies, in `f32` elements,
/// padded the way [`mv_stride`] describes.
pub fn conf_stride(blocks_x: u32, blocks_y: u32, align: u64) -> u32 {
    padded_elems::<f32>(blocks_x as u64 * blocks_y as u64, align)
}

/// `elems` of `T` rounded up so they cover a whole number of `align`
/// byte boundaries.
fn padded_elems<T>(elems: u64, align: u64) -> u32 {
    let size = size_of::<T>() as u64;
    ((elems * size).next_multiple_of(align) / size) as u32
}
