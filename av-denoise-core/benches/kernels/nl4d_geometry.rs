/// `Nl4dParams::default().temporal_radius`.
pub const RADIUS: u32 = 2;
/// `Nl4dParams::default().refine`.
pub const REFINE: u32 = 2;
/// `Nl4dParams::default().spatial_radius`.
pub const SPATIAL_RADIUS: u32 = 9;
/// `collab::MAX_K`, the group size the filter runs at.
pub const K_MAX: u32 = 8;
/// `Nl4dParams::default().lambda_ht`.
pub const LAMBDA_HT: f32 = 4.158;

/// The motion field's block stride.
///
/// It is held at `collab::PATCH_SIZE` so a block boundary lines up with a patch boundary.
pub const BLK_STEP: u32 = 8;
/// The library's default motion block side length, the `blksize` of `MotionCompensationMode::Mvtools`.
pub const BLKSIZE: u32 = 16;

/// Frames in the ring a pass reads.
pub const N_FRAMES: u32 = 2 * RADIUS + 1;
/// The physical ring slot a pass is centred on.
pub const CENTRE_SLOT: u32 = RADIUS;

/// The physical ring slot of each neighbour, skipping the centre.
///
/// Slots run `0..N_FRAMES` and the neighbour at temporal offset `k` sits in slot `k + RADIUS`.
/// The order matches `neighbour_idx_for_k`, ascending `k` on the negative side first and then on
/// the positive side.
pub const NEIGHBOUR_SLOTS: [u32; (2 * RADIUS) as usize] = [0, 1, 3, 4];

/// Sigma the hard-threshold bench filters at.
pub const SIGMA: f32 = 0.02;

/// The motion-field stride one neighbour occupies, in `i32` elements.
///
/// `MotionCtx` pads each neighbour's slice of the motion buffer up to the runtime's buffer-binding
/// alignment and passes the padded count to the kernel as a `#[comptime]` stride. A rig that
/// passes the unpadded count compiles the kernel against a stride the pipeline never uses. Pass
/// `client.properties().memory.alignment` as `align`.
pub fn mv_stride(blocks_x: u32, blocks_y: u32, align: u64) -> u32 {
    let elements = blocks_x as u64 * blocks_y as u64 * 2;
    padded_elems::<i32>(elements, align)
}

/// The confidence stride one neighbour occupies, in `f32` elements, padded like [mv_stride].
pub fn conf_stride(blocks_x: u32, blocks_y: u32, align: u64) -> u32 {
    let elements = blocks_x as u64 * blocks_y as u64;
    padded_elems::<f32>(elements, align)
}

/// `elements` of `T` rounded up to cover a whole number of `align`-byte boundaries.
fn padded_elems<T>(elements: u64, align: u64) -> u32 {
    let element_size = size_of::<T>() as u64;
    let padded_bytes = (elements * element_size).next_multiple_of(align);

    (padded_bytes / element_size) as u32
}
