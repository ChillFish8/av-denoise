use cubecl::prelude::*;
use cubecl::server::Handle;

use super::denoiser::NlmDenoiser;
use super::kernels::{
    gpu_copy,
    gpu_zero_buffers,
    nlm_accumulate,
    nlm_dist_2d_weight,
    nlm_dist_2d_weight_ref,
    nlm_distance,
    nlm_distance_pair,
    nlm_distance_pair_ref,
    nlm_distance_ref,
    nlm_finish,
    nlm_fused_pair_accumulate,
    nlm_fused_pair_accumulate_ref,
    nlm_fused_pair_accumulate_window,
    nlm_fused_pair_accumulate_window_ref,
    nlm_fused_single_window,
    nlm_fused_single_window_ref,
    nlm_horizontal_sum,
    nlm_horizontal_sum_pair,
    nlm_vertical_weight,
    nlm_vweight_pair_accumulate,
};
use super::motion::{
    self,
    confidence_byte_offset,
    run_analyse,
    run_compensate,
    run_confidence_for_neighbour,
};
use super::{BLOCK_1D, BLOCK_X, BLOCK_X_THIN, BLOCK_Y, BLOCK_Y_THIN, MAX_GRID_1D};

/// Derived sizes plus dispatch shape for the per-frame work, bundled so
/// the dispatch helpers don't carry long parallel argument lists.
pub(super) struct LaunchCtx {
    pub(super) total_frame_data: usize,
    pub(super) frame_size: usize,
    pub(super) pixels: usize,
    pub(super) cube_count: CubeCount,
    pub(super) cube_dim: CubeDim,
    /// Alternate shape used by `nlm_accumulate` / `nlm_finish` and the
    /// small-tile `nlm_dist_2d_weight(_ref)` kernels. See
    /// [`BLOCK_X_THIN`].
    pub(super) thin_cube_count: CubeCount,
    pub(super) thin_cube_dim: CubeDim,
}

/// Map a nonzero temporal offset `k` (in `-radius..=radius`) to the
/// neighbour index the analyse/confidence passes use when filling
/// `mv_field_buf`/`confidence_buf` (see `NlmDenoiser::run_motion_compensation`
/// and `run_confidence_pass`). Those passes fill neighbours in the order
/// `k = -radius..-1` first (indices `0..radius-1`), then `k = 1..radius`
/// (indices `radius..2·radius-1`), so this is the inverse of that walk.
fn neighbour_idx_for_k(radius: u32, k: i32) -> u32 {
    debug_assert_ne!(k, 0, "k=0 is the spatial pair, it has no neighbour index");
    debug_assert!(
        k.unsigned_abs() <= radius,
        "k={k} outside the temporal window ±{radius}"
    );
    if k < 0 {
        (k + radius as i32) as u32
    } else {
        (radius as i32 - 1 + k) as u32
    }
}

/// Confidence-kernel arguments for one temporal (k≠0) pair dispatch.
/// Carries whether confidence weighting is active, the forward/backward
/// per-block confidence array views, and the block geometry the
/// kernel needs to map an output pixel to its block index (mirrors
/// `nlm_mc_warp`'s pixel→block mapping). Inactive configurations carry
/// the small placeholder dummy buffer, `use_confidence` set to `false`,
/// and harmless (never read) geometry.
struct ConfidenceArgs<R: Runtime> {
    use_confidence: bool,
    conf_fwd: ArrayArg<R>,
    conf_bwd: ArrayArg<R>,
    step: u32,
    blocks_x: u32,
    blocks_y: u32,
}

impl<R: Runtime> NlmDenoiser<R> {
    fn input_arg(&self, ctx: &LaunchCtx) -> ArrayArg<R> {
        unsafe { ArrayArg::from_raw_parts(self.input_buf.clone(), ctx.total_frame_data) }
    }

    fn reference_arg(&self, ctx: &LaunchCtx) -> ArrayArg<R> {
        let buf = self
            .reference_buf
            .as_ref()
            .expect("reference buffer must exist when use_reference is set");
        unsafe { ArrayArg::from_raw_parts(buf.clone(), ctx.total_frame_data) }
    }

    /// Input array for the temporal (k≠0) kernels. Falls back to the
    /// compensated ring when motion compensation is active; otherwise
    /// identical to [`Self::input_arg`].
    fn input_arg_for_temporal(&self, ctx: &LaunchCtx) -> ArrayArg<R> {
        match self.compensated_input_buf.as_ref() {
            Some(buf) => unsafe { ArrayArg::from_raw_parts(buf.clone(), ctx.total_frame_data) },
            None => self.input_arg(ctx),
        }
    }

    /// Reference array for the temporal (k≠0) `_ref` kernels. Same
    /// fallback as [`Self::input_arg_for_temporal`].
    fn reference_arg_for_temporal(&self, ctx: &LaunchCtx) -> ArrayArg<R> {
        match self.compensated_reference_buf.as_ref() {
            Some(buf) => unsafe { ArrayArg::from_raw_parts(buf.clone(), ctx.total_frame_data) },
            None => self.reference_arg(ctx),
        }
    }

    fn accum_arg(&self, ctx: &LaunchCtx) -> ArrayArg<R> {
        unsafe { ArrayArg::from_raw_parts(self.accum.clone(), ctx.frame_size) }
    }

    fn output_arg(&self, ctx: &LaunchCtx, slot: usize) -> ArrayArg<R> {
        unsafe { ArrayArg::from_raw_parts(self.outputs[slot].clone(), ctx.frame_size) }
    }

    /// Reference ring slot `slot`, viewed as a single frame-sized array
    /// via a byte offset into `reference_buf`. Same byte-offset slicing
    /// pattern as the motion pyramid's per-slot views.
    fn reference_slot_arg(&self, ctx: &LaunchCtx, slot: u32) -> ArrayArg<R> {
        let buf = self
            .reference_buf
            .as_ref()
            .expect("reference buffer must exist for the nlm spatial pilot");
        let byte_offset = (slot as u64) * (ctx.frame_size as u64) * (size_of::<f32>() as u64);
        let handle = buf.clone().offset_start(byte_offset);
        unsafe { ArrayArg::from_raw_parts(handle, ctx.frame_size) }
    }

    fn weight_sum_arg(&self, ctx: &LaunchCtx) -> ArrayArg<R> {
        unsafe { ArrayArg::from_raw_parts(self.weight_sum.clone(), ctx.pixels) }
    }

    fn max_weight_arg(&self, ctx: &LaunchCtx) -> ArrayArg<R> {
        unsafe { ArrayArg::from_raw_parts(self.max_weight.clone(), ctx.pixels) }
    }

    fn weight_buf_arg(&self, ctx: &LaunchCtx) -> ArrayArg<R> {
        unsafe { ArrayArg::from_raw_parts(self.weight_buf.clone(), ctx.pixels) }
    }

    fn raw_fwd_arg(&self, ctx: &LaunchCtx) -> ArrayArg<R> {
        unsafe { ArrayArg::from_raw_parts(self.raw_fwd.clone(), ctx.pixels) }
    }

    fn raw_bwd_arg(&self, ctx: &LaunchCtx) -> ArrayArg<R> {
        unsafe { ArrayArg::from_raw_parts(self.raw_bwd.clone(), ctx.pixels) }
    }

    fn tmp_hsum_arg(&self, ctx: &LaunchCtx) -> ArrayArg<R> {
        unsafe { ArrayArg::from_raw_parts(self.tmp_hsum.clone(), ctx.pixels) }
    }

    fn tmp_hsum_bwd_arg(&self, ctx: &LaunchCtx) -> ArrayArg<R> {
        unsafe { ArrayArg::from_raw_parts(self.tmp_hsum_bwd.clone(), ctx.pixels) }
    }

    /// Build the confidence-kernel arguments for one temporal pair at
    /// offset `q_k` (always nonzero at every call site). `frame_fwd`
    /// reads neighbour `center + q_k`, so its confidence comes from
    /// neighbour `q_k`'s slice. `frame_bwd` reads `center - q_k`, so its
    /// confidence comes from neighbour `-q_k`'s slice (see
    /// `neighbour_idx_for_k`).
    ///
    /// Confidence weighting is active only when `confidence_buf` is
    /// allocated (see `NlmDenoiser::new`) and block geometry exists,
    /// either from `mc_ctx` (MC active) or `confidence_ctx` (the no-MC
    /// confidence pass). Otherwise this falls back to the 1-element
    /// `confidence_dummy` buffer with `use_confidence` set to `false`,
    /// so the kernel never reads it.
    fn confidence_pair_args(&self, q_k: i32) -> ConfidenceArgs<R> {
        let geometry = self.mc_ctx.as_ref().or(self.confidence_ctx.as_ref());
        if let (Some(buf), Some(mc)) = (self.confidence_buf.as_ref(), geometry) {
            let radius = self.params.temporal_radius;
            let fwd_idx = neighbour_idx_for_k(radius, q_k);
            let bwd_idx = neighbour_idx_for_k(radius, -q_k);
            let conf_len = (mc.blocks_x * mc.blocks_y) as usize;
            let fwd_handle = buf.clone().offset_start(confidence_byte_offset(mc, fwd_idx));
            let bwd_handle = buf.clone().offset_start(confidence_byte_offset(mc, bwd_idx));
            ConfidenceArgs {
                use_confidence: true,
                conf_fwd: unsafe { ArrayArg::from_raw_parts(fwd_handle, conf_len) },
                conf_bwd: unsafe { ArrayArg::from_raw_parts(bwd_handle, conf_len) },
                step: mc.step,
                blocks_x: mc.blocks_x,
                blocks_y: mc.blocks_y,
            }
        } else {
            ConfidenceArgs {
                use_confidence: false,
                conf_fwd: unsafe { ArrayArg::from_raw_parts(self.confidence_dummy.clone(), 1) },
                conf_bwd: unsafe { ArrayArg::from_raw_parts(self.confidence_dummy.clone(), 1) },
                step: 1,
                blocks_x: 1,
                blocks_y: 1,
            }
        }
    }

    /// Temporal (k≠0) fused-path step: one launch that computes both
    /// weights in registers and applies the +q / −q contributions.
    fn dispatch_fused_iter(
        &self,
        ctx: &LaunchCtx,
        center_t: u32,
        q_x: i32,
        q_y: i32,
        q_k: i32,
    ) -> Result<(), anyhow::Error> {
        let channels = self.params.channels.count();
        let frame_t = self.phys_frame(center_t as i32);
        let frame_fwd = self.phys_frame(center_t as i32 + q_k);
        let frame_bwd = self.phys_frame(center_t as i32 - q_k);
        // The backward distance compares against a different neighbour
        // depending on whether the temporal offset is zero. For k=0 the
        // pair collapses to a symmetric (+q, +q) self-comparison; for
        // k≠0 the true (+q, −q) cross-frame comparison applies.
        let (bwd_shift_x, bwd_shift_y) = if q_k == 0 { (q_x, q_y) } else { (-q_x, -q_y) };
        let confidence = self.confidence_pair_args(q_k);

        if self.use_reference {
            unsafe {
                nlm_fused_pair_accumulate_ref::launch_unchecked::<R>(
                    &self.client,
                    ctx.cube_count.clone(),
                    ctx.cube_dim,
                    self.params.channels.storage_count() as usize,
                    self.input_arg_for_temporal(ctx),
                    self.reference_arg_for_temporal(ctx),
                    self.accum_arg(ctx),
                    self.weight_sum_arg(ctx),
                    self.max_weight_arg(ctx),
                    confidence.conf_fwd,
                    confidence.conf_bwd,
                    confidence.use_confidence,
                    frame_t,
                    frame_fwd,
                    frame_bwd,
                    q_x,
                    q_y,
                    bwd_shift_x,
                    bwd_shift_y,
                    self.h2_inv_norm,
                    self.noise_offset,
                    self.width,
                    self.height,
                    channels,
                    self.params.patch_radius,
                    BLOCK_X,
                    BLOCK_Y,
                    confidence.step,
                    confidence.blocks_x,
                    confidence.blocks_y,
                );
            }
        } else {
            unsafe {
                nlm_fused_pair_accumulate::launch_unchecked::<R>(
                    &self.client,
                    ctx.cube_count.clone(),
                    ctx.cube_dim,
                    self.params.channels.storage_count() as usize,
                    self.input_arg_for_temporal(ctx),
                    self.accum_arg(ctx),
                    self.weight_sum_arg(ctx),
                    self.max_weight_arg(ctx),
                    confidence.conf_fwd,
                    confidence.conf_bwd,
                    confidence.use_confidence,
                    frame_t,
                    frame_fwd,
                    frame_bwd,
                    q_x,
                    q_y,
                    bwd_shift_x,
                    bwd_shift_y,
                    self.h2_inv_norm,
                    self.noise_offset,
                    self.width,
                    self.height,
                    channels,
                    self.params.patch_radius,
                    BLOCK_X,
                    BLOCK_Y,
                    confidence.step,
                    confidence.blocks_x,
                    confidence.blocks_y,
                );
            }
        }

        Ok(())
    }

    /// Temporal (k≠0) windowed fused step: a single launch covers every
    /// `(q_x, q_y)` in the search window, keeping accum / weight_sum /
    /// max_weight register-resident across the inner q-loop. Collapses
    /// `(2·search_radius + 1)²` per-q launches into one.
    fn dispatch_fused_window_iter(
        &self,
        ctx: &LaunchCtx,
        center_t: u32,
        q_k: i32,
    ) -> Result<(), anyhow::Error> {
        let channels = self.params.channels.count();
        let _stored = self.params.channels.storage_count();
        let frame_t = self.phys_frame(center_t as i32);
        let frame_fwd = self.phys_frame(center_t as i32 + q_k);
        let frame_bwd = self.phys_frame(center_t as i32 - q_k);
        let confidence = self.confidence_pair_args(q_k);

        if self.use_reference {
            unsafe {
                nlm_fused_pair_accumulate_window_ref::launch_unchecked::<R>(
                    &self.client,
                    ctx.cube_count.clone(),
                    ctx.cube_dim,
                    self.params.channels.storage_count() as usize,
                    self.input_arg_for_temporal(ctx),
                    self.reference_arg_for_temporal(ctx),
                    self.accum_arg(ctx),
                    self.weight_sum_arg(ctx),
                    self.max_weight_arg(ctx),
                    confidence.conf_fwd,
                    confidence.conf_bwd,
                    confidence.use_confidence,
                    frame_t,
                    frame_fwd,
                    frame_bwd,
                    self.h2_inv_norm,
                    self.noise_offset,
                    self.width,
                    self.height,
                    channels,
                    self.params.patch_radius,
                    self.params.search_radius,
                    BLOCK_X,
                    BLOCK_Y,
                    confidence.step,
                    confidence.blocks_x,
                    confidence.blocks_y,
                );
            }
        } else {
            unsafe {
                nlm_fused_pair_accumulate_window::launch_unchecked::<R>(
                    &self.client,
                    ctx.cube_count.clone(),
                    ctx.cube_dim,
                    self.params.channels.storage_count() as usize,
                    self.input_arg_for_temporal(ctx),
                    self.accum_arg(ctx),
                    self.weight_sum_arg(ctx),
                    self.max_weight_arg(ctx),
                    confidence.conf_fwd,
                    confidence.conf_bwd,
                    confidence.use_confidence,
                    frame_t,
                    frame_fwd,
                    frame_bwd,
                    self.h2_inv_norm,
                    self.noise_offset,
                    self.width,
                    self.height,
                    channels,
                    self.params.patch_radius,
                    self.params.search_radius,
                    BLOCK_X,
                    BLOCK_Y,
                    confidence.step,
                    confidence.blocks_x,
                    confidence.blocks_y,
                );
            }
        }

        Ok(())
    }

    /// Spatial (k=0) windowed fused step: a single launch covers every
    /// `(q_x, q_y)` in the search window in one direction, exploiting the
    /// symmetry of the patch distance (`w(x, −q) = w(x−q, q)`) so the
    /// full-window single-direction sum equals the half-window paired sum
    /// applied per q.
    fn dispatch_fused_single_window_iter(&self, ctx: &LaunchCtx, center_t: u32) -> Result<(), anyhow::Error> {
        let channels = self.params.channels.count();
        let _stored = self.params.channels.storage_count();
        let frame_t = self.phys_frame(center_t as i32);

        if self.use_reference {
            unsafe {
                nlm_fused_single_window_ref::launch_unchecked::<R>(
                    &self.client,
                    ctx.cube_count.clone(),
                    ctx.cube_dim,
                    self.params.channels.storage_count() as usize,
                    self.input_arg(ctx),
                    self.reference_arg(ctx),
                    self.accum_arg(ctx),
                    self.weight_sum_arg(ctx),
                    self.max_weight_arg(ctx),
                    frame_t,
                    self.h2_inv_norm,
                    self.noise_offset,
                    self.width,
                    self.height,
                    channels,
                    self.params.patch_radius,
                    self.params.search_radius,
                    BLOCK_X,
                    BLOCK_Y,
                );
            }
        } else {
            unsafe {
                nlm_fused_single_window::launch_unchecked::<R>(
                    &self.client,
                    ctx.cube_count.clone(),
                    ctx.cube_dim,
                    self.params.channels.storage_count() as usize,
                    self.input_arg(ctx),
                    self.accum_arg(ctx),
                    self.weight_sum_arg(ctx),
                    self.max_weight_arg(ctx),
                    frame_t,
                    self.h2_inv_norm,
                    self.noise_offset,
                    self.width,
                    self.height,
                    channels,
                    self.params.patch_radius,
                    self.params.search_radius,
                    BLOCK_X,
                    BLOCK_Y,
                );
            }
        }

        Ok(())
    }

    /// Temporal (k≠0) separable-path step: distance_pair →
    /// horizontal_sum_pair → fused vweight+accumulate. The fused
    /// terminal kernel consumes both hsum buffers, so no global weight
    /// buffer is written.
    fn dispatch_separable_iter(
        &self,
        ctx: &LaunchCtx,
        center_t: u32,
        q_x: i32,
        q_y: i32,
        q_k: i32,
    ) -> Result<(), anyhow::Error> {
        let channels = self.params.channels.count();
        let frame_t = self.phys_frame(center_t as i32);
        let frame_fwd = self.phys_frame(center_t as i32 + q_k);
        let frame_bwd = self.phys_frame(center_t as i32 - q_k);

        if self.use_reference {
            unsafe {
                nlm_distance_pair_ref::launch_unchecked::<R>(
                    &self.client,
                    ctx.cube_count.clone(),
                    ctx.cube_dim,
                    self.params.channels.storage_count() as usize,
                    self.reference_arg_for_temporal(ctx),
                    self.raw_fwd_arg(ctx),
                    self.raw_bwd_arg(ctx),
                    frame_t,
                    frame_fwd,
                    frame_bwd,
                    q_x,
                    q_y,
                    self.width,
                    self.height,
                    channels,
                );
            }
        } else {
            unsafe {
                nlm_distance_pair::launch_unchecked::<R>(
                    &self.client,
                    ctx.cube_count.clone(),
                    ctx.cube_dim,
                    self.params.channels.storage_count() as usize,
                    self.input_arg_for_temporal(ctx),
                    self.raw_fwd_arg(ctx),
                    self.raw_bwd_arg(ctx),
                    frame_t,
                    frame_fwd,
                    frame_bwd,
                    q_x,
                    q_y,
                    self.width,
                    self.height,
                    channels,
                );
            }
        }

        unsafe {
            nlm_horizontal_sum_pair::launch_unchecked::<R>(
                &self.client,
                ctx.cube_count.clone(),
                ctx.cube_dim,
                self.raw_fwd_arg(ctx),
                self.raw_bwd_arg(ctx),
                self.tmp_hsum_arg(ctx),
                self.tmp_hsum_bwd_arg(ctx),
                self.width,
                self.height,
                self.params.patch_radius,
                BLOCK_X,
                BLOCK_Y,
            );
        }

        let confidence = self.confidence_pair_args(q_k);
        unsafe {
            nlm_vweight_pair_accumulate::launch_unchecked::<R>(
                &self.client,
                ctx.cube_count.clone(),
                ctx.cube_dim,
                self.params.channels.storage_count() as usize,
                self.tmp_hsum_arg(ctx),
                self.tmp_hsum_bwd_arg(ctx),
                self.input_arg_for_temporal(ctx),
                self.accum_arg(ctx),
                self.weight_sum_arg(ctx),
                self.max_weight_arg(ctx),
                confidence.conf_fwd,
                confidence.conf_bwd,
                confidence.use_confidence,
                frame_fwd,
                frame_bwd,
                q_x,
                q_y,
                self.h2_inv_norm,
                self.noise_offset,
                self.width,
                self.height,
                self.params.patch_radius,
                BLOCK_X,
                BLOCK_Y,
                confidence.step,
                confidence.blocks_x,
                confidence.blocks_y,
            );
        }

        Ok(())
    }

    /// Spatial (k=0) fused-path step: single-tile weight + accumulate.
    /// Cheaper than the paired fused kernel here because the weight
    /// map is symmetric, so a single tile is enough.
    fn dispatch_fused_iter_k0(
        &self,
        ctx: &LaunchCtx,
        center_t: u32,
        q_x: i32,
        q_y: i32,
    ) -> Result<(), anyhow::Error> {
        let channels = self.params.channels.count();
        let frame_t = self.phys_frame(center_t as i32);

        if self.use_reference {
            unsafe {
                nlm_dist_2d_weight_ref::launch_unchecked::<R>(
                    &self.client,
                    ctx.cube_count.clone(),
                    ctx.cube_dim,
                    self.params.channels.storage_count() as usize,
                    self.reference_arg(ctx),
                    self.weight_buf_arg(ctx),
                    frame_t,
                    frame_t,
                    q_x,
                    q_y,
                    self.h2_inv_norm,
                    self.noise_offset,
                    self.width,
                    self.height,
                    channels,
                    self.params.patch_radius,
                    BLOCK_X,
                    BLOCK_Y,
                );
            }
        } else {
            unsafe {
                nlm_dist_2d_weight::launch_unchecked::<R>(
                    &self.client,
                    ctx.cube_count.clone(),
                    ctx.cube_dim,
                    self.params.channels.storage_count() as usize,
                    self.input_arg(ctx),
                    self.weight_buf_arg(ctx),
                    frame_t,
                    frame_t,
                    q_x,
                    q_y,
                    self.h2_inv_norm,
                    self.noise_offset,
                    self.width,
                    self.height,
                    channels,
                    self.params.patch_radius,
                    BLOCK_X,
                    BLOCK_Y,
                );
            }
        }

        unsafe {
            nlm_accumulate::launch_unchecked::<R>(
                &self.client,
                ctx.thin_cube_count.clone(),
                ctx.thin_cube_dim,
                self.params.channels.storage_count() as usize,
                self.input_arg(ctx),
                self.accum_arg(ctx),
                self.weight_sum_arg(ctx),
                self.weight_buf_arg(ctx),
                self.weight_buf_arg(ctx),
                self.max_weight_arg(ctx),
                frame_t,
                frame_t,
                q_x,
                q_y,
                self.width,
                self.height,
            );
        }

        Ok(())
    }

    /// Spatial (k=0) separable-path step: distance → hsum → vweight
    /// (single buffer) → accumulate. Symmetric weight map, so one
    /// buffer is reused for both forward and backward lookups.
    fn dispatch_separable_iter_k0(
        &self,
        ctx: &LaunchCtx,
        center_t: u32,
        q_x: i32,
        q_y: i32,
    ) -> Result<(), anyhow::Error> {
        let channels = self.params.channels.count();
        let frame_t = self.phys_frame(center_t as i32);

        if self.use_reference {
            unsafe {
                nlm_distance_ref::launch_unchecked::<R>(
                    &self.client,
                    ctx.cube_count.clone(),
                    ctx.cube_dim,
                    self.params.channels.storage_count() as usize,
                    self.reference_arg(ctx),
                    self.raw_fwd_arg(ctx),
                    frame_t,
                    frame_t,
                    q_x,
                    q_y,
                    self.width,
                    self.height,
                    channels,
                );
            }
        } else {
            unsafe {
                nlm_distance::launch_unchecked::<R>(
                    &self.client,
                    ctx.cube_count.clone(),
                    ctx.cube_dim,
                    self.params.channels.storage_count() as usize,
                    self.input_arg(ctx),
                    self.raw_fwd_arg(ctx),
                    frame_t,
                    frame_t,
                    q_x,
                    q_y,
                    self.width,
                    self.height,
                    channels,
                );
            }
        }

        unsafe {
            nlm_horizontal_sum::launch_unchecked::<R>(
                &self.client,
                ctx.cube_count.clone(),
                ctx.cube_dim,
                self.raw_fwd_arg(ctx),
                self.tmp_hsum_arg(ctx),
                self.width,
                self.height,
                self.params.patch_radius,
                BLOCK_X,
                BLOCK_Y,
            );
        }

        unsafe {
            nlm_vertical_weight::launch_unchecked::<R>(
                &self.client,
                ctx.cube_count.clone(),
                ctx.cube_dim,
                self.tmp_hsum_arg(ctx),
                self.weight_buf_arg(ctx),
                self.h2_inv_norm,
                self.noise_offset,
                self.width,
                self.height,
                self.params.patch_radius,
                BLOCK_X,
                BLOCK_Y,
            );
        }

        unsafe {
            nlm_accumulate::launch_unchecked::<R>(
                &self.client,
                ctx.thin_cube_count.clone(),
                ctx.thin_cube_dim,
                self.params.channels.storage_count() as usize,
                self.input_arg(ctx),
                self.accum_arg(ctx),
                self.weight_sum_arg(ctx),
                self.weight_buf_arg(ctx),
                self.weight_buf_arg(ctx),
                self.max_weight_arg(ctx),
                frame_t,
                frame_t,
                q_x,
                q_y,
                self.width,
                self.height,
            );
        }

        Ok(())
    }

    /// Run the per-submit motion-compensation sweep: estimate MVs from
    /// the centre to each of the `2·R` neighbours and warp them into
    /// `compensated_*_buf`. The centre slot is copied through
    /// unchanged so temporal kernels can read it uniformly. No-op
    /// when MC is inactive (or no neighbours).
    fn run_motion_compensation(&self, center_t: u32) -> Result<(), anyhow::Error> {
        let Some(mc) = self.mc_ctx.as_ref() else {
            return Ok(());
        };

        let temporal_radius = self.params.temporal_radius;
        if temporal_radius == 0 {
            return Ok(());
        }

        let frame_count = self.params.total_frames();
        let centre_slot = self.phys_frame(center_t as i32);
        let stored_ch = self.params.channels.storage_count();

        let pyramid_input = self
            .pyramid_input
            .as_ref()
            .expect("pyramid_input allocated when mc_ctx is Some");
        let mv_field = self
            .mv_field_buf
            .as_ref()
            .expect("mv_field allocated when mc_ctx is Some");
        let compensated_input = self
            .compensated_input_buf
            .as_ref()
            .expect("compensated_input allocated when mc_ctx is Some");
        // `confidence_buf` only exists when confidence weighting is
        // active (see `NlmDenoiser::new`). MC itself doesn't need it.
        // When it's absent, the fine kernel still requires some buffer
        // for its `confidence` argument, so fall back to the small
        // always-present dummy and tell the kernel not to write it.
        let (confidence_arg, write_confidence): (&Handle, bool) = match self.confidence_buf.as_ref() {
            Some(buf) => (buf, true),
            None => (&self.confidence_dummy, false),
        };
        let thsad_scale = self.params.hq.map_or(1.0, |hq| hq.thsad_scale);
        let sad_noise_floor = motion::sad_noise_floor(mc.blksize, self.sigma_y);
        let thsad = motion::thsad(mc.blksize, thsad_scale);

        // Centre frame: straight passthrough copy so the temporal
        // kernels can read it uniformly from the compensated buffer.
        copy_frame_into_slot_handle::<R>(
            &self.client,
            &self.input_buf,
            compensated_input,
            centre_slot as usize,
            self.width,
            self.height,
            stored_ch,
        );
        if let (Some(ref_src), Some(ref_dst)) = (
            self.reference_buf.as_ref(),
            self.compensated_reference_buf.as_ref(),
        ) {
            copy_frame_into_slot_handle::<R>(
                &self.client,
                ref_src,
                ref_dst,
                centre_slot as usize,
                self.width,
                self.height,
                stored_ch,
            );
        }

        // Use the cleaner of the two buffers for motion estimation:
        // the reference (prefiltered) pyramid when available.
        let analyse_pyramid = self.pyramid_reference.as_ref().unwrap_or(pyramid_input);

        // One analyse + warp per non-centre neighbour. Neighbours run
        // in logical order k = -R .. -1, +1 .. +R; their MV-field index
        // is contiguous so packing keeps the field tight.
        let radius = temporal_radius as i32;
        let mut neighbour_idx: u32 = 0;
        for k in -radius..=radius {
            if k == 0 {
                continue;
            }
            let neighbour_slot = self.phys_frame(center_t as i32 + k);

            run_analyse::<R>(
                &self.client,
                mc,
                self.width,
                self.height,
                frame_count,
                centre_slot,
                neighbour_slot,
                neighbour_idx,
                analyse_pyramid,
                mv_field,
                confidence_arg,
                write_confidence,
                sad_noise_floor,
                thsad,
            )?;

            run_compensate::<R>(
                &self.client,
                mc,
                self.params.channels.count(),
                stored_ch,
                self.width,
                self.height,
                frame_count,
                neighbour_slot,
                neighbour_idx,
                &self.input_buf,
                compensated_input,
                mv_field,
            )?;

            if let (Some(ref_src), Some(ref_dst)) = (
                self.reference_buf.as_ref(),
                self.compensated_reference_buf.as_ref(),
            ) {
                run_compensate::<R>(
                    &self.client,
                    mc,
                    self.params.channels.count(),
                    stored_ch,
                    self.width,
                    self.height,
                    frame_count,
                    neighbour_slot,
                    neighbour_idx,
                    ref_src,
                    ref_dst,
                    mv_field,
                )?;
            }

            neighbour_idx += 1;
        }

        Ok(())
    }

    /// Run the per-submit no-MC confidence sweep. A zero-search-radius
    /// block match from the centre frame to each of the `2·R`
    /// neighbours, writing per-block confidence into `confidence_buf`.
    /// No-op unless the no-MC confidence pass is active (`confidence_ctx`
    /// is `None` whenever motion compensation is active instead, or
    /// confidence is off entirely).
    fn run_confidence_pass(&self, center_t: u32) -> Result<(), anyhow::Error> {
        let Some(ctx) = self.confidence_ctx.as_ref() else {
            return Ok(());
        };

        // `confidence_ctx` is only ever constructed alongside
        // `temporal_radius > 0` (see `NlmDenoiser::new`), so
        // `temporal_radius` is guaranteed non-zero here.
        let temporal_radius = self.params.temporal_radius;

        let frame_count = self.params.total_frames();
        let centre_slot = self.phys_frame(center_t as i32);

        let luma_pyramid = self
            .confidence_pyramid
            .as_ref()
            .expect("confidence_pyramid allocated when confidence_ctx is Some");
        let mv_scratch = self
            .confidence_mv_scratch
            .as_ref()
            .expect("confidence_mv_scratch allocated when confidence_ctx is Some");
        let confidence_buf = self
            .confidence_buf
            .as_ref()
            .expect("confidence_buf allocated when confidence_ctx is Some");

        let thsad_scale = self.params.hq.map_or(1.0, |hq| hq.thsad_scale);
        let sad_noise_floor = motion::sad_noise_floor(ctx.blksize, self.sigma_y);
        let thsad = motion::thsad(ctx.blksize, thsad_scale);

        let radius = temporal_radius as i32;
        let mut neighbour_idx: u32 = 0;
        for k in -radius..=radius {
            if k == 0 {
                continue;
            }
            let neighbour_slot = self.phys_frame(center_t as i32 + k);

            run_confidence_for_neighbour::<R>(
                &self.client,
                ctx,
                self.width,
                self.height,
                frame_count,
                centre_slot,
                neighbour_slot,
                neighbour_idx,
                luma_pyramid,
                mv_scratch,
                confidence_buf,
                sad_noise_floor,
                thsad,
            )?;

            neighbour_idx += 1;
        }

        Ok(())
    }

    fn zero_accumulators(&self, ctx: &LaunchCtx) -> Result<(), anyhow::Error> {
        let grid = (ctx.frame_size as u32).div_ceil(BLOCK_1D).min(MAX_GRID_1D);
        let total_threads = grid * BLOCK_1D;
        unsafe {
            gpu_zero_buffers::launch_unchecked::<R>(
                &self.client,
                CubeCount::new_1d(grid),
                CubeDim::new_1d(BLOCK_1D),
                ArrayArg::from_raw_parts(self.accum.clone(), ctx.frame_size),
                self.weight_sum_arg(ctx),
                self.max_weight_arg(ctx),
                ctx.frame_size as u32,
                ctx.pixels as u32,
                total_threads,
            );
        }

        Ok(())
    }

    /// Shared `nlm_finish` launch, parameterized on where the result
    /// is written. `run_finish` targets an output slot. The nlm-spatial
    /// pilot targets a reference-ring slot instead.
    fn run_finish_to(
        &self,
        ctx: &LaunchCtx,
        center_frame: u32,
        output: ArrayArg<R>,
    ) -> Result<(), anyhow::Error> {
        let channels = self.params.channels.count();
        unsafe {
            nlm_finish::launch_unchecked::<R>(
                &self.client,
                ctx.cube_count.clone(),
                ctx.cube_dim,
                self.params.channels.storage_count() as usize,
                self.input_arg(ctx),
                output,
                ArrayArg::from_raw_parts(self.accum.clone(), ctx.frame_size),
                self.weight_sum_arg(ctx),
                self.max_weight_arg(ctx),
                center_frame,
                self.params.self_weight,
                self.width,
                self.height,
                channels,
            );
        }

        Ok(())
    }

    fn run_finish(&self, ctx: &LaunchCtx, center_t: u32, output_slot: usize) -> Result<(), anyhow::Error> {
        self.run_finish_to(
            ctx,
            self.phys_frame(center_t as i32),
            self.output_arg(ctx, output_slot),
        )
    }

    /// Derive the launch shapes shared by every per-frame dispatch
    /// (main pass and the nlm-spatial pilot alike).
    fn launch_ctx(&self) -> LaunchCtx {
        let width = self.width;
        let height = self.height;
        let stored_ch = self.params.channels.storage_count();
        let total_frames = self.params.total_frames();
        let pixels = (width * height) as usize;
        let frame_size = pixels * stored_ch as usize;

        LaunchCtx {
            total_frame_data: frame_size * total_frames as usize,
            frame_size,
            pixels,
            cube_count: CubeCount::new_2d(width.div_ceil(BLOCK_X), height.div_ceil(BLOCK_Y)),
            cube_dim: CubeDim::new_2d(BLOCK_X, BLOCK_Y),
            thin_cube_count: CubeCount::new_2d(width.div_ceil(BLOCK_X_THIN), height.div_ceil(BLOCK_Y_THIN)),
            thin_cube_dim: CubeDim::new_2d(BLOCK_X_THIN, BLOCK_Y_THIN),
        }
    }

    /// Denoise one freshly pushed frame with the windowed spatial
    /// kernel and store the result in its reference ring slot. Shares
    /// the frame-sized accumulators with the main pass. The in-order
    /// GPU queue makes that safe because the main dispatch zeroes them
    /// again before use.
    pub(super) fn run_nlm_spatial_pilot(&self, slot: u32, strength_scale: f32) -> Result<(), anyhow::Error> {
        let ctx = self.launch_ctx();
        self.zero_accumulators(&ctx)?;

        let channels = self.params.channels.count();
        let pilot_h2 = self.h2_inv_norm / (strength_scale * strength_scale);

        // Always read the noisy input here, never `reference_arg`: for
        // `NlmSpatial`, `reference_buf` is the pilot's own output, not
        // an input to it, even though `use_reference` is true so the
        // *main* pass's kernels pick the `_ref` variants.
        unsafe {
            nlm_fused_single_window::launch_unchecked::<R>(
                &self.client,
                ctx.cube_count.clone(),
                ctx.cube_dim,
                self.params.channels.storage_count() as usize,
                self.input_arg(&ctx),
                self.accum_arg(&ctx),
                self.weight_sum_arg(&ctx),
                self.max_weight_arg(&ctx),
                slot,
                pilot_h2,
                self.input_noise_offset,
                self.width,
                self.height,
                channels,
                self.params.patch_radius,
                self.params.search_radius,
                BLOCK_X,
                BLOCK_Y,
            );
        }

        self.run_finish_to(&ctx, slot, self.reference_slot_arg(&ctx, slot))
    }

    pub(super) fn run_denoise_kernels(&mut self, output_slot: usize) -> Result<(), anyhow::Error> {
        let temporal_radius = self.params.temporal_radius;
        let search_radius = self.params.search_radius as i32;

        let ctx = self.launch_ctx();

        let center_t = temporal_radius;

        // Motion compensation runs before any NLM dispatch so the
        // temporal kernels (k≠0) can read aligned neighbours from
        // `compensated_*_buf`. No-op when MC is inactive.
        self.run_motion_compensation(center_t)?;
        // No-op unless the no-MC confidence pass is active. This is
        // mutually exclusive with `run_motion_compensation` actually
        // doing anything, since `confidence_ctx` is only `Some` when
        // MC isn't.
        self.run_confidence_pass(center_t)?;

        self.zero_accumulators(&ctx)?;
        let window_side = 2 * search_radius + 1;
        let window_area = window_side * window_side;

        // The k≠0 temporal slices cover the full search window (every q
        // there has `linear < 0`), so the non-reference fused path takes
        // the windowed kernel: one launch per q_k that internally loops
        // over every (q_x, q_y). The k=0 slice still uses the per-q
        // half-window dispatch because its weight map is symmetric in q
        // and the single-tile path is cheaper per q.
        //
        // Reference-clip and separable paths still iterate per q until
        // a matching windowed variant is added.
        let k_start = -(temporal_radius as i32);
        let use_windowed = !self.use_separable;
        for q_k in k_start..=0 {
            if use_windowed {
                if q_k != 0 {
                    self.dispatch_fused_window_iter(&ctx, center_t, q_k)?;
                } else {
                    self.dispatch_fused_single_window_iter(&ctx, center_t)?;
                }
                continue;
            }

            for q_y in -search_radius..=search_radius {
                for q_x in -search_radius..=search_radius {
                    let linear = q_k * window_area + q_y * window_side + q_x;
                    if linear >= 0 {
                        continue;
                    }

                    if q_k == 0 {
                        if self.use_separable {
                            self.dispatch_separable_iter_k0(&ctx, center_t, q_x, q_y)?;
                        } else {
                            self.dispatch_fused_iter_k0(&ctx, center_t, q_x, q_y)?;
                        }
                    } else if self.use_separable {
                        self.dispatch_separable_iter(&ctx, center_t, q_x, q_y, q_k)?;
                    } else {
                        self.dispatch_fused_iter(&ctx, center_t, q_x, q_y, q_k)?;
                    }
                }
            }
        }

        self.run_finish(&ctx, center_t, output_slot)?;

        Ok(())
    }
}

/// GPU→GPU copy of one frame from `src`'s slot `slot` into `dst`'s
/// slot `slot`. Both buffers must share the same ring-buffer layout
/// (`total_frames * height * width * stored_ch`). Free function so
/// the motion-compensation dispatcher can call it without tying back
/// into the `NlmDenoiser` impl block (avoids borrow conflicts inside
/// the per-submit method).
fn copy_frame_into_slot_handle<R: Runtime>(
    client: &ComputeClient<R>,
    src: &Handle,
    dst: &Handle,
    slot: usize,
    width: u32,
    height: u32,
    stored_ch: u32,
) {
    let frame_size = width * height * stored_ch;
    let byte_offset = (slot as u64) * (frame_size as u64) * (size_of::<f32>() as u64);
    let src_handle = src.clone().offset_start(byte_offset);
    let dst_handle = dst.clone().offset_start(byte_offset);

    let grid = frame_size.div_ceil(BLOCK_1D).min(MAX_GRID_1D);
    let total_threads = grid * BLOCK_1D;

    unsafe {
        gpu_copy::launch_unchecked::<R>(
            client,
            CubeCount::new_1d(grid),
            CubeDim::new_1d(BLOCK_1D),
            ArrayArg::from_raw_parts(src_handle, frame_size as usize),
            ArrayArg::from_raw_parts(dst_handle, frame_size as usize),
            frame_size,
            total_threads,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::neighbour_idx_for_k;

    /// Walking `k = -radius..=radius` (skipping 0) and incrementing a
    /// counter from 0 must reproduce `neighbour_idx_for_k` exactly, for
    /// every radius. This is the same walk `run_motion_compensation`
    /// and `run_confidence_pass` use to fill `mv_field_buf` /
    /// `confidence_buf`, so it pins down the fill order those passes
    /// established.
    #[test]
    fn matches_the_sequential_fill_order() {
        for radius in 1..=8u32 {
            let mut expected = 0u32;
            for k in -(radius as i32)..=(radius as i32) {
                if k == 0 {
                    continue;
                }
                assert_eq!(neighbour_idx_for_k(radius, k), expected, "radius={radius} k={k}");
                expected += 1;
            }
        }
    }

    /// The forward slice (neighbour `center + q_k`) and backward slice
    /// (neighbour `center - q_k`) must always land on distinct,
    /// in-range indices. A mismatch here would silently apply the
    /// wrong frame's confidence to a temporal weight.
    #[test]
    fn forward_and_backward_indices_are_distinct_and_in_range() {
        for radius in 1..=8u32 {
            for q_k in -(radius as i32)..0 {
                let fwd = neighbour_idx_for_k(radius, q_k);
                let bwd = neighbour_idx_for_k(radius, -q_k);
                assert_ne!(fwd, bwd, "radius={radius} q_k={q_k}");
                assert!(fwd < 2 * radius, "radius={radius} q_k={q_k} fwd={fwd}");
                assert!(bwd < 2 * radius, "radius={radius} q_k={q_k} bwd={bwd}");
            }
        }
    }

    /// Explicit worked example at `radius = 2`, matching the doc
    /// comment's stated ordering. `k = -2, -1` fill indices `0, 1`,
    /// then `k = 1, 2` fill indices `2, 3`.
    #[test]
    fn radius_two_explicit_indices() {
        assert_eq!(neighbour_idx_for_k(2, -2), 0);
        assert_eq!(neighbour_idx_for_k(2, -1), 1);
        assert_eq!(neighbour_idx_for_k(2, 1), 2);
        assert_eq!(neighbour_idx_for_k(2, 2), 3);
    }
}
