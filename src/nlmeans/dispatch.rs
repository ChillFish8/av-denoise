//! Per-frame kernel dispatch for [`NlmDenoiser`]. Lives in its own
//! file so `denoiser.rs` can focus on lifecycle (state machine, ring
//! buffer, public API) and this module owns everything launch-related:
//! arg getters, the per-q dispatch helpers, motion-compensation sweep,
//! and the top-level `run_denoise_kernels` orchestrator.

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
use super::motion::{run_analyse, run_compensate};
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
    /// small-tile `nlm_dist_2d_weight(_ref)` kernels — see
    /// [`super::BLOCK_X_THIN`].
    pub(super) thin_cube_count: CubeCount,
    pub(super) thin_cube_dim: CubeDim,
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
                    frame_t,
                    frame_fwd,
                    frame_bwd,
                    q_x,
                    q_y,
                    bwd_shift_x,
                    bwd_shift_y,
                    self.h2_inv_norm,
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
                nlm_fused_pair_accumulate::launch_unchecked::<R>(
                    &self.client,
                    ctx.cube_count.clone(),
                    ctx.cube_dim,
                    self.params.channels.storage_count() as usize,
                    self.input_arg_for_temporal(ctx),
                    self.accum_arg(ctx),
                    self.weight_sum_arg(ctx),
                    self.max_weight_arg(ctx),
                    frame_t,
                    frame_fwd,
                    frame_bwd,
                    q_x,
                    q_y,
                    bwd_shift_x,
                    bwd_shift_y,
                    self.h2_inv_norm,
                    self.width,
                    self.height,
                    channels,
                    self.params.patch_radius,
                    BLOCK_X,
                    BLOCK_Y,
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
                    frame_t,
                    frame_fwd,
                    frame_bwd,
                    self.h2_inv_norm,
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
                nlm_fused_pair_accumulate_window::launch_unchecked::<R>(
                    &self.client,
                    ctx.cube_count.clone(),
                    ctx.cube_dim,
                    self.params.channels.storage_count() as usize,
                    self.input_arg_for_temporal(ctx),
                    self.accum_arg(ctx),
                    self.weight_sum_arg(ctx),
                    self.max_weight_arg(ctx),
                    frame_t,
                    frame_fwd,
                    frame_bwd,
                    self.h2_inv_norm,
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
                frame_fwd,
                frame_bwd,
                q_x,
                q_y,
                self.h2_inv_norm,
                self.width,
                self.height,
                self.params.patch_radius,
                BLOCK_X,
                BLOCK_Y,
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

    fn run_finish(&self, ctx: &LaunchCtx, center_t: u32, output_slot: usize) -> Result<(), anyhow::Error> {
        let channels = self.params.channels.count();
        unsafe {
            nlm_finish::launch_unchecked::<R>(
                &self.client,
                ctx.cube_count.clone(),
                ctx.cube_dim,
                self.params.channels.storage_count() as usize,
                self.input_arg(ctx),
                self.output_arg(ctx, output_slot),
                ArrayArg::from_raw_parts(self.accum.clone(), ctx.frame_size),
                self.weight_sum_arg(ctx),
                self.max_weight_arg(ctx),
                self.phys_frame(center_t as i32),
                self.params.self_weight,
                self.width,
                self.height,
                channels,
            );
        }

        Ok(())
    }

    pub(super) fn run_denoise_kernels(&mut self, output_slot: usize) -> Result<(), anyhow::Error> {
        let width = self.width;
        let height = self.height;
        let stored_ch = self.params.channels.storage_count();
        let temporal_radius = self.params.temporal_radius;
        let search_radius = self.params.search_radius as i32;
        let total_frames = self.params.total_frames();
        let pixels = (width * height) as usize;
        let frame_size = pixels * stored_ch as usize;

        let ctx = LaunchCtx {
            total_frame_data: frame_size * total_frames as usize,
            frame_size,
            pixels,
            cube_count: CubeCount::new_2d(width.div_ceil(BLOCK_X), height.div_ceil(BLOCK_Y)),
            cube_dim: CubeDim::new_2d(BLOCK_X, BLOCK_Y),
            thin_cube_count: CubeCount::new_2d(width.div_ceil(BLOCK_X_THIN), height.div_ceil(BLOCK_Y_THIN)),
            thin_cube_dim: CubeDim::new_2d(BLOCK_X_THIN, BLOCK_Y_THIN),
        };

        let center_t = temporal_radius;

        // Motion compensation runs before any NLM dispatch so the
        // temporal kernels (k≠0) can read aligned neighbours from
        // `compensated_*_buf`. No-op when MC is inactive.
        self.run_motion_compensation(center_t)?;

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
