use cubecl::prelude::*;
use cubecl::server::Handle;

use super::denoiser::NlmDenoiser;
use super::kernels::{
    gpu_copy,
    gpu_zero_buffers,
    gpu_zero_one,
    nlm_accumulate,
    nlm_distance,
    nlm_distance_pair,
    nlm_distance_pair_ref,
    nlm_distance_ref,
    nlm_finish,
    nlm_fused_pair_accumulate_window,
    nlm_fused_pair_accumulate_window_ref,
    nlm_fused_single_window,
    nlm_fused_single_window_ref,
    nlm_horizontal_sum,
    nlm_horizontal_sum_pair,
    nlm_vertical_weight,
    nlm_vweight_pair_accumulate,
    nlm_weight_ratio_partial,
};
use super::motion::{
    self,
    MotionCtx,
    MotionEstimation,
    confidence_byte_offset,
    run_analyse,
    run_compensate,
    run_confidence_for_neighbour,
    run_seeded_refine,
};
use super::noise::{build_spatial_offset_lut, spatial_offset_factor, spatial_offset_lut_len};
use super::prefilter::PrefilterMode;
use super::{BLOCK_1D, BLOCK_X, BLOCK_X_THIN, BLOCK_Y, BLOCK_Y_THIN, MAX_GRID_1D};

/// The sizes and grid shape one frame's work needs, bundled together so
/// the dispatch helpers do not each carry the same long argument list.
pub(super) struct LaunchCtx {
    pub(super) total_frame_data: usize,
    pub(super) frame_size: usize,
    pub(super) pixels: usize,
    pub(super) cube_count: CubeCount,
    pub(super) cube_dim: CubeDim,
    /// Alternate shape used by `nlm_accumulate`. See [`BLOCK_X_THIN`].
    pub(super) thin_cube_count: CubeCount,
    pub(super) thin_cube_dim: CubeDim,
}

/// Maps a nonzero temporal offset onto the neighbour index the analyse
/// and confidence passes use when filling their buffers.
///
/// Those passes fill the negative offsets first, taking indices 0 up to
/// the radius minus 1, then the positive ones. This is the inverse of
/// that walk.
///
/// See `NlmDenoiser::run_motion_compensation` and `run_confidence_pass`.
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

/// How much of the raw input's noise the `NlmSpatial` pilot's reference
/// image still carries.
///
/// This was measured rather than derived, by sweeping 0, 0.25, 0.5,
/// 0.75, and 1.0 against the HQ variant with the NLM prefilter and
/// motion compensation, at light noise levels on `clean-1080p.mkv`.
///
/// The sweep came back nearly flat, with at most 0.10 dB of spread
/// across the whole grid and both metrics agreeing on direction. Its
/// peak sat at 1.0, meaning no correction at all, which is the opposite
/// of the value chosen here.
///
/// 0 is kept anyway. A floor large enough to swamp `thsad` leaves
/// confidence unable to tell a genuine mismatch apart from noise, no
/// matter whether this one clip happens to show that failure. Because
/// the grid is flat, choosing 0 costs nothing measurable either.
const NLM_SPATIAL_RESIDUAL_FRACTION: f32 = 0.0;

/// How much of the raw input's noise the `Bilateral` prefilter's
/// reference image still carries.
///
/// This was measured the same way as
/// [`NLM_SPATIAL_RESIDUAL_FRACTION`], using the repo's own calibrated
/// `sigma_s` and `sigma_r` pair with motion compensation on.
///
/// Unlike the NLM pilot, this sweep landed cleanly on 0 at every noise
/// level tested, and the margin grew with the noise, from 0.16 dB at
/// the lightest level to 0.57 dB at the heaviest. So 0 here is a real
/// result rather than a judgement call.
///
/// It matches [`NLM_SPATIAL_RESIDUAL_FRACTION`] by coincidence of two
/// different lines of reasoning, not because the two prefilters were
/// shown to share a value. They stay separate constants for that
/// reason.
const BILATERAL_RESIDUAL_FRACTION: f32 = 0.0;

/// The sigma to hand [`motion::sad_noise_floor`] for the
/// motion-compensation block match.
///
/// `run_motion_compensation` matches against the reference pyramid
/// whenever one exists, not the raw input pyramid.
/// [`motion::sad_noise_floor`] models the score two raw noisy copies
/// would produce, so the raw sigma only belongs there when the match
/// really runs on raw pixels.
///
/// A prefilter that runs on the GPU, meaning the NLM pilot pass or the
/// bilateral blur, cleans the frame before the match ever sees it. The
/// raw floor therefore overstates the real one.
///
/// The consequences are not subtle. With the NLM pilot, the default
/// block size, and a sigma of 0.02, the raw floor alone comes to about
/// 5.78 against a threshold of 5.12. That swamps confidence's whole
/// range and pins it at 1.0 everywhere, including genuinely occluded
/// blocks.
///
/// # What is used instead
///
/// Rather than asserting a model of how much noise each prefilter
/// leaves behind, which would only swap one unverified guess for
/// another, the sigma used is the raw sigma times a residual fraction
/// measured per prefilter by a quality sweep. See
/// [`NLM_SPATIAL_RESIDUAL_FRACTION`] and
/// [`BILATERAL_RESIDUAL_FRACTION`].
///
/// 0 means the raw floor contributes nothing, and 1 is the old
/// behaviour this replaced. Those are the sweep grid's two endpoints.
///
/// An `External` reference comes from the caller with unknown noise, and
/// is not something this crate denoised, so it keeps the raw sigma just
/// as `PrefilterMode::None` does.
fn mc_sad_noise_floor_sigma(prefilter: PrefilterMode, sigma_y: f32) -> f32 {
    match prefilter {
        PrefilterMode::NlmSpatial { .. } => sigma_y * NLM_SPATIAL_RESIDUAL_FRACTION,
        PrefilterMode::Bilateral { .. } => sigma_y * BILATERAL_RESIDUAL_FRACTION,
        PrefilterMode::External | PrefilterMode::None => sigma_y,
    }
}

/// The confidence arguments for one temporal pair dispatch.
///
/// This carries whether confidence weighting is on, the forward and
/// backward per-block confidence views, and the block geometry the
/// kernel needs to map an output pixel onto its block. That mapping
/// mirrors the one `nlm_mc_warp` uses.
///
/// When confidence is off, this holds the small placeholder buffer, a
/// false flag, and geometry that is never read.
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

    /// The input array the temporal kernels read.
    ///
    /// With motion compensation active this is the compensated ring.
    /// Otherwise it is the same as [`Self::input_arg`].
    fn input_arg_for_temporal(&self, ctx: &LaunchCtx) -> ArrayArg<R> {
        match self.compensated_input_buf.as_ref() {
            Some(buf) => unsafe { ArrayArg::from_raw_parts(buf.clone(), ctx.total_frame_data) },
            None => self.input_arg(ctx),
        }
    }

    /// The reference array the temporal `_ref` kernels read, following
    /// the same rule as [`Self::input_arg_for_temporal`].
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

    /// The whole reference ring, for a kernel that picks a slot itself.
    ///
    /// Binding one slot instead would need its byte offset to land on
    /// one of the runtime's alignment boundaries, which a
    /// `width * height * stored_ch` frame stride cannot promise.
    fn reference_ring_arg(&self, ctx: &LaunchCtx) -> ArrayArg<R> {
        let buf = self
            .reference_buf
            .as_ref()
            .expect("reference buffer must exist for the nlm spatial pilot");
        unsafe { ArrayArg::from_raw_parts(buf.clone(), ctx.total_frame_data) }
    }

    fn weight_sum_arg(&self, ctx: &LaunchCtx) -> ArrayArg<R> {
        unsafe { ArrayArg::from_raw_parts(self.weight_sum.clone(), ctx.pixels) }
    }

    fn max_weight_arg(&self, ctx: &LaunchCtx) -> ArrayArg<R> {
        unsafe { ArrayArg::from_raw_parts(self.max_weight.clone(), ctx.pixels) }
    }

    /// The weight-squared accumulator argument follows the same
    /// dummy-buffer rule as [`Self::confidence_pair_args`]. It is the real
    /// buffer when `NlmParams::track_weight_sq` allocated one, or the
    /// tiny always-present placeholder otherwise.
    ///
    /// Every call site pairs this with `self.params.track_weight_sq`, so
    /// the placeholder is only ever bound alongside the flag that tells
    /// the kernel not to read or write it.
    fn weight_sq_sum_arg(&self, ctx: &LaunchCtx) -> ArrayArg<R> {
        match self.weight_sq_sum.as_ref() {
            Some(buf) => unsafe { ArrayArg::from_raw_parts(buf.clone(), ctx.pixels) },
            None => unsafe { ArrayArg::from_raw_parts(self.weight_sq_sum_dummy.clone(), 1) },
        }
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

    /// A view of `spatial_offset_lut`, sized for this denoiser's search
    /// radius.
    ///
    /// Only the spatial windowed kernels read it. Every other launch
    /// uses the flat `noise_offset` scalar instead.
    fn spatial_offset_lut_arg(&self) -> ArrayArg<R> {
        let len = spatial_offset_lut_len(self.params.search_radius);
        unsafe { ArrayArg::from_raw_parts(self.spatial_offset_lut.clone(), len) }
    }

    /// Builds the confidence arguments for one temporal pair, at an
    /// offset that is never zero at any call site.
    ///
    /// The forward frame reads the neighbour at a positive offset, so
    /// its confidence comes from that neighbour's slice. The backward
    /// frame reads the negative offset, so its confidence comes from the
    /// opposite slice. See `neighbour_idx_for_k`.
    ///
    /// Confidence weighting only runs when `confidence_buf` was
    /// allocated, which `NlmDenoiser::new` decides, and when block
    /// geometry exists, either from `mc_ctx` with motion compensation on
    /// or from `confidence_ctx` without it.
    ///
    /// Otherwise this falls back to the one-element `confidence_dummy`
    /// buffer with the flag off, so the kernel never reads it.
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

    /// The windowed fused step for a temporal neighbour.
    ///
    /// One launch covers every offset in the search window, keeping the
    /// accumulator, weight sum, and max weight in registers throughout.
    /// That collapses `(2 * search_radius + 1)^2` launches into one.
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
                    self.weight_sq_sum_arg(ctx),
                    confidence.conf_fwd,
                    confidence.conf_bwd,
                    confidence.use_confidence,
                    self.params.track_weight_sq,
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
                    self.weight_sq_sum_arg(ctx),
                    confidence.conf_fwd,
                    confidence.conf_bwd,
                    confidence.use_confidence,
                    self.params.track_weight_sq,
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

    /// The windowed fused step for the frame against itself.
    ///
    /// One launch covers every offset in the search window, walking it
    /// in a single direction. Patch distance reads the same either way,
    /// so a full window in one direction gives exactly the same total as
    /// a half window walked in both.
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
                    self.weight_sq_sum_arg(ctx),
                    self.params.track_weight_sq,
                    frame_t,
                    self.h2_inv_norm,
                    self.spatial_offset_lut_arg(),
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
                    self.weight_sq_sum_arg(ctx),
                    self.params.track_weight_sq,
                    frame_t,
                    self.h2_inv_norm,
                    self.spatial_offset_lut_arg(),
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

    /// The separable-path step for a temporal neighbour.
    ///
    /// It runs the paired distance, then the paired horizontal sums,
    /// then one fused kernel that finishes the vertical sum, the
    /// weights, and the accumulation together.
    ///
    /// That last kernel reads both horizontal-sum buffers itself, so no
    /// weight buffer is ever written to global memory.
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
                self.weight_sq_sum_arg(ctx),
                confidence.conf_fwd,
                confidence.conf_bwd,
                confidence.use_confidence,
                self.params.track_weight_sq,
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

    /// The separable-path step for the frame against itself.
    ///
    /// It runs the distance, the horizontal sums, the vertical sums and
    /// weights, and then the accumulation.
    ///
    /// The weight map reads the same in either direction, so one buffer
    /// serves both the forward and backward lookups.
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

        // The same correlation-adjusted offset the spatial windowed
        // kernel's table holds for this candidate. It is computed
        // directly rather than read back from the table, because this
        // dispatch already knows the exact candidate offset the
        // windowed kernel works out at compile time.
        let offset = self.noise_offset * spatial_offset_factor(q_x, q_y, self.rho_smoothed.unwrap_or(0.0));
        unsafe {
            nlm_vertical_weight::launch_unchecked::<R>(
                &self.client,
                ctx.cube_count.clone(),
                ctx.cube_dim,
                self.tmp_hsum_arg(ctx),
                self.weight_buf_arg(ctx),
                self.h2_inv_norm,
                offset,
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
                self.weight_sq_sum_arg(ctx),
                self.params.track_weight_sq,
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

    /// Works out how the neighbour at temporal offset `k` moved relative
    /// to the centre frame, and writes the result into `mv_field` at
    /// this neighbour's slot.
    ///
    /// Runs the chained compose-and-refine sequence when `Chained`
    /// estimation is active, or a direct coarse-to-fine match otherwise.
    /// `write_confidence` controls whether a per-block score also lands
    /// in `confidence_arg`, the same as `run_analyse` and
    /// `run_seeded_refine` document.
    ///
    /// Returns the neighbour's physical ring slot. This never shifts any
    /// buffer, so both [`Self::run_motion_compensation`], which follows
    /// it with `run_compensate`, and [`Self::run_motion_machinery`],
    /// which does not, can share it without either duplicating the
    /// estimate branch or paying for a shift neither one of them wants
    /// in the other's place.
    #[allow(clippy::too_many_arguments)]
    fn run_motion_estimate(
        &self,
        mc: &MotionCtx,
        analyse_pyramid: &Handle,
        mv_field: &Handle,
        confidence_arg: &Handle,
        write_confidence: bool,
        frame_count: u32,
        centre_slot: u32,
        center_t: u32,
        k: i32,
        neighbour_idx: u32,
        sad_noise_floor: f32,
        thsad: f32,
    ) -> Result<u32, anyhow::Error> {
        let neighbour_slot = self.phys_frame(center_t as i32 + k);

        if self.is_chained() {
            self.run_chain_compose(center_t, k)?;
            let refine_radius = match self
                .params
                .motion_compensation
                .resolved_estimation(self.params.temporal_radius)
            {
                Some(MotionEstimation::Chained { refine_radius }) => refine_radius,
                _ => unreachable!("is_chained() guarantees a resolved Chained estimation"),
            };

            run_seeded_refine::<R>(
                &self.client,
                mc,
                self.width,
                self.height,
                frame_count,
                centre_slot,
                neighbour_slot,
                neighbour_idx,
                refine_radius,
                analyse_pyramid,
                mv_field,
                confidence_arg,
                write_confidence,
                sad_noise_floor,
                thsad,
            )?;
        } else {
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
        }

        Ok(neighbour_slot)
    }

    /// Runs the same per-neighbour motion estimate
    /// [`Self::run_motion_compensation`] does, for every neighbour in
    /// the temporal window, but shifts nothing into the compensated
    /// buffers.
    ///
    /// Returns each neighbour's physical ring slot, in the same
    /// furthest-behind-to-furthest-ahead order their motion-field
    /// indices already follow, which is also the order
    /// `NlmDenoiser::submit_machinery` hands back through
    /// `RingView::neighbour_slots`.
    ///
    /// Returns an empty `Vec` when motion compensation is off, or when
    /// there are no neighbours.
    pub(super) fn run_motion_machinery(&self, center_t: u32) -> Result<Vec<u32>, anyhow::Error> {
        let Some(mc) = self.mc_ctx.as_ref() else {
            return Ok(Vec::new());
        };

        let temporal_radius = self.params.temporal_radius;
        if temporal_radius == 0 {
            return Ok(Vec::new());
        }

        let frame_count = self.params.total_frames();
        let centre_slot = self.phys_frame(center_t as i32);

        let pyramid_input = self
            .pyramid_input
            .as_ref()
            .expect("pyramid_input allocated when mc_ctx is Some");
        let mv_field = self
            .mv_field_buf
            .as_ref()
            .expect("mv_field allocated when mc_ctx is Some");
        // Same placeholder convention `run_motion_compensation` uses:
        // when confidence weighting is off, pass the small dummy buffer
        // and tell the kernel not to write it.
        let (confidence_arg, write_confidence): (&Handle, bool) = match self.confidence_buf.as_ref() {
            Some(buf) => (buf, true),
            None => (&self.confidence_dummy, false),
        };
        let thsad_scale = self.params.hq.map_or(1.0, |hq| hq.thsad_scale);
        let mc_sigma_y = mc_sad_noise_floor_sigma(self.params.prefilter, self.sigma_y);
        let sad_noise_floor = motion::sad_noise_floor(mc.blksize, mc_sigma_y);
        let thsad = motion::thsad(mc.blksize, thsad_scale);

        // Match against the cleaner of the two buffers, the same as
        // `run_motion_compensation`.
        let analyse_pyramid = self.pyramid_reference.as_ref().unwrap_or(pyramid_input);

        let radius = temporal_radius as i32;
        let mut neighbour_idx: u32 = 0;
        let mut slots = Vec::with_capacity((2 * temporal_radius) as usize);
        for k in -radius..=radius {
            if k == 0 {
                continue;
            }

            let neighbour_slot = self.run_motion_estimate(
                mc,
                analyse_pyramid,
                mv_field,
                confidence_arg,
                write_confidence,
                frame_count,
                centre_slot,
                center_t,
                k,
                neighbour_idx,
                sad_noise_floor,
                thsad,
            )?;
            slots.push(neighbour_slot);

            neighbour_idx += 1;
        }

        Ok(slots)
    }

    /// Runs the motion-compensation sweep for one submit.
    ///
    /// It estimates the motion from the centre frame to each neighbour
    /// and shifts them into the compensated buffers.
    ///
    /// The centre slot is copied through unchanged, so the temporal
    /// kernels can read every slot the same way.
    ///
    /// This does nothing when motion compensation is off, or when there
    /// are no neighbours.
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
        // `confidence_buf` only exists when confidence weighting is on,
        // which `NlmDenoiser::new` decides. Motion compensation itself
        // does not need it.
        //
        // The fine kernel still wants some buffer for its `confidence`
        // argument, so when there is none, pass the small always-present
        // dummy and tell the kernel not to write it.
        let (confidence_arg, write_confidence): (&Handle, bool) = match self.confidence_buf.as_ref() {
            Some(buf) => (buf, true),
            None => (&self.confidence_dummy, false),
        };
        let thsad_scale = self.params.hq.map_or(1.0, |hq| hq.thsad_scale);
        let mc_sigma_y = mc_sad_noise_floor_sigma(self.params.prefilter, self.sigma_y);
        let sad_noise_floor = motion::sad_noise_floor(mc.blksize, mc_sigma_y);
        let thsad = motion::thsad(mc.blksize, thsad_scale);

        // The centre frame is copied straight through, so the temporal
        // kernels can read every slot from the compensated buffer.
        copy_frame_into_slot_handle::<R>(
            &self.client,
            &self.input_buf,
            compensated_input,
            centre_slot as usize,
            self.params.total_frames(),
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
                self.params.total_frames(),
                self.width,
                self.height,
                stored_ch,
            );
        }

        // Match against the cleaner of the two buffers, which is the
        // prefiltered reference pyramid whenever one exists.
        let analyse_pyramid = self.pyramid_reference.as_ref().unwrap_or(pyramid_input);

        // One motion estimate and one shift per neighbour. They run in
        // order from the furthest behind to the furthest ahead, skipping
        // the centre, and their motion-field indices are contiguous so
        // the field stays tight.
        let radius = temporal_radius as i32;
        let mut neighbour_idx: u32 = 0;
        for k in -radius..=radius {
            if k == 0 {
                continue;
            }

            let neighbour_slot = self.run_motion_estimate(
                mc,
                analyse_pyramid,
                mv_field,
                confidence_arg,
                write_confidence,
                frame_count,
                centre_slot,
                center_t,
                k,
                neighbour_idx,
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

    /// Scores each neighbour against the centre frame without searching
    /// for motion, once per submit.
    ///
    /// Every block is matched where it stands, and the per-block score
    /// goes into `confidence_buf`.
    ///
    /// This only runs when confidence is on and motion compensation is
    /// off. `confidence_ctx` is `None` in every other case.
    fn run_confidence_pass(&self, center_t: u32) -> Result<(), anyhow::Error> {
        let Some(ctx) = self.confidence_ctx.as_ref() else {
            return Ok(());
        };

        // `confidence_ctx` is only ever built when the temporal radius
        // is above 0, which `NlmDenoiser::new` sees to, so the radius
        // cannot be zero here.
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

        // The weight-squared plane sits outside `gpu_zero_buffers`'
        // fixed three-buffer shape, because it only exists at all when
        // `NlmParams::track_weight_sq` is set.
        if let Some(buf) = self.weight_sq_sum.as_ref() {
            let grid_w = (ctx.pixels as u32).div_ceil(BLOCK_1D).min(MAX_GRID_1D);
            let total_threads_w = grid_w * BLOCK_1D;
            unsafe {
                gpu_zero_one::launch_unchecked::<R>(
                    &self.client,
                    CubeCount::new_1d(grid_w),
                    CubeDim::new_1d(BLOCK_1D),
                    ArrayArg::from_raw_parts(buf.clone(), ctx.pixels),
                    ctx.pixels as u32,
                    total_threads_w,
                );
            }
        }

        Ok(())
    }

    /// The shared `nlm_finish` launch, with the destination left to the
    /// caller.
    ///
    /// `run_finish` writes to an output slot, while the NLM pilot writes
    /// to a reference-ring slot instead.
    fn run_finish_to(
        &self,
        ctx: &LaunchCtx,
        center_frame: u32,
        output_frame: u32,
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
                output_frame,
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
            0,
            self.output_arg(ctx, output_slot),
        )
    }

    /// Works out the launch shapes every per-frame dispatch shares,
    /// covering both the main pass and the NLM pilot.
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

    /// Denoises a freshly pushed frame with the windowed spatial kernel
    /// and stores the result in its reference-ring slot.
    ///
    /// This shares the frame-sized accumulators with the main pass. The
    /// GPU queue runs in order and the main dispatch zeroes them again
    /// before use, so sharing them is safe.
    pub(super) fn run_nlm_spatial_pilot(&self, slot: u32, strength_scale: f32) -> Result<(), anyhow::Error> {
        let ctx = self.launch_ctx();
        self.zero_accumulators(&ctx)?;

        let channels = self.params.channels.count();
        let pilot_h2 = self.h2_inv_norm / (strength_scale * strength_scale);

        // A flat table with no correlation adjustment. The pilot
        // compares noisy input patches directly and keeps the full
        // white-noise floor, which puts it outside the scope of that
        // adjustment. The denoiser's `input_noise_offset` doc explains
        // why.
        //
        // It is built fresh each call rather than cached, because
        // `input_noise_offset` can change between pushes and this is a
        // tiny one-off upload.
        let pilot_lut = build_spatial_offset_lut(self.params.search_radius, 0.0, self.input_noise_offset);
        let pilot_lut_handle = self.client.create_from_slice(f32::as_bytes(&pilot_lut));
        let pilot_lut_arg = unsafe { ArrayArg::<R>::from_raw_parts(pilot_lut_handle, pilot_lut.len()) };

        // Always read the noisy input here, never `reference_arg`. For
        // `NlmSpatial` the reference buffer is the pilot's own output
        // rather than an input to it, even though `use_reference` is
        // true so the main pass picks the `_ref` kernels.
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
                self.weight_sq_sum_arg(&ctx),
                self.params.track_weight_sq,
                slot,
                pilot_h2,
                pilot_lut_arg,
                self.width,
                self.height,
                channels,
                self.params.patch_radius,
                self.params.search_radius,
                BLOCK_X,
                BLOCK_Y,
            );
        }

        self.run_finish_to(&ctx, slot, slot, self.reference_ring_arg(&ctx))
    }

    pub(super) fn run_denoise_kernels(&mut self, output_slot: usize) -> Result<(), anyhow::Error> {
        let temporal_radius = self.params.temporal_radius;
        let search_radius = self.params.search_radius as i32;

        let ctx = self.launch_ctx();

        let center_t = temporal_radius;

        // Motion compensation runs before any NLM dispatch, so the
        // temporal kernels can read already-aligned neighbours from the
        // compensated buffers. It does nothing when motion compensation
        // is off.
        self.run_motion_compensation(center_t)?;
        // This does nothing unless the confidence pass without motion
        // compensation is active. The two never both run, because
        // `confidence_ctx` is only `Some` when motion compensation is
        // off.
        self.run_confidence_pass(center_t)?;

        self.zero_accumulators(&ctx)?;
        let window_side = 2 * search_radius + 1;
        let window_area = window_side * window_side;

        // Every temporal neighbour covers the full search window, so the
        // plain fused path uses the windowed kernel, one launch per
        // neighbour that loops over the offsets itself.
        //
        // The frame against itself still dispatches one offset at a
        // time over half the window, because its weight map reads the
        // same in either direction and the single-tile path is cheaper
        // per offset.
        //
        // The reference-image and separable paths also go one offset at
        // a time, until they gain windowed kernels of their own.
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
                        self.dispatch_separable_iter_k0(&ctx, center_t, q_x, q_y)?;
                    } else {
                        self.dispatch_separable_iter(&ctx, center_t, q_x, q_y, q_k)?;
                    }
                }
            }
        }

        self.run_finish(&ctx, center_t, output_slot)?;

        Ok(())
    }

    /// sqrt(mean over pixels of (sum w^2) / (sum w)^2) for the last
    /// submitted frame, including the centre self weight.
    ///
    /// 1.0 means the window collapsed to a single sample and no noise was
    /// removed there. The value falls as more independent samples join
    /// the average, which is what a second-stage filter needs to know
    /// how much noise this pass actually left behind. See
    /// `nlm_weight_ratio_partial` for the derivation.
    ///
    /// Only meaningful right after a submit made with
    /// `NlmParams::track_weight_sq` set, because that is what keeps
    /// `weight_sq_sum` allocated and filled in. Returns an error
    /// otherwise. This performs a small blocking readback, one `f32` per
    /// dispatched block rather than per pixel.
    ///
    /// This dispatches its own reduction kernel and blocks right behind
    /// it. A caller that has more GPU work to queue for the same frame
    /// should use [`Self::residual_ratio_sqrt_submit`] and
    /// [`Self::read_residual_ratio_sqrt`] instead, so the blocking read
    /// lands on work the GPU has already finished, the same pattern
    /// [`Self::update_noise_estimate`] uses for its own readback.
    pub fn residual_ratio_sqrt(&self) -> Result<f32, anyhow::Error> {
        let pending = self.residual_ratio_sqrt_submit()?;
        self.read_residual_ratio_sqrt(pending)
    }

    /// Queues the residual-ratio reduction kernel for the last submitted
    /// frame and returns a handle to its still-unread result, without
    /// blocking on it.
    ///
    /// Pair this with [`Self::read_residual_ratio_sqrt`] once the caller
    /// has queued more GPU work in between, so the readback lands on
    /// work the GPU has already finished instead of stalling the queue
    /// right behind this dispatch.
    ///
    /// Same `track_weight_sq` requirement as [`Self::residual_ratio_sqrt`].
    pub fn residual_ratio_sqrt_submit(&self) -> Result<PendingResidualRatio, anyhow::Error> {
        let weight_sq_sum = self.weight_sq_sum.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "residual_ratio_sqrt_submit needs NlmParams::track_weight_sq set, so \
                 weight_sq_sum was never allocated"
            )
        })?;

        let ctx = self.launch_ctx();
        let grid = (ctx.pixels as u32).div_ceil(BLOCK_1D).min(MAX_GRID_1D);
        let total_threads = grid * BLOCK_1D;
        let partials = self.client.empty(grid as usize * size_of::<f32>());

        unsafe {
            nlm_weight_ratio_partial::launch_unchecked::<R>(
                &self.client,
                CubeCount::new_1d(grid),
                CubeDim::new_1d(BLOCK_1D),
                self.weight_sum_arg(&ctx),
                ArrayArg::from_raw_parts(weight_sq_sum.clone(), ctx.pixels),
                self.max_weight_arg(&ctx),
                ArrayArg::from_raw_parts(partials.clone(), grid as usize),
                self.params.self_weight,
                ctx.pixels as u32,
                total_threads,
                BLOCK_1D,
            );
        }

        Ok(PendingResidualRatio {
            partials,
            pixels: ctx.pixels,
        })
    }

    /// Reads back a reduction queued by [`Self::residual_ratio_sqrt_submit`].
    ///
    /// Blocks until the GPU finishes the reduction. The wait is cheap
    /// only when the caller queued other GPU work between the submit
    /// and this call, giving the reduction time to finish on its own.
    pub fn read_residual_ratio_sqrt(&self, pending: PendingResidualRatio) -> Result<f32, anyhow::Error> {
        let bytes = self
            .client
            .read_one(pending.partials)
            .map_err(|e| anyhow::anyhow!("residual ratio partials readback failed: {e}"))?;
        let data = f32::from_bytes(&bytes);
        let sum: f32 = data.iter().sum();
        let mean = sum / pending.pixels as f32;
        Ok(mean.sqrt())
    }
}

/// A residual-ratio reduction dispatched by
/// [`NlmDenoiser::residual_ratio_sqrt_submit`] but not yet read back.
pub struct PendingResidualRatio {
    partials: Handle,
    pixels: usize,
}

/// Copies one frame from a slot of `src` into the same slot of `dst`,
/// entirely on the GPU.
///
/// Both buffers have to share the same ring layout.
///
/// This is a free function rather than a method, so the
/// motion-compensation dispatcher can call it without borrowing the
/// denoiser again inside the per-submit method.
///
/// Both rings are bound whole and the kernel picks the slot itself, for
/// the alignment reason [`NlmDenoiser::copy_frame_into_slot`] explains.
#[allow(clippy::too_many_arguments)]
fn copy_frame_into_slot_handle<R: Runtime>(
    client: &ComputeClient<R>,
    src: &Handle,
    dst: &Handle,
    slot: usize,
    frame_count: u32,
    width: u32,
    height: u32,
    stored_ch: u32,
) {
    let frame_size = width * height * stored_ch;
    let ring_len = frame_count as usize * frame_size as usize;
    let offset = slot as u32 * frame_size;

    let grid = frame_size.div_ceil(BLOCK_1D).min(MAX_GRID_1D);
    let total_threads = grid * BLOCK_1D;

    unsafe {
        gpu_copy::launch_unchecked::<R>(
            client,
            CubeCount::new_1d(grid),
            CubeDim::new_1d(BLOCK_1D),
            ArrayArg::from_raw_parts(src.clone(), ring_len),
            ArrayArg::from_raw_parts(dst.clone(), ring_len),
            offset,
            offset,
            frame_size,
            total_threads,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BILATERAL_RESIDUAL_FRACTION,
        NLM_SPATIAL_RESIDUAL_FRACTION,
        PrefilterMode,
        mc_sad_noise_floor_sigma,
        neighbour_idx_for_k,
    };

    /// Walking every offset in order, skipping zero, and counting up
    /// from 0 has to reproduce `neighbour_idx_for_k` exactly at every
    /// radius.
    ///
    /// That is the same walk `run_motion_compensation` and
    /// `run_confidence_pass` use to fill their buffers, so this pins
    /// down the order those passes established.
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

    /// The forward and backward slices must always land on different
    /// indices, both in range.
    ///
    /// Getting that wrong would quietly apply one frame's confidence to
    /// another frame's temporal weight.
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

    #[test]
    fn radius_two_explicit_indices() {
        assert_eq!(neighbour_idx_for_k(2, -2), 0);
        assert_eq!(neighbour_idx_for_k(2, -1), 1);
        assert_eq!(neighbour_idx_for_k(2, 1), 2);
        assert_eq!(neighbour_idx_for_k(2, 2), 3);
    }

    /// Pins the calibrated constant to a literal rather than to a value
    /// worked out from the constant itself.
    ///
    /// That way a future recalibration fails this test instead of
    /// quietly passing again.
    #[test]
    fn nlm_spatial_residual_fraction_is_calibrated_to_zero() {
        assert_eq!(NLM_SPATIAL_RESIDUAL_FRACTION, 0.0);
    }

    #[test]
    fn bilateral_residual_fraction_is_calibrated_to_zero() {
        assert_eq!(BILATERAL_RESIDUAL_FRACTION, 0.0);
    }

    #[test]
    fn mc_sad_noise_floor_sigma_scales_nlm_spatial_by_the_calibrated_fraction() {
        let raw = 0.02f32;
        assert_eq!(
            mc_sad_noise_floor_sigma(PrefilterMode::NlmSpatial { strength_scale: 1.0 }, raw),
            raw * NLM_SPATIAL_RESIDUAL_FRACTION
        );
    }

    #[test]
    fn mc_sad_noise_floor_sigma_scales_bilateral_by_the_calibrated_fraction() {
        let raw = 0.02f32;
        assert_eq!(
            mc_sad_noise_floor_sigma(
                PrefilterMode::Bilateral {
                    sigma_s: 3.0,
                    sigma_r: 0.02
                },
                raw
            ),
            raw * BILATERAL_RESIDUAL_FRACTION
        );
    }

    #[test]
    fn mc_sad_noise_floor_sigma_keeps_raw_sigma_for_none_and_external() {
        let raw = 0.02f32;
        assert_eq!(mc_sad_noise_floor_sigma(PrefilterMode::None, raw), raw);
        assert_eq!(mc_sad_noise_floor_sigma(PrefilterMode::External, raw), raw);
    }
}
