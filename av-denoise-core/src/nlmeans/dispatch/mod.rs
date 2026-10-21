mod motion;

use cubecl::prelude::*;
use cubecl::server::Handle;

pub(super) use self::motion::mc_sad_noise_floor_sigma;
#[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
pub(super) use self::motion::{BILATERAL_RESIDUAL_FRACTION, NLM_SPATIAL_RESIDUAL_FRACTION};
use super::denoiser::NlmDenoiser;
use super::kernels::{
    gpu_zero_buffers,
    nlm_distance,
    nlm_distance_pair,
    nlm_distance_pair_ref,
    nlm_distance_ref,
    nlm_finish,
    nlm_horizontal_sum,
    nlm_horizontal_sum_pair,
    nlm_vertical_weight,
    nlm_vweight_pair_accumulate,
};
use super::motion::{confidence_byte_offset, neighbour_idx_for_k};
use super::noise::{build_spatial_offset_lut, spatial_offset_factor, spatial_offset_lut_len};
use super::{BLOCK_1D, BLOCK_X, BLOCK_Y, MAX_GRID_1D};
use crate::engine::timing::{timed, units_per_span};
use crate::tune::gpu::bind;
use crate::tune::sites::{AccumOutputs, AccumulateSite, ConfidenceBinding, WindowPass, WindowSite};

/// The most dispatches one separable search offset queues.
const SEPARABLE_DISPATCHES_PER_OFFSET: u32 = 4;

/// The sizes and tile launch shape one frame's dispatches share.
///
/// `cube_count` and `cube_dim` are the tile shape of the separable and finish kernels.
pub(super) struct LaunchCtx {
    pub(super) total_frame_data: usize,
    pub(super) frame_size: usize,
    pub(super) pixels: usize,
    pub(super) cube_count: CubeCount,
    pub(super) cube_dim: CubeDim,
}

impl<R: Runtime> NlmDenoiser<R> {
    fn input_arg(&self, ctx: &LaunchCtx) -> ArrayArg<R> {
        bind::<R>(&self.input_buf, ctx.total_frame_data)
    }

    fn reference_arg(&self, ctx: &LaunchCtx) -> ArrayArg<R> {
        let reference = self
            .plain_reference()
            .expect("reference ring must exist when this kernel runs");
        bind::<R>(&reference, ctx.total_frame_data)
    }

    /// The input ring the temporal kernels read, which is the compensated one under motion
    /// compensation.
    fn input_arg_for_temporal(&self, ctx: &LaunchCtx) -> ArrayArg<R> {
        let input = self.temporal_input();
        bind::<R>(&input, ctx.total_frame_data)
    }

    /// The reference ring the temporal `_ref` kernels read, which is the compensated one under
    /// motion compensation.
    fn reference_arg_for_temporal(&self, ctx: &LaunchCtx) -> ArrayArg<R> {
        let reference = self
            .temporal_reference()
            .expect("reference ring must exist when this kernel runs");
        bind::<R>(&reference, ctx.total_frame_data)
    }

    fn accum_arg(&self, ctx: &LaunchCtx) -> ArrayArg<R> {
        unsafe { ArrayArg::from_raw_parts(self.accum.clone(), ctx.frame_size) }
    }

    fn output_arg(&self, ctx: &LaunchCtx, slot: usize) -> ArrayArg<R> {
        unsafe { ArrayArg::from_raw_parts(self.outputs[slot].clone(), ctx.frame_size) }
    }

    /// The whole reference ring, for a kernel that picks its slot itself.
    ///
    /// Binding one slot would need its byte offset to meet the runtime's alignment, which a
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

    fn accum_outputs(&self, ctx: &LaunchCtx) -> AccumOutputs {
        AccumOutputs {
            accum: self.accum.clone(),
            weight_sum: self.weight_sum.clone(),
            max_weight: self.max_weight.clone(),
            accum_len: ctx.frame_size,
            pixels: ctx.pixels,
        }
    }

    /// The input ring the temporal kernels read, which is the compensated one under motion
    /// compensation.
    fn temporal_input(&self) -> Handle {
        let compensated = self.compensated_input_buf.as_ref();
        compensated.unwrap_or(&self.input_buf).clone()
    }

    /// The reference ring the frame-against-itself kernels read, when the pipeline has one.
    fn plain_reference(&self) -> Option<Handle> {
        if !self.use_reference {
            return None;
        }

        let reference = self
            .reference_buf
            .as_ref()
            .expect("reference buffer must exist when use_reference is set");
        Some(reference.clone())
    }

    /// The reference ring the temporal kernels read, which is the compensated one under motion
    /// compensation.
    fn temporal_reference(&self) -> Option<Handle> {
        if !self.use_reference {
            return None;
        }

        let compensated = self.compensated_reference_buf.as_ref();
        let reference = compensated.or(self.reference_buf.as_ref());
        let reference = reference.expect("reference buffer must exist when use_reference is set");
        Some(reference.clone())
    }

    /// Binds the confidence slices for the temporal pair at the nonzero offset `q_k`.
    ///
    /// The forward frame reads the neighbour at `+q_k` and the backward frame the one at `-q_k`,
    /// so each takes its confidence from that neighbour's slice. Weighting runs only when
    /// `confidence_buf` exists and block geometry is available. Otherwise the binding is disabled and
    /// both slices are the placeholder the kernel never reads.
    fn confidence_binding(&self, q_k: i32) -> ConfidenceBinding {
        let confidence_geometry = self.confidence_ctx.as_ref();
        let geometry = self.mc_ctx.as_ref().or(confidence_geometry);

        if let (Some(confidence_buf), Some(geometry)) = (self.confidence_buf.as_ref(), geometry) {
            let radius = self.params.temporal_radius;
            let forward_idx = neighbour_idx_for_k(radius, q_k);
            let backward_idx = neighbour_idx_for_k(radius, -q_k);
            let forward_offset = confidence_byte_offset(geometry, forward_idx);
            let backward_offset = confidence_byte_offset(geometry, backward_idx);
            let forward = confidence_buf.clone().offset_start(forward_offset);
            let backward = confidence_buf.clone().offset_start(backward_offset);

            return ConfidenceBinding {
                enabled: true,
                forward,
                backward,
                len: (geometry.blocks_x * geometry.blocks_y) as usize,
                step: geometry.step,
                blocks_x: geometry.blocks_x,
                blocks_y: geometry.blocks_y,
            };
        }

        ConfidenceBinding {
            enabled: false,
            forward: self.confidence_dummy.clone(),
            backward: self.confidence_dummy.clone(),
            len: 1,
            step: 1,
            blocks_x: 1,
            blocks_y: 1,
        }
    }

    /// A windowed launch over `input` that adds into the frame accumulators.
    fn window_site(
        &self,
        ctx: &LaunchCtx,
        pass: WindowPass,
        input: Handle,
        reference: Option<Handle>,
        frame_t: u32,
        h2_inv_norm: f32,
    ) -> WindowSite<R> {
        let outputs = self.accum_outputs(ctx);

        WindowSite {
            client: self.client.clone(),
            pass,
            input,
            reference,
            ring_len: ctx.total_frame_data,
            outputs,
            frame_t,
            h2_inv_norm,
            width: self.width,
            height: self.height,
            channels: self.params.channels.count(),
            stored_ch: self.params.channels.storage_count(),
            patch_radius: self.params.patch_radius,
            search_radius: self.params.search_radius,
        }
    }

    /// The windowed fused step for a temporal neighbour.
    ///
    /// One launch covers the whole search window and keeps the accumulators in registers, in place
    /// of `(2 * search_radius + 1)^2` launches.
    fn dispatch_fused_window_iter(
        &self,
        ctx: &LaunchCtx,
        center_t: u32,
        q_k: i32,
    ) -> Result<(), anyhow::Error> {
        let frame_t = self.phys_frame(center_t as i32);
        let frame_fwd = self.phys_frame(center_t as i32 + q_k);
        let frame_bwd = self.phys_frame(center_t as i32 - q_k);
        let confidence = self.confidence_binding(q_k);
        let pass = WindowPass::Pair {
            frame_fwd,
            frame_bwd,
            noise_offset: self.noise_offset,
            confidence,
        };
        let input = self.temporal_input();
        let reference = self.temporal_reference();
        let site = self.window_site(ctx, pass, input, reference, frame_t, self.h2_inv_norm);

        self.kernels.run(&site)
    }

    /// The windowed fused step for the frame against itself.
    ///
    /// Patch distance is symmetric, so the full window walked in one direction gives the same total
    /// as a half window walked both ways.
    fn dispatch_fused_single_window_iter(&self, ctx: &LaunchCtx, center_t: u32) -> Result<(), anyhow::Error> {
        let frame_t = self.phys_frame(center_t as i32);
        let offsets_len = spatial_offset_lut_len(self.params.search_radius);
        let pass = WindowPass::Single {
            offsets: self.spatial_offset_lut.handle().clone(),
            offsets_len,
        };
        let input = self.input_buf.clone();
        let reference = self.plain_reference();
        let site = self.window_site(ctx, pass, input, reference, frame_t, self.h2_inv_norm);

        self.kernels.run(&site)
    }

    /// The separable step for a temporal neighbour.
    ///
    /// The last kernel finishes the vertical sum, the weights and the accumulation from both
    /// horizontal-sum buffers, so no weight buffer reaches global memory.
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

        let confidence = self.confidence_binding(q_k);
        let conf_fwd = bind::<R>(&confidence.forward, confidence.len);
        let conf_bwd = bind::<R>(&confidence.backward, confidence.len);

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
                conf_fwd,
                conf_bwd,
                confidence.enabled,
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

    /// The separable step for the frame against itself.
    ///
    /// The weight map is symmetric, so one buffer serves both the forward and backward lookups.
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

        // The same correlation-adjusted offset the spatial table holds for this candidate, computed
        // directly because the candidate is already known here.
        let rho = self.rho_smoothed.unwrap_or(0.0);
        let offset_factor = spatial_offset_factor(q_x, q_y, rho);
        let offset = self.noise_offset * offset_factor;
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

        let outputs = self.accum_outputs(ctx);
        let site = AccumulateSite {
            client: self.client.clone(),
            input: self.input_buf.clone(),
            ring_len: ctx.total_frame_data,
            weights: self.weight_buf.clone(),
            outputs,
            frame: frame_t,
            q_x,
            q_y,
            width: self.width,
            height: self.height,
            stored_ch: self.params.channels.storage_count(),
        };

        self.kernels.run(&site)
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

    /// Launches `nlm_finish` into a destination the caller picks.
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
        let center_frame = self.phys_frame(center_t as i32);
        let output = self.output_arg(ctx, output_slot);

        self.run_finish_to(ctx, center_frame, 0, output)
    }

    /// The sizes and tile shape every per-frame dispatch shares, for the main pass and the NLM pilot.
    fn launch_ctx(&self) -> LaunchCtx {
        let width = self.width;
        let height = self.height;
        let stored_ch = self.params.channels.storage_count();
        let total_frames = self.params.total_frames();
        let pixels = (width * height) as usize;
        let frame_size = pixels * stored_ch as usize;
        let cubes_x = width.div_ceil(BLOCK_X);
        let cubes_y = height.div_ceil(BLOCK_Y);

        LaunchCtx {
            total_frame_data: frame_size * total_frames as usize,
            frame_size,
            pixels,
            cube_count: CubeCount::new_2d(cubes_x, cubes_y),
            cube_dim: CubeDim::new_2d(BLOCK_X, BLOCK_Y),
        }
    }

    /// Denoises a pushed frame with the windowed spatial kernel into its reference-ring slot.
    ///
    /// It shares the frame accumulators with the main pass. That is safe because the GPU queue runs
    /// in order and the main pass zeroes them before use.
    pub(super) fn run_nlm_spatial_pilot(
        &mut self,
        slot: u32,
        strength_scale: f32,
    ) -> Result<(), anyhow::Error> {
        // A flat table with no correlation adjustment, because the pilot compares noisy input
        // patches. Any upload happens before the first dispatch, because an upload ends a device
        // span's timed pass.
        let pilot_lut = build_spatial_offset_lut(self.params.search_radius, 0.0, self.input_noise_offset);
        let pilot_lut_bytes = f32::as_bytes(&pilot_lut);
        self.pilot_offset_lut.update(&self.client, pilot_lut_bytes);
        let pilot_lut_handle = self.pilot_offset_lut.handle().clone();

        let ctx = self.launch_ctx();
        self.zero_accumulators(&ctx)?;

        let pilot_h2 = self.h2_inv_norm / (strength_scale * strength_scale);
        let pass = WindowPass::Single {
            offsets: pilot_lut_handle,
            offsets_len: pilot_lut.len(),
        };
        let input = self.input_buf.clone();

        // Always the noisy input. For `NlmSpatial` the reference ring is this pass's output, even
        // though `use_reference` is set so the main pass picks the `_ref` kernels.
        let site = self.window_site(&ctx, pass, input, None, slot, pilot_h2);
        self.kernels.run(&site)?;

        let reference_ring = self.reference_ring_arg(&ctx);
        self.run_finish_to(&ctx, slot, slot, reference_ring)
    }

    /// Queues every denoise kernel for the current window, in device spans when sampled.
    pub(super) fn run_denoise_kernels(&mut self, output_slot: usize) -> Result<(), anyhow::Error> {
        let center_t = self.params.temporal_radius;
        let client = self.client.clone();

        // Only the motion and confidence launches read the table, and the upload comes before
        // their first dispatch because an upload ends a device span's timed pass.
        if self.mc_ctx.is_some() || self.confidence_ctx.is_some() {
            let neighbour_slots = self.window_neighbour_slots(center_t);
            self.update_neighbour_slots(&neighbour_slots);
        }

        // Both run before any NLM dispatch so the temporal kernels read aligned neighbours.
        // `confidence_ctx` only exists with motion compensation off, so at most one does work.
        self.run_motion_compensation(center_t)?;

        timed(&client, self, move |front| front.run_confidence_pass(center_t))?;

        if self.use_separable {
            return self.run_separable_kernels(output_slot);
        }

        timed(&client, self, move |front| {
            front.run_windowed_kernels(output_slot)
        })
    }

    /// Queues the windowed path, which makes one launch per temporal offset.
    fn run_windowed_kernels(&self, output_slot: usize) -> Result<(), anyhow::Error> {
        let temporal_radius = self.params.temporal_radius;
        let center_t = temporal_radius;
        let ctx = self.launch_ctx();

        self.zero_accumulators(&ctx)?;

        let k_start = -(temporal_radius as i32);
        for q_k in k_start..0 {
            self.dispatch_fused_window_iter(&ctx, center_t, q_k)?;
        }

        self.dispatch_fused_single_window_iter(&ctx, center_t)?;
        self.run_finish(&ctx, center_t, output_slot)
    }

    /// Queues the separable path, which launches per search offset.
    ///
    /// The offsets run in chunks that each fit one device span.
    fn run_separable_kernels(&mut self, output_slot: usize) -> Result<(), anyhow::Error> {
        let center_t = self.params.temporal_radius;
        let client = self.client.clone();

        timed(&client, self, |front| {
            let ctx = front.launch_ctx();
            front.zero_accumulators(&ctx)
        })?;

        let offsets = self.separable_offsets();
        let per_span = units_per_span(SEPARABLE_DISPATCHES_PER_OFFSET);
        for chunk in offsets.chunks(per_span) {
            timed(&client, self, move |front| {
                front.dispatch_separable_offsets(center_t, chunk)
            })?;
        }

        timed(&client, self, move |front| {
            let ctx = front.launch_ctx();
            front.run_finish(&ctx, center_t, output_slot)
        })
    }

    /// Every `(q_k, q_y, q_x)` search offset the separable path visits, in launch order.
    ///
    /// `linear < 0` keeps half of the frame-against-itself window, because its weights are
    /// symmetric.
    fn separable_offsets(&self) -> Vec<(i32, i32, i32)> {
        let temporal_radius = self.params.temporal_radius;
        let search_radius = self.params.search_radius as i32;
        let window_side = 2 * search_radius + 1;
        let window_area = window_side * window_side;
        let k_start = -(temporal_radius as i32);

        let mut offsets = Vec::new();
        for q_k in k_start..=0 {
            for q_y in -search_radius..=search_radius {
                for q_x in -search_radius..=search_radius {
                    let linear = q_k * window_area + q_y * window_side + q_x;
                    if linear < 0 {
                        offsets.push((q_k, q_y, q_x));
                    }
                }
            }
        }

        offsets
    }

    fn dispatch_separable_offsets(
        &self,
        center_t: u32,
        offsets: &[(i32, i32, i32)],
    ) -> Result<(), anyhow::Error> {
        let ctx = self.launch_ctx();

        for &(q_k, q_y, q_x) in offsets {
            if q_k == 0 {
                self.dispatch_separable_iter_k0(&ctx, center_t, q_x, q_y)?;
            } else {
                self.dispatch_separable_iter(&ctx, center_t, q_x, q_y, q_k)?;
            }
        }

        Ok(())
    }
}
