use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::nlmeans::denoiser::NlmDenoiser;
use crate::nlmeans::kernels::gpu_copy;
use crate::nlmeans::motion::{
    self,
    MotionCtx,
    MotionEstimation,
    run_analyse,
    run_compensate,
    run_confidence_for_neighbour,
    run_seeded_refine,
};
use crate::nlmeans::prefilter::PrefilterMode;
use crate::nlmeans::{BLOCK_1D, MAX_GRID_1D};

/// The share of the raw noise the `NlmSpatial` pilot's reference image still carries.
///
/// A sweep from 0 to 1 on `clean-1080p.mkv` is flat to within 0.10 dB and peaks at 1.0. It is 0
/// because a floor large enough to swamp `thsad` leaves confidence unable to tell a real mismatch
/// from noise, and the flat sweep makes that free.
pub(in crate::nlmeans) const NLM_SPATIAL_RESIDUAL_FRACTION: f32 = 0.0;

/// The share of the raw noise the `Bilateral` prefilter's reference image still carries.
///
/// A sweep with motion compensation on peaks at 0 at every noise level, by 0.16 dB at the lightest
/// and 0.57 dB at the heaviest. It stays separate from [NLM_SPATIAL_RESIDUAL_FRACTION] because the
/// two zeros come from different reasons.
pub(in crate::nlmeans) const BILATERAL_RESIDUAL_FRACTION: f32 = 0.0;

/// The sigma to pass [motion::sad_noise_floor] for the motion-compensation block match.
///
/// The match runs on the reference pyramid whenever one exists, and a GPU prefilter has already
/// cleaned it, so the raw floor overstates the real one. With the NLM pilot, the default block
/// size and a sigma of 0.02, the raw floor is about 5.78 against a threshold of 5.12, which pins
/// confidence at 1.0 even on occluded blocks. The raw sigma is therefore scaled by each
/// prefilter's measured residual fraction, and `PrefilterMode::None` keeps it whole.
pub(in crate::nlmeans) fn mc_sad_noise_floor_sigma(prefilter: PrefilterMode, sigma_y: f32) -> f32 {
    match prefilter {
        PrefilterMode::NlmSpatial { .. } => sigma_y * NLM_SPATIAL_RESIDUAL_FRACTION,
        PrefilterMode::Bilateral { .. } => sigma_y * BILATERAL_RESIDUAL_FRACTION,
        PrefilterMode::None => sigma_y,
    }
}

impl<R: Runtime> NlmDenoiser<R> {
    /// Estimates how the neighbour at temporal offset `k` moved and writes it to its `mv_field`
    /// slot.
    ///
    /// `Chained` estimation composes the pair fields and refines the seed, and otherwise a direct
    /// coarse-to-fine match runs. It shifts no buffer and returns the neighbour's physical slot.
    #[expect(
        clippy::too_many_arguments,
        reason = "the dispatch threads through every buffer and shape the kernel binds"
    )]
    fn run_motion_estimate(
        &self,
        motion_ctx: &MotionCtx,
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
            self.run_chain_compose(center_t, k, neighbour_idx)?;
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
                motion_ctx,
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
                motion_ctx,
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

    /// Runs the motion estimate for every neighbour in the window without shifting any buffer.
    ///
    /// It returns each neighbour's physical slot in logical ring order, skipping the centre, or an
    /// empty list when motion compensation is off or there are no neighbours.
    pub(in crate::nlmeans) fn run_motion_machinery(&self, center_t: u32) -> Result<Vec<u32>, anyhow::Error> {
        let Some(motion_ctx) = self.mc_ctx.as_ref() else {
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
        let (confidence_arg, write_confidence): (&Handle, bool) = match self.confidence_buf.as_ref() {
            Some(buf) => (buf, true),
            None => (&self.confidence_dummy, false),
        };
        let thsad_scale = self.params.hq.map_or(1.0, |hq| hq.thsad_scale);
        let mc_sigma_y = mc_sad_noise_floor_sigma(self.params.prefilter, self.sigma_y);
        let sad_noise_floor = motion::sad_noise_floor(motion_ctx.blksize, mc_sigma_y);
        let thsad = motion::thsad(motion_ctx.blksize, thsad_scale);

        // Match against the reference pyramid when one exists, because it is cleaner.
        let analyse_pyramid = self.pyramid_reference.as_ref().unwrap_or(pyramid_input);

        let mut neighbour_idx: u32 = 0;
        let mut slots = Vec::with_capacity((frame_count - 1) as usize);
        for logical in 0..frame_count {
            if logical == center_t {
                continue;
            }

            let k = logical as i32 - center_t as i32;
            let neighbour_slot = self.run_motion_estimate(
                motion_ctx,
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

    /// Estimates each neighbour's motion and shifts it into the compensated rings.
    ///
    /// The centre slot is copied through unchanged so the temporal kernels read every slot the same
    /// way. It does nothing when motion compensation is off or there are no neighbours.
    pub(super) fn run_motion_compensation(&self, center_t: u32) -> Result<(), anyhow::Error> {
        let Some(motion_ctx) = self.mc_ctx.as_ref() else {
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
        // Without confidence weighting the fine kernel still needs a buffer, so it gets the dummy
        // and is told not to write it.
        let (confidence_arg, write_confidence): (&Handle, bool) = match self.confidence_buf.as_ref() {
            Some(buf) => (buf, true),
            None => (&self.confidence_dummy, false),
        };
        let thsad_scale = self.params.hq.map_or(1.0, |hq| hq.thsad_scale);
        let mc_sigma_y = mc_sad_noise_floor_sigma(self.params.prefilter, self.sigma_y);
        let sad_noise_floor = motion::sad_noise_floor(motion_ctx.blksize, mc_sigma_y);
        let thsad = motion::thsad(motion_ctx.blksize, thsad_scale);

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

        // Match against the reference pyramid when one exists, because it is cleaner.
        let analyse_pyramid = self.pyramid_reference.as_ref().unwrap_or(pyramid_input);

        // Neighbours run from the furthest behind to the furthest ahead, skipping the centre, so
        // their motion-field indices stay contiguous.
        let radius = temporal_radius as i32;
        let mut neighbour_idx: u32 = 0;
        for k in -radius..=radius {
            if k == 0 {
                continue;
            }

            let neighbour_slot = self.run_motion_estimate(
                motion_ctx,
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
                motion_ctx,
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
                    motion_ctx,
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

    /// Scores each neighbour against the centre frame with every block matched where it stands.
    ///
    /// It runs only when confidence is on and motion compensation is off, the one case where
    /// `confidence_ctx` exists.
    pub(super) fn run_confidence_pass(&self, center_t: u32) -> Result<(), anyhow::Error> {
        let Some(ctx) = self.confidence_ctx.as_ref() else {
            return Ok(());
        };

        // `confidence_ctx` only exists for a temporal radius above 0.
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
}

/// Copies one frame from a slot of `src` into the same slot of `dst`, which share a ring layout.
///
/// Both rings are bound whole and the kernel picks the slot, because a slot's byte offset rarely
/// meets the GPU's `min_storage_buffer_offset_alignment`. It is a free function so the motion pass
/// can call it without re-borrowing the denoiser.
#[expect(
    clippy::too_many_arguments,
    reason = "the dispatch threads through every buffer and shape the kernel binds"
)]
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
