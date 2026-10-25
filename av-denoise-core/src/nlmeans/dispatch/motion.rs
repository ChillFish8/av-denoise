use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::engine::timing::{timed, units_per_span};
use crate::engine::{BLOCK_1D, MAX_GRID_1D};
use crate::motion::kernels::mc_chain_compose;
use crate::nlmeans::denoiser::NlmDenoiser;
use crate::nlmeans::kernels::gpu_copy;
use crate::nlmeans::motion::{
    self,
    MatchTargets,
    MotionCtx,
    MotionEstimation,
    mv_field_byte_offset,
    run_analyse,
    run_compensate,
    run_confidence,
    run_seeded_refine,
};
use crate::nlmeans::prefilter::PrefilterMode;
use crate::noise::{DERIVED_CONFIDENCE_SAD_FLOOR, DERIVED_SAD_FLOOR};

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

/// The dispatches one neighbour's chained seed queues.
const DISPATCHES_PER_COMPOSE: u32 = 1;

/// The most dispatches one neighbour's warps queue.
const DISPATCHES_PER_WARPED_NEIGHBOUR: u32 = 2;

/// The sigma to pass [motion::sad_noise_floor] for the motion-compensation block match.
///
/// The match runs on the reference pyramid whenever one exists, and a GPU prefilter has already
/// cleaned it, so the raw floor overstates the real one. With the NLM pilot, the default block
/// size and a sigma of 0.02, the raw floor is about 5.78 against a threshold of 5.12, which pins
/// confidence at 1.0 even on occluded blocks. The raw sigma is therefore scaled by each
/// prefilter's measured residual fraction, and `PrefilterMode::None` keeps it whole.
pub(crate) fn mc_sad_noise_floor_sigma(prefilter: PrefilterMode, sigma_y: f32) -> f32 {
    match prefilter {
        PrefilterMode::NlmSpatial { .. } => sigma_y * NLM_SPATIAL_RESIDUAL_FRACTION,
        PrefilterMode::Bilateral { .. } => sigma_y * BILATERAL_RESIDUAL_FRACTION,
        PrefilterMode::None => sigma_y,
    }
}

impl<R: Runtime> NlmDenoiser<R> {
    /// Each neighbour's physical slot for a pass centred on `center_t`, in logical order skipping
    /// the centre.
    pub(in crate::nlmeans) fn window_neighbour_slots(&self, center_t: u32) -> Vec<u32> {
        let frame_count = self.params.total_frames();
        let mut slots = Vec::with_capacity((frame_count - 1) as usize);

        for logical in 0..frame_count {
            if logical == center_t {
                continue;
            }

            let slot = self.ring_slot(logical);
            slots.push(slot);
        }

        slots
    }

    /// The temporal offset of every neighbour of logical position `center_t`, oldest first.
    fn neighbour_offsets(&self, center_t: u32) -> Vec<i32> {
        let frame_count = self.params.total_frames();

        (0..frame_count)
            .filter(|&logical| logical != center_t)
            .map(|logical| logical as i32 - center_t as i32)
            .collect()
    }

    /// Estimates every neighbour's motion without shifting any buffer.
    ///
    /// The neighbour-slot table must already name the window's neighbours. It does nothing when
    /// motion compensation is off or there are no neighbours.
    pub(in crate::nlmeans) fn run_motion_machinery(&mut self, center_t: u32) -> Result<(), anyhow::Error> {
        if self.mc_ctx.is_none() || self.params.temporal_radius == 0 {
            return Ok(());
        }

        let client = self.client.clone();

        self.compose_chained_seeds(center_t)?;

        timed(&client, self, move |front| front.match_neighbours(center_t))
    }

    /// Composes every neighbour's seed from the pair ring when `Chained` estimation is active.
    ///
    /// The neighbours run in chunks that each fit one device span.
    fn compose_chained_seeds(&mut self, center_t: u32) -> Result<(), anyhow::Error> {
        if !self.is_chained() {
            return Ok(());
        }

        let client = self.client.clone();
        let offsets = self.neighbour_offsets(center_t);
        let per_span = units_per_span(DISPATCHES_PER_COMPOSE);

        for (chunk_index, chunk) in offsets.chunks(per_span).enumerate() {
            let first_idx = (chunk_index * per_span) as u32;
            timed(&client, self, move |front| {
                front.compose_seeds(center_t, chunk, first_idx)
            })?;
        }

        Ok(())
    }

    /// Composes the seeds of the neighbours at `offsets`, whose motion-field indices start at
    /// `first_idx`.
    fn compose_seeds(&self, center_t: u32, offsets: &[i32], first_idx: u32) -> Result<(), anyhow::Error> {
        let indexed = (first_idx..).zip(offsets.iter());
        for (neighbour_idx, &k) in indexed {
            self.run_chain_compose(center_t, k, neighbour_idx)?;
        }

        Ok(())
    }

    /// Matches every neighbour against the centre frame and writes each one's `mv_field` slice.
    ///
    /// It reads the neighbours from the neighbour-slot table, and writes confidence too when the
    /// denoiser has a confidence buffer.
    fn match_neighbours(&self, center_t: u32) -> Result<(), anyhow::Error> {
        let motion_ctx = self
            .mc_ctx
            .as_ref()
            .expect("motion estimates run only with motion compensation active");

        let frame_count = self.params.total_frames();
        let neighbours = frame_count - 1;
        let centre_slot = self.phys_frame(center_t as i32);
        let neighbour_slots = self.neighbour_slots_handle();

        let pyramid_input = self
            .pyramid_input
            .as_ref()
            .expect("pyramid_input allocated when mc_ctx is Some");
        let mv_field = self
            .mv_field_buf
            .as_ref()
            .expect("mv_field allocated when mc_ctx is Some");
        // Without confidence weighting the fine kernel still needs a buffer, so it gets the dummy
        // and is told not to write it.
        let (confidence_arg, write_confidence): (&Handle, bool) = match self.confidence_buf.as_ref() {
            Some(buf) => (buf, true),
            None => (&self.confidence_dummy, false),
        };
        let thsad_scale = self.params.hq.map_or(1.0, |hq| hq.thsad_scale);
        let thsad = motion::thsad(motion_ctx.blksize, thsad_scale);

        // Match against the reference pyramid when one exists, because it is cleaner.
        let analyse_pyramid = self.pyramid_reference.as_ref().unwrap_or(pyramid_input);

        let mv_len = neighbours as usize * motion_ctx.mv_stride() as usize;
        // A placeholder `confidence` holds a single value.
        let conf_len = if write_confidence {
            neighbours as usize * motion_ctx.conf_stride() as usize
        } else {
            1
        };

        let targets = MatchTargets {
            pyramid: analyse_pyramid,
            frame_count,
            centre_slot,
            neighbour_slots: &neighbour_slots,
            neighbours,
            mv_field,
            mv_len,
            confidence: confidence_arg,
            conf_len,
            write_confidence,
        };

        self.run_match(motion_ctx, &targets, thsad)
    }

    /// Runs the block match for every neighbour in `targets` in one launch per level.
    ///
    /// `Chained` estimation refines the composed seeds. Otherwise a direct coarse-to-fine match runs.
    /// The block match scores confidence against the derived noise values' SAD floor.
    fn run_match(
        &self,
        motion_ctx: &MotionCtx,
        targets: &MatchTargets<'_>,
        thsad: f32,
    ) -> Result<(), anyhow::Error> {
        let noise = self.derived_handle();
        let estimation = self
            .params
            .motion_compensation
            .resolved_estimation(self.params.temporal_radius);

        if let Some(MotionEstimation::Chained { refine_radius }) = estimation {
            return run_seeded_refine::<R>(
                &self.client,
                motion_ctx,
                self.width,
                self.height,
                targets,
                refine_radius,
                &noise,
                DERIVED_SAD_FLOOR,
                thsad,
            );
        }

        run_analyse::<R>(
            &self.client,
            motion_ctx,
            self.width,
            self.height,
            targets,
            &noise,
            DERIVED_SAD_FLOOR,
            thsad,
        )
    }

    /// Estimates each neighbour's motion and shifts it into the compensated rings.
    ///
    /// The centre slot is copied through unchanged so the temporal kernels read every slot the same
    /// way. The neighbour-slot table must already name the window's neighbours. It does nothing when
    /// motion compensation is off or there are no neighbours. The warps run in chunks that each fit
    /// one device span.
    pub(super) fn run_motion_compensation(&mut self, center_t: u32) -> Result<(), anyhow::Error> {
        if self.mc_ctx.is_none() || self.params.temporal_radius == 0 {
            return Ok(());
        }

        let client = self.client.clone();

        timed(&client, self, move |front| front.copy_centre_through(center_t));

        self.compose_chained_seeds(center_t)?;
        timed(&client, self, move |front| front.match_neighbours(center_t))?;

        // Neighbours run from the furthest behind to the furthest ahead, skipping the centre, so
        // their motion-field indices stay contiguous.
        let offsets = self.neighbour_offsets(center_t);
        let per_span = units_per_span(DISPATCHES_PER_WARPED_NEIGHBOUR);
        for (chunk_index, chunk) in offsets.chunks(per_span).enumerate() {
            let first_idx = (chunk_index * per_span) as u32;
            timed(&client, self, move |front| {
                front.warp_neighbours(center_t, chunk, first_idx)
            })?;
        }

        Ok(())
    }

    /// Copies the centre slot unchanged into the compensated input and reference rings.
    fn copy_centre_through(&self, center_t: u32) {
        let compensated_input = self
            .compensated_input_buf
            .as_ref()
            .expect("compensated_input allocated when mc_ctx is Some");

        let centre_slot = self.phys_frame(center_t as i32);
        let stored_ch = self.params.channels.storage_count();

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
    }

    /// Shifts the neighbours at `offsets`, whose motion-field indices start at `first_idx`, into
    /// the compensated rings.
    fn warp_neighbours(&self, center_t: u32, offsets: &[i32], first_idx: u32) -> Result<(), anyhow::Error> {
        let motion_ctx = self
            .mc_ctx
            .as_ref()
            .expect("motion compensation runs only with motion compensation active");

        let frame_count = self.params.total_frames();
        let stored_ch = self.params.channels.storage_count();

        let mv_field = self
            .mv_field_buf
            .as_ref()
            .expect("mv_field allocated when mc_ctx is Some");
        let compensated_input = self
            .compensated_input_buf
            .as_ref()
            .expect("compensated_input allocated when mc_ctx is Some");

        let indexed = (first_idx..).zip(offsets.iter());
        for (neighbour_idx, &k) in indexed {
            let neighbour_slot = self.phys_frame(center_t as i32 + k);

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
        }

        Ok(())
    }

    /// Scores every neighbour against the centre frame with each block matched where it stands.
    ///
    /// It runs only when confidence is on and motion compensation is off, the one case where
    /// `confidence_ctx` exists. The neighbour-slot table must already name the window's neighbours.
    pub(super) fn run_confidence_pass(&self, center_t: u32) -> Result<(), anyhow::Error> {
        let Some(ctx) = self.confidence_ctx.as_ref() else {
            return Ok(());
        };

        let frame_count = self.params.total_frames();
        // `confidence_ctx` only exists for a temporal radius above 0.
        let neighbours = 2 * self.params.temporal_radius;
        let centre_slot = self.phys_frame(center_t as i32);
        let neighbour_slots = self.neighbour_slots_handle();

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
        let thsad = motion::thsad(ctx.blksize, thsad_scale);
        let noise = self.derived_handle();

        let mv_len = neighbours as usize * ctx.mv_stride() as usize;
        let conf_len = neighbours as usize * ctx.conf_stride() as usize;

        let targets = MatchTargets {
            pyramid: luma_pyramid,
            frame_count,
            centre_slot,
            neighbour_slots: &neighbour_slots,
            neighbours,
            mv_field: mv_scratch,
            mv_len,
            confidence: confidence_buf,
            conf_len,
            write_confidence: true,
        };

        run_confidence::<R>(
            &self.client,
            ctx,
            self.width,
            self.height,
            &targets,
            &noise,
            DERIVED_CONFIDENCE_SAD_FLOOR,
            thsad,
        )
    }

    /// Joins the adjacent-frame fields into one motion field for the neighbour at offset `k`.
    ///
    /// `k` must be nonzero and land inside the ring, at most twice the temporal radius either way.
    /// The walk takes one hop per step out from the centre `center_t`, following the older-to-newer
    /// field for a positive `k` and the newer-to-older field for a negative one. It does nothing
    /// unless `Chained` estimation is active.
    pub(crate) fn run_chain_compose(
        &self,
        center_t: u32,
        k: i32,
        neighbour_idx: u32,
    ) -> Result<(), anyhow::Error> {
        let Some(motion_ctx) = self.mc_ctx.as_ref() else {
            return Ok(());
        };

        if !self.is_chained() || k == 0 {
            return Ok(());
        }

        let radius = self.params.temporal_radius;
        let far_edge = 2 * radius as i32;
        debug_assert!(
            (0..=far_edge).contains(&(center_t as i32 + k)),
            "center_t={center_t} k={k} lands outside the ring 0..={far_edge}"
        );
        debug_assert!(
            neighbour_idx < 2 * radius,
            "neighbour_idx={neighbour_idx} outside the field, which holds {} neighbours",
            2 * radius,
        );

        let pair_ring = self
            .pair_ring_buf
            .as_ref()
            .expect("pair_ring allocated when Chained is active");
        let mv_field = self
            .mv_field_buf
            .as_ref()
            .expect("mv_field allocated when mc_ctx is Some");

        let forward = k > 0;
        let steps = k.unsigned_abs();
        let start_gap = if forward {
            center_t as i32
        } else {
            center_t as i32 - 1
        };
        let start_pair_slot = self.pair_slot(start_gap);
        let pair_ring_slots = motion::pair_ring_slot_count(radius);
        let pair_ring_len = pair_ring_slots as usize * motion_ctx.pair_slot_stride() as usize;

        dispatch_chain_compose::<R>(
            &self.client,
            motion_ctx,
            self.width,
            self.height,
            pair_ring_slots,
            pair_ring_len,
            start_pair_slot,
            forward,
            steps,
            pair_ring,
            mv_field,
            neighbour_idx,
        )
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
    let cube_count = CubeCount::new_1d(grid);
    let cube_dim = CubeDim::new_1d(BLOCK_1D);
    let src_handle = src.clone();
    let dst_handle = dst.clone();

    unsafe {
        let src_arg = ArrayArg::from_raw_parts(src_handle, ring_len);
        let dst_arg = ArrayArg::from_raw_parts(dst_handle, ring_len);

        gpu_copy::launch_unchecked::<R>(
            client,
            cube_count,
            cube_dim,
            src_arg,
            dst_arg,
            offset,
            offset,
            frame_size,
            total_threads,
        );
    }
}

/// Launches `mc_chain_compose` into this neighbour's `mv_field` slot.
///
/// The padded strides are passed explicitly, because the kernel reads the whole pair ring as one
/// array and must step through it as `pair_byte_offset` does.
#[expect(
    clippy::too_many_arguments,
    reason = "the dispatch threads through every buffer and shape the kernel binds"
)]
fn dispatch_chain_compose<R: Runtime>(
    client: &ComputeClient<R>,
    motion_ctx: &MotionCtx,
    width: u32,
    height: u32,
    pair_ring_slots: u32,
    pair_ring_len: usize,
    start_pair_slot: u32,
    forward: bool,
    steps: u32,
    pair_ring: &Handle,
    mv_field: &Handle,
    neighbour_idx: u32,
) -> Result<(), anyhow::Error> {
    let mv_offset = mv_field_byte_offset(motion_ctx, neighbour_idx);
    let mv_slot = mv_field.clone().offset_start(mv_offset);
    let mv_slot_len = (motion_ctx.blocks_x as usize) * (motion_ctx.blocks_y as usize) * 2;

    // One single-thread cube per block, since there is no per-candidate work to share.
    let grid = CubeCount::new_2d(motion_ctx.blocks_x, motion_ctx.blocks_y);
    let dim = CubeDim::new_2d(1, 1);
    let pair_direction_stride = motion_ctx.pair_direction_stride();
    let pair_slot_stride = motion_ctx.pair_slot_stride();
    let pair_ring_handle = pair_ring.clone();

    unsafe {
        let pair_ring_arg = ArrayArg::from_raw_parts(pair_ring_handle, pair_ring_len);
        let mv_slot_arg = ArrayArg::from_raw_parts(mv_slot, mv_slot_len);

        mc_chain_compose::launch_unchecked::<R>(
            client,
            grid,
            dim,
            pair_ring_arg,
            mv_slot_arg,
            start_pair_slot,
            forward,
            steps,
            pair_ring_slots,
            pair_direction_stride,
            pair_slot_stride,
            motion_ctx.step,
            width,
            height,
            motion_ctx.blocks_x,
            motion_ctx.blocks_y,
        );
    }

    Ok(())
}
