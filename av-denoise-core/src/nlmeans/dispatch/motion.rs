use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::engine::timing::{timed, units_per_span};
use crate::engine::{BLOCK_1D, MAX_GRID_1D, copy_words_into};
use crate::motion::kernels::mc_chain_compose;
use crate::motion::truemotion::TruemotionTargets;
use crate::motion::{
    self,
    Escalation,
    FieldLayout,
    MatchTargets,
    MotionCtx,
    MotionEstimation,
    RingRole,
    map_luma_fields,
    mv_field_byte_offset,
    run_analyse,
    run_compensate,
    run_confidence,
    run_seeded_refine,
};
use crate::nlmeans::denoiser::NlmDenoiser;
use crate::nlmeans::kernels::gpu_copy;
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

/// How far a front that reads luma motion searches around each mapped luma vector.
pub(in crate::nlmeans) const MAPPED_REFINE_RADIUS: u32 = 1;

/// The dispatches one neighbour's chained seed queues.
const DISPATCHES_PER_COMPOSE: u32 = 1;

/// The most dispatches one neighbour's warps queue.
const DISPATCHES_PER_WARPED_NEIGHBOUR: u32 = 2;

/// The confidence buffer a window search binds, and whether the search writes it.
struct WindowConfidence<'a> {
    handle: &'a Handle,
    /// `f32` values `handle` holds.
    len: usize,
    written: bool,
}

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

impl NlmDenoiser {
    /// The `thsad` every motion search of this front scores its matches with.
    pub(in crate::nlmeans) fn search_thsad(&self, motion_ctx: &MotionCtx) -> f32 {
        let thsad_scale = self.params.hq.map_or(1.0, |hq| hq.thsad_scale);
        motion::thsad(motion_ctx.blksize, thsad_scale)
    }

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

        self.estimate_window_motion(center_t)
    }

    /// Queues the window's motion search, or the refine of the luma pass's vectors on a front that
    /// reads them.
    fn estimate_window_motion(&mut self, center_t: u32) -> Result<(), anyhow::Error> {
        let client = self.client.clone();

        if self.reads_luma_motion() {
            return timed(&client, self, move |front| front.refine_luma_motion(center_t));
        }

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
    /// denoiser has a confidence buffer. A `Direct` match also searches around each neighbour's
    /// camera model prediction, from the gap table the window last uploaded.
    fn match_neighbours(&self, center_t: u32) -> Result<(), anyhow::Error> {
        let motion_ctx = self
            .mc_ctx
            .as_ref()
            .expect("motion estimates run only with motion compensation active");

        let frame_count = self.params.total_frames();
        let neighbours = frame_count - 1;
        let centre_slot = self.phys_frame(center_t as i32);
        let neighbour_slots = self.neighbour_slots_handle();

        let mv_field = self.window_field();
        let confidence = self.window_confidence(motion_ctx, neighbours);
        let thsad = self.search_thsad(motion_ctx);
        let analyse_pyramid = self.window_pyramid();
        let mv_len = neighbours as usize * motion_ctx.mv_stride() as usize;

        // The chained seeds already carry the camera's motion from their seeded pair searches.
        let predictions = if self.is_chained() {
            None
        } else {
            self.compose_window_seeds()
        };

        // The window's derived noise is the floor confidence scores against, so escalation reads it.
        let escalation = self.escalation_buf.as_ref().map(|state| Escalation {
            state,
            reads_floor: true,
        });

        // The window's derived noise is also the floor the truemotion penalty scales with. Chained
        // seeds come from pair fields that already ended with the stage, so the refine skips it.
        let truemotion = self.truemotion_targets(true).filter(|_| !self.is_chained());

        let targets = MatchTargets {
            pyramid: analyse_pyramid,
            frame_count,
            centre_slot,
            neighbour_slots: &neighbour_slots,
            neighbours,
            mv_field,
            mv_len,
            confidence: confidence.handle,
            conf_len: confidence.len,
            write_confidence: confidence.written,
            predictions,
            sad: None,
            escalation,
            truemotion,
        };

        self.run_match(motion_ctx, &targets, thsad)?;
        self.share_window_motion(motion_ctx, centre_slot, neighbours);

        Ok(())
    }

    /// Maps the window fields just searched into the luma motion ring, on a front that writes it.
    fn share_window_motion(&self, motion_ctx: &MotionCtx, centre_slot: u32, neighbours: u32) {
        let Some(RingRole::Writer(ring)) = self.luma_motion.as_ref() else {
            return;
        };

        let centre_frame = self.slot_frame(centre_slot);
        let slot = ring.window_slot(centre_frame);
        let luma = FieldLayout::window(motion_ctx);
        let chroma = ring.window_layout();
        let subsampling = ring.subsampling();
        let mv_field = self.window_field();

        map_luma_fields(
            &self.client,
            mv_field,
            luma,
            &slot,
            chroma,
            subsampling,
            neighbours,
        );
    }

    /// Refines the luma pass's mapped vectors for the window centred on logical position `center_t`.
    ///
    /// Each vector is searched [MAPPED_REFINE_RADIUS] pixels around on this front's own plane, and
    /// the search scores its confidence too.
    fn refine_luma_motion(&self, center_t: u32) -> Result<(), anyhow::Error> {
        let Some(RingRole::Reader(ring)) = self.luma_motion.as_ref() else {
            anyhow::bail!("only a front that reads luma motion refines it");
        };

        let refine_ctx = self
            .refine_ctx
            .as_ref()
            .expect("a front that reads luma motion has a refine search");

        let frame_count = self.params.total_frames();
        let neighbours = frame_count - 1;
        let centre_slot = self.phys_frame(center_t as i32);
        let centre_frame = self.slot_frame(centre_slot);
        let Some(mapped) = ring.read_window(centre_frame) else {
            anyhow::bail!("the luma pass has not shared the motion of stream frame {centre_frame}");
        };

        let mv_field = self.window_field();
        let mv_len = neighbours as usize * refine_ctx.mv_stride() as usize;
        copy_words_into(&self.client, &mapped, mv_field, mv_len);

        let neighbour_slots = self.neighbour_slots_handle();
        let confidence = self.window_confidence(refine_ctx, neighbours);
        let targets = MatchTargets {
            pyramid: self.window_pyramid(),
            frame_count,
            centre_slot,
            neighbour_slots: &neighbour_slots,
            neighbours,
            mv_field,
            mv_len,
            confidence: confidence.handle,
            conf_len: confidence.len,
            write_confidence: confidence.written,
            predictions: None,
            sad: None,
            escalation: None,
            truemotion: None,
        };

        let noise = self.derived_handle();
        let thsad = self.search_thsad(refine_ctx);

        run_seeded_refine(
            &self.client,
            refine_ctx,
            self.width,
            self.height,
            &targets,
            MAPPED_REFINE_RADIUS,
            &noise,
            DERIVED_SAD_FLOOR,
            thsad,
        )
    }

    /// The window motion field, one slice per neighbour.
    fn window_field(&self) -> &Handle {
        self.mv_field_buf
            .as_ref()
            .expect("mv_field allocated when mc_ctx is Some")
    }

    /// The pyramid window searches match on, the reference pyramid when one exists since it is
    /// cleaner.
    fn window_pyramid(&self) -> &Handle {
        let pyramid_input = self
            .pyramid_input
            .as_ref()
            .expect("pyramid_input allocated when mc_ctx is Some");

        self.pyramid_reference.as_ref().unwrap_or(pyramid_input)
    }

    /// The confidence buffer a window search of `neighbours` neighbours writes.
    ///
    /// Without confidence weighting the fine kernel still needs a buffer, so it gets the one-value
    /// dummy and is told not to write it.
    fn window_confidence(&self, motion_ctx: &MotionCtx, neighbours: u32) -> WindowConfidence<'_> {
        match self.confidence_buf.as_ref() {
            Some(handle) => WindowConfidence {
                handle,
                len: neighbours as usize * motion_ctx.conf_stride() as usize,
                written: true,
            },
            None => WindowConfidence {
                handle: &self.confidence_dummy,
                len: 1,
                written: false,
            },
        }
    }

    /// Sets the truemotion stage's penalty to `field_lambda`, which 0 turns off.
    ///
    /// It does nothing without motion compensation.
    pub(crate) fn set_field_lambda(&mut self, field_lambda: f32) {
        let tuned = self
            .mc_ctx
            .take()
            .map(|motion_ctx| motion_ctx.with_field_lambda(field_lambda));
        self.mc_ctx = tuned;
    }

    /// The truemotion stage that ends a motion search, when the search runs one.
    ///
    /// With `reads_floor`, the penalty scales with the floor in the search's noise.
    pub(crate) fn truemotion_targets(&self, reads_floor: bool) -> Option<TruemotionTargets<'_>> {
        let [seeds, field, changes] = self.truemotion_bufs.as_ref()?;
        let targets = TruemotionTargets {
            kernels: &self.kernels,
            seeds,
            field,
            changes,
            reads_floor,
        };

        Some(targets)
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
            return run_seeded_refine(
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

        run_analyse(
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

        self.estimate_window_motion(center_t)?;

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
        let channels = self.params.channels.count();

        copy_frame_into_slot_handle(
            &self.client,
            &self.input_buf,
            compensated_input,
            centre_slot as usize,
            self.params.total_frames(),
            self.width,
            self.height,
            channels,
        );
        if let (Some(ref_src), Some(ref_dst)) = (
            self.reference_buf.as_ref(),
            self.compensated_reference_buf.as_ref(),
        ) {
            copy_frame_into_slot_handle(
                &self.client,
                ref_src,
                ref_dst,
                centre_slot as usize,
                self.params.total_frames(),
                self.width,
                self.height,
                channels,
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
        let channels = self.params.channels.count();

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

            run_compensate(
                &self.client,
                motion_ctx,
                channels,
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
                run_compensate(
                    &self.client,
                    motion_ctx,
                    channels,
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

        let thsad = self.search_thsad(ctx);
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
            predictions: None,
            sad: None,
            escalation: None,
            truemotion: None,
        };

        run_confidence(
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

        dispatch_chain_compose(
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
fn copy_frame_into_slot_handle(
    client: &Client,
    src: &Handle,
    dst: &Handle,
    slot: usize,
    frame_count: u32,
    width: u32,
    height: u32,
    channels: u32,
) {
    let frame_size = width * height * channels;
    let ring_len = frame_count as usize * frame_size as usize;
    let offset = slot as u32 * frame_size;

    let grid = frame_size.div_ceil(BLOCK_1D).min(MAX_GRID_1D);
    let total_threads = grid * BLOCK_1D;
    let cube_count = CubeCount::new_1d(grid);
    let cube_dim = CubeDim::new_1d(BLOCK_1D);
    let src_handle = src.clone();
    let dst_handle = dst.clone();

    unsafe {
        let src_arg = BufferArg::from_raw_parts(src_handle, ring_len);
        let dst_arg = BufferArg::from_raw_parts(dst_handle, ring_len);

        gpu_copy::launch_unchecked(
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
fn dispatch_chain_compose(
    client: &Client,
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
        let pair_ring_arg = BufferArg::from_raw_parts(pair_ring_handle, pair_ring_len);
        let mv_slot_arg = BufferArg::from_raw_parts(mv_slot, mv_slot_len);

        mc_chain_compose::launch_unchecked(
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
