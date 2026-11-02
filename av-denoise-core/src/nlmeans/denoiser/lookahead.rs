use std::ops::RangeInclusive;

use anyhow::Context;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::NlmDenoiser;
use super::noise::measures_noise;
use super::seeds::PairSeeds;
use crate::engine::align::StorageAlign;
use crate::engine::timing::timed;
use crate::engine::{DevicePlane, IngestTarget, SampleFormat, copy_words_into, ingest};
use crate::motion::{
    self,
    Escalation,
    FieldLayout,
    MatchTargets,
    MotionCtx,
    RingRole,
    map_luma_fields,
    run_analyse,
    run_pyramid_build,
    run_seeded_refine,
};
use crate::nlmeans::dispatch::MAPPED_REFINE_RADIUS;
use crate::nlmeans::params::NlmParams;
use crate::noise::{DERIVED_SAD_FLOOR, NOISE_WINDOW_RADIUS, StatsMotion};

/// The frames the look-ahead holds before releasing the oldest into the frame ring.
pub(crate) const LOOKAHEAD_FRAMES: u32 = NOISE_WINDOW_RADIUS;

/// The frames a push waits in the look-ahead of a front built from `params`.
///
/// It is [LOOKAHEAD_FRAMES] for a front that measures the noise and 0 otherwise.
pub(crate) fn lookahead_depth_for(params: &NlmParams) -> usize {
    if measures_noise(params) {
        LOOKAHEAD_FRAMES as usize
    } else {
        0
    }
}

/// The pair-field direction from the older frame to the newer one.
const OLDER_TO_NEWER: u32 = 0;
/// The pair-field direction from the newer frame to the older one.
const NEWER_TO_OLDER: u32 = 1;

/// A ring of the newest pushed frames, which the noise is read from before they enter the frame
/// ring.
///
/// Each push reads the frame's noise against the frame before it, along the motion between the two
/// when motion compensation is on, and records the reading. A frame enters the frame ring
/// [LOOKAHEAD_FRAMES] pushes later, so every frame a noise window reaches is read by then.
pub(crate) struct LookAhead {
    /// The held frames, one slot per frame in the frame ring's layout.
    input: Handle,
    /// The luma pyramid of each held frame, present with motion compensation.
    pyramid: Option<Handle>,
    /// One single-slot table per held slot, so a search can name any held frame as its neighbour.
    slot_tables: Vec<Handle>,
    /// The motion between the newest frame and the one before it, laid out like one pair-ring slot.
    pair_field: Option<Handle>,
    /// Frames pushed since the stream started.
    pushed: u32,
    /// Frames released into the frame ring since the stream started.
    released: u32,
}

impl LookAhead {
    /// The look-ahead of a front with `frame_bytes` per frame, searching with `mc_ctx`.
    pub(super) fn new(
        client: &Client,
        frame_bytes: usize,
        mc_ctx: Option<&MotionCtx>,
        width: u32,
        height: u32,
    ) -> Self {
        let slots = LOOKAHEAD_FRAMES as usize;
        let input = client.empty(frame_bytes * slots);

        let pyramid = mc_ctx.map(|motion_ctx| {
            let levels = motion_ctx.pyramid_levels;
            pyramid_ring(client, width, height, levels, motion_ctx.align)
        });

        let pair_field = mc_ctx.map(|motion_ctx| {
            let pair_bytes = motion_ctx.pair_slot_bytes() as usize;
            client.empty(pair_bytes)
        });

        let slot_tables = if mc_ctx.is_some() {
            (0..LOOKAHEAD_FRAMES)
                .map(|slot| {
                    let table = [slot];
                    let table_bytes = u32::as_bytes(&table);
                    client.create_from_slice(table_bytes)
                })
                .collect()
        } else {
            Vec::new()
        };

        Self {
            input,
            pyramid,
            slot_tables,
            pair_field,
            pushed: 0,
            released: 0,
        }
    }

    /// The frames pushed but not yet released.
    pub(super) fn held(&self) -> u32 {
        self.pushed - self.released
    }

    fn pyramid(&self) -> &Handle {
        self.pyramid
            .as_ref()
            .expect("the look-ahead has a pyramid with motion compensation")
    }

    fn pair_field(&self) -> &Handle {
        self.pair_field
            .as_ref()
            .expect("the look-ahead has a pair field with motion compensation")
    }

    pub(super) fn reset(&mut self) {
        self.pushed = 0;
        self.released = 0;
    }

    /// Replaces the pyramid ring with one that holds only the full-resolution level.
    pub(super) fn keep_one_pyramid_level(
        &mut self,
        client: &Client,
        width: u32,
        height: u32,
        align: StorageAlign,
    ) {
        let pyramid = pyramid_ring(client, width, height, 1, align);
        self.pyramid = Some(pyramid);
    }

    /// The look-ahead's buffers by the names the buffer sizes list them under.
    #[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
    pub(crate) fn buffers_for_test(&self) -> [(&'static str, Option<&Handle>); 3] {
        [
            ("noise look-ahead ring", Some(&self.input)),
            ("look-ahead pyramid ring", self.pyramid.as_ref()),
            ("look-ahead pair field", self.pair_field.as_ref()),
        ]
    }
}

/// A pyramid of `levels` levels for every held frame.
fn pyramid_ring(client: &Client, width: u32, height: u32, levels: u32, align: StorageAlign) -> Handle {
    let pyramid_pixels = motion::pyramid_pixels_per_frame(width, height, levels, align);
    let pyramid_bytes = pyramid_pixels * LOOKAHEAD_FRAMES as usize * size_of::<f32>();

    client.empty(pyramid_bytes)
}

/// The look-ahead slot of stream frame `frame`.
fn lookahead_slot(frame: u32) -> u32 {
    frame % LOOKAHEAD_FRAMES
}

/// The pair-field directions a search between two frames writes.
fn searched_directions(both_ways: bool) -> RangeInclusive<u32> {
    if both_ways {
        OLDER_TO_NEWER..=NEWER_TO_OLDER
    } else {
        NEWER_TO_OLDER..=NEWER_TO_OLDER
    }
}

impl NlmDenoiser {
    /// Pushes `planes` into the look-ahead, releasing its oldest frame first when it is full.
    pub(super) fn push_into_lookahead(
        &mut self,
        planes: &[DevicePlane<'_>],
        format: SampleFormat,
    ) -> Result<(), anyhow::Error> {
        let held = self.lookahead().held();
        if held == LOOKAHEAD_FRAMES {
            self.release_held_frame()?;
        }

        let frame = self.lookahead().pushed;
        let slot = lookahead_slot(frame);
        let pixels = self.width * self.height;
        let channels = self.params.channels.count();
        let frame_len = pixels * channels;
        let target = IngestTarget {
            ring: &self.lookahead().input,
            ring_len: LOOKAHEAD_FRAMES as usize * frame_len as usize,
            offset: slot * frame_len,
            pixels,
            channels,
        };

        ingest(&self.client, planes, format, &self.placeholder, target);

        // Each stage gets its own device span, since a span times at most one compute pass.
        let client = self.client.clone();
        timed(&client, self, move |front| front.search_lookahead(frame))?;
        timed(&client, self, move |front| front.read_lookahead_noise(frame))?;

        let rereads_first_frame = self.mc_ctx.is_some() && frame == 1;
        if rereads_first_frame {
            timed(&client, self, |front| front.reread_first_frame())?;
        }

        self.lookahead_mut().pushed += 1;

        Ok(())
    }

    /// Moves the oldest held frame into the next frame-ring slot and runs its per-frame stages.
    ///
    /// Returns `false` when the look-ahead holds nothing.
    pub(crate) fn release_held_frame(&mut self) -> Result<bool, anyhow::Error> {
        let Some(lookahead) = self.lookahead.as_ref() else {
            return Ok(false);
        };

        if lookahead.held() == 0 {
            return Ok(false);
        }

        let frame = lookahead.released;
        debug_assert_eq!(
            frame as usize, self.real_pushes,
            "frames leave the look-ahead in stream order"
        );

        let source = lookahead.input.clone();
        let source_slot = lookahead_slot(frame) as usize;
        let total_frames = self.params.total_frames() as usize;
        let slot = self.ring_head % total_frames;
        let input_buf = self.input_buf.clone();
        let client = self.client.clone();

        timed(&client, self, move |front| {
            let held_slots = LOOKAHEAD_FRAMES as usize;
            front.copy_frame_into_slot(&input_buf, slot, &source, source_slot, held_slots);
            front.mirror_search_slot(slot);
        });

        self.lookahead_mut().released += 1;
        self.run_post_upload_stages(slot)?;

        Ok(true)
    }

    /// The frames the look-ahead holds, 0 for a front without one.
    pub(crate) fn held_frames(&self) -> usize {
        self.lookahead
            .as_ref()
            .map_or(0, |lookahead| lookahead.held() as usize)
    }

    /// The frames a push waits in the look-ahead before it enters the frame ring.
    ///
    /// It is 0 for a front that does not measure the noise.
    pub(crate) fn lookahead_depth(&self) -> usize {
        if self.lookahead.is_some() {
            LOOKAHEAD_FRAMES as usize
        } else {
            0
        }
    }

    fn lookahead(&self) -> &LookAhead {
        self.lookahead
            .as_ref()
            .expect("a front that measures the noise has a look-ahead")
    }

    fn lookahead_mut(&mut self) -> &mut LookAhead {
        self.lookahead
            .as_mut()
            .expect("a front that measures the noise has a look-ahead")
    }

    /// Builds the pyramid of stream frame `frame` and searches its motion to the frame before it.
    ///
    /// The second frame of a stream is also searched the other way, so the first frame can be read
    /// against it, and so is every frame under `Chained` estimation. A front that reads luma motion
    /// refines the luma pass's pair instead. It does nothing without motion compensation.
    fn search_lookahead(&self, frame: u32) -> Result<(), anyhow::Error> {
        let Some(motion_ctx) = self.mc_ctx.as_ref() else {
            return Ok(());
        };

        let slot = lookahead_slot(frame);
        let pyramid_ctx = self.refine_ctx.as_ref().unwrap_or(motion_ctx);
        self.build_lookahead_pyramid(pyramid_ctx, slot)?;

        let Some(previous) = frame.checked_sub(1) else {
            return Ok(());
        };

        let previous_slot = lookahead_slot(previous);
        let both_ways = frame == 1 || self.is_chained();
        if self.reads_luma_motion() {
            return self.refine_lookahead_motion(frame, previous_slot, slot, both_ways);
        }

        let seeds = self.compose_pair_seeds(previous, frame, both_ways);
        self.search_lookahead_pair(motion_ctx, previous_slot, slot, both_ways, seeds)?;
        self.share_lookahead_motion(motion_ctx, frame, both_ways);

        Ok(())
    }

    /// Maps the pair just searched for stream frame `frame` into the luma motion ring, on a front
    /// that writes it.
    fn share_lookahead_motion(&self, motion_ctx: &MotionCtx, frame: u32, both_ways: bool) {
        let Some(RingRole::Writer(ring)) = self.luma_motion.as_ref() else {
            return;
        };

        let directions = searched_directions(both_ways);
        let first = *directions.start();
        let fields = directions.count() as u32;

        let luma = FieldLayout::pair(motion_ctx);
        let luma_offset = luma.field_byte_offset(first);
        let luma_field = self.lookahead().pair_field().clone().offset_start(luma_offset);

        let chroma = ring.pair_layout();
        let chroma_offset = chroma.field_byte_offset(first);
        let slot = ring.lookahead_slot(frame);
        let chroma_field = slot.offset_start(chroma_offset);

        map_luma_fields(
            &self.client,
            &luma_field,
            luma,
            &chroma_field,
            chroma,
            ring.subsampling(),
            fields,
        );
    }

    /// Refines the luma pass's mapped pair for stream frame `frame`, held in `newer_slot`, against
    /// the frame in `older_slot`.
    ///
    /// Each vector is searched [MAPPED_REFINE_RADIUS] pixels around on this front's own plane.
    fn refine_lookahead_motion(
        &self,
        frame: u32,
        older_slot: u32,
        newer_slot: u32,
        both_ways: bool,
    ) -> Result<(), anyhow::Error> {
        let Some(RingRole::Reader(ring)) = self.luma_motion.as_ref() else {
            anyhow::bail!("only a front that reads luma motion refines it");
        };

        let Some(mapped) = ring.read_lookahead(frame) else {
            anyhow::bail!("the luma pass has not shared the look-ahead motion of stream frame {frame}");
        };

        let refine_ctx = self
            .refine_ctx
            .as_ref()
            .expect("a front that reads luma motion has a refine search");

        // Only the searched directions are copied, since only those are refined and read.
        let directions = searched_directions(both_ways);
        let first = *directions.start();
        let fields = directions.count() as u32;
        let layout = FieldLayout::pair(refine_ctx);
        let offset = layout.field_byte_offset(first);
        let source = mapped.offset_start(offset);
        let lookahead = self.lookahead();
        let pair_field = lookahead.pair_field();
        let destination = pair_field.clone().offset_start(offset);
        let words = (fields * layout.stride) as usize;
        copy_words_into(&self.client, &source, &destination, words);

        let noise = self.derived_handle();
        let thsad = self.search_thsad(refine_ctx);
        for direction in searched_directions(both_ways) {
            let (centre_slot, neighbour_slot) = if direction == OLDER_TO_NEWER {
                (older_slot, newer_slot)
            } else {
                (newer_slot, older_slot)
            };

            let direction_offset = motion::pair_byte_offset(refine_ctx, 0, direction);
            let mv_field = pair_field.clone().offset_start(direction_offset);
            let targets = MatchTargets {
                pyramid: lookahead.pyramid(),
                frame_count: LOOKAHEAD_FRAMES,
                centre_slot,
                neighbour_slots: &lookahead.slot_tables[neighbour_slot as usize],
                neighbours: 1,
                mv_field: &mv_field,
                mv_len: refine_ctx.pair_direction_len() as usize,
                confidence: &self.confidence_dummy,
                conf_len: 1,
                write_confidence: false,
                predictions: None,
                sad: None,
                escalation: None,
                truemotion: None,
            };

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
            .context("look-ahead refine dispatch failed")?;
        }

        Ok(())
    }

    /// Reads and records the noise of stream frame `frame`, just written to the look-ahead.
    ///
    /// With motion compensation the frame is read along the motion to the frame before it.
    /// Without, it is read against the frame before it as it stands. A stream's first frame has
    /// nothing before it, so it keeps only its Immerkær values. Each reading also pools into the
    /// colour cells, on a front that measures them.
    fn read_lookahead_noise(&self, frame: u32) -> Result<(), anyhow::Error> {
        let input = &self.lookahead().input;
        let slot = lookahead_slot(frame);
        let previous = frame.checked_sub(1).map(lookahead_slot);

        let (Some(motion_ctx), Some(previous)) = (self.mc_ctx.as_ref(), previous) else {
            self.noise.queue_stats(input, slot, previous)?;
            self.noise.queue_reading(slot)?;
            self.noise.queue_colour_cells(slot, frame)?;
            self.noise.record_reading(slot, frame);
            return Ok(());
        };

        self.noise.queue_spatial_stats(input, slot)?;
        self.read_along_motion(motion_ctx, slot, previous, NEWER_TO_OLDER, frame)
    }

    /// Reads the stream's first frame again along the motion to the second, and re-records it.
    fn reread_first_frame(&self) -> Result<(), anyhow::Error> {
        let motion_ctx = self
            .mc_ctx
            .as_ref()
            .expect("only a motion-compensated front re-reads the first frame");
        let first_slot = lookahead_slot(0);
        let second_slot = lookahead_slot(1);

        self.read_along_motion(motion_ctx, first_slot, second_slot, OLDER_TO_NEWER, 0)
    }

    fn build_lookahead_pyramid(&self, motion_ctx: &MotionCtx, slot: u32) -> Result<(), anyhow::Error> {
        let lookahead = self.lookahead();
        let pyramid = lookahead.pyramid();

        run_pyramid_build(
            &self.client,
            motion_ctx,
            self.width,
            self.height,
            LOOKAHEAD_FRAMES,
            slot,
            &lookahead.input,
            pyramid,
            self.params.channels.count(),
        )
        .context("look-ahead pyramid build dispatch failed")
    }

    /// Searches the motion between the held frames in `older_slot` and `newer_slot`.
    ///
    /// It searches from the newer frame to the older one, and also the other way when `both_ways`.
    /// With `seeds`, each direction also searches around its camera model prediction.
    ///
    /// The fields only feed the noise reading, so the search runs no truemotion stage. The stage's
    /// smoother vectors leave misalignment in the reading's residual, which reads the noise high.
    fn search_lookahead_pair(
        &self,
        motion_ctx: &MotionCtx,
        older_slot: u32,
        newer_slot: u32,
        both_ways: bool,
        seeds: Option<PairSeeds<'_>>,
    ) -> Result<(), anyhow::Error> {
        let lookahead = self.lookahead();
        let pyramid = lookahead.pyramid();
        let pair_field = lookahead.pair_field();
        let noise = match seeds.as_ref() {
            Some(seeds) => seeds.noise.clone(),
            None => self.derived_handle(),
        };
        let escalation_state = self.escalation_buf.as_ref();
        // The look-ahead's noise is zeroed or belongs to an earlier window, so it measures its floor.
        let escalation = escalation_state.map(|state| Escalation {
            state,
            reads_floor: false,
        });
        let thsad = self.search_thsad(motion_ctx);

        if both_ways {
            let predictions = seeds.as_ref().and_then(PairSeeds::both_ways);
            return motion::run_pair_analyse(
                &self.client,
                motion_ctx,
                self.width,
                self.height,
                LOOKAHEAD_FRAMES,
                older_slot,
                newer_slot,
                0,
                pyramid,
                &lookahead.slot_tables,
                pair_field,
                &self.confidence_dummy,
                &noise,
                predictions,
                escalation_state,
                None,
                thsad,
            )
            .context("look-ahead pair analyse dispatch failed");
        }

        let newer_to_older_prediction = seeds.as_ref().map(|seeds| seeds.newer_to_older);

        let direction_offset = motion::pair_byte_offset(motion_ctx, 0, NEWER_TO_OLDER);
        let newer_to_older = pair_field.clone().offset_start(direction_offset);
        let targets = MatchTargets {
            pyramid,
            frame_count: LOOKAHEAD_FRAMES,
            centre_slot: newer_slot,
            neighbour_slots: &lookahead.slot_tables[older_slot as usize],
            neighbours: 1,
            mv_field: &newer_to_older,
            mv_len: motion_ctx.pair_direction_len() as usize,
            confidence: &self.confidence_dummy,
            conf_len: 1,
            write_confidence: false,
            predictions: newer_to_older_prediction,
            sad: None,
            escalation,
            truemotion: None,
        };

        run_analyse(
            &self.client,
            motion_ctx,
            self.width,
            self.height,
            &targets,
            &noise,
            DERIVED_SAD_FLOOR,
            thsad,
        )
        .context("look-ahead analyse dispatch failed")
    }

    /// Reads the held frame in `slot` against `neighbour` along the pair field's `direction`, and
    /// records it as stream frame `frame`.
    fn read_along_motion(
        &self,
        motion_ctx: &MotionCtx,
        slot: u32,
        neighbour: u32,
        direction: u32,
        frame: u32,
    ) -> Result<(), anyhow::Error> {
        let lookahead = self.lookahead();
        let pair_field = lookahead.pair_field();
        let direction_stride = motion_ctx.pair_direction_stride();
        let motion = StatsMotion {
            mv_field: pair_field,
            mv_len: motion_ctx.pair_slot_stride() as usize,
            mv_offset: direction * direction_stride,
            step: motion_ctx.step,
            blocks_x: motion_ctx.blocks_x,
            blocks_y: motion_ctx.blocks_y,
        };

        self.noise
            .queue_compensated_stats(&lookahead.input, slot, neighbour, motion)?;
        self.noise.queue_reading(slot)?;
        self.noise.queue_colour_cells(slot, frame)?;
        self.noise.record_reading(slot, frame);

        Ok(())
    }
}
