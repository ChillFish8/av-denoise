use anyhow::Context;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::NlmDenoiser;
use super::noise::measures_noise;
use crate::engine::timing::timed;
use crate::engine::{DevicePlane, IngestTarget, SampleFormat, ingest};
use crate::motion::{self, MatchTargets, MotionCtx, run_analyse, run_pyramid_build};
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
            let pyramid_pixels =
                motion::pyramid_pixels_per_frame(width, height, motion_ctx.pyramid_levels, motion_ctx.align);
            let pyramid_bytes = pyramid_pixels * slots * size_of::<f32>();
            client.empty(pyramid_bytes)
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

/// The look-ahead slot of stream frame `frame`.
fn lookahead_slot(frame: u32) -> u32 {
    frame % LOOKAHEAD_FRAMES
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
    /// against it. It does nothing without motion compensation.
    fn search_lookahead(&self, frame: u32) -> Result<(), anyhow::Error> {
        let Some(motion_ctx) = self.mc_ctx.as_ref() else {
            return Ok(());
        };

        let slot = lookahead_slot(frame);
        self.build_lookahead_pyramid(motion_ctx, slot)?;

        let Some(previous) = frame.checked_sub(1) else {
            return Ok(());
        };

        let previous_slot = lookahead_slot(previous);
        let both_ways = frame == 1;
        self.search_lookahead_pair(motion_ctx, previous_slot, slot, both_ways)
    }

    /// Reads and records the noise of stream frame `frame`, just written to the look-ahead.
    ///
    /// With motion compensation the frame is read along the motion to the frame before it.
    /// Without, it is read against the frame before it as it stands. A stream's first frame has
    /// nothing before it, so it keeps only its Immerkær values.
    fn read_lookahead_noise(&self, frame: u32) -> Result<(), anyhow::Error> {
        let input = &self.lookahead().input;
        let slot = lookahead_slot(frame);
        let previous = frame.checked_sub(1).map(lookahead_slot);

        let (Some(motion_ctx), Some(previous)) = (self.mc_ctx.as_ref(), previous) else {
            self.noise.queue_stats(input, slot, previous)?;
            self.noise.queue_reading(slot)?;
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
    /// `Chained` estimation runs the pair analyse, which searches both ways. `Direct` estimation
    /// searches from the newer frame to the older one, and also the other way when `both_ways`.
    fn search_lookahead_pair(
        &self,
        motion_ctx: &MotionCtx,
        older_slot: u32,
        newer_slot: u32,
        both_ways: bool,
    ) -> Result<(), anyhow::Error> {
        let lookahead = self.lookahead();
        let pyramid = lookahead.pyramid();
        let pair_field = lookahead.pair_field();
        let noise = self.derived_handle();

        if self.is_chained() || both_ways {
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
                None,
            )
            .context("look-ahead pair analyse dispatch failed");
        }

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
            predictions: None,
            sad: None,
        };

        run_analyse(
            &self.client,
            motion_ctx,
            self.width,
            self.height,
            &targets,
            &noise,
            DERIVED_SAD_FLOOR,
            1.0,
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
        self.noise.record_reading(slot, frame);

        Ok(())
    }
}
