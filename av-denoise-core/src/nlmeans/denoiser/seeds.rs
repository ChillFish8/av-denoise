use cubecl::prelude::*;
use cubecl::server::Handle;

use super::NlmDenoiser;
use crate::motion::{
    AFFINE_WORDS,
    NO_PAIR,
    RingRole,
    SharedMotion,
    compose_gaps,
    pyramid_pixels_per_frame,
    run_compose_models,
};
use crate::nlmeans::dispatch::MAPPED_REFINE_RADIUS;
use crate::nlmeans::mirror::MirroredBuffer;
use crate::noise::DERIVED_WORDS;
use crate::tune::gpu::zeroed_words;

/// One pair's predictions and the noise values its searches score the seed preference against.
pub(in crate::nlmeans) struct PairSeeds<'a> {
    /// The older-to-newer prediction, present when the pair was composed both ways.
    pub older_to_newer: Option<&'a Handle>,
    /// The newer-to-older prediction.
    pub newer_to_older: &'a Handle,
    /// Zeroed noise values, so a pair search keeps its seed only on exact ties.
    pub noise: &'a Handle,
}

impl<'a> PairSeeds<'a> {
    /// The older-to-newer prediction, then the newer-to-older one, when composed both ways.
    pub fn both_ways(&self) -> Option<[&'a Handle; 2]> {
        let older_to_newer = self.older_to_newer?;

        Some([older_to_newer, self.newer_to_older])
    }
}

/// The camera model predictions a front's motion searches start from.
///
/// Pair predictions come from single-entry gap tables built once, so a pair search uploads
/// nothing. A window's gap table is uploaded from the host whenever the window moves.
pub(in crate::nlmeans) struct ModelSeeds {
    motion: SharedMotion,
    /// One single-entry gap table per model-ring slot, then one holding [NO_PAIR].
    pair_gaps: Vec<Handle>,
    /// The pair's model from its older frame to its newer one.
    older_to_newer: Handle,
    /// The pair's model from its newer frame to its older one.
    newer_to_older: Handle,
    /// Zeroed noise values for the pair searches.
    pair_noise: Handle,
    /// The gap table of the last window, with an entry per neighbour per step.
    window_gaps: MirroredBuffer,
    /// Each neighbour's prediction for the last window.
    window_predictions: Handle,
    /// The neighbours of a window, which is also the most steps one is away.
    neighbours: u32,
    /// The neighbours behind the centre of the last window.
    past_neighbours: u32,
}

impl ModelSeeds {
    /// The predictions of a front with a temporal window `temporal_radius` either side.
    pub(super) fn new(client: &Client, motion: SharedMotion, temporal_radius: u32) -> Self {
        let depth = motion.ring.depth();
        let pair_gaps = (0..depth)
            .chain([NO_PAIR])
            .map(|gap| {
                let table = [gap];
                let table_bytes = u32::as_bytes(&table);
                client.create_from_slice(table_bytes)
            })
            .collect();

        let prediction_bytes = AFFINE_WORDS as usize * size_of::<f32>();
        let older_to_newer = client.empty(prediction_bytes);
        let newer_to_older = client.empty(prediction_bytes);
        let pair_noise = zeroed_words(client, DERIVED_WORDS as usize);

        let neighbours = 2 * temporal_radius;
        let gap_count = (neighbours * neighbours) as usize;
        let no_gaps = vec![NO_PAIR; gap_count];
        let no_gap_bytes = u32::as_bytes(&no_gaps);
        let window_gaps = MirroredBuffer::new(client, no_gap_bytes);
        let window_bytes = (neighbours * AFFINE_WORDS) as usize * size_of::<f32>();
        let window_predictions = client.empty(window_bytes);

        Self {
            motion,
            pair_gaps,
            older_to_newer,
            newer_to_older,
            pair_noise,
            window_gaps,
            window_predictions,
            neighbours,
            past_neighbours: temporal_radius,
        }
    }

    /// Composes the pair's prediction from stream frame `newer` back to `older`, and from `older`
    /// to `newer` too when `both_ways`.
    ///
    /// A pair that does not step exactly one frame predicts no motion.
    pub(super) fn compose_pair(
        &self,
        client: &Client,
        older: u32,
        newer: u32,
        both_ways: bool,
    ) -> PairSeeds<'_> {
        let frames = [older, newer];
        let gaps = compose_gaps(&frames, 1, 1, &self.motion.ring);
        let table_index = match gaps[0] {
            NO_PAIR => self.motion.ring.depth(),
            slot => slot,
        };
        let table = &self.pair_gaps[table_index as usize];
        let scale_x = self.motion.scale_x;
        let scale_y = self.motion.scale_y;

        if both_ways {
            run_compose_models(
                client,
                &self.motion.ring,
                table,
                &self.older_to_newer,
                1,
                0,
                1,
                scale_x,
                scale_y,
            );
        }

        run_compose_models(
            client,
            &self.motion.ring,
            table,
            &self.newer_to_older,
            1,
            1,
            1,
            scale_x,
            scale_y,
        );

        let older_to_newer = both_ways.then_some(&self.older_to_newer);

        PairSeeds {
            older_to_newer,
            newer_to_older: &self.newer_to_older,
            noise: &self.pair_noise,
        }
    }

    /// Uploads the gap table of a window holding `window_frames`, centred on position `centre`.
    pub(super) fn update_window(&mut self, client: &Client, window_frames: &[u32], centre: u32) {
        let max_steps = self.neighbours as usize;
        let gaps = compose_gaps(window_frames, centre as usize, max_steps, &self.motion.ring);
        let gap_bytes = u32::as_bytes(&gaps);

        self.window_gaps.update(client, gap_bytes);
        self.past_neighbours = centre;
    }

    /// Composes every neighbour's prediction for the window last uploaded.
    pub(super) fn compose_window(&self, client: &Client) -> &Handle {
        let gaps = self.window_gaps.handle();
        let scale_x = self.motion.scale_x;
        let scale_y = self.motion.scale_y;

        run_compose_models(
            client,
            &self.motion.ring,
            gaps,
            &self.window_predictions,
            self.neighbours,
            self.past_neighbours,
            self.neighbours,
            scale_x,
            scale_y,
        );

        &self.window_predictions
    }
}

impl NlmDenoiser {
    /// Starts every motion search from the camera models in `motion`, and joins its luma motion ring.
    ///
    /// A front that reads luma motion runs no search, so it takes no camera models. It does nothing
    /// without motion compensation, and must be called before the first push.
    pub(crate) fn set_shared_motion(&mut self, motion: SharedMotion) {
        if self.mc_ctx.is_none() {
            return;
        }

        let role = motion.luma_motion.clone();
        if motion.reads_luma_motion() {
            self.drop_motion_search();
        } else {
            let seeds = ModelSeeds::new(&self.client, motion, self.params.temporal_radius);
            self.seeds = Some(seeds);
        }

        self.luma_motion = role;
    }

    /// Swaps the motion search's buffers for the single-level refine of a front that reads luma motion.
    fn drop_motion_search(&mut self) {
        let motion_ctx = self.motion_ctx();
        let refine_ctx = motion_ctx.single_level(MAPPED_REFINE_RADIUS);
        let pyramid_pixels = pyramid_pixels_per_frame(self.width, self.height, 1, refine_ctx.align);
        let frames = self.params.total_frames() as usize;
        let pyramid_bytes = pyramid_pixels * frames * size_of::<f32>();

        let input_pyramid = self.client.empty(pyramid_bytes);
        self.pyramid_input = Some(input_pyramid);
        if self.pyramid_reference.is_some() {
            let reference_pyramid = self.client.empty(pyramid_bytes);
            self.pyramid_reference = Some(reference_pyramid);
        }

        if let Some(lookahead) = self.lookahead.as_mut() {
            lookahead.keep_one_pyramid_level(&self.client, self.width, self.height, refine_ctx.align);
        }

        self.truemotion_bufs = None;
        self.pair_ring_buf = None;
        self.ring_slot_tables.clear();
        self.escalation_buf = None;
        self.refine_ctx = Some(refine_ctx);
    }

    /// Whether the front refines the luma pass's motion instead of searching its own.
    pub(in crate::nlmeans) fn reads_luma_motion(&self) -> bool {
        matches!(self.luma_motion, Some(RingRole::Reader(_)))
    }

    /// Uploads the gap table of the window centred on logical position `center_t`.
    ///
    /// Only a `Direct` search reads it, and the upload must come before the search's device span.
    pub(in crate::nlmeans) fn update_window_seeds(&mut self, center_t: u32) {
        if self.seeds.is_none() || self.is_chained() {
            return;
        }

        let total_frames = self.params.total_frames();
        let window_frames: Vec<u32> = (0..total_frames)
            .map(|logical| {
                let slot = self.ring_slot(logical);
                self.slot_frame(slot)
            })
            .collect();

        let Some(seeds) = self.seeds.as_mut() else {
            return;
        };

        seeds.update_window(&self.client, &window_frames, center_t);
    }

    /// Composes the predictions of the window last uploaded, when the front has camera models.
    pub(in crate::nlmeans) fn compose_window_seeds(&self) -> Option<&Handle> {
        let seeds = self.seeds.as_ref()?;
        let predictions = seeds.compose_window(&self.client);

        Some(predictions)
    }

    /// Composes the pair's prediction from stream frame `newer` back to `older`, and from `older`
    /// to `newer` too when `both_ways`, when the front has camera models.
    pub(in crate::nlmeans) fn compose_pair_seeds(
        &self,
        older: u32,
        newer: u32,
        both_ways: bool,
    ) -> Option<PairSeeds<'_>> {
        let seeds = self.seeds.as_ref()?;
        let predictions = seeds.compose_pair(&self.client, older, newer, both_ways);

        Some(predictions)
    }
}
