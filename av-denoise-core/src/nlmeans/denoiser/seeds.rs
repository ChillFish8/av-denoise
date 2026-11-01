use cubecl::prelude::*;
use cubecl::server::Handle;

use super::NlmDenoiser;
use crate::motion::{AFFINE_WORDS, NO_PAIR, SharedMotion, compose_gaps, run_compose_models};
use crate::nlmeans::mirror::MirroredBuffer;
use crate::noise::DERIVED_WORDS;
use crate::tune::gpu::zeroed_words;

/// One pair's predictions and the noise values its searches score the seed preference against.
pub(in crate::nlmeans) struct PairSeeds<'a> {
    /// The older-to-newer prediction, then the newer-to-older one.
    pub predictions: [&'a Handle; 2],
    /// Zeroed noise values, so a pair search keeps its seed only on exact ties.
    ///
    /// The front's own noise values belong to an earlier window, or to the stream before a reset.
    pub noise: &'a Handle,
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

    /// Composes the pair from stream frame `older` to `newer` both ways.
    ///
    /// A pair that does not step exactly one frame predicts no motion.
    pub(super) fn compose_pair(&self, client: &Client, older: u32, newer: u32) -> PairSeeds<'_> {
        let frames = [older, newer];
        let gaps = compose_gaps(&frames, 1, 1, &self.motion.ring);
        let table_index = match gaps[0] {
            NO_PAIR => self.motion.ring.depth(),
            slot => slot,
        };
        let table = &self.pair_gaps[table_index as usize];
        let scale_x = self.motion.scale_x;
        let scale_y = self.motion.scale_y;

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

        PairSeeds {
            predictions: [&self.older_to_newer, &self.newer_to_older],
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
    /// Starts every motion search from the camera models in `motion`.
    ///
    /// It does nothing without motion compensation, and must be called before the first push.
    pub(crate) fn set_shared_motion(&mut self, motion: SharedMotion) {
        if self.mc_ctx.is_none() {
            return;
        }

        let seeds = ModelSeeds::new(&self.client, motion, self.params.temporal_radius);
        self.seeds = Some(seeds);
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

    /// Composes both directions of the pair from stream frame `older` to `newer`, when the front
    /// has camera models.
    pub(in crate::nlmeans) fn compose_pair_seeds(&self, older: u32, newer: u32) -> Option<PairSeeds<'_>> {
        let seeds = self.seeds.as_ref()?;
        let predictions = seeds.compose_pair(&self.client, older, newer);

        Some(predictions)
    }
}
