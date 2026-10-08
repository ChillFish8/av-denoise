use anyhow::Context;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::NlmDenoiser;
use crate::nlmeans::kernels::{gpu_cast_f16, gpu_copy};
use crate::nlmeans::prefilter::PrefilterMode;
use crate::nlmeans::{BLOCK_1D, MAX_GRID_1D};

impl<R: Runtime> NlmDenoiser<R> {
    pub(super) fn advance_ring(&mut self) {
        let total_frames = self.params.total_frames() as usize;
        self.ring_head += 1;
        if self.frames_loaded < total_frames {
            self.frames_loaded += 1;
        }

        self.real_pushes += 1;
    }

    /// Copies frame `src_slot` of `src` into `slot` of `dst` on the GPU.
    ///
    /// `dst` uses the ring layout of `input_buf` and `src` holds `src_slots` frames. Both handles
    /// are bound whole, because a slot's byte offset rarely meets the GPU's
    /// `min_storage_buffer_offset_alignment`.
    fn copy_frame_into_slot(
        &self,
        dst: &Handle,
        slot: usize,
        src: &Handle,
        src_slot: usize,
        src_slots: usize,
    ) {
        let stored_ch = self.params.channels.storage_count();
        let frame_size = self.width * self.height * stored_ch;
        let dst_slots = self.params.total_frames() as usize;

        let grid = frame_size.div_ceil(BLOCK_1D).min(MAX_GRID_1D);
        let total_threads = grid * BLOCK_1D;

        unsafe {
            gpu_copy::launch_unchecked::<R>(
                &self.client,
                CubeCount::new_1d(grid),
                CubeDim::new_1d(BLOCK_1D),
                ArrayArg::from_raw_parts(src.clone(), src_slots * frame_size as usize),
                ArrayArg::from_raw_parts(dst.clone(), dst_slots * frame_size as usize),
                src_slot as u32 * frame_size,
                slot as u32 * frame_size,
                frame_size,
                total_threads,
            )
        };
    }

    /// Allocates the f16 search ring, a slot-for-slot f16 copy of the input ring.
    ///
    /// Every later write to an input slot is mirrored into it, so it must be called before the
    /// first push.
    pub(crate) fn enable_search_ring(&mut self) {
        debug_assert_eq!(
            self.frames_loaded, 0,
            "the search ring must be enabled before the first push"
        );

        let total_frames = self.params.total_frames() as usize;
        let stored_ch = self.params.channels.storage_count() as usize;
        let frame_len = (self.width * self.height) as usize * stored_ch;
        let bytes = frame_len * total_frames * size_of::<half::f16>();
        let search_buf = self.client.empty(bytes);
        self.search_buf = Some(search_buf);
    }

    /// Copies input slot `slot` into the search ring as f16, when the search ring exists.
    pub(in crate::nlmeans) fn mirror_search_slot(&self, slot: usize) {
        let Some(search_buf) = self.search_buf.as_ref() else {
            return;
        };

        let total_frames = self.params.total_frames() as usize;
        let stored_ch = self.params.channels.storage_count();
        let frame_len = self.width * self.height * stored_ch;
        let ring_len = frame_len as usize * total_frames;
        let grid = frame_len.div_ceil(BLOCK_1D).min(MAX_GRID_1D);
        let total_threads = grid * BLOCK_1D;
        let offset = slot as u32 * frame_len;

        unsafe {
            gpu_cast_f16::launch_unchecked::<R>(
                &self.client,
                CubeCount::new_1d(grid),
                CubeDim::new_1d(BLOCK_1D),
                ArrayArg::from_raw_parts(self.input_buf.clone(), ring_len),
                ArrayArg::from_raw_parts(search_buf.clone(), ring_len),
                offset,
                frame_len,
                total_threads,
            );
        }
    }

    /// Copies the first pushed frame into the leading ring slots so the window starts balanced.
    ///
    /// It does nothing with shifted edges on, since those windows stop at the clip start instead.
    pub(super) fn prime_leading_edge_if_first(&mut self) -> Result<(), anyhow::Error> {
        if self.shifted_edges {
            return Ok(());
        }

        let temporal_radius = self.params.temporal_radius as usize;

        if temporal_radius == 0 || self.frames_loaded != 1 {
            return Ok(());
        }

        for _ in 0..temporal_radius {
            self.duplicate_last_frame()?;
            self.frames_loaded += 1;
        }

        Ok(())
    }

    /// Copies the last pushed frame into the next ring slot and refreshes that slot's state.
    pub(in crate::nlmeans) fn duplicate_last_frame(&mut self) -> Result<(), anyhow::Error> {
        let total_frames = self.params.total_frames() as usize;
        let last_slot = (self.ring_head - 1) % total_frames;
        let next_slot = self.ring_head % total_frames;

        // Slots never overlap, so copying within the same buffer is safe.
        let input_buf = self.input_buf.clone();
        self.copy_frame_into_slot(&input_buf, next_slot, &input_buf, last_slot, total_frames);
        self.mirror_search_slot(next_slot);

        // `NlmSpatial` rebuilds this slot's reference with the pilot below.
        if !matches!(self.params.prefilter, PrefilterMode::NlmSpatial { .. })
            && let Some(reference_buf) = self.reference_buf.clone()
        {
            self.copy_frame_into_slot(&reference_buf, next_slot, &reference_buf, last_slot, total_frames);
        }

        // Every per-slot buffer is refreshed so a later pass never reads an older frame's state here.
        if let PrefilterMode::NlmSpatial { strength_scale } = self.params.prefilter {
            self.run_nlm_spatial_pilot(next_slot as u32, strength_scale)
                .context("nlm spatial pilot dispatch failed")?;
        }

        self.build_pyramids_for_slot(next_slot as u32)?;
        self.build_confidence_pyramid_for_slot(next_slot as u32)?;
        self.run_noise_estimate_for_slot(next_slot as u32)?;
        self.zero_temporal_stats_for_slot(next_slot as u32);
        // This runs before `ring_head` advances so `pair_slot(0)` matches the push-time slot.
        self.zero_pair_slot_for_duplicate();

        self.ring_head += 1;

        Ok(())
    }

    /// The physical slot holding the oldest frame in the window.
    fn ring_start(&self) -> u32 {
        let total_frames = self.params.total_frames() as usize;
        (self.ring_head % total_frames) as u32
    }

    /// The physical `input_buf` slot of a logical frame index within the window.
    pub(in crate::nlmeans) fn phys_frame(&self, logical: i32) -> u32 {
        let total_frames = self.params.total_frames() as i32;
        let wrapped = logical.rem_euclid(total_frames);
        let start = self.ring_start() as i32;
        ((start + wrapped).rem_euclid(total_frames)) as u32
    }

    /// The pair-ring slot holding the motion between two neighbouring frames.
    ///
    /// At push time the gap index is 0 and `ring_head` has not advanced yet. At compose time the
    /// gap index is measured from the window centre, which cancels how far `ring_head` has moved
    /// since, so both land on the same slot.
    pub(in crate::nlmeans) fn pair_slot(&self, gap_index: i32) -> u32 {
        let radius = self.params.temporal_radius as i32;
        debug_assert!(
            radius > 0,
            "pair ring is only meaningful when temporal_radius > 0"
        );

        let pair_slots = 2 * radius;
        ((self.ring_head as i32 + gap_index).rem_euclid(pair_slots)) as u32
    }
}
