use std::marker::PhantomData;
use std::ops::Range;

use cubecl::prelude::*;
use cubecl::server::Handle;

use super::denoiser::{AccumClear, Nl4dDenoiser, accumulator_lens, noise_curve_upload, strength_map_upload};
use super::nl4d_pool_ratio;
use super::regularise::run_regularise;
use super::snapshot::LastFields;
use crate::collab::geometry::{ref_count, refs_along};
use crate::collab::grid_frames;
use crate::collab::kernels::aggregate::{collab_zero_accum, weight_scale};
use crate::collab::kernels::packed::collab_zero_packed;
use crate::engine::timing::timed;
use crate::nlmeans::{MAX_GRID_1D, RingView};
use crate::noise::STRENGTH_MAP_OFF;
use crate::tune::sites::{
    CollabFilterSite,
    CollabInputs,
    CollabOutputs,
    CollabSearchSite,
    CollabShape,
    CollabSite,
    CollabValues,
    SearchOutputs,
};

/// The per-pass buffers and values `collab_fused` reads beside the denoiser's own state.
struct FusedPass {
    mv_field: Handle,
    confidence: Handle,
    neighbour_slots: Handle,
    noise_curve: Handle,
    strength_map: Handle,
    curve_valid: u32,
    map_mode: u32,
    weight_norm: f32,
    stored_ch: u32,
    /// Values in the frame ring, which the f16 search ring matches.
    ring_len: usize,
    accum_ring_len: usize,
    wsum_ring_len: usize,
    neighbours: u32,
}

impl<R: Runtime> Nl4dDenoiser<R> {
    /// Runs the grouping, filtering and aggregation kernels for one pass centred on `view`.
    ///
    /// Each filtered member scatters into the accumulator region of the frame it came from, so one
    /// pass adds to every region in the ring. `clear` picks which regions are zeroed first. The
    /// newest region is the slot `temporal_radius` ahead of the centre, which the newest frame has
    /// just taken over.
    pub(super) fn run_pass(&mut self, view: &RingView, clear: AccumClear) -> Result<(), anyhow::Error> {
        let centre_slot = view.centre_slot;
        let client = self.front.compute_client().clone();

        let stored_ch = self.channels.storage_count();
        let channels_count = self.channels.count();
        let pixels = (self.width * self.height) as usize;
        let frame_len = pixels * stored_ch as usize;
        let total_frames = 1 + 2 * self.temporal_radius;
        let ring_len = frame_len * total_frames as usize;
        let (accum_ring_len, wsum_ring_len) = accumulator_lens(pixels, stored_ch, total_frames, self.packed);
        let neighbours = 2 * self.temporal_radius;

        let neighbour_slots_buf = view.neighbour_slots_buf.clone();

        let sigmas = self.front.current_sigmas_temporal_only();
        let mut sigma_host = vec![0.0f32; stored_ch as usize];
        sigma_host[..channels_count as usize].copy_from_slice(&sigmas[..channels_count as usize]);
        let sigma_bytes = f32::as_bytes(&sigma_host);
        self.sigma_buf.update(&client, sigma_bytes);
        let weight_norm = weight_scale(sigma_host[0], &self.dct_profile);

        let curve_ratios = self.front.current_noise_curve().map(|curve| curve.ratios);
        let (ratios, curve_valid) = noise_curve_upload(curve_ratios, self.apply_noise_map);
        let ratio_bytes = f32::as_bytes(&ratios);
        self.noise_curve_buf.update(&client, ratio_bytes);
        let noise_curve_buf = self.noise_curve_buf.handle().clone();

        let classes = self.front.current_quarter_classes();
        if let Some(classes) = classes {
            let class_dims = (classes.cols(), classes.rows());
            let map_dims = (self.map_cols as usize, self.map_rows as usize);
            assert_eq!(
                class_dims, map_dims,
                "quarter classes must cover the strength map"
            );
        }

        let map_upload = strength_map_upload(classes, curve_valid, self.luma_map, self.chroma_map_boost);
        let (strength_map_buf, map_mode) = match map_upload {
            Some((multipliers, mode)) => {
                let multiplier_bytes = f32::as_bytes(&multipliers);
                self.strength_map_buf.update(&client, multiplier_bytes);
                let strength_map = self.strength_map_buf.handle().clone();

                (strength_map, mode)
            },
            None => (self.unit_map.clone(), STRENGTH_MAP_OFF),
        };

        let cleared_slots = match clear {
            AccumClear::WholeRing => 0..total_frames,
            AccumClear::NewestRegion => {
                let newest_slot = (centre_slot + self.temporal_radius) % total_frames;

                // A region normalise zeroed since the last scatter is already empty.
                let already_zeroed = self.zeroed_regions[newest_slot as usize];
                if already_zeroed {
                    newest_slot..newest_slot
                } else {
                    newest_slot..newest_slot + 1
                }
            },
            AccumClear::Nothing => 0..0,
        };

        // The pass scatters into every region, so none stays zeroed past it.
        self.zeroed_regions.fill(false);

        #[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
        {
            self.region_clears += cleared_slots.len() as u64;
        }

        self.passes_run += 1;

        // The regularise launch and the scatter run in separate device spans, so each span fits
        // one compute pass.
        let (mv_field, confidence) = timed(&client, self, |denoiser| denoiser.regularise_fields(view))?;

        self.last_fields = Some(LastFields {
            mv_field: mv_field.clone(),
            confidence: confidence.clone(),
            mv_stride: view.mv_stride,
            conf_stride: view.conf_stride,
            neighbours,
        });

        let pass = FusedPass {
            mv_field,
            confidence,
            neighbour_slots: neighbour_slots_buf,
            noise_curve: noise_curve_buf,
            strength_map: strength_map_buf,
            curve_valid,
            map_mode,
            weight_norm,
            stored_ch,
            ring_len,
            accum_ring_len,
            wsum_ring_len,
            neighbours,
        };

        timed(&client, self, move |denoiser| {
            denoiser.scatter_pass(view, cleared_slots, pass)
        })
    }

    /// Regularises the view's motion fields when the field regulariser is on.
    ///
    /// Returns the motion field and confidence the pass groups with.
    fn regularise_fields(&self, view: &RingView) -> Result<(Handle, Handle), anyhow::Error> {
        let (Some(reg_mv), Some(reg_conf)) = (self.reg_mv.as_ref(), self.reg_conf.as_ref()) else {
            let fields = (view.mv_field.clone(), view.confidence.clone());
            return Ok(fields);
        };

        let motion_ctx = self.front.motion_ctx();
        let sad_noise_floor = self.front.sad_noise_floor_value();
        let thsad = self.front.thsad_value();
        run_regularise::<R>(
            &self.kernels,
            motion_ctx,
            view,
            self.width,
            self.height,
            self.field_lambda,
            sad_noise_floor,
            thsad,
            reg_mv,
            reg_conf,
        )?;

        let fields = (reg_mv.clone(), reg_conf.clone());
        Ok(fields)
    }

    /// Zeroes the accumulator regions in `cleared_slots`, then groups, filters and scatters one pass.
    fn scatter_pass(
        &self,
        view: &RingView,
        cleared_slots: Range<u32>,
        pass: FusedPass,
    ) -> Result<(), anyhow::Error> {
        let client = self.front.compute_client().clone();
        let pixels = (self.width * self.height) as usize;
        let frame_len = pixels * pass.stored_ch as usize;
        let accum_ring_len = pass.accum_ring_len;

        let zero_dim = 256u32;
        // Clamped to the 65,535 workgroup limit, which a 4:4:4 4K frame or an 8K luma plane alone
        // exceeds. `collab_zero_accum` strides, so a clamped launch still reaches every slot.
        let zero_workgroups_one_frame = (frame_len as u32).div_ceil(zero_dim).min(MAX_GRID_1D);
        let zero_grid_one_frame = CubeCount::new_1d(zero_workgroups_one_frame);
        let zero_total_threads_one_frame = zero_workgroups_one_frame * zero_dim;
        let zero_cube_dim = CubeDim::new_1d(zero_dim);

        unsafe {
            // One dispatch over the whole ring exceeds the 65,535 workgroup limit at
            // `temporal_radius = 4` for a 1080p luma plane. A rejected dispatch leaves undefined
            // memory that a fresh stream would aggregate as real, so each region gets its own.
            for slot in cleared_slots {
                let accum = self.accum.clone();
                let wsum = self.wsum.clone();
                let accum_arg = ArrayArg::from_raw_parts(accum, accum_ring_len);
                let wsum_arg = ArrayArg::from_raw_parts(wsum, pass.wsum_ring_len);

                // A packed region spans no more threads than a split one, so the same grid covers it.
                if self.packed {
                    collab_zero_packed::launch_unchecked::<R>(
                        &client,
                        zero_grid_one_frame.clone(),
                        zero_cube_dim,
                        wsum_arg,
                        accum_arg,
                        slot * pixels as u32,
                        pixels as u32,
                        pass.stored_ch - 1,
                        zero_total_threads_one_frame,
                    );
                } else {
                    collab_zero_accum::launch_unchecked::<R>(
                        &client,
                        zero_grid_one_frame.clone(),
                        zero_cube_dim,
                        accum_arg,
                        wsum_arg,
                        slot * pixels as u32,
                        pixels as u32,
                        pass.stored_ch,
                        zero_total_threads_one_frame,
                    );
                }
            }
        }

        // The f32 search binds the f32 ring as its placeholder, so its kernel declares no f16 type.
        if self.f16_search {
            let search_handle = view
                .search_input
                .clone()
                .expect("the f16 search enables the search ring");
            let search_len = pass.ring_len;
            self.launch_collab_fused::<half::f16>(&client, view, search_handle, search_len, pass)?;
        } else {
            let placeholder = view.input.clone();
            let search_len = pass.stored_ch as usize;
            self.launch_collab_fused::<f32>(&client, view, placeholder, search_len, pass)?;
        }

        Ok(())
    }

    /// Launches `collab_fused` for one pass, with its search ring read as `S`.
    ///
    /// With a member buffer, the pass runs `collab_search` then `collab_filter` in its place.
    fn launch_collab_fused<S: Float>(
        &self,
        client: &ComputeClient<R>,
        view: &RingView,
        search_ring: Handle,
        search_len: usize,
        pass: FusedPass,
    ) -> Result<(), anyhow::Error> {
        let neighbours = pass.neighbours;
        let mv_len = (neighbours * view.mv_stride) as usize;
        let conf_len = (neighbours * view.conf_stride) as usize;
        let neighbour_slots_len = view.neighbour_slots.len();
        let map_len = (self.map_cols * self.map_rows) as usize;
        let refs_x = refs_along(self.width);
        let refs_y = refs_along(self.height);
        let refs = ref_count(self.width, self.height);
        let motion_ctx = self.front.motion_ctx();
        let frames_per_volume = grid_frames(self.temporal_radius);
        let pool_ratio = nl4d_pool_ratio(self.channels);

        let inputs = CollabInputs {
            ring: view.input.clone(),
            ring_len: pass.ring_len,
            search_ring,
            search_len,
            mv_field: pass.mv_field,
            mv_len: mv_len.max(1),
            confidence: pass.confidence,
            conf_len: conf_len.max(1),
            neighbour_slots: pass.neighbour_slots,
            neighbour_slots_len: neighbour_slots_len.max(1),
            sigma: self.sigma_buf.handle().clone(),
            noise_curve: pass.noise_curve,
            strength_map: pass.strength_map,
            map_len,
            dct_profile: self.dct_profile_buf.clone(),
            kaiser: self.kaiser_buf.clone(),
        };

        // The kernel's centre slot must be the ring view's centre, or a member is grouped against
        // one frame and scattered into another.
        let values = CollabValues {
            centre_slot: view.centre_slot,
            c_min: self.c_min,
            lambda_ht: self.lambda_ht,
            curve_valid: pass.curve_valid,
            map_mode: pass.map_mode,
            weight_scale: pass.weight_norm,
            accum_scale: self.accum_scale,
            pool_ratio,
        };
        let shape = CollabShape {
            f16_search: self.f16_search,
            radius: self.temporal_radius,
            grid_frames: frames_per_volume,
            refine: self.refine,
            mv_stride: view.mv_stride,
            conf_stride: view.conf_stride,
            blk_step: motion_ctx.step,
            blksize: motion_ctx.blksize,
            blocks_x: motion_ctx.blocks_x,
            blocks_y: motion_ctx.blocks_y,
            width: self.width,
            height: self.height,
            channels: self.channels.count(),
            k_max: self.k_max,
            stored_ch: pass.stored_ch,
            spatial_radius: self.spatial_radius,
            refs_x,
            refs_y,
            map_cols: self.map_cols,
            map_rows: self.map_rows,
            pooled: self.pooled_threshold,
            uniform_walk_only: self.uniform_walk_only,
            packed: self.packed,
        };
        let outputs = CollabOutputs {
            accum: self.accum.clone(),
            wsum: self.wsum.clone(),
            group_weight: self.group_weight.clone(),
            accum_len: pass.accum_ring_len,
            wsum_len: pass.wsum_ring_len,
            refs,
            packed: self.packed,
        };
        // Both split launches queue on one client, so the filter sees the search's writes.
        if let Some(members) = &self.members {
            let search_outputs = SearchOutputs {
                members: members.clone(),
                refs,
            };
            let search_site = CollabSearchSite::<R, S> {
                client: client.clone(),
                inputs: inputs.clone(),
                values,
                shape,
                outputs: search_outputs,
                search: PhantomData,
            };
            let filter_site = CollabFilterSite::<R> {
                client: client.clone(),
                inputs,
                members: members.clone(),
                values,
                shape,
                outputs,
            };

            self.kernels.run(&search_site)?;
            self.kernels.run(&filter_site)
        } else {
            let site = CollabSite::<R, S> {
                client: client.clone(),
                inputs,
                values,
                shape,
                outputs,
                search: PhantomData,
            };

            self.kernels.run(&site)
        }
    }
}
