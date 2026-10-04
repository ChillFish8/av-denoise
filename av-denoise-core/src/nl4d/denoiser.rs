use cubecl::prelude::*;
use cubecl::server::Handle;

use super::grain::{GrainChunk, GrainExport, GrainGeometry};
use super::nl4d_pool_ratio;
use super::params::Nl4dParams;
use super::regularise::run_regularise;
use super::snapshot::{LastFields, MotionSnapshot, read_snapshot};
use crate::collab::geometry::{fused_cubes_x, ref_count, refs_along, strength_map_dims};
use crate::collab::kernels::aggregate::{
    collab_normalise,
    collab_zero_accum,
    cross_frame_accum_scale,
    kaiser_window,
    weight_scale,
};
use crate::collab::kernels::fused::{STRENGTH_MAP_ALL, STRENGTH_MAP_LUMA, STRENGTH_MAP_OFF, collab_fused};
use crate::collab::kernels::transforms::dct_noise_profile;
use crate::collab::{MAX_K, PATCH_SIZE, grid_frames, needs_warp_uniform_search};
use crate::engine::{DevicePlane, SampleFormat};
use crate::nlmeans::{
    BLOCK_X,
    BLOCK_Y,
    ChannelMode,
    MAX_GRID_1D,
    NOISE_CURVE_BINS,
    NlmDenoiser,
    QuarterClasses,
    RingView,
    StrengthMapParams,
};

/// Which accumulator regions a pass zeroes before scattering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AccumClear {
    /// Every region, on a stream's first pass.
    WholeRing,
    /// Only the region about to be reused by the newest frame.
    NewestRegion,
    /// None, for edge passes whose regions already hold live contributions.
    Nothing,
}

/// A finished accumulator region and the ring slot of the frame after it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CompletedRegion {
    pub slot: u32,
    pub next: Option<u32>,
}

/// How the current stream began.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamStart {
    /// The stream starts a scene, so its first frames get off-centre head passes.
    SceneStart,
    /// The stream picks up mid-clip from priming pushes, so it runs no head passes.
    Continuation,
}

/// Groups similar 8x8 patches across a motion-compensated window and denoises each group jointly.
///
/// The NLMeans front end supplies the frame ring, motion field and confidence scores, and no NLM
/// weighting runs. Each pass groups patches by searching the centre frame around each reference
/// and each neighbour frame around where motion predicts the patch moved. It shrinks each group's
/// coefficients in the transform domain and scatters the filtered members back into an
/// accumulator ring, which `collab_normalise` turns into finished frames.
///
/// A pass scatters each member into the frame it came from, so a frame finishes only once every
/// pass that can reach it has run. Latency is `2 * temporal_radius` pushes.
pub struct Nl4dDenoiser<R: Runtime> {
    front: NlmDenoiser<R>,
    width: u32,
    height: u32,
    channels: ChannelMode,
    /// Whether the threshold pools each coefficient with its neighbours.
    pooled_threshold: bool,
    temporal_radius: u32,
    refine: u32,
    spatial_radius: u32,
    lambda_ht: f32,
    c_min: f32,
    k_max: u32,
    /// Whether `collab_fused` runs its warp-uniform search.
    warp_uniform: bool,
    /// The fixed-point scale the cross-frame accumulator ring counts in.
    accum_scale: f32,

    group_weight: Handle,
    sigma_buf: Handle,
    dct_profile_buf: Handle,
    /// The aggregation window's 8 taps, all ones when `kaiser_beta` is 0.
    kaiser_buf: Handle,
    /// The correlation profile kept on the host, so the weight normalisation needs no readback.
    dct_profile: [f32; 8],
    /// Fixed-point accumulators the filter scatters into, one region per slot of the frame ring.
    ///
    /// A pass contributes to every frame in the ring, so a frame's region stays live for as long
    /// as the frame sits in the ring.
    accum: Handle,
    wsum: Handle,
    /// Two output buffers, alternated so one frame's kernels can overlap the previous frame's
    /// readback.
    outputs: [Handle; 2],
    next_output_slot: usize,
    /// Passes run for the current stream.
    ///
    /// At zero the accumulator ring may hold a previous stream's stale contributions, so the
    /// stream's first pass zeroes all of it.
    passes_run: u32,
    stream_start: StreamStart,
    /// The field buffers the last pass handed the fused kernel.
    last_fields: Option<LastFields>,
    field_lambda: f32,
    /// The regularised motion field, allocated only when `field_lambda > 0.0`.
    reg_mv: Option<Handle>,
    reg_conf: Option<Handle>,
    /// Whether passes apply the luma noise curve, set when the noise map is on and the denoiser
    /// filters luma.
    apply_noise_map: bool,
    /// Set when the noise map applies and its multipliers are not all 1.0.
    luma_map: Option<StrengthMapParams>,
    /// The chroma flat boost, set when the noise map is on and the boost is not 1.0.
    chroma_map_boost: Option<f32>,
    /// A strength map of all 1.0, bound by every pass with no map to apply.
    unit_map: Handle,
    map_cols: u32,
    map_rows: u32,
    /// Present only when grain export is on.
    grain: Option<GrainExport>,
}

impl<R: Runtime> Nl4dDenoiser<R> {
    /// Builds the denoiser, rejecting invalid `params` and frames smaller than one patch.
    pub fn new(
        client: &ComputeClient<R>,
        mut params: Nl4dParams,
        width: u32,
        height: u32,
    ) -> Result<Self, String> {
        params.validate()?;

        if width < PATCH_SIZE || height < PATCH_SIZE {
            return Err(format!(
                "frame dimensions {width}x{height} must be at least {p}x{p} for the \
                 collaborative filter's patch grid",
                p = PATCH_SIZE,
            ));
        }

        // The ring view `submit_machinery` returns walks the front end's own window, so its radius
        // must match the grouping radius.
        params.nlm.temporal_radius = params.temporal_radius;
        params.nlm.validate().map_err(|error| error.to_string())?;

        let nlm_params = params.nlm.clone();
        let mut front = NlmDenoiser::new(client, nlm_params, width, height);

        let channels = params.nlm.channels;
        let apply_noise_map = params.noise_map && channels != ChannelMode::Chroma;
        let luma_map_params = StrengthMapParams {
            flat_boost: params.flat_boost,
            shadow_soften: params.shadow_soften,
        };
        let luma_map = (apply_noise_map && !luma_map_params.is_identity()).then_some(luma_map_params);
        let chroma_map_applies =
            params.noise_map && channels == ChannelMode::Chroma && params.chroma_flat_boost != 1.0;
        let chroma_map_boost = chroma_map_applies.then_some(params.chroma_flat_boost);
        front.set_luma_noise_fields(apply_noise_map || chroma_map_boost.is_some());
        let texture_cut_applies = apply_noise_map && params.flat_texture_cut < 1.0;
        let texture_cut = texture_cut_applies.then_some(params.flat_texture_cut);
        front.set_flat_texture_cut(texture_cut);
        front.set_shifted_edges(true);

        let stored_ch = channels.storage_count();
        let k_max = MAX_K;
        let refs = ref_count(width, height);
        let pixels = (width * height) as usize;
        let frame_len = pixels * stored_ch as usize;

        let group_weight = client.empty(refs * size_of::<f32>());
        let sigma_host = vec![0.0f32; stored_ch as usize];
        let sigma_bytes = f32::as_bytes(&sigma_host);
        let sigma_buf = client.create_from_slice(sigma_bytes);

        // The correlation profile is purely spatial and has no `rho` knob, so it is built once from
        // the white-noise default.
        let dct_profile = dct_noise_profile(0.0);
        let dct_profile_bytes = f32::as_bytes(&dct_profile);
        let dct_profile_buf = client.create_from_slice(dct_profile_bytes);
        let kaiser_taps = kaiser_window(params.kaiser_beta);
        let kaiser_bytes = f32::as_bytes(&kaiser_taps);
        let kaiser_buf = client.create_from_slice(kaiser_bytes);

        let (map_cols, map_rows) = strength_map_dims(width, height);
        let unit_map_host = vec![1.0f32; (map_cols * map_rows) as usize];
        let unit_map_bytes = f32::as_bytes(&unit_map_host);
        let unit_map = client.create_from_slice(unit_map_bytes);

        // `client.empty` memory is not zeroed, so a stream's first pass zeroes the whole ring.
        let ring_frames = 1 + 2 * params.temporal_radius;
        let accum = client.empty(frame_len * ring_frames as usize * size_of::<i32>());
        let wsum = client.empty(pixels * ring_frames as usize * size_of::<i32>());
        let outputs = [
            client.empty(frame_len * size_of::<f32>()),
            client.empty(frame_len * size_of::<f32>()),
        ];

        // `motion_ctx()` panics without motion compensation, which `validate` already requires.
        let (reg_mv, reg_conf) = if params.field_lambda > 0.0 {
            let motion_ctx = front.motion_ctx();
            let neighbours = 2 * params.temporal_radius as u64;
            let mv_bytes = neighbours * motion_ctx.mv_field_bytes_per_neighbour();
            let conf_bytes = neighbours * motion_ctx.confidence_bytes_per_neighbour();
            let reg_mv = client.empty(mv_bytes as usize);
            let reg_conf = client.empty(conf_bytes as usize);

            (Some(reg_mv), Some(reg_conf))
        } else {
            (None, None)
        };

        let exports_grain = params.grain_export && channels != ChannelMode::Chroma;
        let grain = if exports_grain {
            let motion_ctx = front.motion_ctx();
            let geometry = GrainGeometry {
                width,
                height,
                stored_ch,
                blocks_x: motion_ctx.blocks_x,
                blocks_y: motion_ctx.blocks_y,
                step: motion_ctx.step,
                ring_frames,
            };
            Some(GrainExport::new(client, geometry))
        } else {
            None
        };

        let warp_uniform = needs_warp_uniform_search(client);
        let accum_scale = cross_frame_accum_scale(params.spatial_radius, params.temporal_radius);

        Ok(Self {
            front,
            width,
            height,
            channels,
            pooled_threshold: params.pooled_threshold,
            temporal_radius: params.temporal_radius,
            refine: params.refine,
            spatial_radius: params.spatial_radius,
            lambda_ht: params.lambda_ht,
            c_min: params.c_min,
            k_max,
            warp_uniform,
            accum_scale,
            group_weight,
            sigma_buf,
            dct_profile_buf,
            dct_profile,
            kaiser_buf,
            accum,
            wsum,
            outputs,
            next_output_slot: 0,
            passes_run: 0,
            stream_start: StreamStart::SceneStart,
            last_fields: None,
            field_lambda: params.field_lambda,
            reg_mv,
            reg_conf,
            apply_noise_map,
            luma_map,
            chroma_map_boost,
            unit_map,
            map_cols,
            map_rows,
            grain,
        })
    }

    /// Runs the grouping passes a submit owes and returns the region they completed.
    ///
    /// The push that fills a scene's ring runs head passes centred on the ring's first
    /// `temporal_radius` frames, then the pass centred on its middle frame. Every later submit runs
    /// one pass centred on the middle frame.
    ///
    /// Once more than `temporal_radius` passes have run, each middle pass completes the region
    /// `temporal_radius` frames behind it. A continuation stream skips the head passes, so its
    /// first `temporal_radius` submits after the ring fills return `None`.
    pub(crate) fn submit_passes(&mut self) -> Result<Option<CompletedRegion>, anyhow::Error> {
        if !self.front.window_ready() {
            return Ok(None);
        }

        let radius = self.temporal_radius;
        let opens_scene = self.passes_run == 0 && self.stream_start == StreamStart::SceneStart;
        let mut clear = if self.passes_run == 0 {
            AccumClear::WholeRing
        } else {
            AccumClear::NewestRegion
        };

        if opens_scene {
            for centre in 0..radius {
                let view = self.machinery_at(centre)?;
                self.run_pass(&view, clear)?;
                self.save_grain_vectors(view.centre_slot, centre, 2 * radius);
                clear = AccumClear::Nothing;
            }
        }

        let view = self.machinery_at(radius)?;
        self.run_pass(&view, clear)?;
        self.save_grain_vectors(view.centre_slot, radius, 2 * radius);

        if self.passes_run <= radius {
            return Ok(None);
        }

        let total_frames = 1 + 2 * radius;
        let slot = (view.centre_slot + total_frames - radius) % total_frames;
        let next = Some(self.front.ring_slot(1));

        Ok(Some(CompletedRegion { slot, next }))
    }

    /// Normalises a finished region into the next output slot and returns that slot's buffer.
    pub(crate) fn read_region(&mut self, region: CompletedRegion) -> Handle {
        let (handle, output_slot) = self.normalise_region(region.slot);
        self.measure_grain(region.slot, region.next, output_slot);

        handle
    }

    pub(crate) fn push_planes(
        &mut self,
        planes: &[DevicePlane<'_>],
        format: SampleFormat,
    ) -> Result<(), anyhow::Error> {
        self.front.push_planes(planes, format)
    }

    /// A 4-byte handle to bind for planes a kernel never reads.
    pub(crate) fn placeholder(&self) -> &Handle {
        self.front.placeholder()
    }

    pub(crate) fn compute_client(&self) -> &ComputeClient<R> {
        self.front.compute_client()
    }

    pub(crate) fn frame_shape(&self) -> (u32, u32, ChannelMode) {
        (self.width, self.height, self.channels)
    }

    /// Marks the current stream as picking up mid-clip, so it runs no head passes.
    ///
    /// Only has an effect before the stream's first pass.
    pub(crate) fn mark_continuation(&mut self) {
        if self.passes_run == 0 {
            self.stream_start = StreamStart::Continuation;
        }
    }

    /// Runs the front end's motion and noise machinery for a pass centred on logical position `centre`.
    fn machinery_at(&mut self, centre: u32) -> Result<RingView, anyhow::Error> {
        let view = self.front.submit_machinery(centre)?;
        let view = view.expect("the ring is full whenever a pass runs");

        Ok(view)
    }

    /// Runs the passes a stream's end owes and returns the regions to read out, in emit order.
    ///
    /// A stream that filled its ring runs tail passes centred on its last `temporal_radius`
    /// frames, then reads out the last `2 * temporal_radius` regions. A shorter stream pads the
    /// ring with copies of its last frame, runs every real frame as a centre and reads out every
    /// real frame.
    pub(crate) fn finish_passes(&mut self) -> Result<Vec<CompletedRegion>, anyhow::Error> {
        let emit = self.flush_target() as u32;
        if emit == 0 {
            return Ok(Vec::new());
        }

        let radius = self.temporal_radius;
        let total_frames = 1 + 2 * radius;
        let short_stream = self.front.real_pushes() < total_frames as usize;
        if short_stream {
            self.front.fill_ring_with_last_frame()?;
        }

        let (centres, last_real) = if short_stream {
            (0..emit, emit - 1)
        } else {
            (radius + 1..2 * radius + 1, 2 * radius)
        };

        // A full ring with no pass run was primed through pushes alone, so its accumulators hold
        // stale data and the first tail pass clears the whole ring.
        let mut clear = if short_stream || self.passes_run == 0 {
            AccumClear::WholeRing
        } else {
            AccumClear::Nothing
        };
        for centre in centres {
            let view = self.machinery_at(centre)?;
            self.run_pass(&view, clear)?;
            self.save_grain_vectors(view.centre_slot, centre, last_real);
            clear = AccumClear::Nothing;
        }

        let first_region = last_real + 1 - emit;
        let mut regions = Vec::with_capacity(emit as usize);
        for logical in first_region..=last_real {
            let slot = self.front.ring_slot(logical);
            let next = (logical < last_real).then(|| self.front.ring_slot(logical + 1));
            regions.push(CompletedRegion { slot, next });
        }

        Ok(regions)
    }

    /// Drops the current stream and returns to a fresh denoiser's state, keeping every allocation.
    ///
    /// The pass counter resets, so the next stream's first pass clears the previous stream's
    /// contributions from the accumulator ring. The next stream starts a scene unless it is marked
    /// a continuation.
    pub fn reset_stream(&mut self) {
        self.front.reset_stream_state();
        self.next_output_slot = 0;
        self.passes_run = 0;
        self.stream_start = StreamStart::SceneStart;
        self.last_fields = None;

        if let Some(grain) = self.grain.as_mut() {
            grain.reset_stream();
        }
    }

    /// The motion field and confidence the last pass gave the fused kernel.
    ///
    /// A synchronous readback for measurement tooling. It is not a stable interface.
    #[doc(hidden)]
    pub fn motion_snapshot(&self) -> Option<MotionSnapshot> {
        let fields = self.last_fields.as_ref()?;
        let motion_ctx = self.front.motion_ctx();
        let snapshot = read_snapshot(
            self.front.compute_client(),
            fields,
            self.temporal_radius,
            motion_ctx.blocks_x,
            motion_ctx.blocks_y,
            motion_ctx.step,
            motion_ctx.blksize,
        );

        Some(snapshot)
    }

    /// Reads back the grain chunks measured since the last call. Empty when export is off.
    ///
    /// Measured chunks stay on the GPU until drained, so callers drain after each flush.
    pub fn drain_grain_chunks(&mut self) -> Result<Vec<GrainChunk>, anyhow::Error> {
        let Some(grain) = self.grain.as_mut() else {
            return Ok(Vec::new());
        };

        let chunks = grain.drain(self.front.compute_client())?;

        Ok(chunks)
    }

    #[cfg(test)]
    pub(crate) fn has_grain_export(&self) -> bool {
        self.grain.is_some()
    }

    /// How many measured frames had a saved grain entry to their next frame.
    #[cfg(test)]
    pub(crate) fn grain_measured_with_entry(&self) -> u32 {
        self.grain.as_ref().map_or(0, |grain| grain.measured_with_entry())
    }

    #[cfg(test)]
    pub(crate) fn front_for_test(&self) -> &NlmDenoiser<R> {
        &self.front
    }

    /// How many tail frames [Self::finish_passes] must emit.
    ///
    /// A stream that filled its ring holds `2 * temporal_radius` unread regions. A shorter stream
    /// holds every frame it pushed.
    fn flush_target(&self) -> usize {
        let real_pushes = self.front.real_pushes();
        if real_pushes == 0 {
            0
        } else {
            real_pushes.min(2 * self.temporal_radius as usize)
        }
    }

    /// Runs the grouping, filtering and aggregation kernels for one pass centred on `view`.
    ///
    /// Each filtered member scatters into the accumulator region of the frame it came from, so one
    /// pass adds to every region in the ring. `clear` picks which regions are zeroed first. The
    /// newest region is the slot `temporal_radius` ahead of the centre, which the newest frame has
    /// just taken over.
    fn run_pass(&mut self, view: &RingView, clear: AccumClear) -> Result<(), anyhow::Error> {
        // The kernel's centre slot must be the ring view's centre, or a member is grouped against
        // one frame and scattered into another.
        let centre_slot = view.centre_slot;

        let client = self.front.compute_client().clone();

        let stored_ch = self.channels.storage_count();
        let channels_count = self.channels.count();
        let pixels = (self.width * self.height) as usize;
        let frame_len = pixels * stored_ch as usize;
        let total_frames = 1 + 2 * self.temporal_radius;
        let ring_len = frame_len * total_frames as usize;
        let accum_ring_len = frame_len * total_frames as usize;
        let wsum_ring_len = pixels * total_frames as usize;

        let neighbours = 2 * self.temporal_radius;
        let mv_len = (neighbours * view.mv_stride) as usize;
        let conf_len = (neighbours * view.conf_stride) as usize;

        let neighbour_slot_bytes = u32::as_bytes(&view.neighbour_slots);
        let neighbour_slots_buf = client.create_from_slice(neighbour_slot_bytes);

        let sigmas = self.front.current_sigmas_temporal_only();
        let mut sigma_host = vec![0.0f32; stored_ch as usize];
        sigma_host[..channels_count as usize].copy_from_slice(&sigmas[..channels_count as usize]);
        let sigma_bytes = f32::as_bytes(&sigma_host);
        self.sigma_buf = client.create_from_slice(sigma_bytes);
        let weight_norm = weight_scale(sigma_host[0], &self.dct_profile);

        let curve_ratios = self.front.current_noise_curve().map(|curve| curve.ratios);
        let (ratios, curve_valid) = noise_curve_upload(curve_ratios, self.apply_noise_map);
        let ratio_bytes = f32::as_bytes(&ratios);
        let noise_curve_buf = client.create_from_slice(ratio_bytes);

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
                let strength_map_buf = client.create_from_slice(multiplier_bytes);

                (strength_map_buf, mode)
            },
            None => (self.unit_map.clone(), STRENGTH_MAP_OFF),
        };
        let map_len = (self.map_cols * self.map_rows) as usize;

        let refs_x = refs_along(self.width);
        let refs_y = refs_along(self.height);
        let refs = ref_count(self.width, self.height);

        let collab_cubes_x = fused_cubes_x(self.width);
        let collab_grid = CubeCount::new_2d(collab_cubes_x, refs_y);
        let collab_dim = CubeDim::new_1d(64);
        let zero_dim = 256u32;
        // Clamped to the 65,535 workgroup limit, which a 4:4:4 4K frame or an 8K luma plane alone
        // exceeds. `collab_zero_accum` strides, so a clamped launch still reaches every slot.
        let zero_workgroups_one_frame = (frame_len as u32).div_ceil(zero_dim).min(MAX_GRID_1D);
        let zero_grid_one_frame = CubeCount::new_1d(zero_workgroups_one_frame);
        let zero_total_threads_one_frame = zero_workgroups_one_frame * zero_dim;

        let motion_ctx = self.front.motion_ctx();
        let blk_step = motion_ctx.step;
        let blksize = motion_ctx.blksize;
        let blocks_x = motion_ctx.blocks_x;
        let blocks_y = motion_ctx.blocks_y;

        let cleared_slots = match clear {
            AccumClear::WholeRing => 0..total_frames,
            AccumClear::NewestRegion => {
                let newest_slot = (centre_slot + self.temporal_radius) % total_frames;
                newest_slot..newest_slot + 1
            },
            AccumClear::Nothing => 0..0,
        };

        self.passes_run += 1;

        let (mv_field, confidence) = match (self.reg_mv.as_ref(), self.reg_conf.as_ref()) {
            (Some(reg_mv), Some(reg_conf)) => {
                let sad_noise_floor = self.front.sad_noise_floor_value();
                let thsad = self.front.thsad_value();
                run_regularise::<R>(
                    &client,
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

                (reg_mv.clone(), reg_conf.clone())
            },
            _ => (view.mv_field.clone(), view.confidence.clone()),
        };

        self.last_fields = Some(LastFields {
            mv_field: mv_field.clone(),
            confidence: confidence.clone(),
            mv_stride: view.mv_stride,
            conf_stride: view.conf_stride,
            neighbours,
        });

        let frames_per_volume = grid_frames(self.temporal_radius);

        unsafe {
            // One dispatch over the whole ring exceeds the 65,535 workgroup limit at
            // `temporal_radius = 4` for a 1080p luma plane. A rejected dispatch leaves undefined
            // memory that a fresh stream would aggregate as real, so each region gets its own.
            for slot in cleared_slots {
                collab_zero_accum::launch_unchecked::<R>(
                    &client,
                    zero_grid_one_frame.clone(),
                    CubeDim::new_1d(zero_dim),
                    ArrayArg::from_raw_parts(self.accum.clone(), accum_ring_len),
                    ArrayArg::from_raw_parts(self.wsum.clone(), wsum_ring_len),
                    slot * pixels as u32,
                    pixels as u32,
                    stored_ch,
                    zero_total_threads_one_frame,
                );
            }

            let pool_ratio = nl4d_pool_ratio(self.channels);

            collab_fused::launch_unchecked::<R>(
                &client,
                collab_grid,
                collab_dim,
                stored_ch as usize,
                ArrayArg::from_raw_parts(view.input.clone(), ring_len),
                ArrayArg::from_raw_parts(mv_field.clone(), mv_len.max(1)),
                ArrayArg::from_raw_parts(confidence.clone(), conf_len.max(1)),
                ArrayArg::from_raw_parts(neighbour_slots_buf, view.neighbour_slots.len().max(1)),
                ArrayArg::from_raw_parts(self.sigma_buf.clone(), stored_ch as usize),
                ArrayArg::from_raw_parts(noise_curve_buf, NOISE_CURVE_BINS),
                ArrayArg::from_raw_parts(strength_map_buf, map_len),
                ArrayArg::from_raw_parts(self.dct_profile_buf.clone(), 8),
                ArrayArg::from_raw_parts(self.kaiser_buf.clone(), PATCH_SIZE as usize),
                ArrayArg::from_raw_parts(self.accum.clone(), accum_ring_len),
                ArrayArg::from_raw_parts(self.wsum.clone(), wsum_ring_len),
                ArrayArg::from_raw_parts(self.group_weight.clone(), refs),
                centre_slot,
                self.c_min,
                self.lambda_ht,
                curve_valid,
                map_mode,
                weight_norm,
                self.accum_scale,
                self.warp_uniform,
                self.temporal_radius,
                frames_per_volume,
                self.refine,
                view.mv_stride,
                view.conf_stride,
                blk_step,
                blksize,
                blocks_x,
                blocks_y,
                self.width,
                self.height,
                channels_count,
                self.k_max,
                stored_ch,
                self.spatial_radius,
                refs_x,
                self.map_cols,
                self.map_rows,
                pool_ratio,
                self.pooled_threshold,
            );
        }

        Ok(())
    }

    /// Saves the last pass's vectors from the centre to the next frame for grain export.
    ///
    /// `centre` is the pass's logical ring position and `last_real` the last logical position
    /// holding a real frame.
    fn save_grain_vectors(&mut self, centre_slot: u32, centre: u32, last_real: u32) {
        let Some(grain) = self.grain.as_mut() else {
            return;
        };

        let fields = self
            .last_fields
            .as_ref()
            .expect("a pass ran before its vectors are saved");
        // The motion field numbers neighbours in logical order and skips the centre, so the
        // frame at `centre + 1` is neighbour `centre`.
        let next_neighbour = (centre < last_real).then_some(centre);
        let client = self.front.compute_client();
        grain.save_vectors(client, fields, centre_slot, next_neighbour);
    }

    /// Measures the grain of the frame just normalised into `output_slot`.
    fn measure_grain(&mut self, slot_t: u32, slot_next: Option<u32>, output_slot: usize) {
        let Some(grain) = self.grain.as_mut() else {
            return;
        };

        let client = self.front.compute_client();
        let input = self.front.input_ring();
        grain.measure(client, input, &self.outputs, slot_t, slot_next, output_slot);
    }

    /// Normalises the accumulator region at `region_slot` into the next output buffer.
    ///
    /// Returns the output buffer and its index. The region is left as it was, and a later pass
    /// clears it before reuse.
    fn normalise_region(&mut self, region_slot: u32) -> (Handle, usize) {
        let client = self.front.compute_client().clone();
        let stored_ch = self.channels.storage_count();
        let pixels = (self.width * self.height) as usize;
        let frame_len = pixels * stored_ch as usize;
        let total_frames = 1 + 2 * self.temporal_radius;
        let accum_ring_len = frame_len * total_frames as usize;
        let wsum_ring_len = pixels * total_frames as usize;
        let agg_cubes_x = self.width.div_ceil(BLOCK_X);
        let agg_cubes_y = self.height.div_ceil(BLOCK_Y);
        let agg_grid = CubeCount::new_2d(agg_cubes_x, agg_cubes_y);
        let agg_dim = CubeDim::new_2d(BLOCK_X, BLOCK_Y);

        let slot = self.next_output_slot;
        self.next_output_slot = (slot + 1) % self.outputs.len();

        unsafe {
            collab_normalise::launch_unchecked::<R>(
                &client,
                agg_grid,
                agg_dim,
                stored_ch as usize,
                ArrayArg::from_raw_parts(self.accum.clone(), accum_ring_len),
                ArrayArg::from_raw_parts(self.wsum.clone(), wsum_ring_len),
                ArrayArg::from_raw_parts(self.outputs[slot].clone(), frame_len),
                region_slot * pixels as u32,
                self.width,
                self.height,
                self.channels.count(),
                stored_ch,
            );
        }

        (self.outputs[slot].clone(), slot)
    }
}

/// The noise curve a pass uploads, and the flag that tells the kernel to apply it.
///
/// A missing curve, or one that does not apply, uploads zeroes with the flag at 0.
pub(super) fn noise_curve_upload(
    ratios: Option<[f32; NOISE_CURVE_BINS]>,
    applies: bool,
) -> ([f32; NOISE_CURVE_BINS], u32) {
    match ratios {
        Some(ratios) if applies => (ratios, 1),
        _ => ([0.0f32; NOISE_CURVE_BINS], 0),
    }
}

/// The strength map a pass uploads, and the mode the kernel applies it in.
///
/// `None` binds the unit map with the map off. The luma map needs a curve the pass applies, and
/// the chroma map needs only the quarter classes.
pub(super) fn strength_map_upload(
    classes: Option<&QuarterClasses>,
    curve_valid: u32,
    luma_map: Option<StrengthMapParams>,
    chroma_map_boost: Option<f32>,
) -> Option<(Vec<f32>, u32)> {
    let classes = classes?;

    if let Some(params) = luma_map
        && curve_valid == 1
    {
        let multipliers = classes.luma_multipliers(params);
        return Some((multipliers, STRENGTH_MAP_LUMA));
    }

    let boost = chroma_map_boost?;
    let multipliers = classes.chroma_multipliers(boost);
    Some((multipliers, STRENGTH_MAP_ALL))
}
