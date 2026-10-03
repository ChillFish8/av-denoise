use cubecl::prelude::*;
use cubecl::server::Handle;

use super::grain::{GrainChunk, GrainExport, GrainGeometry};
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
use crate::denoiser::{DenoiserError, FrameOutput, OutputFormat, nl4d_pool_ratio};
use crate::nlmeans::{
    BLOCK_X,
    BLOCK_Y,
    ChannelMode,
    Depth,
    MAX_GRID_1D,
    NOISE_CURVE_BINS,
    NlmDenoiser,
    Pending,
    QuarterClasses,
    RingView,
    StrengthMapParams,
    start_readback,
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

/// How the current stream began.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamStart {
    /// The stream starts a scene, so its first frames get off-centre head passes.
    SceneStart,
    /// The stream picks up mid-clip from priming pushes, so it runs no head passes.
    Continuation,
}

/// Groups similar 8x8 patches across a motion-compensated temporal
/// window and denoises each group jointly.
///
/// This drives the NLMeans front end, but only for its machinery, the
/// frame ring, the motion field, and the confidence scores built by
/// [`NlmDenoiser::submit_machinery`].
/// No NLM weighting kernel ever runs. Instead, every submit hands the
/// noisy ring to [`collab_fused`], which groups patches by searching
/// both the centre frame spatially and each neighbour frame around
/// where motion compensation predicts a patch moved, shrinks each
/// group's coefficients in the transform domain, and scatters the
/// filtered members back into the accumulator ring.
/// [`collab_normalise`] then turns one region of that ring into a
/// finished frame.
///
/// Every pass scatters its filtered members into whichever frame each one
/// came from, not only the centre frame, so a frame's own output finishes
/// only once every pass that can reach it has run.
///
/// Latency is `2 * temporal_radius` pushes, twice the front end's own
/// window depth. [`Self::denoise_submit`] returns `None` while the front
/// end's window is still filling, and [`Self::flush`] drains the frames
/// still held once the input stream ends.
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
    /// Whether [`collab_fused`] runs its warp-uniform search, decided
    /// once from the runtime this denoiser was built on. See
    /// [`needs_warp_uniform_search`].
    warp_uniform: bool,
    /// The fixed-point scale the cross-frame accumulator ring counts in,
    /// from
    /// [`crate::collab::kernels::aggregate::cross_frame_accum_scale`].
    ///
    /// Both radii it derives from are fixed for the denoiser's lifetime,
    /// so this is worked out once here rather than on every
    /// [`Self::run_pass`] call.
    accum_scale: f32,

    group_weight: Handle,
    sigma_buf: Handle,
    dct_profile_buf: Handle,
    /// The aggregation window's 8 taps, built once from the caller's
    /// `kaiser_beta`. Eight ones when that is 0.
    kaiser_buf: Handle,
    /// The correlation profile kept on the host too, so the weight
    /// normalisation can be derived from it every submit without a
    /// device readback.
    dct_profile: [f32; 8],
    /// Fixed-point accumulators the filter scatters into, one weighted
    /// value per covering patch.
    ///
    /// These hold `1 + 2 * temporal_radius` frames' worth of pixels back
    /// to back, one region per physical ring slot of the front end's own
    /// frame ring.
    ///
    /// A pass contributes to every frame in the ring, so a frame's region
    /// stays live across every pass run while it sits in the ring. See
    /// [`Self::denoise_submit`] for when a region is read back.
    accum: Handle,
    wsum: Handle,
    /// Two output buffers, alternated so one frame's kernels can overlap
    /// the previous frame's readback.
    outputs: [Handle; 2],
    next_output_slot: usize,
    /// The format every readback this denoiser starts comes back in.
    output_format: OutputFormat,
    /// Packed-word destinations, one per entry of `outputs`, allocated
    /// only in wire mode.
    ///
    /// These buffers rotate on the same slot counter, so each is free again exactly
    /// when the `f32` slot it is packed from is free.
    wire_outputs: Option<[Handle; 2]>,
    /// How many passes [`Self::run_pass`] has run for the current stream.
    ///
    /// The stream's first pass zeroes the whole of `accum`/`wsum`, so a
    /// zero here also means the ring may hold a previous stream's stale
    /// contributions. It resets to 0 in [`Self::reset_stream`].
    passes_run: u32,
    /// How the current stream began, set by [`Self::mark_continuation`].
    stream_start: StreamStart,
    /// The field buffers the last pass handed the fused kernel, for
    /// [`Self::motion_snapshot`].
    last_fields: Option<LastFields>,
    /// See [`Nl4dParams::field_lambda`].
    field_lambda: f32,
    /// The regularised motion field and confidence, laid out like the
    /// front end's own, allocated only when `field_lambda > 0.0`.
    reg_mv: Option<Handle>,
    reg_conf: Option<Handle>,
    /// Whether passes apply the luma noise curve.
    ///
    /// Set when [Nl4dParams::noise_map](crate::nl4d::Nl4dParams::noise_map)
    /// is on and the denoiser filters luma.
    apply_noise_map: bool,
    /// The luma map's multipliers, set when `apply_noise_map` is and they are not all 1.0.
    luma_map: Option<StrengthMapParams>,
    /// The flat boost a chroma denoiser applies, set when the noise map is on and the boost is not
    /// 1.0.
    chroma_map_boost: Option<f32>,
    /// A strength map of all 1.0, bound by every pass with no map to apply.
    unit_map: Handle,
    map_cols: u32,
    map_rows: u32,
    /// The grain measurement state, present only when grain export is on.
    grain: Option<GrainExport>,
}

impl<R: Runtime> Nl4dDenoiser<R> {
    /// Builds a new denoiser.
    ///
    /// Rejects an invalid `params` (see [`Nl4dParams::validate`]) and a
    /// frame smaller than one collaborative patch on either axis.
    pub fn new(
        client: &ComputeClient<R>,
        params: Nl4dParams,
        width: u32,
        height: u32,
    ) -> Result<Self, String> {
        Self::with_output_format(client, params, width, height, OutputFormat::F32)
    }

    /// Builds a new denoiser whose readbacks come back in `output_format`.
    ///
    /// [`OutputFormat::Wire`] gives the denoiser a packed-word buffer
    /// per output slot, so a readback quantises on the GPU and only the
    /// wire bytes cross the bus.
    ///
    /// Rejects the same `params` and dimensions [`Self::new`] does.
    pub fn with_output_format(
        client: &ComputeClient<R>,
        mut params: Nl4dParams,
        width: u32,
        height: u32,
        output_format: OutputFormat,
    ) -> Result<Self, String> {
        params.validate()?;

        if width < PATCH_SIZE || height < PATCH_SIZE {
            return Err(format!(
                "frame dimensions {width}x{height} must be at least {p}x{p} for the \
                 collaborative filter's patch grid",
                p = PATCH_SIZE,
            ));
        }

        // The front end's own temporal radius has to match the grouping
        // radius exactly, since `submit_machinery`'s ring view walks the
        // front end's own window, so this is forced here rather than
        // trusted to the caller.
        params.nlm.temporal_radius = params.temporal_radius;
        params.nlm.validate().map_err(|e| e.to_string())?;

        // The front end only supplies the ring, motion field, and
        // confidence scores the collaborative stage reads. Its own
        // buffers never leave the GPU, so it stays in `f32` whatever
        // format this denoiser hands back.
        let mut front =
            NlmDenoiser::with_output_format(client, params.nlm.clone(), width, height, OutputFormat::F32);

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
        let sigma_buf = client.create_from_slice(f32::as_bytes(&vec![0.0f32; stored_ch as usize]));
        // The correlation profile is purely spatial and this denoiser
        // exposes no `rho` knob, so it is built once here from the
        // white-noise default rather than every submit.
        let dct_profile = dct_noise_profile(0.0);
        let dct_profile_buf = client.create_from_slice(f32::as_bytes(&dct_profile));
        // The window depends only on `kaiser_beta`, which cannot change
        // over a denoiser's life, so it is built here rather than every
        // submit.
        let kaiser_buf = client.create_from_slice(f32::as_bytes(&kaiser_window(params.kaiser_beta)));

        let (map_cols, map_rows) = strength_map_dims(width, height);
        let unit_map_host = vec![1.0f32; (map_cols * map_rows) as usize];
        let unit_map = client.create_from_slice(f32::as_bytes(&unit_map_host));

        // One region per physical ring slot of the front end's own frame
        // ring, `1 + 2 * temporal_radius` of them, see the `accum` field
        // doc for why. The ring is zeroed in full by a stream's first pass
        // rather than here, since `client.empty` gives no guarantee its memory
        // starts zeroed.
        let ring_frames = 1 + 2 * params.temporal_radius;
        let accum = client.empty(frame_len * ring_frames as usize * size_of::<i32>());
        let wsum = client.empty(pixels * ring_frames as usize * size_of::<i32>());
        let outputs = [
            client.empty(frame_len * size_of::<f32>()),
            client.empty(frame_len * size_of::<f32>()),
        ];
        let wire_outputs = match output_format {
            OutputFormat::F32 => None,
            OutputFormat::Wire { depth } => {
                let samples = pixels as u32 * channels.count();
                let words = samples.div_ceil(depth.wire_pack().samples_per_word()) as usize;
                Some([
                    client.empty(words * size_of::<u32>()),
                    client.empty(words * size_of::<u32>()),
                ])
            },
        };

        // `motion_ctx()` panics without motion compensation, and
        // `validate` above already requires it, so this is safe here.
        let (reg_mv, reg_conf) = if params.field_lambda > 0.0 {
            let mc = front.motion_ctx();
            let neighbours = 2 * params.temporal_radius as u64;
            (
                Some(client.empty((neighbours * mc.mv_field_bytes_per_neighbour()) as usize)),
                Some(client.empty((neighbours * mc.confidence_bytes_per_neighbour()) as usize)),
            )
        } else {
            (None, None)
        };

        let exports_grain = params.grain_export && channels != ChannelMode::Chroma;
        let grain = if exports_grain {
            let mc = front.motion_ctx();
            let geometry = GrainGeometry {
                width,
                height,
                stored_ch,
                blocks_x: mc.blocks_x,
                blocks_y: mc.blocks_y,
                step: mc.step,
                ring_frames,
            };
            Some(GrainExport::new(client, geometry))
        } else {
            None
        };

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
            warp_uniform: needs_warp_uniform_search(client),
            accum_scale: cross_frame_accum_scale(params.spatial_radius, params.temporal_radius),
            group_weight,
            sigma_buf,
            dct_profile_buf,
            dct_profile,
            kaiser_buf,
            accum,
            wsum,
            outputs,
            next_output_slot: 0,
            output_format,
            wire_outputs,
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

    /// Pushes a new frame into the front end's ring buffer.
    ///
    /// `frame` holds `width * height * channels` `f32` values in
    /// `[0, 1]`, matching [`NlmDenoiser::push_frame`].
    pub fn push_frame(&mut self, frame: &[f32]) {
        self.front.push_frame(frame);
    }

    /// Pushes a new frame held as wire bytes into the front end's ring
    /// buffer.
    ///
    /// `planes` holds one `width * height` plane per channel at `depth`,
    /// matching [`NlmDenoiser::push_frame_wire`].
    pub fn push_frame_wire(&mut self, planes: &[&[u8]], depth: Depth) {
        self.front.push_frame_wire(planes, depth);
    }

    /// Runs one submit's worth of grouping, filtering, and aggregation,
    /// and starts the readback.
    ///
    /// Returns `Ok(None)` while the front end's ring is still filling.
    ///
    /// The push that fills a scene's ring runs head passes centred on the
    /// ring's first `temporal_radius` frames, then the pass centred on its
    /// middle frame. Every later submit runs one pass centred on the
    /// ring's middle frame.
    ///
    /// Once more than `temporal_radius` passes have run, each pass centred
    /// on the ring's middle completes the region `temporal_radius` frames
    /// behind it, and that region is read back. A continuation stream
    /// skips the head passes, so its first `temporal_radius` submits after
    /// the ring fills return `Ok(None)`. Latency stays
    /// `2 * temporal_radius` pushes for a scene start.
    ///
    /// There are two output slots, so at most two [`Pending`]s from this
    /// denoiser may be outstanding at once. A third concurrent submit
    /// reuses the oldest one's slot and silently corrupts it.
    ///
    /// The frame comes back in the [`OutputFormat`] this denoiser was
    /// built with. [`OutputFormat::Wire`] quantises and packs the frame
    /// on the GPU before the readback, so only the wire bytes cross the
    /// bus.
    pub fn denoise_submit(&mut self) -> Result<Option<Pending<R>>, DenoiserError> {
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
        let completed_slot = (view.centre_slot + total_frames - radius) % total_frames;
        let (handle, slot) = self.normalise_region(completed_slot);
        let next_slot = self.front.ring_slot(1);
        self.measure_grain(completed_slot, Some(next_slot), slot);
        let wire_dst = self.wire_outputs.as_ref().map(|outputs| &outputs[slot]);
        let pending = self.start_readback(handle, wire_dst, self.output_format);

        Ok(Some(pending))
    }

    /// Marks the current stream as picking up mid-clip, so it runs no head passes.
    ///
    /// Only has an effect before the stream's first pass.
    pub(crate) fn mark_continuation(&mut self) {
        if self.passes_run == 0 {
            self.stream_start = StreamStart::Continuation;
        }
    }

    /// Runs the front end's motion and noise machinery for a pass centred on logical ring position `centre`.
    fn machinery_at(&mut self, centre: u32) -> Result<RingView, DenoiserError> {
        let view = self.front.submit_machinery(centre)?;
        let view = view.expect("the ring is full whenever a pass runs");
        Ok(view)
    }

    /// Submits and waits for the result in one call.
    ///
    /// Prefer [`Self::denoise_submit`] when the caller can hold a frame
    /// in flight.
    ///
    /// The frame comes back in the [`OutputFormat`] this denoiser was
    /// built with.
    pub fn denoise(&mut self) -> Result<Option<FrameOutput>, DenoiserError> {
        let Some(pending) = self.denoise_submit()? else {
            return Ok(None);
        };
        Ok(Some(pending.wait()?))
    }

    /// Produces the frames still held at the end of a stream.
    ///
    /// A stream that filled its ring runs off-centre passes centred on its
    /// last `temporal_radius` frames, then reads out the last
    /// `2 * temporal_radius` frames' regions. A stream too short to fill
    /// its ring pads the ring with copies of its last frame, runs every
    /// real frame as a centre, then reads out every real frame.
    ///
    /// `sink` is called once per frame, in order, and the frame it
    /// receives is only valid for that call. It arrives in the
    /// [`OutputFormat`] this denoiser was built with, quantised by the
    /// same pack kernel as every streaming frame.
    pub fn flush(&mut self, mut sink: impl FnMut(&FrameOutput)) -> Result<(), DenoiserError> {
        let emit = self.flush_target() as u32;
        if emit == 0 {
            self.reset_stream();
            return Ok(());
        }

        let radius = self.temporal_radius;
        let total_frames = 1 + 2 * radius;
        let short_stream = self.front.real_pushes() < total_frames as usize;
        if short_stream {
            self.front.fill_ring_with_last_frame();
        }

        let (centres, last_real) = if short_stream {
            (0..emit, emit - 1)
        } else {
            (radius + 1..2 * radius + 1, 2 * radius)
        };

        // A full ring with no pass run means a caller primed every slot
        // through pushes alone and never called `denoise_submit`, so
        // `accum`/`wsum` are still whatever the last stream, or nothing
        // at all, left in them. The tail path's first pass has to clear
        // the whole ring in that case, the same as a short stream does.
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

        // Every output slot is free here. A caller reaches a flush only
        // once its streaming readbacks have landed, and each readback
        // below blocks before the next region reuses a slot.
        let first_region = last_real + 1 - emit;
        for logical in first_region..=last_real {
            let region_slot = self.front.ring_slot(logical);
            let (handle, slot) = self.normalise_region(region_slot);
            let next_slot = (logical < last_real).then(|| self.front.ring_slot(logical + 1));
            self.measure_grain(region_slot, next_slot, slot);
            let wire_dst = self.wire_outputs.as_ref().map(|outputs| &outputs[slot]);
            let pending = self.start_readback(handle, wire_dst, self.output_format);
            let frame = pending.wait()?;
            sink(&frame);
        }

        self.reset_stream();

        Ok(())
    }

    /// Drops the current stream and returns to the state a fresh
    /// denoiser starts in, keeping every GPU allocation.
    ///
    /// Clears the front end's own stream state plus the cross-frame
    /// accumulator's pass counter and output slot, so a window primed
    /// after this call never reads a previous window's stale
    /// contributions out of the fixed-point `accum`/`wsum` ring. The next
    /// stream starts a scene unless it is marked a continuation.
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

    /// The motion field and confidence the last pass gave the fused
    /// kernel, or `None` before any pass has run.
    ///
    /// This is a synchronous readback for measurement tooling, not a
    /// stable interface.
    #[doc(hidden)]
    pub fn motion_snapshot(&self) -> Option<MotionSnapshot> {
        let fields = self.last_fields.as_ref()?;
        let mc = self.front.motion_ctx();
        Some(read_snapshot(
            self.front.compute_client(),
            fields,
            self.temporal_radius,
            mc.blocks_x,
            mc.blocks_y,
            mc.step,
            mc.blksize,
        ))
    }

    /// Reads back the grain chunks measured since the last call. Empty when export is off.
    ///
    /// Measured chunks stay on the GPU until drained, so callers drain after each flush.
    pub fn drain_grain_chunks(&mut self) -> Result<Vec<GrainChunk>, DenoiserError> {
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

    /// The front end this denoiser drives.
    #[cfg(test)]
    pub(crate) fn front_for_test(&self) -> &NlmDenoiser<R> {
        &self.front
    }

    /// How many tail frames [`Self::flush`] must emit for the stream
    /// pushed so far.
    ///
    /// A stream that filled its ring holds `2 * temporal_radius` frames
    /// whose regions are not yet read out. A shorter stream holds every
    /// frame it pushed.
    fn flush_target(&self) -> usize {
        let real_pushes = self.front.real_pushes();
        if real_pushes == 0 {
            0
        } else {
            real_pushes.min(2 * self.temporal_radius as usize)
        }
    }

    /// Runs the grouping, filtering, and aggregation kernels for one pass.
    ///
    /// The pass is centred on the physical ring slot `view.centre_slot`.
    /// It groups the centre frame against every other frame in the ring,
    /// and each filtered member scatters into the region of
    /// `self.accum`/`self.wsum` for the frame it came from (see
    /// [`collab_fused`]'s scatter). So one pass adds to every region in
    /// the ring, not just the centre's.
    ///
    /// Before scattering, `clear` picks which regions to zero.
    /// [`AccumClear::WholeRing`] zeroes every slot, since nothing has
    /// cleared a new stream's ring.
    /// [`AccumClear::NewestRegion`] does the same for the slot
    /// `temporal_radius` ahead of the centre, which the newest frame has
    /// just taken over. [`AccumClear::Nothing`] leaves every region as it
    /// is, for an edge pass that sees no new frame.
    fn run_pass(&mut self, view: &RingView, clear: AccumClear) -> Result<(), DenoiserError> {
        // The frame-slot contract: `collab_fused`'s `centre_slot` and
        // the ring view's own centre must be the same physical slot, or
        // a member gets grouped against one frame and scattered as
        // though it belonged to another.
        let centre_slot = view.centre_slot;

        let client = self.front.compute_client().clone();

        let stored_ch = self.channels.storage_count();
        let channels_count = self.channels.count();
        let pixels = (self.width * self.height) as usize;
        let frame_len = pixels * stored_ch as usize;
        let total_frames = 1 + 2 * self.temporal_radius;
        let ring_len = frame_len * total_frames as usize;

        // The accumulators' own ring, one region per physical slot of
        // the frame ring above, the same `total_frames` count.
        let accum_ring_len = frame_len * total_frames as usize;
        let wsum_ring_len = pixels * total_frames as usize;

        let neighbours = 2 * self.temporal_radius;
        let mv_len = (neighbours * view.mv_stride) as usize;
        let conf_len = (neighbours * view.conf_stride) as usize;

        let neighbour_slots_buf = client.create_from_slice(u32::as_bytes(&view.neighbour_slots));

        let sigmas = self.front.current_sigmas_temporal_only();
        let mut sigma_host = vec![0.0f32; stored_ch as usize];
        sigma_host[..channels_count as usize].copy_from_slice(&sigmas[..channels_count as usize]);
        self.sigma_buf = client.create_from_slice(f32::as_bytes(&sigma_host));
        let wnorm = weight_scale(sigma_host[0], &self.dct_profile);

        let curve_ratios = self.front.current_noise_curve().map(|curve| curve.ratios);
        let (ratios, curve_valid) = noise_curve_upload(curve_ratios, self.apply_noise_map);
        let noise_curve_buf = client.create_from_slice(f32::as_bytes(&ratios));

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
            Some((multipliers, mode)) => (client.create_from_slice(f32::as_bytes(&multipliers)), mode),
            None => (self.unit_map.clone(), STRENGTH_MAP_OFF),
        };
        let map_len = (self.map_cols * self.map_rows) as usize;

        let refs_x = refs_along(self.width);
        let refs_y = refs_along(self.height);
        let refs = ref_count(self.width, self.height);

        // The kernel packs eight references into one 64-lane cube, so
        // its grid is an eighth as wide as the reference grid along x.
        let collab_grid = CubeCount::new_2d(fused_cubes_x(self.width), refs_y);
        let collab_dim = CubeDim::new_1d(64);
        let zero_dim = 256u32;
        // Sized for one frame's worth of the ring, and issued once per
        // region the pass clears.
        //
        // Still clamped to the GPU's 65,535-workgroups-per-dimension
        // limit, because one frame alone can exceed it. A 4:4:4 4K frame
        // or an 8K luma plane both need more than that at 256 threads
        // each. `collab_zero_accum` strides, so a clamped launch still
        // reaches every slot in the frame.
        let zero_workgroups_one_frame = (frame_len as u32).div_ceil(zero_dim).min(MAX_GRID_1D);
        let zero_grid_one_frame = CubeCount::new_1d(zero_workgroups_one_frame);
        let zero_total_threads_one_frame = zero_workgroups_one_frame * zero_dim;

        let mc = self.front.motion_ctx();
        let blk_step = mc.step;
        let blksize = mc.blksize;
        let blocks_x = mc.blocks_x;
        let blocks_y = mc.blocks_y;

        // The physical slots whose regions this pass resets before
        // scattering.
        let cleared_slots = match clear {
            AccumClear::WholeRing => 0..total_frames,
            AccumClear::NewestRegion => {
                let newest_slot = (centre_slot + self.temporal_radius) % total_frames;
                newest_slot..newest_slot + 1
            },
            AccumClear::Nothing => 0..0,
        };

        self.passes_run += 1;

        // The field the fused kernel reads, the regularised one when the
        // pass is on.
        let (mv_field, confidence) = match (self.reg_mv.as_ref(), self.reg_conf.as_ref()) {
            (Some(mv), Some(conf)) => {
                run_regularise::<R>(
                    &client,
                    mc,
                    view,
                    self.width,
                    self.height,
                    self.field_lambda,
                    self.front.sad_noise_floor_value(),
                    self.front.thsad_value(),
                    mv,
                    conf,
                )
                .map_err(DenoiserError::Other)?;
                (mv.clone(), conf.clone())
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

        unsafe {
            // Clearing the whole ring in one dispatch would need
            // `accum_ring_len.div_ceil(zero_dim)` workgroups, which grows
            // with `total_frames`. At `temporal_radius = 4` a 1080p luma
            // plane alone needs 72,900, already over the GPU's 65,535
            // limit. A rejected dispatch would leave the ring holding
            // `client.empty`'s undefined memory instead of zero, which a
            // fresh stream's first frames would then aggregate as though
            // it were real. So each cleared region gets its own dispatch.
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
                wnorm,
                self.accum_scale,
                self.warp_uniform,
                self.temporal_radius,
                grid_frames(self.temporal_radius),
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
        grain.save_vectors(self.front.compute_client(), fields, centre_slot, next_neighbour);
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

    /// Normalises the accumulator region at physical slot `region_slot` into the next output buffer.
    ///
    /// Returns the output buffer and its index. The region is left as it
    /// was, and a later pass clears it before reuse.
    fn normalise_region(&mut self, region_slot: u32) -> (Handle, usize) {
        let client = self.front.compute_client().clone();
        let stored_ch = self.channels.storage_count();
        let pixels = (self.width * self.height) as usize;
        let frame_len = pixels * stored_ch as usize;
        let total_frames = 1 + 2 * self.temporal_radius;
        let accum_ring_len = frame_len * total_frames as usize;
        let wsum_ring_len = pixels * total_frames as usize;
        let agg_grid = CubeCount::new_2d(self.width.div_ceil(BLOCK_X), self.height.div_ceil(BLOCK_Y));
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

    /// Starts an async readback of `handle`, wrapped in the same
    /// [`Pending`] type [`NlmDenoiser`] returns.
    ///
    /// `wire_dst` is the packed-word buffer belonging to the slot
    /// `handle` came from, and is `None` for an `f32` readback.
    fn start_readback(&self, handle: Handle, wire_dst: Option<&Handle>, format: OutputFormat) -> Pending<R> {
        let pixels = (self.width * self.height) as usize;
        start_readback(
            self.front.compute_client(),
            handle,
            wire_dst,
            self.channels.count(),
            self.channels.storage_count(),
            pixels,
            format,
        )
    }

    /// The packed-word destinations, which are `Some` only in wire mode.
    #[cfg(test)]
    pub(crate) fn wire_outputs_for_test(&self) -> Option<&[Handle; 2]> {
        self.wire_outputs.as_ref()
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
