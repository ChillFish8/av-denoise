use cubecl::prelude::*;
use cubecl::server::Handle;

use super::params::Nl4dParams;
use crate::collab::geometry::{fused_cubes_x, ref_count, refs_along};
use crate::collab::kernels::aggregate::{
    collab_normalise,
    collab_zero_accum,
    cross_frame_accum_scale,
    kaiser_window,
    weight_scale,
};
use crate::collab::kernels::fused::collab_fused;
use crate::collab::kernels::transforms::dct_noise_profile;
use crate::collab::{MAX_K, PATCH_SIZE};
use crate::denoiser::{DenoiserError, FrameOutput, OutputFormat};
use crate::nlmeans::{
    BLOCK_X,
    BLOCK_Y,
    ChannelMode,
    Depth,
    MAX_GRID_1D,
    NlmDenoiser,
    Pending,
    RingView,
    start_readback,
};

/// Groups similar 8x8 patches across a motion-compensated temporal
/// window and denoises each group jointly.
///
/// This drives the NLMeans front end, but only for its machinery, the
/// frame ring, the motion field, and the confidence scores built by
/// [`NlmDenoiser::submit_machinery`]/[`NlmDenoiser::flush_step_machinery`].
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
/// only once every pass that can reach it has run, which is
/// `temporal_radius` passes on either side of it.
///
/// Latency is therefore `2 * temporal_radius` pushes, twice the front
/// end's own window depth. [`Self::denoise_submit`] returns `None` while
/// the front end's window is still filling and for the further passes
/// this cross-frame accumulation needs, and [`Self::flush`] drains the
/// frames still held once the input stream ends.
pub struct Nl4dDenoiser<R: Runtime> {
    front: NlmDenoiser<R>,
    width: u32,
    height: u32,
    channels: ChannelMode,
    temporal_radius: u32,
    refine: u32,
    spatial_radius: u32,
    lambda_ht: f32,
    c_min: f32,
    /// See [`Nl4dParams::mismatch_scale`].
    mismatch_scale: f32,
    confidence_variance: bool,
    k_max: u32,
    /// The fixed-point scale the cross-frame accumulator ring counts in,
    /// from
    /// [`crate::collab::kernels::aggregate::cross_frame_accum_scale`].
    ///
    /// Both radii it derives from are fixed for the denoiser's lifetime,
    /// so this is worked out once here rather than on every
    /// [`Self::run_collab_stage`] call.
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
    /// A pass centred on frame `u` contributes to every frame in
    /// `u - temporal_radius ..= u + temporal_radius`, so a frame's region
    /// stays live across that many consecutive passes before
    /// [`Self::run_collab_stage`] reads it back and clears it for reuse.
    /// See that method for the exact scheduling.
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
    /// How many passes [`Self::run_collab_stage`] has run for the
    /// current stream.
    ///
    /// Pass 0 also full-zeroes `accum`/`wsum`, since every later pass
    /// only zeroes the one region it is about to reuse (see
    /// [`Self::run_collab_stage`]). This therefore doubles as "does the
    /// ring still hold a previous stream's stale contributions". It
    /// resets to 0 at the end of [`Self::flush`], alongside the front
    /// end's own stream state.
    ///
    /// A pass emits an output only once `passes_run > temporal_radius`.
    /// The earliest a frame's region has contributions from every pass
    /// that can reach it is `temporal_radius` passes after the pass
    /// centred on that frame.
    ///
    /// [`Self::denoise_submit`] and [`Self::flush`] read this through
    /// [`Self::run_collab_stage`]'s return value rather than checking it
    /// themselves.
    passes_run: u32,
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
        let front =
            NlmDenoiser::with_output_format(client, params.nlm.clone(), width, height, OutputFormat::F32);

        let channels = params.nlm.channels;
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
        // One region per physical ring slot of the front end's own frame
        // ring, `1 + 2 * temporal_radius` of them, see the `accum` field
        // doc for why. The ring is zeroed in full by pass 0 rather than
        // here, since `client.empty` gives no guarantee its memory
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

        Ok(Self {
            front,
            width,
            height,
            channels,
            temporal_radius: params.temporal_radius,
            refine: params.refine,
            spatial_radius: params.spatial_radius,
            lambda_ht: params.lambda_ht,
            c_min: params.c_min,
            mismatch_scale: params.mismatch_scale,
            confidence_variance: params.confidence_variance,
            k_max,
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
    /// Returns `Ok(None)` while the front end's temporal window is still
    /// filling, and also for `temporal_radius` further submits after
    /// that, while the earliest frames' cross-frame accumulation is
    /// still gathering contributions from passes that have not run yet.
    /// End-to-end latency is therefore `2 * temporal_radius` pushes, not
    /// `temporal_radius`, see [`Self::run_collab_stage`].
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
        let Some(view) = self.front.submit_machinery()? else {
            return Ok(None);
        };
        let Some((handle, slot)) = self.run_collab_stage(&view)? else {
            return Ok(None);
        };
        let wire_dst = self.wire_outputs.as_ref().map(|w| &w[slot]);
        Ok(Some(self.start_readback(handle, wire_dst, self.output_format)))
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
    /// For the last few frames the front end keeps its temporal window
    /// full by repeating the final pushed frame, exactly as
    /// [`NlmDenoiser::flush`] does. `sink` is called once per frame
    /// produced, and the frame it receives is only valid for that call.
    /// It arrives in the [`OutputFormat`] this denoiser was built with,
    /// quantised by the same pack kernel as every streaming frame.
    ///
    /// This drives [`NlmDenoiser::flush_step_machinery`] for
    /// [`Self::flush_target`] emissions, which is `2 * temporal_radius`
    /// duplicate-driven passes for a long enough stream, twice what the
    /// front end's own [`NlmDenoiser::flush_target`] would give.
    ///
    /// The front end only has to let every real frame finish a turn as a
    /// pass's own centre. This cross-frame stage also has to let every
    /// real frame finish gathering the `temporal_radius` trailing passes
    /// its own accumulation needs, which is another `temporal_radius`
    /// pushes' worth of passes.
    ///
    /// A call to [`NlmDenoiser::flush_step_machinery`] runs a pass
    /// whether or not it emits, so the loop below keeps calling it until
    /// enough passes have actually emitted.
    pub fn flush(&mut self, mut sink: impl FnMut(&FrameOutput)) -> Result<(), DenoiserError> {
        let target = self.flush_target();
        let mut emitted = 0usize;

        // Every output slot is free here. A caller reaches a flush only
        // once its streaming readbacks have landed, and the readback
        // below blocks, so no other readback is ever reading the slot
        // this step is handed.
        while emitted < target {
            if let Some(view) = self.front.flush_step_machinery()?
                && let Some((handle, slot)) = self.run_collab_stage(&view)?
            {
                let wire_dst = self.wire_outputs.as_ref().map(|w| &w[slot]);
                let pending = self.start_readback(handle, wire_dst, self.output_format);
                let frame = pending.wait()?;
                sink(&frame);
                emitted += 1;
            }
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
    /// contributions out of the fixed-point `accum`/`wsum` ring.
    pub fn reset_stream(&mut self) {
        self.front.reset_stream_state();
        self.next_output_slot = 0;
        self.passes_run = 0;
    }

    /// How many tail frames [`Self::flush`] must emit for the stream
    /// pushed so far.
    ///
    /// Mirrors [`NlmDenoiser::flush_target`]'s shape exactly, `real
    /// pushes so far, capped at a fixed multiple of the radius`, but at
    /// `2 * temporal_radius` rather than `temporal_radius`. The front
    /// end's own target only accounts for letting every real frame run
    /// as a pass's centre; this stage also needs every real frame to
    /// finish gathering its own `temporal_radius` trailing passes once
    /// it has been a centre, which doubles how many duplicate-driven
    /// passes the tail needs.
    fn flush_target(&self) -> usize {
        let real_pushes = self.front.real_pushes();
        if real_pushes == 0 {
            0
        } else {
            real_pushes.min(2 * self.temporal_radius as usize)
        }
    }

    /// Runs the grouping, filtering, and aggregation kernels for one
    /// pass, and returns the handle of whichever output slot the
    /// completed frame landed in along with that slot's index, once one
    /// has completed.
    ///
    /// # Scheduling
    ///
    /// A pass is centred on one physical ring slot, `view.centre_slot`,
    /// and every member it groups and filters scatters into its own
    /// frame's region of `self.accum`/`self.wsum`, whichever physical
    /// slot that member actually came from (see [`collab_fused`]'s
    /// scatter). Since this pass searches the centre frame
    /// spatially and each of the `temporal_radius` frames on either side
    /// of it, this pass's own contributions land across every region in
    /// `centre_slot - temporal_radius ..= centre_slot + temporal_radius`
    /// (mod the ring length), not just the centre's own region.
    ///
    /// A region only receives its full set of contributions once every
    /// pass that can reach it has run, which for the region at
    /// `centre_slot - temporal_radius` is this pass itself, the last and
    /// latest of the `1 + 2 * temporal_radius` passes able to write into
    /// it. That region is therefore exactly what this pass completes,
    /// and is read back into `output` and cleared for its next
    /// occupant. The region at `centre_slot + temporal_radius`, this
    /// pass's own newest edge, is symmetrically the one about to be
    /// reused for the first time since it was last completed, so it is
    /// the one this pass clears ahead of scattering into it, rather than
    /// clearing the whole ring on every call.
    ///
    /// [`self.passes_run`](Self::passes_run) counts how many passes have
    /// run for the current stream, including this one. Pass 0 clears the
    /// whole ring rather than only its newest edge, because nothing else
    /// has ever cleared the rest of it, whether this is the denoiser's
    /// first stream or a later one reusing the same buffers (see
    /// [`Self::flush`], which resets the counter but leaves the GPU
    /// buffers as they were). A pass only has every contribution its
    /// completed region can ever receive once `temporal_radius` further
    /// passes have run beyond the one centred on that region, so this
    /// returns `None` for the first `temporal_radius` passes of a
    /// stream, whose completed region is still short of contributions
    /// from passes that have not run yet.
    fn run_collab_stage(&mut self, view: &RingView) -> Result<Option<(Handle, usize)>, DenoiserError> {
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

        let refs_x = refs_along(self.width);
        let refs_y = refs_along(self.height);
        let refs = ref_count(self.width, self.height);

        // The kernel packs eight references into one 64-lane cube, so
        // its grid is an eighth as wide as the reference grid along x.
        let collab_grid = CubeCount::new_2d(fused_cubes_x(self.width), refs_y);
        let collab_dim = CubeDim::new_1d(64);
        let agg_grid = CubeCount::new_2d(self.width.div_ceil(BLOCK_X), self.height.div_ceil(BLOCK_Y));
        let agg_dim = CubeDim::new_2d(BLOCK_X, BLOCK_Y);
        let zero_dim = 256u32;
        // Sized for one frame's worth of the ring, the region a
        // steady-state pass clears. Pass 0 issues this same dispatch once
        // per ring slot, see the `pass_index == 0` branch below.
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
        // The kernel takes the two multiplied together, see
        // `mismatch_sigma2`. The confidence score itself stays derived
        // from the unscaled threshold, which is why this is applied here
        // rather than inside the front end.
        let mismatch_thsad = self.front.thsad_value() * self.mismatch_scale;

        // See the doc comment above for why these two slots are what
        // this pass clears and completes. `total_frames` is added
        // before the subtraction so the operands to `%` stay
        // non-negative regardless of how `centre_slot` and
        // `temporal_radius` compare.
        let newest_slot = (centre_slot + self.temporal_radius) % total_frames;
        let completed_slot = (centre_slot + total_frames - self.temporal_radius) % total_frames;

        let pass_index = self.passes_run;
        self.passes_run += 1;

        unsafe {
            if pass_index == 0 {
                // Clearing the whole ring in one dispatch would need
                // `accum_ring_len.div_ceil(zero_dim)` workgroups, which
                // grows with `total_frames`. At `temporal_radius = 4` a
                // 1080p luma plane alone needs 72,900, already over the
                // GPU's 65,535 limit. A rejected dispatch would leave the
                // ring holding `client.empty`'s undefined memory instead
                // of zero, which a fresh stream's first frames would then
                // aggregate as though it were real.
                //
                // Issuing `zero_grid_one_frame` once per ring slot keeps
                // every dispatch the size the steady-state branch below
                // already relies on, whatever `total_frames` is.
                for slot in 0..total_frames {
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
            } else {
                collab_zero_accum::launch_unchecked::<R>(
                    &client,
                    zero_grid_one_frame,
                    CubeDim::new_1d(zero_dim),
                    ArrayArg::from_raw_parts(self.accum.clone(), accum_ring_len),
                    ArrayArg::from_raw_parts(self.wsum.clone(), wsum_ring_len),
                    newest_slot * pixels as u32,
                    pixels as u32,
                    stored_ch,
                    zero_total_threads_one_frame,
                );
            }

            collab_fused::launch_unchecked::<R>(
                &client,
                collab_grid,
                collab_dim,
                stored_ch as usize,
                ArrayArg::from_raw_parts(view.input.clone(), ring_len),
                ArrayArg::from_raw_parts(view.mv_field.clone(), mv_len.max(1)),
                ArrayArg::from_raw_parts(view.confidence.clone(), conf_len.max(1)),
                ArrayArg::from_raw_parts(neighbour_slots_buf, view.neighbour_slots.len().max(1)),
                ArrayArg::from_raw_parts(self.sigma_buf.clone(), stored_ch as usize),
                ArrayArg::from_raw_parts(self.dct_profile_buf.clone(), 8),
                ArrayArg::from_raw_parts(self.kaiser_buf.clone(), PATCH_SIZE as usize),
                ArrayArg::from_raw_parts(self.accum.clone(), accum_ring_len),
                ArrayArg::from_raw_parts(self.wsum.clone(), wsum_ring_len),
                ArrayArg::from_raw_parts(self.group_weight.clone(), refs),
                centre_slot,
                // `collab_fused` has no admission gate (see its own doc
                // comment), so a constant subtracted from every
                // candidate's distance can never change which ones the
                // selection picks. Any value is exact here; 0.0 is the
                // simplest one that says so.
                0.0f32,
                self.c_min,
                mismatch_thsad,
                self.lambda_ht,
                wnorm,
                self.accum_scale,
                self.confidence_variance,
                self.temporal_radius,
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
            );
        }

        // Every pass scatters its contributions regardless, but the
        // region it completes, `completed_slot`, only holds every
        // contribution it will ever receive once `temporal_radius`
        // further passes have run beyond the one centred on it. See the
        // doc comment above for the full argument.
        if pass_index < self.temporal_radius {
            return Ok(None);
        }

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
                completed_slot * pixels as u32,
                self.width,
                self.height,
                channels_count,
                stored_ch,
            );
        }

        Ok(Some((self.outputs[slot].clone(), slot)))
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
