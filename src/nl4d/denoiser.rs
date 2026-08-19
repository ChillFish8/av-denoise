use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::collab::geometry::{member_buf_len, ref_count, refs_along};
use crate::collab::kernels::aggregate::{
    CROSS_FRAME_ACCUM_SCALE,
    collab_normalise,
    collab_zero_accum,
    weight_scale,
};
use crate::collab::kernels::filter_ht::collab_filter_ht;
use crate::collab::kernels::group_temporal::collab_group_temporal;
use crate::collab::kernels::transforms::dct_noise_profile;
use crate::collab::{MAX_K, PATCH_SIZE};
use crate::denoiser::DenoiserError;
use crate::nlmeans::{BLOCK_X, BLOCK_Y, ChannelMode, NlmDenoiser, Pending, RingView};

use super::params::Nl4dParams;

/// Groups similar 8x8 patches across a motion-compensated temporal
/// window and denoises each group jointly.
///
/// This drives the same front end [`crate::nl3d::Nl3dDenoiser`] uses,
/// but only for its machinery, the frame ring, the motion field, and
/// the confidence scores built by
/// [`NlmDenoiser::submit_machinery`]/[`NlmDenoiser::flush_step_machinery`].
/// No NLM weighting kernel ever runs. Instead, every submit groups
/// patches straight out of the noisy ring with
/// [`collab_group_temporal`], searching both the centre frame spatially
/// and each neighbour frame around where motion compensation predicts a
/// patch moved, then shrinks the group's coefficients with
/// [`collab_filter_ht`] in its temporal mode and aggregates the result
/// with [`collab_normalise`].
///
/// Every pass scatters its filtered members into whichever frame each
/// one actually came from, not only the centre frame, so a frame's own
/// output only finishes once every pass that can reach it,
/// `temporal_radius` on either side of it, has run. Latency is therefore
/// `2 * temporal_radius` pushes, twice [`crate::nl3d::Nl3dDenoiser`]'s
/// own `temporal_radius`. [`Self::denoise_submit`] returns `None` while
/// the front end's window is still filling and for the further passes
/// this cross-frame accumulation still needs, and [`Self::flush`] drains
/// the frames still held once the input stream ends.
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
    mismatch_scale: f32,
    confidence_variance: bool,
    k_max: u32,

    member_pos: Handle,
    member_frame: Handle,
    member_count: Handle,
    /// [`collab_group_temporal`]'s per-member mismatch-variance output,
    /// and `collab_filter_ht`'s `member_sig2` argument on the way back
    /// in. Always the real `refs * k_max` size, whatever
    /// `confidence_variance` is, since the grouping kernel always fills
    /// it. `confidence_variance` gates whether `collab_filter_ht` reads
    /// it back, through that call's own `use_member_sigma`, not whether
    /// this buffer holds real values.
    member_sig2: Handle,
    /// `collab_filter_ht`'s `filtered` argument. This denoiser never
    /// sets `emit_filtered`, so a one-element placeholder is valid here
    /// too.
    filtered_dummy: Handle,
    group_weight: Handle,
    sigma_buf: Handle,
    dct_profile_buf: Handle,
    /// The correlation profile kept on the host too, so the weight
    /// normalisation can be derived from it every submit without a
    /// device readback.
    dct_profile: [f32; 8],
    /// Fixed-point accumulators the filter scatters into, one weighted
    /// value per covering patch. Unlike a single-frame collaborative
    /// filter, these hold `1 + 2 * temporal_radius` frames' worth of
    /// pixels back to back, one region per physical ring slot of the
    /// front end's own frame ring. A pass centred on frame `u`
    /// contributes to every frame in `u - temporal_radius ..= u +
    /// temporal_radius`, so a frame's own region stays live across that
    /// many consecutive passes before [`Self::run_collab_stage`] reads
    /// it back and clears it for reuse. See that method for the exact
    /// scheduling.
    accum: Handle,
    wsum: Handle,
    /// Two output buffers, alternated so one frame's kernels can overlap
    /// the previous frame's readback.
    outputs: [Handle; 2],
    next_output_slot: usize,
    /// How many passes [`Self::run_collab_stage`] has run for the
    /// current stream.
    ///
    /// Pass 0 also full-zeroes `accum`/`wsum`, since every later pass
    /// only zeroes the one region it is about to reuse (see
    /// [`Self::run_collab_stage`]), so this doubles as "does the ring
    /// still hold a previous stream's stale contributions". It resets
    /// to 0 at the end of [`Self::flush`], alongside the front end's own
    /// stream state.
    ///
    /// A pass only emits an output once `passes_run > temporal_radius`,
    /// because the earliest a frame's region has received contributions
    /// from every pass that can reach it, `t - temporal_radius ..= t +
    /// temporal_radius`, is `temporal_radius` passes after the pass
    /// centred on `t` itself. [`Self::denoise_submit`] and
    /// [`Self::flush`] both read this through
    /// [`Self::run_collab_stage`]'s return value rather than checking it
    /// themselves.
    passes_run: u32,
}

impl<R: Runtime> Nl4dDenoiser<R> {
    /// Builds a new denoiser.
    ///
    /// Rejects an invalid `params` (see [`Nl4dParams::validate`]) and a
    /// frame smaller than one collaborative patch on either axis.
    pub fn new(client: &ComputeClient<R>, mut params: Nl4dParams, width: u32, height: u32) -> Result<Self, String> {
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

        let front = NlmDenoiser::new(client, params.nlm.clone(), width, height);

        let channels = params.nlm.channels;
        let stored_ch = channels.storage_count();
        let k_max = MAX_K;
        let refs = ref_count(width, height);
        let pos_len = member_buf_len(width, height, k_max);
        let pixels = (width * height) as usize;
        let frame_len = pixels * stored_ch as usize;

        let member_pos = client.empty(pos_len * size_of::<u32>());
        let member_frame = client.empty(pos_len * size_of::<u32>());
        let member_count = client.empty(refs * size_of::<u32>());
        let member_sig2 = client.empty(pos_len * size_of::<f32>());
        let filtered_dummy = client.empty(stored_ch as usize * size_of::<f32>());
        let group_weight = client.empty(refs * size_of::<f32>());
        let sigma_buf = client.create_from_slice(f32::as_bytes(&vec![0.0f32; stored_ch as usize]));
        // The correlation profile is purely spatial and this denoiser
        // exposes no `rho` knob, so it is built once here from the
        // white-noise default rather than every submit.
        let dct_profile = dct_noise_profile(0.0);
        let dct_profile_buf = client.create_from_slice(f32::as_bytes(&dct_profile));
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
            member_pos,
            member_frame,
            member_count,
            member_sig2,
            filtered_dummy,
            group_weight,
            sigma_buf,
            dct_profile_buf,
            dct_profile,
            accum,
            wsum,
            outputs,
            next_output_slot: 0,
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
    pub fn denoise_submit(&mut self) -> Result<Option<Pending<R>>, DenoiserError> {
        let Some(view) = self.front.submit_machinery()? else {
            return Ok(None);
        };
        let Some(handle) = self.run_collab_stage(&view)? else {
            return Ok(None);
        };
        Ok(Some(self.start_readback(handle)))
    }

    /// Submits and waits for the result in one call.
    ///
    /// Prefer [`Self::denoise_submit`] when the caller can hold a frame
    /// in flight.
    pub fn denoise(&mut self) -> Result<Option<Vec<f32>>, DenoiserError> {
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
    /// produced, and the slice it receives is only valid for that call.
    ///
    /// This drives [`NlmDenoiser::flush_step_machinery`] for
    /// [`Self::flush_target`] emissions, which is `2 * temporal_radius`
    /// duplicate-driven passes for a long enough stream, twice what the
    /// front end's own [`NlmDenoiser::flush_target`] would give. The
    /// front end only has to let every real frame finish a turn as a
    /// pass's own centre; this cross-frame stage also has to let every
    /// real frame finish gathering the `temporal_radius` trailing
    /// passes its own accumulation needs, which is `temporal_radius`
    /// pushes' worth of passes beyond that. Every call to
    /// [`NlmDenoiser::flush_step_machinery`] still runs a pass regardless
    /// of whether it emits, so the loop below keeps calling it until
    /// enough passes have actually emitted, the same way it always has.
    pub fn flush(&mut self, mut sink: impl FnMut(&[f32])) -> Result<(), DenoiserError> {
        let target = self.flush_target();
        let mut emitted = 0usize;

        while emitted < target {
            if let Some(view) = self.front.flush_step_machinery()?
                && let Some(handle) = self.run_collab_stage(&view)?
            {
                let pending = self.start_readback(handle);
                let frame = pending.wait()?;
                sink(&frame);
                emitted += 1;
            }
        }

        self.front.reset_stream_state();
        self.next_output_slot = 0;
        self.passes_run = 0;

        Ok(())
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
    /// completed frame landed in, once one has completed.
    ///
    /// # Scheduling
    ///
    /// A pass is centred on one physical ring slot, `view.centre_slot`,
    /// and every member it groups and filters scatters into its own
    /// frame's region of `self.accum`/`self.wsum`, whichever physical
    /// slot that member actually came from (see
    /// [`crate::collab::kernels::filter_ht::collab_filter_ht`]'s
    /// `temporal` mode). Since this pass searches the centre frame
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
    fn run_collab_stage(&mut self, view: &RingView) -> Result<Option<Handle>, DenoiserError> {
        // The frame-slot contract: `collab_group_temporal`'s
        // `centre_slot` and `collab_filter_ht`'s `frame` must be the
        // same physical ring slot, or a member gets grouped against one
        // frame and scattered as though it belonged to another. Both
        // launches below read this one local, computed here exactly
        // once, rather than each recomputing their own.
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
        let pos_len = member_buf_len(self.width, self.height, self.k_max);

        let group_grid = CubeCount::new_2d(refs_x, refs_y);
        let group_dim = CubeDim::new_2d(8, 8);
        let agg_grid = CubeCount::new_2d(self.width.div_ceil(BLOCK_X), self.height.div_ceil(BLOCK_Y));
        let agg_dim = CubeDim::new_2d(BLOCK_X, BLOCK_Y);
        let zero_dim = 256u32;
        // Sized for one frame's worth of the ring, the region a
        // steady-state pass clears. Pass 0 clears the whole ring
        // instead, with its own grid sized for that below.
        let zero_grid_one_frame = CubeCount::new_1d((frame_len as u32).div_ceil(zero_dim));

        let mc = self.front.motion_ctx();
        let blk_step = mc.step;
        let blksize = mc.blksize;
        let blocks_x = mc.blocks_x;
        let blocks_y = mc.blocks_y;
        let thsad = self.front.thsad_value();

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
                collab_zero_accum::launch_unchecked::<R>(
                    &client,
                    CubeCount::new_1d((accum_ring_len as u32).div_ceil(zero_dim)),
                    CubeDim::new_1d(zero_dim),
                    ArrayArg::from_raw_parts(self.accum.clone(), accum_ring_len),
                    ArrayArg::from_raw_parts(self.wsum.clone(), wsum_ring_len),
                    0u32,
                    wsum_ring_len as u32,
                    stored_ch,
                );
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
                );
            }

            collab_group_temporal::launch_unchecked::<R>(
                &client,
                group_grid.clone(),
                group_dim,
                stored_ch as usize,
                ArrayArg::from_raw_parts(view.input.clone(), ring_len),
                ArrayArg::from_raw_parts(view.mv_field.clone(), mv_len.max(1)),
                ArrayArg::from_raw_parts(view.confidence.clone(), conf_len.max(1)),
                ArrayArg::from_raw_parts(self.member_pos.clone(), pos_len),
                ArrayArg::from_raw_parts(self.member_frame.clone(), pos_len),
                ArrayArg::from_raw_parts(self.member_count.clone(), refs),
                ArrayArg::from_raw_parts(self.member_sig2.clone(), pos_len),
                centre_slot,
                ArrayArg::from_raw_parts(neighbour_slots_buf, view.neighbour_slots.len().max(1)),
                // `collab_group_temporal` has no admission gate (see its
                // own doc comment), so a constant subtracted from every
                // candidate's distance can never change which ones the
                // argmin selection below picks. Any value is exact here;
                // 0.0 is the simplest one that says so.
                0.0f32,
                self.c_min,
                thsad,
                self.mismatch_scale,
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
                self.spatial_radius,
                refs_x,
            );

            collab_filter_ht::launch_unchecked::<R>(
                &client,
                group_grid,
                group_dim,
                stored_ch as usize,
                ArrayArg::from_raw_parts(view.input.clone(), ring_len),
                ArrayArg::from_raw_parts(self.member_pos.clone(), pos_len),
                ArrayArg::from_raw_parts(self.member_frame.clone(), pos_len),
                ArrayArg::from_raw_parts(self.member_count.clone(), refs),
                ArrayArg::from_raw_parts(self.member_sig2.clone(), pos_len),
                ArrayArg::from_raw_parts(self.accum.clone(), accum_ring_len),
                ArrayArg::from_raw_parts(self.wsum.clone(), wsum_ring_len),
                ArrayArg::from_raw_parts(self.filtered_dummy.clone(), 1),
                ArrayArg::from_raw_parts(self.group_weight.clone(), refs),
                centre_slot,
                ArrayArg::from_raw_parts(self.sigma_buf.clone(), stored_ch as usize),
                ArrayArg::from_raw_parts(self.dct_profile_buf.clone(), 8),
                self.lambda_ht,
                wnorm,
                CROSS_FRAME_ACCUM_SCALE,
                self.confidence_variance,
                false,
                false,
                true,
                self.width,
                self.height,
                channels_count,
                self.k_max,
                stored_ch,
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

        Ok(Some(self.outputs[slot].clone()))
    }

    /// Starts an async readback of `handle`, wrapped in the same
    /// [`Pending`] type [`NlmDenoiser`] returns.
    fn start_readback(&self, handle: Handle) -> Pending<R> {
        let client = self.front.compute_client().clone();
        let fut = Box::pin(async move { client.read_async(vec![handle]).await });
        let pixels = (self.width * self.height) as usize;
        Pending::new(fut, self.channels.count(), self.channels.storage_count(), pixels)
    }
}
