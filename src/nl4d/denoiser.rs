use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::collab::geometry::{member_buf_len, ref_count, refs_along};
use crate::collab::kernels::aggregate::{collab_normalise, collab_zero_accum, weight_scale};
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
/// Latency is `temporal_radius` pushes, the same as any other temporal
/// denoiser in this crate. [`Self::denoise_submit`] returns `None` while
/// the front end's window is still filling, and [`Self::flush`] drains
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
    k_max: u32,

    member_pos: Handle,
    member_frame: Handle,
    member_count: Handle,
    /// `collab_filter_ht`'s `member_sig2` argument. This denoiser never
    /// sets `use_member_sigma`, so a one-element placeholder is valid
    /// here, the same pattern `CollabPipeline` uses.
    member_sig2_dummy: Handle,
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
    /// value per covering patch, cleared before each submit and divided
    /// out by `collab_normalise` after it.
    accum: Handle,
    wsum: Handle,
    /// Two output buffers, alternated so one frame's kernels can overlap
    /// the previous frame's readback.
    outputs: [Handle; 2],
    next_output_slot: usize,
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
        let member_sig2_dummy = client.empty(size_of::<f32>());
        let filtered_dummy = client.empty(stored_ch as usize * size_of::<f32>());
        let group_weight = client.empty(refs * size_of::<f32>());
        let sigma_buf = client.create_from_slice(f32::as_bytes(&vec![0.0f32; stored_ch as usize]));
        // The correlation profile is purely spatial and this denoiser
        // exposes no `rho` knob, so it is built once here from the
        // white-noise default rather than every submit.
        let dct_profile = dct_noise_profile(0.0);
        let dct_profile_buf = client.create_from_slice(f32::as_bytes(&dct_profile));
        let accum = client.empty(frame_len * size_of::<i32>());
        let wsum = client.empty(pixels * size_of::<i32>());
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
            k_max,
            member_pos,
            member_frame,
            member_count,
            member_sig2_dummy,
            filtered_dummy,
            group_weight,
            sigma_buf,
            dct_profile_buf,
            dct_profile,
            accum,
            wsum,
            outputs,
            next_output_slot: 0,
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
    /// filling, the same convention every temporal denoiser in this
    /// crate follows.
    ///
    /// There are two output slots, so at most two [`Pending`]s from this
    /// denoiser may be outstanding at once. A third concurrent submit
    /// reuses the oldest one's slot and silently corrupts it.
    pub fn denoise_submit(&mut self) -> Result<Option<Pending<R>>, DenoiserError> {
        let Some(view) = self.front.submit_machinery()? else {
            return Ok(None);
        };
        let handle = self.run_collab_stage(&view)?;
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
    pub fn flush(&mut self, mut sink: impl FnMut(&[f32])) -> Result<(), DenoiserError> {
        let target = self.front.flush_target();
        let mut emitted = 0usize;

        while emitted < target {
            if let Some(view) = self.front.flush_step_machinery()? {
                let handle = self.run_collab_stage(&view)?;
                let pending = self.start_readback(handle);
                let frame = pending.wait()?;
                sink(&frame);
                emitted += 1;
            }
        }

        self.front.reset_stream_state();
        self.next_output_slot = 0;

        Ok(())
    }

    /// Runs the grouping, filtering, and aggregation kernels for one
    /// ring view, writing into whichever of the two output slots is next
    /// in rotation, and returns that slot's handle.
    fn run_collab_stage(&mut self, view: &RingView) -> Result<Handle, DenoiserError> {
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
        let zero_grid = CubeCount::new_1d((frame_len as u32).div_ceil(zero_dim));

        let mc = self.front.motion_ctx();
        let blk_step = mc.step;
        let blocks_x = mc.blocks_x;
        let blocks_y = mc.blocks_y;

        let slot = self.next_output_slot;
        self.next_output_slot = (slot + 1) % self.outputs.len();

        unsafe {
            collab_zero_accum::launch_unchecked::<R>(
                &client,
                zero_grid,
                CubeDim::new_1d(zero_dim),
                ArrayArg::from_raw_parts(self.accum.clone(), frame_len),
                ArrayArg::from_raw_parts(self.wsum.clone(), pixels),
                pixels as u32,
                stored_ch,
            );

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
                centre_slot,
                ArrayArg::from_raw_parts(neighbour_slots_buf, view.neighbour_slots.len().max(1)),
                // `collab_group_temporal` has no admission gate (see its
                // own doc comment), so a constant subtracted from every
                // candidate's distance can never change which ones the
                // argmin selection below picks. Any value is exact here;
                // 0.0 is the simplest one that says so.
                0.0f32,
                self.c_min,
                self.temporal_radius,
                self.refine,
                view.mv_stride,
                view.conf_stride,
                blk_step,
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
                ArrayArg::from_raw_parts(self.member_sig2_dummy.clone(), 1),
                ArrayArg::from_raw_parts(self.accum.clone(), frame_len),
                ArrayArg::from_raw_parts(self.wsum.clone(), pixels),
                ArrayArg::from_raw_parts(self.filtered_dummy.clone(), 1),
                ArrayArg::from_raw_parts(self.group_weight.clone(), refs),
                centre_slot,
                ArrayArg::from_raw_parts(self.sigma_buf.clone(), stored_ch as usize),
                ArrayArg::from_raw_parts(self.dct_profile_buf.clone(), 8),
                self.lambda_ht,
                wnorm,
                false,
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

            collab_normalise::launch_unchecked::<R>(
                &client,
                agg_grid,
                agg_dim,
                stored_ch as usize,
                ArrayArg::from_raw_parts(self.accum.clone(), frame_len),
                ArrayArg::from_raw_parts(self.wsum.clone(), pixels),
                ArrayArg::from_raw_parts(self.outputs[slot].clone(), frame_len),
                self.width,
                self.height,
                channels_count,
                stored_ch,
            );
        }

        Ok(self.outputs[slot].clone())
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
