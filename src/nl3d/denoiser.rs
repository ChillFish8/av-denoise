use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::collab::{CollabParams, CollabPipeline, PATCH_SIZE};
use crate::nlmeans::{
    ChannelMode,
    GpuOutput,
    HqParams,
    NlmDenoiser,
    NlmParams,
    Pending,
    PendingResidualRatio,
};

/// Parameters for the two-stage nl3d cascade.
#[derive(Debug, Clone)]
pub struct Nl3dParams {
    /// Front-end nlmeans parameters. `hq` must be `Some`, because the
    /// collaborative stage derives its own shrinkage sigma from the
    /// front end's noise estimate. The constructor turns
    /// `track_weight_sq` on by itself, since the residual-noise
    /// measurement the collaborative stage needs depends on it.
    pub nlm: NlmParams,
    /// A multiplier applied to the front end's strength before it runs.
    ///
    /// The default sits below 1.0, so the front end filters gently and
    /// leaves structured residual noise in its output. The collaborative
    /// stage removes that residual through grouped transform shrinkage,
    /// which recovers more real detail than pushing the front end harder
    /// to remove the same noise on its own would. Calibrated to 0.5, see
    /// [`crate::Nl3dOptions`]'s struct-level docs for the sweep this came
    /// from.
    pub front_strength_scale: f32,
    /// Parameters for the collaborative filter that runs second.
    pub collab: CollabParams,
    /// A calibratable multiplier on the residual sigma passed to the
    /// collaborative stage.
    ///
    /// The residual sigma is the front end's per-channel noise estimate
    /// scaled by how much noise that pass actually left behind. This
    /// field is an extra knob on top of that analytic value, for tuning
    /// without touching either stage's own parameters. Calibrated to
    /// 1.9, see [`crate::Nl3dOptions`]'s struct-level docs for the sweep
    /// this came from.
    pub residual_sigma_scale: f32,
}

impl Default for Nl3dParams {
    fn default() -> Self {
        Self {
            nlm: NlmParams {
                hq: Some(HqParams::default()),
                ..NlmParams::default()
            },
            front_strength_scale: 0.5,
            collab: CollabParams::default(),
            residual_sigma_scale: 1.9,
        }
    }
}

/// A cascade that runs a frame through non-local means and then a
/// collaborative-filter cleanup pass, without the frame leaving the GPU
/// between the two.
pub struct Nl3dDenoiser<R: Runtime> {
    client: ComputeClient<R>,
    front: NlmDenoiser<R>,
    collab: CollabPipeline<R>,
    /// The collaborative stage's own two output buffers, alternated the
    /// same way `NlmDenoiser`'s output ring is, so one frame's
    /// collaborative-filter kernels can overlap the previous frame's
    /// readback.
    outputs: [Handle; 2],
    next_output_slot: usize,
    width: u32,
    height: u32,
    channels: ChannelMode,
    residual_sigma_scale: f32,
    /// The residual ratio measured on the previous frame, reused for
    /// this frame's sigma. `None` only before the first frame has ever
    /// been measured.
    ///
    /// Noise level moves slowly from one frame to the next, so a ratio
    /// measured one frame ago is a good estimate for the frame in front
    /// of the collaborative stage right now, and reusing it avoids
    /// blocking on a value the GPU has not finished computing yet. See
    /// `run_collab_stage` for how this is kept up to date.
    last_ratio: Option<f32>,
    /// A ratio reduction queued for the frame just submitted, not yet
    /// read back.
    ///
    /// Read on the *next* call to `run_collab_stage`, by which point the
    /// GPU has had this frame's own collaborative-stage kernels, this
    /// frame's readback, and the next frame's front-end kernels queued
    /// behind it, so the read lands on already-finished work instead of
    /// stalling the queue right behind the dispatch that produced it.
    pending_ratio: Option<PendingResidualRatio>,
}

impl<R: Runtime> Nl3dDenoiser<R> {
    /// Builds a new cascade.
    ///
    /// Rejects a front end without HQ enabled, a front end and
    /// collaborative stage that disagree on which channels they process,
    /// a frame too small for the collaborative filter's 8x8 patches, and
    /// a non-finite or out-of-range `front_strength_scale` or
    /// `residual_sigma_scale`.
    ///
    /// `front_strength_scale` is applied to `params.nlm.strength` before
    /// the front end is built, and `params.nlm.track_weight_sq` is
    /// forced on, so neither has to be set by the caller.
    pub fn new(
        client: &ComputeClient<R>,
        mut params: Nl3dParams,
        width: u32,
        height: u32,
    ) -> Result<Self, anyhow::Error> {
        if params.nlm.hq.is_none() {
            anyhow::bail!(
                "nl3d requires nlm.hq to be Some, got None. The collaborative stage needs \
                 the front end's estimated noise sigma to shrink its own coefficients by"
            );
        }

        if params.nlm.channels != params.collab.channels {
            anyhow::bail!(
                "nlm.channels={:?} and collab.channels={:?} must match, so both stages \
                 agree on which planes they are filtering",
                params.nlm.channels,
                params.collab.channels,
            );
        }

        if width < PATCH_SIZE || height < PATCH_SIZE {
            anyhow::bail!(
                "frame dimensions {width}x{height} must be at least {p}x{p} for the \
                 collaborative filter's patch grid",
                p = PATCH_SIZE,
            );
        }

        if !(params.front_strength_scale.is_finite() && params.front_strength_scale > 0.0) {
            anyhow::bail!(
                "front_strength_scale must be finite and greater than 0, got {}. A \
                 non-positive scale leaves the front end with a strength of 0 or less, \
                 which nlmeans itself rejects",
                params.front_strength_scale,
            );
        }

        if !(params.residual_sigma_scale.is_finite() && params.residual_sigma_scale >= 0.0) {
            anyhow::bail!(
                "residual_sigma_scale must be finite and 0 or greater, got {}",
                params.residual_sigma_scale,
            );
        }

        params.nlm.strength *= params.front_strength_scale;
        params.nlm.track_weight_sq = true;
        params.nlm.validate()?;

        let front = NlmDenoiser::new(client, params.nlm.clone(), width, height);
        let collab = CollabPipeline::new(client, params.collab, width, height)?;

        let stored_ch = params.nlm.channels.storage_count();
        let frame_bytes = (width * height * stored_ch) as usize * size_of::<f32>();
        let outputs = [client.empty(frame_bytes), client.empty(frame_bytes)];

        Ok(Self {
            client: client.clone(),
            front,
            collab,
            outputs,
            next_output_slot: 0,
            width,
            height,
            channels: params.nlm.channels,
            residual_sigma_scale: params.residual_sigma_scale,
            last_ratio: None,
            pending_ratio: None,
        })
    }

    /// Pushes a new frame into the front end's ring buffer.
    ///
    /// `frame` holds `width * height * channels` `f32` values in `[0,
    /// 1]`, matching `NlmDenoiser::push_frame`.
    pub fn push_frame(&mut self, frame: &[f32]) {
        self.front.push_frame(frame);
    }

    /// Runs both stages of the cascade over the current window and
    /// starts the readback.
    ///
    /// Returns `Ok(None)` while the front end's temporal window is still
    /// filling.
    ///
    /// There are two output slots, alternated the same way
    /// `NlmDenoiser::denoise_submit`'s are, so at most two `Pending`s
    /// from this denoiser may be outstanding at once. A third concurrent
    /// submit reuses the oldest one's slot and silently corrupts it.
    pub fn denoise_submit(&mut self) -> Result<Option<Pending<R>>, anyhow::Error> {
        let Some(gpu) = self.front.denoise_submit_gpu()? else {
            return Ok(None);
        };
        let handle = self.run_collab_stage(gpu)?;
        Ok(Some(self.start_readback(handle)))
    }

    /// Submits and waits for the result in one call.
    ///
    /// Prefer `denoise_submit` when the caller can hold a frame in
    /// flight, which lets one frame's kernels overlap the previous
    /// frame's readback.
    pub fn denoise(&mut self) -> Result<Option<Vec<f32>>, anyhow::Error> {
        let Some(pending) = self.denoise_submit()? else {
            return Ok(None);
        };
        Ok(Some(pending.wait()?))
    }

    /// Produces the frames still held at the end of a stream, running
    /// each one through both stages before it is read back.
    ///
    /// For the last few frames the front end keeps its temporal window
    /// full by repeating the final pushed frame, exactly as
    /// `NlmDenoiser::flush` does. `sink` is called once per frame
    /// produced, and the slice it receives is only valid for that call.
    pub fn flush(&mut self, mut sink: impl FnMut(&[f32])) -> Result<(), anyhow::Error> {
        let target = self.front.flush_target();
        let mut emitted = 0usize;

        while emitted < target {
            if let Some(gpu) = self.front.flush_step_gpu()? {
                let handle = self.run_collab_stage(gpu)?;
                let pending = self.start_readback(handle);
                let frame = pending.wait()?;
                sink(&frame);
                emitted += 1;
            }
        }

        self.front.reset_stream_state();
        self.next_output_slot = 0;
        // A future push starts a new stream, with its own first frame
        // that has no previous ratio to reuse. Discard both fields so
        // that frame is handled the same explicit way the very first
        // frame ever pushed to this denoiser is, rather than reusing a
        // ratio measured on the stream that just ended.
        self.last_ratio = None;
        self.pending_ratio = None;

        Ok(())
    }

    /// Runs the collaborative stage over one front-end output, writing
    /// into whichever of the two output slots is next in rotation, and
    /// returns that slot's handle.
    ///
    /// `gpu` is consumed here, its handle read as the collaborative
    /// stage's input and never held onto afterward, so the two-slot
    /// lifetime rule documented on `GpuOutput` is respected without this
    /// denoiser ever needing to track it itself.
    ///
    /// # The residual sigma
    ///
    /// The collaborative stage shrinks its coefficients according to how
    /// noisy it believes its input is. Its actual input is `gpu`, the
    /// front end's already-denoised frame, not the original noisy
    /// source, so passing the original per-channel sigma would tell it
    /// to expect far more noise than is really left and it would
    /// over-smooth badly.
    ///
    /// The correct value is the original sigma scaled down by how much
    /// noise the front end actually removed. `resolve_ratio` supplies
    /// exactly that fraction, measured from whatever the front end's own
    /// last pass left in its accumulators. `current_sigmas` gives the
    /// original per-channel estimate those accumulators were built from.
    /// Multiplying the two together, per channel, gives the sigma the
    /// residual noise actually has. `residual_sigma_scale` is then
    /// applied last, a further calibratable multiplier on top of that
    /// analytic value, for tuning without touching either stage's own
    /// parameters.
    fn run_collab_stage(&mut self, gpu: GpuOutput) -> Result<Handle, anyhow::Error> {
        let ratio = self.resolve_ratio()?;
        let base_sigmas = self.front.current_sigmas();
        let count = self.channels.count() as usize;

        let mut sigmas = vec![0.0f32; count];
        for (dst, &base) in sigmas.iter_mut().zip(base_sigmas[..count].iter()) {
            *dst = base * ratio * self.residual_sigma_scale;
        }

        // Queue this frame's own ratio reduction while its accumulators
        // are still fresh, but do not read it back yet. `resolve_ratio`
        // reads it on the *next* call instead, once the collaborative
        // stage kernels below, this frame's readback, and the next
        // frame's front-end kernels have all been queued behind it, so
        // that read lands on already-finished work instead of stalling
        // the queue right behind this dispatch.
        self.pending_ratio = Some(self.front.residual_ratio_sqrt_submit()?);

        let slot = self.next_output_slot;
        self.next_output_slot = (slot + 1) % self.outputs.len();

        self.collab
            .run_two_stage(&gpu.handle, &sigmas, &self.outputs[slot])?;

        Ok(self.outputs[slot].clone())
    }

    /// Resolves the residual ratio this frame's collaborative stage
    /// should shrink by.
    ///
    /// Noise level moves slowly from one frame to the next, so a ratio
    /// measured one frame ago is a good estimate for the frame in front
    /// of the collaborative stage right now. This reads back whatever
    /// reduction the previous call to `run_collab_stage` queued (see
    /// `pending_ratio`), which by now the GPU has had a full frame's
    /// worth of other work to finish behind, and caches it in
    /// `last_ratio` in case a later call needs it before its own
    /// reduction lands.
    ///
    /// The very first call of a stream has nothing queued yet, since no
    /// frame has been measured before it. That one case falls back to a
    /// synchronous submit-and-read, the one stall a stream ever pays for
    /// this value, rather than guessing at a ratio with no basis at all.
    fn resolve_ratio(&mut self) -> Result<f32, anyhow::Error> {
        if let Some(pending) = self.pending_ratio.take() {
            let ratio = self.front.read_residual_ratio_sqrt(pending)?;
            self.last_ratio = Some(ratio);
            return Ok(ratio);
        }

        if let Some(ratio) = self.last_ratio {
            return Ok(ratio);
        }

        let ratio = self.front.residual_ratio_sqrt()?;
        self.last_ratio = Some(ratio);
        Ok(ratio)
    }

    /// Starts an async readback of `handle`, wrapped in the same
    /// `Pending` type `NlmDenoiser` returns, so a caller sees one
    /// uniform API for both stages.
    fn start_readback(&self, handle: Handle) -> Pending<R> {
        let client = self.client.clone();
        let fut = Box::pin(async move { client.read_async(vec![handle]).await });
        let pixels = (self.width * self.height) as usize;
        Pending::new(fut, self.channels.count(), self.channels.storage_count(), pixels)
    }
}
