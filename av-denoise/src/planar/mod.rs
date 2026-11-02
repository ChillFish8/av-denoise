mod reseed;

#[cfg(test)]
mod tests;

use std::sync::Arc;

use anyhow::Context;
use av_denoise_core::tune::{TuningConfig, TuningFrame, tune_frames};
use av_denoise_core::{ChromaLayout, ChromaSiting, DenoisingMode, EngineSpec, GrainChunk, WindowSpan};
use cubecl::prelude::Client;

pub use self::reseed::ReseedWindow;
use crate::backend::accelerate::Accelerator;
use crate::backend::{Device, DevicePlacement};
pub use crate::host::ChannelIntent;
use crate::host::{Algorithm, DenoiserError, DenoiserOptions, Depth, HostDenoiser, PlanarTiming};
use crate::tune::{BudgetRequest, TuneReport, TuneRequest, TuningMode, tuning_config};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Subsampling {
    Yuv420,
    Yuv422,
    Yuv444,
}

impl Subsampling {
    /// The engine's name for this layout.
    pub fn chroma_layout(self) -> ChromaLayout {
        match self {
            Subsampling::Yuv420 => ChromaLayout::Yuv420,
            Subsampling::Yuv422 => ChromaLayout::Yuv422,
            Subsampling::Yuv444 => ChromaLayout::Yuv444,
        }
    }

    /// The chroma plane size for a `width` by `height` frame.
    ///
    /// Halved axes round up, so an odd dimension keeps the extra sample, matching what y4m and ffmpeg do.
    pub fn chroma_dims(self, width: u32, height: u32) -> (u32, u32) {
        let layout = self.chroma_layout();
        let dims = layout.chroma_dims(width, height);
        dims.expect("a subsampled layout has chroma planes")
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FrameLayout {
    pub width: u32,
    pub height: u32,
    pub subsampling: Subsampling,
    pub depth: Depth,
    /// Where the chroma samples sit. It is carried for callers and never changes the output.
    pub chroma_siting: ChromaSiting,
}

impl FrameLayout {
    pub fn luma_pixels(&self) -> usize {
        (self.width as usize) * (self.height as usize)
    }

    pub fn chroma_dims(&self) -> (u32, u32) {
        self.subsampling.chroma_dims(self.width, self.height)
    }

    pub fn chroma_pixels(&self) -> usize {
        let (chroma_width, chroma_height) = self.chroma_dims();
        (chroma_width as usize) * (chroma_height as usize)
    }

    /// Wire size of the luma plane.
    pub fn luma_bytes(&self) -> usize {
        self.luma_pixels() * self.depth.bytes_per_sample()
    }

    /// Wire size of one chroma plane.
    pub fn chroma_bytes(&self) -> usize {
        self.chroma_pixels() * self.depth.bytes_per_sample()
    }

    /// A full neutral chroma plane, used when a source has no chroma.
    pub fn neutral_chroma_plane(&self) -> Vec<u8> {
        let samples = self.chroma_pixels();
        let neutral = self.depth.neutral_chroma();
        fill_plane(samples, neutral, self.depth)
    }
}

/// Builds a plane of `samples` copies of `value` in wire-byte form.
pub fn fill_plane(samples: usize, value: u16, depth: Depth) -> Vec<u8> {
    match depth.bytes_per_sample() {
        1 => vec![value as u8; samples],
        _ => {
            let word = value.to_le_bytes();
            let mut plane = Vec::with_capacity(samples * 2);
            for _ in 0..samples {
                plane.extend_from_slice(&word);
            }

            plane
        },
    }
}

/// A planar YUV frame holding little-endian wire bytes.
///
/// Plane lengths come from [FrameLayout], so `y.len()` is `layout.luma_bytes()` and both `u.len()`
/// and `v.len()` are `layout.chroma_bytes()`.
#[derive(Debug, Clone, PartialEq)]
pub struct Planes {
    pub y: Vec<u8>,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
}

impl Planes {
    /// Rebuilds a frame from one denoised frame's per-plane buffers.
    fn from_buffers(planes: Vec<Vec<u8>>) -> Self {
        let count = planes.len();
        let array: Result<[Vec<u8>; 3], _> = planes.try_into();
        let Ok([y_plane, u_plane, v_plane]) = array else {
            panic!("expected 3 planes from a HostDenoiser, got {count}");
        };

        Self {
            y: y_plane,
            u: u_plane,
            v: v_plane,
        }
    }
}

/// The per-plane option set a caller resolves once and passes into [PlanarDenoiser::create].
#[derive(Debug, Clone)]
pub struct PlaneOptions {
    pub accelerators: Vec<Accelerator>,
    pub device: Device,
    pub intent: ChannelIntent,
    /// Whether to clean each frame on its own or across a temporal window.
    ///
    /// This wins over `NlmeansOptions.mode` and `Nl4dOptions.temporal_radius` inside `algorithm`.
    pub mode: DenoisingMode,
    /// Which denoising algorithm to run, along with the settings only that algorithm reads.
    pub algorithm: Algorithm,
    /// Strength override for the luma denoiser, which wins over the algorithm's `tuning.strength`.
    ///
    /// Only the two NLM algorithms read it.
    pub luma_strength: Option<f32>,
    /// Strength override for the chroma denoiser, which wins over the algorithm's `tuning.strength`.
    ///
    /// Only the two NLM algorithms read it.
    pub chroma_strength: Option<f32>,
    /// Luma override for nl4d's `lambda_ht`, the temporal grouping stage's hard threshold.
    ///
    /// It wins over `algorithm`'s value, which itself falls back to a calibrated per-plane default
    /// when neither is set.
    pub luma_lambda_ht: Option<f32>,
    /// Chroma override for nl4d's `lambda_ht`, the temporal grouping stage's hard threshold.
    ///
    /// It wins over `algorithm`'s value, which itself falls back to a calibrated per-plane default
    /// when neither is set.
    pub chroma_lambda_ht: Option<f32>,
    /// Whether the planes run tuned kernels.
    ///
    /// With [TuningMode::Auto], [PlanarDenoiser::tune] runs a tuning pass and rebuilds them from its
    /// result.
    pub tuning: TuningMode,
    /// Picks only bit-identical candidates and the baseline profile.
    pub deterministic: bool,
}

impl PlaneOptions {
    /// The most frames a [PlanarDenoiser] built from these options holds before it emits one.
    pub fn max_held_frames(&self, layout: FrameLayout) -> usize {
        let spec = plane_spec(self, layout);
        spec.max_held_frames()
    }

    /// The options for the denoiser of a frame laid out as `layout`.
    ///
    /// Every plane quantises to the layout's depth on the GPU.
    pub(crate) fn denoiser_options(&self, layout: FrameLayout) -> DenoiserOptions {
        let chroma = layout.subsampling.chroma_layout();

        DenoiserOptions::builder()
            .intent(self.intent)
            .chroma(chroma)
            .mode(self.mode)
            .algorithm(self.algorithm)
            .maybe_luma_strength(self.luma_strength)
            .maybe_chroma_strength(self.chroma_strength)
            .maybe_luma_lambda_ht(self.luma_lambda_ht)
            .maybe_chroma_lambda_ht(self.chroma_lambda_ht)
            .depth(layout.depth)
            .chroma_siting(layout.chroma_siting)
            .tuning(self.tuning)
            .deterministic(self.deterministic)
            .build()
    }
}

/// The engine spec `options` builds for a frame laid out as `layout`.
pub(crate) fn plane_spec(options: &PlaneOptions, layout: FrameLayout) -> EngineSpec {
    let denoiser_options = options.denoiser_options(layout);
    denoiser_options.engine_spec(layout.width, layout.height)
}

/// Reads the result of a [PlanarDenoiser::push] for the push, drain and retry loop.
///
/// `Ok(false)` means the push landed. `Ok(true)` means the queue was full, so the caller should drain
/// one output and push again. Any error other than `QueueFull` is passed on rather than discarded.
pub fn push_needs_retry(result: Result<(), DenoiserError>) -> Result<bool, anyhow::Error> {
    match result {
        Ok(()) => Ok(false),
        Err(DenoiserError::QueueFull) => Ok(true),
        Err(other) => Err(other.into()),
    }
}

/// The push a [PlanarDenoiser] runs against its host, either [HostDenoiser::push] or
/// [HostDenoiser::push_priming].
type WirePush = fn(&mut HostDenoiser, &[&[u8]]) -> Result<(), DenoiserError>;

/// The device, engine and uploaded frames one tuning pass runs on.
struct TuningPass {
    device: cubecl::Device,
    client: Client,
    spec: EngineSpec,
    frames: Vec<TuningFrame>,
}

impl TuningPass {
    fn run(&self, config: &TuningConfig, request: &TuneRequest) -> TuneReport {
        let specs = [self.spec];

        tune_frames(&self.device, &self.client, &specs, &self.frames, config, request)
    }
}

/// Wraps the `HostDenoiser` that cleans one planar YUV source.
///
/// The caller pushes planar frames in and gets planar frames out. Planes the intent leaves alone come
/// out unchanged, paired with the frame they came in with.
pub struct PlanarDenoiser {
    layout: FrameLayout,
    /// `None` only after [Self::tune] freed the host and could not rebuild it.
    host: Option<HostDenoiser>,
    temporal_radius: u32,
    /// The backend the host was built on.
    accelerator: Accelerator,
    window_span: WindowSpan,
    options: PlaneOptions,
    /// Where the host opened its client, kept so a rebuild lands on the same device.
    placement: Arc<DevicePlacement>,
}

impl PlanarDenoiser {
    pub fn create(options: &PlaneOptions, layout: FrameLayout) -> Result<Self, anyhow::Error> {
        let placement = DevicePlacement::shared();
        let placement = Arc::new(placement);
        Self::create_placed(options, layout, &placement)
    }

    /// [Self::create] on a wgpu device of its own when the denoiser runs on Vulkan.
    ///
    /// The device sits on the adapter `options.device` selects and has its own queue, so this
    /// denoiser's readbacks never wait behind another denoiser's work. It loads the same tuning
    /// results as the shared device, and [Self::tune] keeps it on this device. Other runtimes build
    /// exactly as [Self::create] does.
    ///
    /// Each call opens a new device, which stays open until the process exits because CubeCL never
    /// frees a client.
    pub fn create_on_own_device(options: &PlaneOptions, layout: FrameLayout) -> Result<Self, anyhow::Error> {
        let placement = DevicePlacement::own();
        let placement = Arc::new(placement);
        Self::create_placed(options, layout, &placement)
    }

    /// [Self::create] on the client `placement` gives.
    fn create_placed(
        options: &PlaneOptions,
        layout: FrameLayout,
        placement: &Arc<DevicePlacement>,
    ) -> Result<Self, anyhow::Error> {
        let (chroma_width, chroma_height) = layout.chroma_dims();

        if chroma_width == 0 || chroma_height == 0 {
            anyhow::bail!(
                "frame dimensions {}x{} are too small for subsampling {:?}",
                layout.width,
                layout.height,
                layout.subsampling
            );
        }

        let denoiser_options = options.denoiser_options(layout);
        let host = HostDenoiser::create_with(
            &options.accelerators,
            &options.device,
            placement,
            layout.width,
            layout.height,
            denoiser_options,
        )?;

        let temporal_radius = match options.mode {
            DenoisingMode::Spacial => 0,
            DenoisingMode::Temporal { radius } => radius,
        };

        let accelerator = host.selected_accelerator();
        let window_span = host.window_span();

        Ok(Self {
            layout,
            host: Some(host),
            temporal_radius,
            accelerator,
            window_span,
            options: options.clone(),
            placement: placement.clone(),
        })
    }

    #[cfg(all(test, feature = "vulkan"))]
    pub(crate) fn placement(&self) -> &Arc<DevicePlacement> {
        &self.placement
    }

    #[cfg(all(test, feature = "vulkan"))]
    pub(crate) fn host(&self) -> Option<&HostDenoiser> {
        self.host.as_ref()
    }

    #[cfg(all(test, feature = "vulkan"))]
    pub(crate) fn host_mut(&mut self) -> Option<&mut HostDenoiser> {
        self.host.as_mut()
    }

    /// Tunes this denoiser's engine on `frames` within `budget`, and saves the result.
    ///
    /// The pass reads the [tuning_config](crate::tune::tuning_config) of this denoiser's options, so it
    /// shares a store and the deterministic flag with the rebuild. The frames are looped as often as
    /// tuning needs and no output from them is returned. The host's engine is freed while the pass
    /// runs, then rebuilt from the stored results on the same device, ready for a fresh stream. A pass
    /// that is skipped up front leaves the host untouched.
    ///
    /// A failed rebuild is an error, and the denoiser then rejects every push.
    pub fn tune(
        &mut self,
        frames: &[Planes],
        budget: BudgetRequest,
        request: &TuneRequest,
    ) -> Result<TuneReport, anyhow::Error> {
        let defaults = tuning_config(&self.options);
        let config = TuningConfig { budget, ..defaults };

        let prepared = self.prepare_tuning(frames, &config);
        let pass = match prepared {
            Ok(pass) => pass,
            Err(reason) => return Ok(TuneReport::skipped(reason)),
        };

        self.drop_host();
        let report = pass.run(&config, request);
        drop(pass);

        self.rebuild_settled()?;
        Ok(report)
    }

    /// Tunes on `frames` like [Self::tune], dropping this denoiser before the pass runs.
    pub(crate) fn tune_and_drop(
        self,
        frames: &[Planes],
        config: &TuningConfig,
        request: &TuneRequest,
    ) -> TuneReport {
        let prepared = self.prepare_tuning(frames, config);
        let pass = match prepared {
            Ok(pass) => pass,
            Err(reason) => return TuneReport::skipped(reason),
        };

        drop(self);
        pass.run(config, request)
    }

    /// Uploads `frames` once for a tuning pass, or says why no pass runs.
    fn prepare_tuning(&self, frames: &[Planes], config: &TuningConfig) -> Result<TuningPass, String> {
        if config.mode == TuningMode::Off {
            return Err("tuning is off".to_string());
        }

        if frames.is_empty() {
            return Err("there are no frames to tune on".to_string());
        }

        let Some(host) = self.host.as_ref() else {
            return Err("there is no engine to tune".to_string());
        };

        let mut uploaded = Vec::with_capacity(frames.len());

        for planes in frames {
            let checked = self.check_planes(planes);
            if let Err(error) = checked {
                return Err(format!("{error:#}"));
            }

            let handles = host.upload_frame(&[&planes.y, &planes.u, &planes.v]);
            let frame = TuningFrame::new(vec![handles]);
            uploaded.push(frame);
        }

        Ok(TuningPass {
            device: host.selected().clone(),
            client: host.client().clone(),
            spec: host.spec(),
            frames: uploaded,
        })
    }

    /// Frees the host, so its engine releases the device's memory.
    fn drop_host(&mut self) {
        self.host = None;
    }

    /// Rebuilds the host on the same placement from the stored results, with tuning settled.
    fn rebuild_settled(&mut self) -> Result<(), anyhow::Error> {
        let tuning = self.options.tuning.after_tuning();
        let settled = PlaneOptions {
            tuning,
            ..self.options.clone()
        };

        let rebuilt = Self::create_placed(&settled, self.layout, &self.placement);
        let planar = rebuilt.context("cannot rebuild the denoiser after tuning")?;
        *self = planar;

        Ok(())
    }

    /// The host, or an error once [Self::tune] has freed it and could not rebuild it.
    fn live_host(&mut self) -> Result<&mut HostDenoiser, DenoiserError> {
        let Some(host) = self.host.as_mut() else {
            let error = anyhow::anyhow!("the denoiser has no engine because its rebuild after tuning failed");
            return Err(DenoiserError::Other(error));
        };

        Ok(host)
    }

    /// The temporal radius the underlying denoiser runs at.
    pub fn temporal_radius(&self) -> u32 {
        self.temporal_radius
    }

    /// Pushes one planar frame.
    ///
    /// On `QueueFull` the caller should receive one frame and then retry the whole call. Any other
    /// error is passed on unchanged. `QueueFull` is raised before anything changes, so a retry
    /// replays the frame cleanly.
    pub fn push(&mut self, planes: &Planes) -> Result<(), DenoiserError> {
        self.push_with(planes, HostDenoiser::push)
    }

    /// Uploads one planar frame into the temporal window without starting a denoise.
    fn push_priming(&mut self, planes: &Planes) -> Result<(), DenoiserError> {
        self.push_with(planes, HostDenoiser::push_priming)
    }

    /// Runs `push_frame` against the host with every plane of `planes`.
    ///
    /// The planes go over as wire bytes, so the normalisation and the channel interleave both happen
    /// on the GPU.
    fn push_with(&mut self, planes: &Planes, push_frame: WirePush) -> Result<(), DenoiserError> {
        self.check_planes(planes)?;

        let host = self.live_host()?;
        push_frame(host, &[&planes.y, &planes.u, &planes.v])
    }

    /// Rejects a frame whose plane lengths do not match the layout.
    fn check_planes(&self, planes: &Planes) -> Result<(), DenoiserError> {
        let luma_bytes = self.layout.luma_bytes();
        let chroma_bytes = self.layout.chroma_bytes();
        let expected = [
            ("Y", planes.y.len(), luma_bytes),
            ("U", planes.u.len(), chroma_bytes),
            ("V", planes.v.len(), chroma_bytes),
        ];

        for (name, length, expected_length) in expected {
            if length != expected_length {
                let message = format!("plane {name} holds {length} bytes, expected {expected_length}");
                let error = av_denoise_core::Error::PlaneMismatch(message);
                return Err(DenoiserError::Engine(error));
            }
        }

        Ok(())
    }

    /// Blocks until the oldest frame in flight lands and returns it.
    ///
    /// Returns `Ok(None)` when nothing is in flight.
    pub fn recv(&mut self) -> Result<Option<Planes>, anyhow::Error> {
        let host = self.live_host()?;
        let received = host.recv()?;

        Ok(received.map(Planes::from_buffers))
    }

    /// Reads back the luma grain chunks measured since the last call, in frame order.
    ///
    /// Measured chunks stay on the GPU until drained, so callers drain after each flush.
    pub fn drain_grain_chunks(&mut self) -> Result<Vec<GrainChunk>, anyhow::Error> {
        let Some(host) = self.host.as_mut() else {
            return Ok(Vec::new());
        };

        let chunks = host.drain_grain_chunks()?;

        Ok(chunks)
    }

    /// Drains the frames in flight and the temporal tail.
    ///
    /// `sink` is called once per emitted planar frame.
    pub fn flush(&mut self, mut sink: impl FnMut(Planes)) -> Result<(), anyhow::Error> {
        let host = self.live_host()?;
        host.flush(|buffers| {
            let planes = Planes::from_buffers(buffers);
            sink(planes);
        })?;

        Ok(())
    }

    /// Samples the device time of one frame in every `every` on each denoised plane.
    pub fn sample_device_time(&mut self, every: u32) {
        if let Some(host) = self.host.as_mut() {
            host.sample_device_time(every);
        }
    }

    /// Where each denoised plane's time went so far.
    pub fn timing(&mut self) -> PlanarTiming {
        let Some(host) = self.host.as_mut() else {
            return PlanarTiming::default();
        };

        host.timing()
    }

    /// The backend the denoised planes run on.
    pub fn selected_accelerator(&self) -> Accelerator {
        self.accelerator
    }

    /// The number of frames behind and ahead of a target frame a [Self::reseed] window must supply.
    pub fn window_span(&self) -> WindowSpan {
        self.window_span
    }
}
