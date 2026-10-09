use std::ops::Range;
use std::sync::Mutex;

use anyhow::{Error, Result, anyhow, bail};
use av_denoise::{
    EdgePadding,
    FrameLayout,
    MAX_PENDING,
    PlanarDenoiser,
    Planes,
    ReseedWindow,
    WarmUp,
    push_needs_retry,
};
use vapoursynth::core::CoreRef;
use vapoursynth::plugins::{Filter, FrameContext};
use vapoursynth::prelude::{API, FrameRef, FrameRefMut, Node, Property};
use vapoursynth::video_info::{Resolution, VideoInfo};

use crate::frames::{pack_plane, shifted_window_range, unpack_plane_into, window_indices};
use crate::params::{AlgorithmKind, RawFormat, RawParams, layout_from_format, plane_options_from};
use crate::stream::{OutputBuffer, Outstanding, Plan, Step, StreamGeometry, StreamPlanner, lookback_for};
use crate::{init_logging, pin_plugin_library};

/// The running pipeline, the planner tracking it, and the outputs no request has taken yet.
///
/// VapourSynth may call `get_frame` from several threads, so this sits behind a mutex. One pipeline is
/// enough because the GPU is the bottleneck.
struct State {
    denoiser: PlanarDenoiser,
    planner: StreamPlanner,
    buffer: OutputBuffer<Planes>,
    /// The cold-cache queue place this filter holds until its first frame is rendered.
    ///
    /// CubeCL compiles a kernel on its first dispatch rather than when the denoiser is built, so the
    /// place is held across the first frame. A process that builds the filter and never pulls a frame
    /// keeps the place until it exits, making other workers wait out the queue's limit before compiling
    /// for themselves, which is rare enough to accept.
    warm_up: Option<WarmUp>,
}

impl State {
    /// Gives up this filter's place in the cold-cache queue once a frame has compiled and cached every
    /// kernel it needs.
    fn finish_warm_up(&mut self) {
        if let Some(warm_up) = self.warm_up.take() {
            warm_up.finish();
        }
    }
}

/// A denoising filter backed by one [PlanarDenoiser] stream shared by every request.
///
/// `avd.NLMeans` and `avd.NL4D` both build one, differing only in the algorithm the denoiser runs.
pub struct Denoise<'core> {
    source: Node<'core>,
    layout: FrameLayout,
    geometry: StreamGeometry,
    /// Requests `get_frame_initial` has registered, kept apart from [State] so registering one never
    /// waits on GPU work.
    outstanding: Mutex<Outstanding>,
    state: Mutex<State>,
}

impl<'core> Denoise<'core> {
    /// Builds an `avd.NLMeans` or `avd.NL4D` filter.
    ///
    /// Rejects the source's format and resolution before touching the GPU, including a
    /// variable-resolution source, which [FrameLayout] has no way to represent.
    pub(crate) fn create(
        _api: API,
        core: CoreRef<'core>,
        source: Node<'core>,
        algorithm_kind: AlgorithmKind,
        raw: &RawParams,
    ) -> Result<Self, Error> {
        init_logging();
        pin_plugin_library();

        // cubecl spawns its codegen thread when the denoiser below is created, so the stack limit is
        // raised first. `export_vapoursynth_plugin!` owns the plugin's entry point, so there is no
        // earlier hook of ours to do it in.
        // SAFETY: best-effort mutation at the earliest hook this plugin gets. The host may already
        // have other threads touching the environment, so this cannot guarantee exclusive access, but
        // the alternative is a hard abort during codegen.
        unsafe { av_denoise::raise_codegen_stack_limit() };

        let info = source.info();

        let (width, height) = match info.resolution {
            Property::Constant(resolution) => (resolution.width as u32, resolution.height as u32),
            Property::Variable => {
                bail!("clips with variable resolution are not supported");
            },
        };

        let format = info.format;
        let raw_format = RawFormat {
            sample_type: format.sample_type(),
            bits_per_sample: format.bits_per_sample(),
            subsampling_w: format.sub_sampling_w(),
            subsampling_h: format.sub_sampling_h(),
            color_family: format.color_family(),
        };

        let layout = layout_from_format(raw_format, width, height)?;
        let plane_options = plane_options_from(raw, algorithm_kind, layout)?;

        // Av1an runs one of these per chunk, so without a cache every chunk pays the ten seconds it
        // takes to compile the kernels. The queue keeps the first wave of chunks from all paying it at
        // once.
        av_denoise::install_compilation_cache_once();
        let cache_key = av_denoise::kernel_key(&plane_options, layout);
        let warm_up = WarmUp::begin(cache_key);

        let denoiser = PlanarDenoiser::create(&plane_options, layout)?;
        let span = denoiser.window_span();

        let core_info = core.info();
        let lookback = lookback_for(core_info.num_threads);
        let last_frame = info.num_frames.saturating_sub(1);
        let geometry = StreamGeometry {
            span,
            lookback,
            lead: MAX_PENDING,
            last_frame,
        };

        let planner = StreamPlanner::new(geometry);
        let state = State {
            denoiser,
            planner,
            buffer: OutputBuffer::new(),
            warm_up,
        };

        Ok(Self {
            source,
            layout,
            geometry,
            outstanding: Mutex::new(Outstanding::default()),
            state: Mutex::new(state),
        })
    }

    /// Renders output frame `n` from the shared stream.
    ///
    /// The stream is caught up, rebuilt, or skipped when `n` is already buffered, as the planner decides.
    /// Any error leaves the stream dead, so the next request rebuilds it.
    fn render(&self, n: usize, fetch: impl Fn(usize) -> Result<Planes, Error>) -> Result<Planes, Error> {
        let mut state = self.state.lock().expect("denoiser mutex poisoned");

        let result = self.render_locked(&mut state, n, &fetch);
        if result.is_err() {
            state.planner.kill();
        }

        result
    }

    fn render_locked(
        &self,
        state: &mut State,
        n: usize,
        fetch: &impl Fn(usize) -> Result<Planes, Error>,
    ) -> Result<Planes, Error> {
        let buffered = state.buffer.contains(n);
        let plan = {
            let outstanding = self.outstanding.lock().expect("outstanding mutex poisoned");
            state.planner.plan(n, buffered, &outstanding)
        };

        match plan {
            Plan::Serve => {},
            Plan::Advance { steps } => {
                self.run_steps(state, steps, fetch)?;
            },
            Plan::Reseed { start, steps } => {
                self.reseed_at(state, start, fetch)?;
                self.run_steps(state, steps, fetch)?;
            },
        }

        receive_all(state)?;

        let denoised = state
            .buffer
            .take(n)
            .ok_or_else(|| anyhow!("output frame {n} was never produced"))?;

        let capacity = self.geometry.buffer_capacity();
        state.buffer.evict_to(capacity);
        state.finish_warm_up();

        Ok(denoised)
    }

    /// Rebuilds the stream so it has produced output `start`, buffering it and any clip-end tail.
    fn reseed_at(
        &self,
        state: &mut State,
        start: usize,
        fetch: &impl Fn(usize) -> Result<Planes, Error>,
    ) -> Result<(), Error> {
        let span = self.geometry.span;
        let last_frame = self.geometry.last_frame;

        if span.edges == EdgePadding::Repeat {
            let indices = window_indices(start, span.behind, span.ahead, last_frame);
            let window = fetch_all(&indices, fetch)?;
            let denoised = state.denoiser.reseed(&window)?;
            state.buffer.insert(start, denoised);
            state.planner.reseeded(start);
            return Ok(());
        }

        let range = shifted_window_range(start, span.behind, span.ahead, last_frame);
        let first_index = *range.start();
        let last_index = *range.end();
        let indices: Vec<usize> = range.collect();
        let window = fetch_all(&indices, fetch)?;

        let request = ReseedWindow {
            frames: &window,
            target: start - first_index,
            at_clip_start: first_index == 0,
            at_clip_end: last_index == last_frame,
        };

        let outputs = state.denoiser.reseed_window(request)?;
        for (offset, denoised) in outputs.into_iter().enumerate() {
            state.buffer.insert(start + offset, denoised);
        }

        state.planner.reseeded(start);
        Ok(())
    }

    /// Runs each step in order, buffering every output the stream hands back along the way.
    fn run_steps(
        &self,
        state: &mut State,
        steps: Range<usize>,
        fetch: &impl Fn(usize) -> Result<Planes, Error>,
    ) -> Result<(), Error> {
        for step in steps {
            match self.geometry.step(step) {
                Step::Push(source) => {
                    let planes = fetch(source)?;
                    push_with_drain(state, &planes)?;
                },
                Step::Flush => {
                    flush_into_buffer(state)?;
                },
            }
        }

        Ok(())
    }

    /// Renders output frame `n` and wraps it in a frame carrying the source frame's properties.
    ///
    /// Source frames are packed only when the stream actually pushes them.
    fn serve(&self, core: CoreRef<'core>, context: FrameContext, n: usize) -> Result<FrameRef<'core>, Error> {
        let depth_bytes = self.layout.depth.bytes_per_sample();
        let fetch = |index: usize| -> Result<Planes, Error> {
            let frame = self
                .source
                .get_frame_filter(context, index)
                .ok_or_else(|| anyhow!("couldn't get source frame {index}"))?;
            let packed = pack_frame(&frame, depth_bytes);
            Ok(packed)
        };

        let planes = self.render(n, fetch)?;

        let props_source = self
            .source
            .get_frame_filter(context, n)
            .ok_or_else(|| anyhow!("couldn't get source frame {n}"))?;
        let format = props_source.format();
        let resolution = Resolution {
            width: self.layout.width as usize,
            height: self.layout.height as usize,
        };

        // SAFETY: the frame's plane data starts uninitialized, but
        // `unpack_into_frame` below writes every byte of every plane
        // before the frame is returned to VapourSynth.
        let mut output_frame =
            unsafe { FrameRefMut::new_uninitialized(core, Some(&props_source), format, resolution) };
        unpack_into_frame(&mut output_frame, &planes, depth_bytes);

        Ok(output_frame.into())
    }
}

fn fetch_all(
    indices: &[usize],
    fetch: &impl Fn(usize) -> Result<Planes, Error>,
) -> Result<Vec<Planes>, Error> {
    indices.iter().map(|&index| fetch(index)).collect()
}

/// Pushes one frame, first receiving one output into the buffer when the queue is full.
fn push_with_drain(state: &mut State, planes: &Planes) -> Result<(), Error> {
    let first_attempt = state.denoiser.push(planes);
    if push_needs_retry(first_attempt)? {
        receive_one(state)?;
        state.denoiser.push(planes)?;
    }

    state.planner.pushed();
    Ok(())
}

/// Receives the oldest in-flight output into the buffer.
fn receive_one(state: &mut State) -> Result<(), Error> {
    let index = state
        .planner
        .received()
        .ok_or_else(|| anyhow!("no output is in flight to receive"))?;

    receive_output(state, index)
}

/// Receives every in-flight output into the buffer.
///
/// A readback left in flight could be collected by a request on another thread, whose GPU stream has no
/// ordering against the one that started it. Receiving everything keeps each readback inside the call
/// that began it.
fn receive_all(state: &mut State) -> Result<(), Error> {
    while let Some(index) = state.planner.received() {
        receive_output(state, index)?;
    }

    Ok(())
}

fn receive_output(state: &mut State, index: usize) -> Result<(), Error> {
    let received = state.denoiser.recv()?;
    let denoised = received.ok_or_else(|| anyhow!("the stream yielded nothing for output {index}"))?;
    state.buffer.insert(index, denoised);
    Ok(())
}

/// Flushes the stream's in-flight and tail outputs into the buffer. The stream ends.
fn flush_into_buffer(state: &mut State) -> Result<(), Error> {
    let mut outputs = Vec::new();
    state.denoiser.flush(|planes| outputs.push(planes))?;

    let indices = state.planner.flushed();
    if outputs.len() != indices.len() {
        bail!(
            "the clip's final flush produced {} frames, expected {}",
            outputs.len(),
            indices.len()
        );
    }

    for (index, denoised) in indices.zip(outputs) {
        state.buffer.insert(index, denoised);
    }

    Ok(())
}

/// Packs one source frame's three planes into a [Planes], dropping each plane's row padding.
fn pack_frame(frame: &FrameRef, depth_bytes: usize) -> Planes {
    let pack = |plane: usize| -> Vec<u8> {
        let stride = frame.stride(plane);
        let height = frame.height(plane);
        let width_bytes = frame.width(plane) * depth_bytes;
        // SAFETY: `stride * height` is exactly the byte range VapourSynth
        // allocated for this plane, and `frame` outlives the slice.
        let data = unsafe { std::slice::from_raw_parts(frame.data_ptr(plane), stride * height) };
        pack_plane(data, stride, width_bytes, height)
    };

    let y_plane = pack(0);
    let u_plane = pack(1);
    let v_plane = pack(2);

    Planes {
        y: y_plane,
        u: u_plane,
        v: v_plane,
    }
}

/// Writes a denoised [Planes] into a freshly allocated output frame.
fn unpack_into_frame(frame: &mut FrameRefMut, planes: &Planes, depth_bytes: usize) {
    let sources = [&planes.y, &planes.u, &planes.v];
    for (plane, packed) in sources.into_iter().enumerate() {
        let stride = frame.stride(plane);
        let height = frame.height(plane);
        let width_bytes = frame.width(plane) * depth_bytes;
        // SAFETY: `stride * height` is exactly the byte range VapourSynth
        // allocated for this plane.
        let data = unsafe { std::slice::from_raw_parts_mut(frame.data_ptr_mut(plane), stride * height) };
        unpack_plane_into(data, stride, width_bytes, height, packed);
    }
}

impl<'core> Filter<'core> for Denoise<'core> {
    fn video_info(&self, _api: API, _core: CoreRef<'core>) -> Vec<VideoInfo<'core>> {
        vec![self.source.info()]
    }

    fn get_frame_initial(
        &self,
        _api: API,
        _core: CoreRef<'core>,
        context: FrameContext,
        output_index: usize,
    ) -> Result<Option<FrameRef<'core>>, Error> {
        {
            let mut outstanding = self.outstanding.lock().expect("outstanding mutex poisoned");
            outstanding.register(output_index);
        }

        let range = self.geometry.request_range(output_index);
        for index in range {
            self.source.request_frame_filter(context, index);
        }

        Ok(None)
    }

    fn get_frame(
        &self,
        _api: API,
        core: CoreRef<'core>,
        context: FrameContext,
        output_index: usize,
    ) -> Result<FrameRef<'core>, Error> {
        let result = self.serve(core, context, output_index);

        {
            let mut outstanding = self.outstanding.lock().expect("outstanding mutex poisoned");
            outstanding.finish(output_index);
        }

        result
    }
}
