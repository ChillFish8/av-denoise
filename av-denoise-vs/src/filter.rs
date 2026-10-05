use std::collections::HashMap;
use std::sync::Mutex;

use anyhow::{Error, Result, anyhow};
use av_denoise::{EdgePadding, FrameLayout, PlanarDenoiser, Planes, ReseedWindow, WarmUp, WindowSpan};
use vapoursynth::core::CoreRef;
use vapoursynth::plugins::{Filter, FrameContext};
use vapoursynth::prelude::{API, FrameRef, FrameRefMut, Node, Property};
use vapoursynth::video_info::{Resolution, VideoInfo};

use crate::frames::{TailCache, pack_plane, shifted_window_range, unpack_plane_into, window_indices};
use crate::params::{AlgorithmKind, RawFormat, RawParams, layout_from_format, plane_options_from};
use crate::{init_logging, pin_plugin_library};

/// The running pipeline and the output frame it last produced.
///
/// VapourSynth may call `get_frame` from several threads, so the pipeline sits behind a mutex and
/// requests queue on it. One pipeline is enough because the GPU is the bottleneck.
struct State {
    denoiser: PlanarDenoiser,
    last_served: Option<usize>,
    /// The cold-cache queue place this filter holds until its first frame is rendered.
    ///
    /// CubeCL compiles a kernel on its first dispatch rather than when the denoiser is built, so the
    /// place is held across the first frame. A process that builds the filter and never pulls a frame
    /// keeps the place until it exits, making other workers wait out the queue's limit before compiling
    /// for themselves, which is rare enough to accept.
    warm_up: Option<WarmUp>,
    /// Outputs from the clip's final flush that no request has taken yet.
    tail: Option<TailCache>,
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

/// A denoising filter backed by one [PlanarDenoiser] pipeline.
///
/// `avd.NLMeans` and `avd.NL4D` both build one, differing only in the algorithm the denoiser runs.
pub struct Denoise<'core> {
    source: Node<'core>,
    layout: FrameLayout,
    /// How many source frames a window needs behind and ahead of its output frame.
    ///
    /// nlmeans and nl4d report different spans, so this comes from [PlanarDenoiser::window_span]
    /// rather than being assumed symmetric.
    span: WindowSpan,
    source_len: usize,
    state: Mutex<State>,
}

impl<'core> Denoise<'core> {
    /// Builds an `avd.NLMeans` or `avd.NL4D` filter.
    ///
    /// Rejects the source's format and resolution before touching the GPU, including a
    /// variable-resolution source, which [FrameLayout] has no way to represent.
    pub(crate) fn create(
        _api: API,
        _core: CoreRef<'core>,
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
                anyhow::bail!("clips with variable resolution are not supported");
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

        let state = State {
            denoiser,
            last_served: None,
            warm_up,
            tail: None,
        };

        Ok(Self {
            source,
            layout,
            span,
            source_len: info.num_frames,
            state: Mutex::new(state),
        })
    }

    /// The ordered source indices `reseed` or `reseed_window` needs for an output frame.
    ///
    /// `EdgePadding::Repeat` repeats the boundary frame so `reseed` sees the exact window length it
    /// expects. `EdgePadding::Shifted` stops at either end of the clip instead, with no repeats, for
    /// `reseed_window`.
    fn window(&self, output_index: usize) -> Vec<usize> {
        let last_frame = self.source_len - 1;
        match self.span.edges {
            EdgePadding::Repeat => {
                window_indices(output_index, self.span.behind, self.span.ahead, last_frame)
            },
            EdgePadding::Shifted => {
                let range = shifted_window_range(output_index, self.span.behind, self.span.ahead, last_frame);
                range.collect()
            },
        }
    }

    /// The source indices an output frame needs, deduplicated so each one is requested and fetched
    /// from VapourSynth only once.
    ///
    /// Only `EdgePadding::Repeat` windows can repeat an index, at either end of the clip.
    /// [Self::window] is already non-decreasing, so the sort is a no-op kept for clarity.
    fn unique_window(&self, output_index: usize) -> Vec<usize> {
        let mut indices = self.window(output_index);
        indices.sort_unstable();
        indices.dedup();
        indices
    }

    /// Renders one output frame, applying the hybrid fast/rebuild policy.
    ///
    /// A sequential request, straight after the last frame produced, pushes one frame through the
    /// running stream. Under shifted edges, the request that reaches the clip's end flushes the stream
    /// once instead, and the frames after it are served from the tail cache. Anything else, including
    /// frame 0, rebuilds the stream from an explicit window, which costs more but is correct from any
    /// starting point.
    fn render(
        &self,
        output_index: usize,
        fetch: impl Fn(usize) -> Result<Planes, Error>,
    ) -> Result<Planes, Error> {
        let mut state = self.state.lock().expect("denoiser mutex poisoned");
        let last_frame = self.source_len - 1;
        let shifted = self.span.edges == EdgePadding::Shifted;

        // The anchor is cleared before anything touches the pipeline. Every path below either sets it
        // again or leaves through `?`, so an error can never leave it claiming a position the stream
        // has moved past.
        let sequential = state.last_served == Some(output_index.wrapping_sub(1)) && output_index > 0;
        state.last_served = None;

        let cached = state.tail.as_mut().and_then(|tail| tail.take(output_index));
        if let Some(denoised) = cached {
            state.last_served = Some(output_index);
            return Ok(denoised);
        }

        state.tail = None;

        let window_end = output_index + self.span.ahead;

        // This request is the first whose window would run past the clip's last frame. The previous
        // request pushed that last frame into a live stream, and no tail cache covers this index, so
        // flushing the stream yields this frame and every later one.
        if sequential && shifted && window_end == last_frame + 1 {
            let mut outputs = Vec::new();
            state.denoiser.flush(|planes| outputs.push(planes))?;

            let mut outputs = outputs.into_iter();
            let denoised = outputs
                .next()
                .ok_or_else(|| anyhow!("the clip's final flush produced no frame"))?;
            let rest: Vec<Planes> = outputs.collect();
            let tail = TailCache::new(output_index + 1, rest);
            state.tail = Some(tail);
            state.last_served = Some(output_index);
            state.finish_warm_up();
            return Ok(denoised);
        }

        if sequential && (!shifted || window_end <= last_frame) {
            let next_index = window_end.min(last_frame);
            let frame = fetch(next_index)?;
            state.denoiser.push(&frame)?;
            if let Some(denoised) = state.denoiser.recv()? {
                state.last_served = Some(output_index);
                state.finish_warm_up();
                return Ok(denoised);
            }

            // The stream did not yield, so fall through and rebuild.
        }

        let indices = self.window(output_index);
        let window: Vec<Planes> = indices
            .iter()
            .map(|&index| fetch(index))
            .collect::<Result<_, _>>()?;

        let denoised = if shifted {
            let first_index = indices[0];
            let last_index = *indices
                .last()
                .ok_or_else(|| anyhow!("window produced no frame indices"))?;
            let request = ReseedWindow {
                frames: &window,
                target: output_index - first_index,
                at_clip_start: first_index == 0,
                at_clip_end: last_index == last_frame,
            };

            let mut outputs = state.denoiser.reseed_window(request)?.into_iter();
            let denoised = outputs
                .next()
                .ok_or_else(|| anyhow!("reseed produced no frame"))?;
            let rest: Vec<Planes> = outputs.collect();
            if !rest.is_empty() {
                let tail = TailCache::new(output_index + 1, rest);
                state.tail = Some(tail);
            }

            denoised
        } else {
            state.denoiser.reseed(&window)?
        };

        state.last_served = Some(output_index);
        state.finish_warm_up();
        Ok(denoised)
    }
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
        for index in self.unique_window(output_index) {
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
        let mut frames: HashMap<usize, FrameRef<'core>> = HashMap::new();
        for index in self.unique_window(output_index) {
            let frame = self
                .source
                .get_frame_filter(context, index)
                .ok_or_else(|| anyhow!("couldn't get source frame {index}"))?;
            frames.insert(index, frame);
        }

        let depth_bytes = self.layout.depth.bytes_per_sample();
        let fetch = |index: usize| -> Result<Planes, Error> {
            let frame = frames
                .get(&index)
                .expect("get_frame_initial requested the same window as get_frame");
            let packed = pack_frame(frame, depth_bytes);
            Ok(packed)
        };

        let planes = self.render(output_index, fetch)?;

        let props_source = frames
            .get(&output_index)
            .expect("the window always includes output_index");
        let format = props_source.format();
        let resolution = Resolution {
            width: self.layout.width as usize,
            height: self.layout.height as usize,
        };

        // SAFETY: the frame's plane data starts uninitialized, but
        // `unpack_into_frame` below writes every byte of every plane
        // before the frame is returned to VapourSynth.
        let mut output_frame =
            unsafe { FrameRefMut::new_uninitialized(core, Some(props_source), format, resolution) };
        unpack_into_frame(&mut output_frame, &planes, depth_bytes);

        Ok(output_frame.into())
    }
}
