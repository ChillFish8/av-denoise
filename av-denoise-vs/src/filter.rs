//! The `avd.NLMeans` and `avd.NL4D` filters.
//!
//! Both share one `Denoise` filter type and one GPU pipeline underneath.
//! They differ only in the [`AlgorithmKind`] their creation function
//! passes to [`plane_options_from`].

use std::collections::HashMap;
use std::sync::Mutex;

use anyhow::{Error, Result, anyhow};
use av_denoise_core::{FrameLayout, PlanarDenoiser, Planes, WarmUp, WindowSpan};
use vapoursynth::core::CoreRef;
use vapoursynth::plugins::{Filter, FrameContext};
use vapoursynth::prelude::{API, FrameRef, FrameRefMut, Node, Property};
use vapoursynth::video_info::{Resolution, VideoInfo};

use crate::frames::{pack_plane, unpack_plane_into, window_indices};
use crate::params::{AlgorithmKind, RawFormat, RawParams, layout_from_format, plane_options_from};

/// The running pipeline and the output frame it last produced.
///
/// VapourSynth may call `get_frame` from several threads, so the
/// pipeline sits behind a mutex and requests queue on it. One pipeline
/// is enough because the GPU is the bottleneck.
struct State {
    denoiser: PlanarDenoiser,
    /// The output frame index the pipeline is positioned after.
    last: Option<usize>,
    /// The cold-cache queue place this filter holds, until the first
    /// frame proves the kernels are compiled and cached.
    ///
    /// CubeCL compiles a kernel when it is first dispatched rather than
    /// when the denoiser is built, so the place has to be held across a
    /// frame and cannot be given up at the end of creation.
    ///
    /// A process that builds the filter and then renders nothing keeps
    /// the place until it exits, and other workers wait out the queue's
    /// limit before compiling for themselves. That needs a long lived
    /// process holding a node it never pulls a frame from, which is rare
    /// enough to accept.
    warm_up: Option<WarmUp>,
}

impl State {
    /// Gives up this filter's place in the cold-cache queue, now that a
    /// frame has been through and every kernel it needs is compiled and
    /// written to the cache.
    ///
    /// Does nothing after the first call, and nothing at all when the
    /// filter never took a place.
    fn finish_warm_up(&mut self) {
        if let Some(warm_up) = self.warm_up.take() {
            warm_up.finish();
        }
    }
}

/// A denoising filter backed by one [`PlanarDenoiser`] pipeline.
///
/// `avd.NLMeans` and `avd.NL4D` both build one of these, differing only
/// in the algorithm baked into `state.denoiser` at creation.
pub struct Denoise<'core> {
    source: Node<'core>,
    layout: FrameLayout,
    /// How many source frames a window at output frame `n` needs behind
    /// and ahead of `n`, read once from
    /// [`PlanarDenoiser::window_span`] at creation. nlmeans and nl4d
    /// report different spans, so this is asked for rather than
    /// assumed symmetric.
    span: WindowSpan,
    /// The source clip's frame count, read once at creation.
    source_len: usize,
    state: Mutex<State>,
}

impl<'core> Denoise<'core> {
    /// Builds the shared parts of a `avd.NLMeans` or `avd.NL4D` filter.
    ///
    /// Raises the stack limit first, since a `PlanarDenoiser` is created
    /// below and cubecl only spawns its codegen thread once that
    /// happens. Rejects the source's format and resolution before
    /// touching the GPU, and rejects a variable-resolution source, which
    /// `FrameLayout` has no way to represent.
    pub(crate) fn create(
        _api: API,
        _core: CoreRef<'core>,
        source: Node<'core>,
        algorithm_kind: AlgorithmKind,
        raw: &RawParams,
    ) -> Result<Self, Error> {
        // `export_vapoursynth_plugin!` expands to the whole body of the
        // plugin's entry point, so there is no earlier hook of ours to
        // raise the stack limit in.
        // SAFETY: best-effort mutation at the earliest hook this plugin
        // gets. The host may already have other threads touching the
        // environment, so this cannot guarantee exclusive access, but the
        // alternative is a hard abort during codegen.
        unsafe { av_denoise_core::raise_codegen_stack_limit() };

        let info = source.info();

        let (width, height) = match info.resolution {
            Property::Constant(res) => (res.width as u32, res.height as u32),
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

        // Av1an runs one of these per chunk, so without a cache every
        // chunk pays the ten seconds it takes to compile the kernels.
        // The queue below keeps the first wave of chunks from all paying
        // it at once.
        av_denoise_core::install_compilation_cache_once();
        let warm_up = WarmUp::begin(av_denoise_core::kernel_key(&plane_options, layout));

        let denoiser = PlanarDenoiser::create(&plane_options, layout)?;
        let span = denoiser.window_span();

        Ok(Self {
            source,
            layout,
            span,
            source_len: info.num_frames,
            state: Mutex::new(State {
                denoiser,
                last: None,
                warm_up,
            }),
        })
    }

    /// The full, ordered source indices around output frame `n`,
    /// exactly as `reseed` needs them, boundary repeats included.
    fn window(&self, n: usize) -> Vec<usize> {
        window_indices(n, self.span.behind, self.span.ahead, self.source_len - 1)
    }

    /// The source indices output frame `n` needs, deduplicated so each
    /// one is requested and fetched from VapourSynth only once.
    ///
    /// A window near either end of the clip repeats its boundary frame,
    /// which [`Self::window`] preserves since `reseed` needs the exact
    /// count. This is sorted since `window_indices` is already
    /// non-decreasing, so sorting is a no-op kept for clarity.
    fn unique_window(&self, n: usize) -> Vec<usize> {
        let mut indices = self.window(n);
        indices.sort_unstable();
        indices.dedup();
        indices
    }

    /// Renders one output frame, applying the hybrid fast/rebuild policy.
    ///
    /// A request for the frame straight after the last one produced
    /// pushes a single frame through the running stream. Anything else,
    /// including frame 0, abandons the stream and rebuilds it from an
    /// explicit window, which costs more but is correct from any
    /// starting point.
    fn render(&self, n: usize, fetch: impl Fn(usize) -> Result<Planes, Error>) -> Result<Planes, Error> {
        let mut state = self.state.lock().expect("denoiser mutex poisoned");
        let last_frame = self.source_len - 1;

        // Read the anchor, then clear it before anything touches the pipeline.
        //
        // Every path below either reaches a `state.last = Some(n)` or leaves through `?`,
        // so an error out of `fetch`, `push`, `recv`, or `reseed` can never leave the
        // anchor claiming a position the stream has moved past.
        let sequential = state.last == Some(n.wrapping_sub(1)) && n > 0;
        state.last = None;

        if sequential {
            let ahead = (n + self.span.ahead).min(last_frame);
            state.denoiser.push(&fetch(ahead)?)?;
            if let Some(out) = state.denoiser.recv()? {
                state.last = Some(n);
                state.finish_warm_up();
                return Ok(out);
            }
            // The stream did not yield, so fall through and rebuild.
        }

        let window: Vec<Planes> = self.window(n).into_iter().map(fetch).collect::<Result<_, _>>()?;

        let out = state.denoiser.reseed(&window)?;
        state.last = Some(n);
        state.finish_warm_up();
        Ok(out)
    }
}

/// Packs one source frame's three planes into a [`Planes`], dropping
/// each plane's row padding.
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

    Planes {
        y: pack(0),
        u: pack(1),
        v: pack(2),
    }
}

/// Writes a denoised [`Planes`] into a freshly allocated output frame.
fn unpack_into_frame(frame: &mut FrameRefMut, planes: &Planes, depth_bytes: usize) {
    let sources = [&planes.y, &planes.u, &planes.v];
    for (plane, src) in sources.into_iter().enumerate() {
        let stride = frame.stride(plane);
        let height = frame.height(plane);
        let width_bytes = frame.width(plane) * depth_bytes;
        // SAFETY: `stride * height` is exactly the byte range VapourSynth
        // allocated for this plane.
        let data = unsafe { std::slice::from_raw_parts_mut(frame.data_ptr_mut(plane), stride * height) };
        unpack_plane_into(data, stride, width_bytes, height, src);
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
        n: usize,
    ) -> Result<Option<FrameRef<'core>>, Error> {
        for idx in self.unique_window(n) {
            self.source.request_frame_filter(context, idx);
        }
        Ok(None)
    }

    fn get_frame(
        &self,
        _api: API,
        core: CoreRef<'core>,
        context: FrameContext,
        n: usize,
    ) -> Result<FrameRef<'core>, Error> {
        let mut frames: HashMap<usize, FrameRef<'core>> = HashMap::new();
        for idx in self.unique_window(n) {
            let frame = self
                .source
                .get_frame_filter(context, idx)
                .ok_or_else(|| anyhow!("couldn't get source frame {idx}"))?;
            frames.insert(idx, frame);
        }

        let depth_bytes = self.layout.depth.bytes_per_sample();
        let fetch = |idx: usize| -> Result<Planes, Error> {
            let frame = frames
                .get(&idx)
                .expect("get_frame_initial requested the same window as get_frame");
            Ok(pack_frame(frame, depth_bytes))
        };

        let planes = self.render(n, fetch)?;

        let prop_src = frames.get(&n).expect("the window always includes n");
        let format = prop_src.format();
        let resolution = Resolution {
            width: self.layout.width as usize,
            height: self.layout.height as usize,
        };

        // SAFETY: the frame's plane data starts uninitialized, but
        // `unpack_into_frame` below writes every byte of every plane
        // before the frame is returned to VapourSynth.
        let mut out = unsafe { FrameRefMut::new_uninitialized(core, Some(prop_src), format, resolution) };
        unpack_into_frame(&mut out, &planes, depth_bytes);

        Ok(out.into())
    }
}
