//! The `avd.Passthrough` filter.
//!
//! Returns source frames unchanged. It exists to prove the plugin build,
//! link, and load path works before any real denoising filter is wired in.

use anyhow::{Error, anyhow};
use vapoursynth::core::CoreRef;
use vapoursynth::plugins::{Filter, FrameContext};
use vapoursynth::prelude::{API, FrameRef, Node};
use vapoursynth::video_info::VideoInfo;

/// Raises the stack size cubecl's kernel codegen thread inherits.
///
/// Codegen unrolls the windowed NLM kernel body `(2R+1)^2` times, which
/// overflows the default 2 MiB stack at a search radius of 5 or more and
/// aborts the host process. `export_vapoursynth_plugin!` expands to the
/// whole body of the plugin's entry point, so there is no earlier hook of
/// ours to set this in. Setting it as the first statement of the shared
/// filter-creation function is early enough: cubecl only spawns its
/// codegen thread once a denoiser is created, later in this same function.
pub(crate) fn raise_stack_limit() {
    if std::env::var_os("RUST_MIN_STACK").is_none() {
        // SAFETY: called before any denoiser thread spawns, and before any
        // other thread in this process could be reading the environment.
        unsafe { std::env::set_var("RUST_MIN_STACK", "16777216") };
    }
}

/// A filter that returns the source clip's frames unchanged.
pub struct Passthrough<'core> {
    source: Node<'core>,
}

impl<'core> Passthrough<'core> {
    /// Wraps a source clip so its frames pass through unchanged.
    pub(crate) fn new(source: Node<'core>) -> Self {
        Self { source }
    }
}

impl<'core> Filter<'core> for Passthrough<'core> {
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
        self.source.request_frame_filter(context, n);
        Ok(None)
    }

    fn get_frame(
        &self,
        _api: API,
        _core: CoreRef<'core>,
        context: FrameContext,
        n: usize,
    ) -> Result<FrameRef<'core>, Error> {
        self.source
            .get_frame_filter(context, n)
            .ok_or_else(|| anyhow!("Couldn't get the source frame"))
    }
}
