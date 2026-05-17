mod bilateral;

pub use bilateral::bilateral_radius;
use cubecl::prelude::*;
use cubecl::server::Handle;

/// How the per-frame reference clip is produced.
///
/// `Bilateral` and any future GPU-internal variants run a kernel
/// during `push_frame`. `External` requires the caller to supply a
/// reference frame via [`super::NlmDenoiser::push_frame_with_reference`].
/// `None` disables the reference path entirely (zero-cost).
#[non_exhaustive]
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub enum PrefilterMode {
    #[default]
    None,
    External,
    Bilateral {
        sigma_s: f32,
        sigma_r: f32,
    },
}

impl PrefilterMode {
    /// Whether the denoiser needs to allocate the reference ring buffer.
    pub(crate) fn needs_reference_buf(self) -> bool {
        !matches!(self, Self::None)
    }

    /// Whether the variant computes its reference on the GPU during
    /// `push_frame` (as opposed to consuming a caller-supplied clip).
    pub(crate) fn is_gpu_internal(self) -> bool {
        matches!(self, Self::Bilateral { .. })
    }
}

/// Inputs for a single-slot prefilter dispatch. Lives only for the
/// duration of one `push_frame`, so borrows on the denoiser's buffers
/// are sound.
pub(crate) struct PrefilterCtx<'a> {
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    pub stored_ch: u32,
    pub frame_count: u32,
    pub frame: u32,
    pub input_buf: &'a Handle,
    pub reference_buf: &'a Handle,
}

/// Dispatch the GPU prefilter for the most recently uploaded frame.
/// `None` and `External` are no-ops.
pub(crate) fn run_prefilter<R: Runtime>(
    mode: PrefilterMode,
    client: &ComputeClient<R>,
    ctx: &PrefilterCtx<'_>,
) -> Result<(), anyhow::Error> {
    match mode {
        PrefilterMode::None | PrefilterMode::External => Ok(()),
        PrefilterMode::Bilateral { sigma_s, sigma_r } => {
            bilateral::run_bilateral::<R>(client, ctx, sigma_s, sigma_r)
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn none_requires_no_reference_buffer() {
        assert!(!PrefilterMode::None.needs_reference_buf());
        assert!(!PrefilterMode::None.is_gpu_internal());
    }

    #[test]
    fn external_needs_buffer_but_not_gpu() {
        assert!(PrefilterMode::External.needs_reference_buf());
        assert!(!PrefilterMode::External.is_gpu_internal());
    }

    #[test]
    fn bilateral_is_gpu_internal() {
        let m = PrefilterMode::Bilateral {
            sigma_s: 3.0,
            sigma_r: 0.02,
        };

        assert!(m.needs_reference_buf());
        assert!(m.is_gpu_internal());
    }

    #[test]
    fn bilateral_radius_truncates_at_two_sigma() {
        assert_eq!(bilateral_radius(0.1), 1);
        assert_eq!(bilateral_radius(1.0), 2);
        assert_eq!(bilateral_radius(3.0), 6);
        assert_eq!(bilateral_radius(3.5), 7);
    }
}
