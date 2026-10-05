//! The hardware backends kernels can run on

use strum_macros::{Display, EnumIter, EnumString, IntoStaticStr};

/// A hardware backend that kernels can run on.
///
/// It names a backend rather than a specific GPU, which is chosen separately with
/// [Device](crate::Device). Only the backends whose crate feature is enabled exist.
///
/// Every accelerator runs kernels on a GPU. There is no software backend, because the collaborative
/// filter aggregates through atomic floating-point adds and cubecl's CPU runtime does not implement
/// atomics. [Device::Cpu](crate::Device::Cpu) still selects a software device where the platform
/// offers one, such as lavapipe under Vulkan.
#[derive(Debug, Copy, Clone, Eq, PartialEq, IntoStaticStr, EnumString, EnumIter, Display)]
#[strum(serialize_all = "snake_case")]
pub enum Accelerator {
    #[cfg(any(feature = "cuda", docsrs))]
    #[cfg_attr(docsrs, doc(cfg(feature = "cuda")))]
    /// Runs kernels through the Nvidia CUDA backend, on Nvidia GPUs only.
    Cuda,
    #[cfg(any(feature = "vulkan", docsrs))]
    #[cfg_attr(docsrs, doc(cfg(feature = "vulkan")))]
    /// Runs kernels through the wgpu Vulkan backend.
    ///
    /// This is the lightest and most portable option, because it works on any platform and GPU that
    /// supports basic compute shaders.
    Vulkan,
    #[cfg(any(feature = "metal", docsrs))]
    #[cfg_attr(docsrs, doc(cfg(feature = "metal")))]
    /// Runs kernels through the wgpu Metal backend.
    ///
    /// This is the only option on Apple Silicon.
    Metal,
    #[cfg(any(feature = "rocm", docsrs))]
    #[cfg_attr(docsrs, doc(cfg(feature = "rocm")))]
    /// Runs kernels through the AMD ROCm backend, on AMD GPUs only.
    ///
    /// ROCm is not the recommended backend for AMD GPUs. It is slower and often hit by driver issues,
    /// so Vulkan is almost always faster and less buggy.
    Rocm,
}

/// Returns every accelerator this build enables, in the order to try them.
///
/// ```no_run
/// use av_denoise::accelerate::get_default_accelerators;
///
/// let preferred = get_default_accelerators();
/// # let _ = preferred;
/// ```
pub fn get_default_accelerators() -> Vec<Accelerator> {
    use strum::IntoEnumIterator;

    let mut accelerators = Vec::new();
    for enabled in Accelerator::iter() {
        accelerators.push(enabled);
    }

    accelerators
}
