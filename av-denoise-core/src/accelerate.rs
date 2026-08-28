//! The hardware backends kernels can run on.
//!
//! An [`Accelerator`] names a backend rather than a specific piece of
//! hardware. Which physical GPU it lands on is chosen separately, with
//! [`crate::Device`].
//!
//! Only the backends whose crate feature is enabled exist at compile
//! time, so a build without the `cuda` feature has no `Accelerator::Cuda`
//! variant at all.
//!
//! [`Denoiser::create`](crate::Denoiser::create) takes a list of these
//! and uses the first one that starts successfully, which lets a program
//! prefer a fast backend and quietly fall back to a slower one.
//!
//! ```no_run
//! use av_denoise_core::accelerate::get_default_accelerators;
//!
//! // Every backend this build supports, in the order to try them.
//! let preferred = get_default_accelerators();
//! # let _ = preferred;
//! ```
//!
//! A list can also be written out by hand, such as
//! `vec![Accelerator::Cuda, Accelerator::Vulkan]` to prefer the vendor
//! backend and fall back to the portable one.
//!
//! Every accelerator here runs kernels on a GPU. There is no software
//! backend, because the collaborative filter aggregates through atomic
//! floating-point adds and cubecl's CPU runtime does not implement
//! atomics. [`crate::Device::Cpu`] still selects a software *device*
//! where the platform offers one, such as lavapipe under Vulkan.

use strum_macros::{Display, EnumIter, EnumString, IntoStaticStr};

#[derive(Debug, Copy, Clone, Eq, PartialEq, IntoStaticStr, EnumString, EnumIter, Display)]
#[strum(serialize_all = "snake_case")]
/// A hardware backend that kernels can run on.
pub enum Accelerator {
    #[cfg(any(feature = "cuda", docsrs))]
    #[cfg_attr(docsrs, doc(cfg(feature = "cuda")))]
    /// Runs kernels through the Nvidia CUDA backend.
    ///
    /// Nvidia GPUs only.
    Cuda,
    #[cfg(any(feature = "rocm", docsrs))]
    #[cfg_attr(docsrs, doc(cfg(feature = "rocm")))]
    /// Runs kernels through the AMD ROCm backend.
    ///
    /// AMD GPUs only.
    Rocm,
    #[cfg(any(feature = "vulkan", docsrs))]
    #[cfg_attr(docsrs, doc(cfg(feature = "vulkan")))]
    /// Runs kernels through the wgpu Vulkan backend.
    ///
    /// This is the lightest and most portable option, because it works
    /// on any platform and GPU that supports basic compute shaders.
    Vulkan,
    #[cfg(any(feature = "metal", docsrs))]
    #[cfg_attr(docsrs, doc(cfg(feature = "metal")))]
    /// Runs kernels through the wgpu Metal backend.
    ///
    /// This is the only option on Apple Silicon.
    Metal,
}

/// Returns every accelerator this build enables, in the order to try
/// them.
pub fn get_default_accelerators() -> Vec<Accelerator> {
    use strum::IntoEnumIterator;

    let mut accelerator = Vec::new();
    for enabled in Accelerator::iter() {
        accelerator.push(enabled);
    }

    accelerator
}
