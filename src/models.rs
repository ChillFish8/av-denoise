use strum_macros::{Display, EnumIter, EnumString, IntoStaticStr};

#[derive(Debug, Copy, Clone, Eq, PartialEq, IntoStaticStr, EnumString, EnumIter, Display)]
#[strum(serialize_all = "snake_case")]
/// A hardware accelerator that can be used to compute any target metrics.
pub enum Accelerator {
    #[cfg(feature = "cuda")]
    /// Run kernels using the Nvidia CUDA backend.
    ///
    /// Nvidia GPUs only (duh.)
    Cuda,
    #[cfg(feature = "rocm")]
    /// Run kernels using the AMD ROCm backend.
    ///
    /// AMD GPUs only (duh.)
    Rocm,
    #[cfg(feature = "vulkan")]
    /// Run kernels using the WGPU Vulkan backend.
    ///
    /// This is the most lightweight and portable accelerator
    /// because it supports all platforms and GPUs that support basic
    /// compute shaders.
    Vulkan,
    #[cfg(feature = "metal")]
    /// Run kernels using the WGPU Metal backend.
    ///
    /// This is the only accelerator available for Apple Silicon.
    Metal,
    #[cfg(feature = "cpu")]
    /// Run kernels using the CPU JIT compiler.
    Cpu,
}

/// Returns the enabled, default accelerators, in order of what to attempt.
#[allow(dead_code)]
pub fn get_default_accelerators() -> Vec<Accelerator> {
    use strum::IntoEnumIterator;

    let mut accelerator = Vec::new();
    for enabled in Accelerator::iter() {
        accelerator.push(enabled);
    }

    accelerator
}
