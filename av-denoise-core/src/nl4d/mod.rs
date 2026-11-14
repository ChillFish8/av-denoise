//! Collaborative denoising across a window of frames
//!
//! Similar 8x8 patches are grouped and filtered together. Each group holds a few patches from the
//! centre frame, each followed through its neighbour frames along its motion vector. Grain differs
//! from frame to frame while texture stays, so a transform along time separates the two.
//!
//! - [Nl4d], the engine
//! - [Nl4dOptions] and the per-preset defaults it is built from
//! - per-frame film grain measurements for a grain data file and AV1 grain tables

mod collab_pass;
pub(crate) mod denoiser;
mod engine;
pub mod grain;
pub(crate) mod harness;
pub(crate) mod kernels;
mod options;
pub(crate) mod params;
pub(crate) mod snapshot;

// The tests run against a real GPU runtime, so they need a wgpu-backed feature.
#[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
pub(crate) mod tests;

#[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
pub(crate) use self::denoiser::Nl4dDenoiser;
pub(crate) use self::engine::Nl4d;
pub use self::options::Nl4dOptions;
#[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
pub(crate) use self::options::nl4d_default_lambda_ht;
pub(crate) use self::options::nl4d_pool_ratio;
#[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
pub(crate) use self::options::resolve_params;
#[cfg(test)]
pub(crate) use self::params::MAX_KAISER_BETA;
pub(crate) use self::params::Nl4dParams;
pub(crate) use self::snapshot::MotionSnapshot;
