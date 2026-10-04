#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, doc(auto_cfg))]
#![doc = include_str!("../README.md")]

mod backend;
pub mod cache;
mod host;
mod planar;
pub mod stack;
pub mod warmup;

/// Planar frame types and helpers.
pub mod frame {
    pub use crate::planar::{
        ChannelIntent,
        FrameLayout,
        PlanarDenoiser,
        PlaneOptions,
        Planes,
        ReseedWindow,
        Subsampling,
        f32_to_plane,
        fill_plane,
        interleave_uv_to_f32,
        interleave_yuv_to_f32,
        plane_to_f32,
        push_needs_retry,
        unpack_uv_from_f32,
    };
}

pub use av_denoise_core::{
    ChannelMode,
    DEFAULT_PILOT_STRENGTH_SCALE,
    DenoisingMode,
    DevicePlane,
    EdgePadding,
    Engine,
    Error,
    Geometry,
    GrainChunk,
    HqParams,
    KERNEL_HASH,
    MotionCompensationMode,
    MotionEstimation,
    MotionSearch,
    Nl4d,
    Nl4dOptions,
    NlmTuning,
    Nlmeans,
    NlmeansAlgorithm,
    NlmeansHqOptions,
    NlmeansOptions,
    NlmeansVariant,
    PrefilterMode,
    Preset,
    SampleFormat,
    SceneGrain,
    WindowSpan,
    build_table,
    nl4d_default_lambda_ht,
    nl4d_spatial_radius_for,
    nl4d_temporal_radius_for,
    nlmeans_search_radius_for,
    nlmeans_temporal_radius_for,
    nlmeans_variant_for,
    parse_prefilter,
};
#[doc(hidden)]
pub use av_denoise_core::{nl4d, nlmeans};

pub use self::backend::{Device, accelerate, device, enumerate, sniff};
pub use self::cache::{
    COMPILATION_CACHE_ENV,
    CacheError,
    compilation_cache_dir,
    default_cache_dir,
    install_compilation_cache,
    install_compilation_cache_at,
    install_compilation_cache_once,
};
pub use self::host::{
    Algorithm,
    DenoiserError,
    DenoiserOptions,
    Depth,
    HostDenoiser,
    MAX_PENDING,
    UnsupportedDepthError,
};
pub use self::planar::{
    ChannelIntent,
    FrameLayout,
    PlanarDenoiser,
    PlaneOptions,
    Planes,
    ReseedWindow,
    Subsampling,
    push_needs_retry,
};
pub use self::stack::{CODEGEN_STACK_BYTES, codegen_stack_is_sufficient, raise_codegen_stack_limit};
pub use self::warmup::{WarmUp, kernel_key};
