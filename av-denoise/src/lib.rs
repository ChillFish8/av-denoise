#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, doc(auto_cfg))]
#![doc = include_str!("../README.md")]

mod backend;
pub mod cache;
mod host;
mod planar;
pub mod stack;
#[doc(hidden)]
pub mod testing;
pub mod tune;
pub mod warmup;

#[cfg(feature = "avds")]
pub use av_denoise_core::avds;
pub use av_denoise_core::{
    ChromaLayout,
    ChromaSiting,
    DEFAULT_FINE_NOISE,
    DEFAULT_PILOT_STRENGTH_SCALE,
    DenoisingMode,
    DevicePlane,
    EdgePadding,
    Error as EngineError,
    FitOptions,
    FrameGrain,
    GrainChunk,
    GrainChunks,
    HqParams,
    KERNEL_HASH,
    MotionCompensationMode,
    MotionEstimation,
    MotionSearch,
    Nl4dOptions,
    NlmTuning,
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
    frame_rows,
    nlmeans_search_radius_for,
    nlmeans_temporal_radius_for,
    nlmeans_variant_for,
    parse_prefilter,
    search_escalation_for,
};

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
    ChannelIntent,
    DenoiserError,
    DenoiserOptions,
    Depth,
    HostDenoiser,
    MAX_PENDING,
    PlanarTiming,
    PlaneTiming,
    UnsupportedDepthError,
};
pub use self::planar::{
    FrameLayout,
    PlanarDenoiser,
    PlaneOptions,
    Planes,
    ReseedWindow,
    Subsampling,
    fill_plane,
    push_needs_retry,
};
pub use self::stack::{CODEGEN_STACK_BYTES, codegen_stack_is_sufficient, raise_codegen_stack_limit};
pub use self::tune::{TUNE_ENV, TuningMode};
pub use self::warmup::{WarmUp, kernel_key};
