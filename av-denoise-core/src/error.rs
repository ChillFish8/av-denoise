/// Errors returned by the denoising engines.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid options, {0}")]
    InvalidOptions(String),
    #[error("invalid geometry, {0}")]
    InvalidGeometry(String),
    #[error("plane mismatch, {0}")]
    PlaneMismatch(String),
    #[error("every ready frame must be emitted before the next push")]
    OutputsPending,
    #[error("no frame is ready to emit")]
    NothingToEmit,
    #[error("context frames must come before the first push of a stream")]
    ContextAfterPush,
    #[error("an earlier call failed, reset the engine before using it again")]
    NeedsReset,
    #[error(transparent)]
    Gpu(#[from] anyhow::Error),
}
