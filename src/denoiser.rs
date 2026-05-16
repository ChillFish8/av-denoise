use crate::nlmeans::ChannelMode;

#[derive(bon::Builder)]
/// The denoiser configuration options.
pub struct DenoiserOptions {
    /// What channels to apply denoising to the frame.
    pub channel_mode: ChannelMode,
    /// The denoising mode to apply to the video stream.
    pub mode: DenoisingMode,
    /// An optional prefilter to apply the frame to use for the weighting
    /// of the frame, allowing for better detail retention and accuracy.
    pub prefilter: (),
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
/// The denoising mode to apply to the video stream.
pub enum DenoisingMode {
    /// Standard spacial denoising.
    Spacial,
    /// Temporal-aware denoising with a temporal radius specified.
    Temporal(usize),
}

/// The denoiser is the high-level object for taking in a stream
/// of input video frames, and producing an output stream of denoised frames.
pub struct Denoiser {

}