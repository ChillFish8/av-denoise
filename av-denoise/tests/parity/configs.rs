use av_denoise::{
    Algorithm,
    ChannelIntent,
    DenoisingMode,
    Depth,
    FrameLayout,
    MotionCompensationMode,
    MotionSearch,
    Nl4dOptions,
    NlmeansHqOptions,
    NlmeansOptions,
    PlaneOptions,
    Subsampling,
    TuningMode,
};

use super::clips::ClipKind;

pub struct ParityConfig {
    pub name: &'static str,
    pub clip: ClipKind,
    pub options: PlaneOptions,
    pub layout: FrameLayout,
    pub frames: usize,
    pub reseed: Reseed,
    /// Runs through a luma-only `HostDenoiser` pushing one plane.
    pub greyscale: bool,
}

/// Which reseed a config runs after streaming.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Reseed {
    None,
    /// A full window around a mid-clip target.
    Middle,
    /// An explicit window that starts at the clip's first frame.
    ClipStart,
    /// An explicit window that ends at the clip's last frame.
    ClipEnd,
}

const WIDTH: u32 = 160;
const HEIGHT: u32 = 128;

fn layout(subsampling: Subsampling, depth: Depth) -> FrameLayout {
    sized_layout(WIDTH, HEIGHT, subsampling, depth)
}

fn sized_layout(width: u32, height: u32, subsampling: Subsampling, depth: Depth) -> FrameLayout {
    FrameLayout {
        width,
        height,
        subsampling,
        depth,
    }
}

fn options(intent: ChannelIntent, mode: DenoisingMode, algorithm: Algorithm) -> PlaneOptions {
    PlaneOptions {
        accelerators: vec![av_denoise::testing::test_accelerator()],
        device: av_denoise::testing::test_device(),
        intent,
        mode,
        algorithm,
        luma_strength: None,
        chroma_strength: None,
        luma_lambda_ht: None,
        chroma_lambda_ht: None,
        tuning: TuningMode::Off,
        deterministic: false,
    }
}

pub fn all() -> Vec<ParityConfig> {
    let temporal = DenoisingMode::Temporal { radius: 2 };
    let fast = Algorithm::Nlmeans(NlmeansOptions::default());
    let hq = Algorithm::NlmeansHq(NlmeansHqOptions::default());
    let nl4d = Algorithm::Nl4d(Nl4dOptions::default());
    let grain_options = Nl4dOptions {
        grain_export: true,
        ..Nl4dOptions::default()
    };
    let nl4d_grain = Algorithm::Nl4d(grain_options);
    let yuv420_8 = layout(Subsampling::Yuv420, Depth::Eight);
    let yuv420_10 = layout(Subsampling::Yuv420, Depth::Ten);
    let yuv420_12 = layout(Subsampling::Yuv420, Depth::Twelve);
    let yuv422_10 = layout(Subsampling::Yuv422, Depth::Ten);
    let yuv422_12 = layout(Subsampling::Yuv422, Depth::Twelve);
    let yuv444_10 = layout(Subsampling::Yuv444, Depth::Ten);
    let yuv444_12 = layout(Subsampling::Yuv444, Depth::Twelve);
    let odd_420_8 = sized_layout(161, 127, Subsampling::Yuv420, Depth::Eight);
    let wide = DenoisingMode::Temporal { radius: 3 };
    let motion_search = MotionSearch::default();
    let compensated = NlmeansOptions {
        motion_compensation: MotionCompensationMode::from(motion_search),
        ..NlmeansOptions::default()
    };
    let hq_compensated_options = NlmeansHqOptions {
        nlm: compensated,
        ..NlmeansHqOptions::default()
    };
    let hq_compensated = Algorithm::NlmeansHq(hq_compensated_options);
    let nlm_fast_spatial_8 = config(
        "nlm_fast_spatial_8",
        ChannelIntent::LumaChroma,
        DenoisingMode::Spacial,
        fast,
        yuv420_8,
        6,
        false,
    );
    let nlm_fast_temporal_8 = config(
        "nlm_fast_temporal_8",
        ChannelIntent::LumaChroma,
        temporal,
        fast,
        yuv420_8,
        10,
        false,
    );
    let nlm_hq_temporal_10 = config(
        "nlm_hq_temporal_10",
        ChannelIntent::LumaChroma,
        temporal,
        hq,
        yuv420_10,
        10,
        false,
    );
    let nlm_hq_short_8 = config(
        "nlm_hq_short_8",
        ChannelIntent::Luma,
        temporal,
        hq,
        yuv420_8,
        2,
        false,
    );
    let nl4d_luma_8 = config(
        "nl4d_luma_8",
        ChannelIntent::Luma,
        temporal,
        nl4d,
        yuv420_8,
        12,
        false,
    );
    let nl4d_chroma_10 = config(
        "nl4d_chroma_10",
        ChannelIntent::Chroma,
        temporal,
        nl4d,
        yuv420_10,
        12,
        false,
    );
    let nl4d_lumachroma_10 = config(
        "nl4d_lumachroma_10",
        ChannelIntent::LumaChroma,
        temporal,
        nl4d,
        yuv420_10,
        12,
        false,
    );
    let nl4d_short_8 = config(
        "nl4d_short_8",
        ChannelIntent::Luma,
        temporal,
        nl4d,
        yuv420_8,
        3,
        false,
    );
    let nl4d_grain_8 = config(
        "nl4d_grain_8",
        ChannelIntent::Luma,
        temporal,
        nl4d_grain,
        yuv420_8,
        12,
        false,
    );
    let nlm_hq_reseed_8 = config(
        "nlm_hq_reseed_8",
        ChannelIntent::LumaChroma,
        temporal,
        hq,
        yuv420_8,
        12,
        true,
    );
    let nlm_hq_mc_8 = config(
        "nlm_hq_mc_8",
        ChannelIntent::LumaChroma,
        temporal,
        hq_compensated,
        yuv420_8,
        12,
        false,
    );
    let nlm_hq_chained_8 = config(
        "nlm_hq_chained_8",
        ChannelIntent::Luma,
        wide,
        hq_compensated,
        yuv420_8,
        12,
        false,
    );
    let nl4d_chained_8 = config(
        "nl4d_chained_8",
        ChannelIntent::Luma,
        wide,
        nl4d,
        yuv420_8,
        14,
        false,
    );
    let nl4d_reseed_8 = config(
        "nl4d_reseed_8",
        ChannelIntent::LumaChroma,
        temporal,
        nl4d,
        yuv420_8,
        12,
        true,
    );

    let mut configs = vec![
        nlm_fast_spatial_8,
        nlm_fast_temporal_8,
        nlm_hq_temporal_10,
        nlm_hq_short_8,
        nl4d_luma_8,
        nl4d_chroma_10,
        nl4d_lumachroma_10,
        nl4d_short_8,
        nl4d_grain_8,
        nlm_hq_reseed_8,
        nlm_hq_mc_8,
        nlm_hq_chained_8,
        nl4d_chained_8,
        nl4d_reseed_8,
    ];

    let lumachroma = ChannelIntent::LumaChroma;
    let luma = ChannelIntent::Luma;
    let chroma = ChannelIntent::Chroma;

    // Channel intents on the nlmeans hq path.
    configs.push(config("nlm_hq_luma_8", luma, temporal, hq, yuv420_8, 12, false));
    configs.push(config(
        "nlm_hq_chroma_8",
        chroma,
        temporal,
        hq,
        yuv420_8,
        12,
        false,
    ));

    // Subsamplings and depths.
    configs.push(config(
        "nl4d_lumachroma_12",
        lumachroma,
        temporal,
        nl4d,
        yuv420_12,
        12,
        false,
    ));
    configs.push(config(
        "nlm_hq_lumachroma_12",
        lumachroma,
        temporal,
        hq,
        yuv420_12,
        12,
        false,
    ));
    configs.push(config(
        "nl4d_lumachroma_422_10",
        lumachroma,
        temporal,
        nl4d,
        yuv422_10,
        12,
        false,
    ));
    configs.push(config(
        "nl4d_lumachroma_422_12",
        lumachroma,
        temporal,
        nl4d,
        yuv422_12,
        12,
        false,
    ));
    configs.push(config(
        "nl4d_lumachroma_444_10",
        lumachroma,
        temporal,
        nl4d,
        yuv444_10,
        12,
        false,
    ));
    configs.push(config(
        "nl4d_lumachroma_444_12",
        lumachroma,
        temporal,
        nl4d,
        yuv444_12,
        12,
        false,
    ));
    configs.push(config(
        "nlm_hq_lumachroma_422_12",
        lumachroma,
        temporal,
        hq,
        yuv422_12,
        12,
        false,
    ));
    configs.push(config(
        "nlm_hq_lumachroma_444_10",
        lumachroma,
        temporal,
        hq,
        yuv444_10,
        12,
        false,
    ));

    // Deterministic kernel selection.
    let mut deterministic = config(
        "nl4d_deterministic_8",
        lumachroma,
        temporal,
        nl4d,
        yuv420_8,
        12,
        false,
    );
    deterministic.options.deterministic = true;
    configs.push(deterministic);

    // Odd dimensions.
    configs.push(config(
        "nl4d_odd_8",
        lumachroma,
        temporal,
        nl4d,
        odd_420_8,
        12,
        false,
    ));
    configs.push(config(
        "nlm_hq_odd_8",
        lumachroma,
        temporal,
        hq,
        odd_420_8,
        12,
        false,
    ));

    // Reseed for single-channel intents.
    configs.push(reseeding("nlm_hq_reseed_luma_8", luma, hq, Reseed::Middle));
    configs.push(reseeding("nlm_hq_reseed_chroma_8", chroma, hq, Reseed::Middle));
    configs.push(reseeding(
        "nl4d_reseed_luma_start_8",
        luma,
        nl4d,
        Reseed::ClipStart,
    ));
    configs.push(reseeding("nl4d_reseed_luma_end_8", luma, nl4d, Reseed::ClipEnd));
    configs.push(reseeding(
        "nl4d_reseed_chroma_start_8",
        chroma,
        nl4d,
        Reseed::ClipStart,
    ));
    configs.push(reseeding(
        "nl4d_reseed_chroma_end_8",
        chroma,
        nl4d,
        Reseed::ClipEnd,
    ));
    configs.push(reseeding(
        "nl4d_reseed_lumachroma_start_8",
        lumachroma,
        nl4d,
        Reseed::ClipStart,
    ));
    configs.push(reseeding(
        "nl4d_reseed_lumachroma_end_8",
        lumachroma,
        nl4d,
        Reseed::ClipEnd,
    ));

    // Greyscale through the host denoiser.
    let mut grey_nl4d = config("nl4d_greyscale_8", luma, temporal, nl4d, yuv420_8, 12, false);
    grey_nl4d.greyscale = true;
    configs.push(grey_nl4d);
    let mut grey_hq = config("nlm_hq_greyscale_8", luma, temporal, hq, yuv420_8, 12, false);
    grey_hq.greyscale = true;
    configs.push(grey_hq);

    configs
}

fn reseeding(
    name: &'static str,
    intent: ChannelIntent,
    algorithm: Algorithm,
    reseed: Reseed,
) -> ParityConfig {
    let temporal = DenoisingMode::Temporal { radius: 2 };
    let yuv420_8 = layout(Subsampling::Yuv420, Depth::Eight);
    let mut reseeding = config(name, intent, temporal, algorithm, yuv420_8, 12, false);
    reseeding.reseed = reseed;
    reseeding
}

fn config(
    name: &'static str,
    intent: ChannelIntent,
    mode: DenoisingMode,
    algorithm: Algorithm,
    layout: FrameLayout,
    frames: usize,
    reseed: bool,
) -> ParityConfig {
    let reseed = if reseed { Reseed::Middle } else { Reseed::None };
    let plane_options = options(intent, mode, algorithm);

    ParityConfig {
        name,
        clip: ClipKind::Moving,
        options: plane_options,
        layout,
        frames,
        reseed,
        greyscale: false,
    }
}
