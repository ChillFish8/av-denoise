use av_denoise::accelerate::Accelerator;
use av_denoise::{
    Algorithm,
    ChannelIntent,
    DenoisingMode,
    Depth,
    Device,
    FrameLayout,
    Nl4dOptions,
    NlmeansHqOptions,
    NlmeansOptions,
    PlaneOptions,
    Subsampling,
};

pub struct ParityConfig {
    pub name: &'static str,
    pub options: PlaneOptions,
    pub layout: FrameLayout,
    pub frames: usize,
    pub reseed: bool,
}

const WIDTH: u32 = 160;
const HEIGHT: u32 = 128;

fn layout(subsampling: Subsampling, depth: Depth) -> FrameLayout {
    FrameLayout {
        width: WIDTH,
        height: HEIGHT,
        subsampling,
        depth,
    }
}

fn options(intent: ChannelIntent, mode: DenoisingMode, algorithm: Algorithm) -> PlaneOptions {
    PlaneOptions {
        accelerators: vec![Accelerator::Vulkan],
        device: Device::Default,
        intent,
        mode,
        algorithm,
        luma_strength: None,
        chroma_strength: None,
        luma_lambda_ht: None,
        chroma_lambda_ht: None,
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
    let yuv444_8 = layout(Subsampling::Yuv444, Depth::Eight);

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
    let nlm_hq_yuv_8 = config(
        "nlm_hq_yuv_8",
        ChannelIntent::YuvFused,
        temporal,
        hq,
        yuv444_8,
        10,
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
    let nl4d_yuv_8 = config(
        "nl4d_yuv_8",
        ChannelIntent::YuvFused,
        temporal,
        nl4d,
        yuv444_8,
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

    vec![
        nlm_fast_spatial_8,
        nlm_fast_temporal_8,
        nlm_hq_temporal_10,
        nlm_hq_short_8,
        nlm_hq_yuv_8,
        nl4d_luma_8,
        nl4d_chroma_10,
        nl4d_lumachroma_10,
        nl4d_yuv_8,
        nl4d_short_8,
        nl4d_grain_8,
        nlm_hq_reseed_8,
    ]
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
    let plane_options = options(intent, mode, algorithm);

    ParityConfig {
        name,
        options: plane_options,
        layout,
        frames,
        reseed,
    }
}
