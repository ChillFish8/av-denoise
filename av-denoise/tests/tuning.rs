#![cfg(feature = "vulkan")]

use av_denoise::accelerate::Accelerator;
use av_denoise::tune::{
    BudgetRequest,
    TuneOutcome,
    TuneRequest,
    TuneStore,
    is_tuned,
    selected_device,
    tune_planes,
};
use av_denoise::{
    Algorithm,
    ChannelIntent,
    DenoisingMode,
    Depth,
    Device,
    FrameLayout,
    Nl4dOptions,
    PlanarDenoiser,
    PlaneOptions,
    Planes,
    Subsampling,
    TuningMode,
    install_compilation_cache_at,
    push_needs_retry,
};

const WIDTH: u32 = 96;
const HEIGHT: u32 = 64;

fn layout() -> FrameLayout {
    FrameLayout {
        width: WIDTH,
        height: HEIGHT,
        subsampling: Subsampling::Yuv420,
        depth: Depth::Eight,
    }
}

/// `count` frames of a ramp plus per-frame hashed grain.
fn frames(count: usize) -> Vec<Planes> {
    let layout = layout();
    let luma_pixels = layout.luma_pixels() as u32;
    let chroma_pixels = layout.chroma_pixels() as u32;

    (0..count as u32)
        .map(|frame| {
            let luma: Vec<u8> = (0..luma_pixels)
                .map(|index| {
                    let ramp = (index % WIDTH) * 2;
                    let grain = index.wrapping_mul(2_654_435_761).wrapping_add(frame * 131) >> 27;
                    let value = 30 + ramp + grain;
                    value.min(255) as u8
                })
                .collect();
            let chroma: Vec<u8> = (0..chroma_pixels)
                .map(|index| {
                    let grain = index.wrapping_mul(40_503).wrapping_add(frame * 7) >> 28;
                    (118 + grain) as u8
                })
                .collect();

            Planes {
                y: luma,
                u: chroma.clone(),
                v: chroma,
            }
        })
        .collect()
}

fn options(algorithm: Algorithm, tuning: TuningMode, deterministic: bool) -> PlaneOptions {
    PlaneOptions {
        accelerators: vec![Accelerator::Vulkan],
        device: Device::Default,
        intent: ChannelIntent::LumaChroma,
        mode: DenoisingMode::Temporal { radius: 1 },
        algorithm,
        luma_strength: None,
        chroma_strength: None,
        luma_lambda_ht: None,
        chroma_lambda_ht: None,
        tuning,
        deterministic,
    }
}

fn denoise(plane_options: &PlaneOptions, clip: &[Planes]) -> Vec<Planes> {
    let frame_layout = layout();
    let mut planar = PlanarDenoiser::create(plane_options, frame_layout).expect("the denoiser builds");
    let mut outputs = Vec::new();

    for frame in clip {
        let pushed = planar.push(frame);
        let needs_retry = push_needs_retry(pushed).expect("the push succeeds");
        if needs_retry {
            let received = planar.recv().expect("the receive succeeds");
            outputs.extend(received);
            planar.push(frame).expect("the retried push succeeds");
        }
    }

    planar
        .flush(|frame| outputs.push(frame))
        .expect("the flush succeeds");
    outputs
}

fn planes_equal(left: &[Planes], right: &[Planes]) -> bool {
    left.len() == right.len()
        && left.iter().zip(right).all(|(left_frame, right_frame)| {
            left_frame.y == right_frame.y && left_frame.u == right_frame.u && left_frame.v == right_frame.v
        })
}

#[test]
fn short_tunes_store_results_until_a_later_run_loads_them() {
    let cache = tempfile::tempdir().expect("a temporary directory");
    install_compilation_cache_at(cache.path()).expect("the cache installs");

    let algorithm = Algorithm::Nl4d(Nl4dOptions::default());
    let plane_options = options(algorithm, TuningMode::Auto, false);
    let frame_layout = layout();
    let clip = frames(13);
    let request = TuneRequest {
        budget: BudgetRequest::Seconds(90.0),
        force: false,
    };

    let first = tune_planes(&plane_options, frame_layout, &clip, &request);
    assert_eq!(first.outcome, TuneOutcome::Tuned, "{first:?}");
    assert!(first.saved);

    let device = selected_device(&plane_options.accelerators, &plane_options.device).expect("a device");
    let store = TuneStore::open_default().expect("caching is on");
    let stored = store.load(&device.identity);
    let mut settled = stored.sites.len() + stored.failed.len();
    assert!(settled > 0);

    // A short budget can leave keys that later runs tune before the result is complete.
    let mut later = tune_planes(&plane_options, frame_layout, &clip, &request);
    let mut passes = 2;
    while later.outcome == TuneOutcome::Tuned {
        assert!(later.saved, "pass {passes} saved nothing");

        let stored = store.load(&device.identity);
        let now_settled = stored.sites.len() + stored.failed.len();
        let complete = is_tuned(&plane_options, frame_layout);
        assert!(
            now_settled > settled || complete,
            "pass {passes} left {settled} settled keys untouched"
        );
        // Every pass that leaves the result incomplete settles at least one more key.
        assert!(
            passes <= now_settled + 1,
            "{passes} passes for {now_settled} keys"
        );

        settled = now_settled;
        later = tune_planes(&plane_options, frame_layout, &clip, &request);
        passes += 1;
    }

    assert_eq!(
        later.outcome,
        TuneOutcome::AlreadyTuned,
        "{later:?} after {passes} passes"
    );
    assert!(is_tuned(&plane_options, frame_layout));
}

#[test]
fn deterministic_output_equals_baseline_output() {
    let cache = tempfile::tempdir().expect("a temporary directory");
    install_compilation_cache_at(cache.path()).expect("the cache installs");

    let algorithm = Algorithm::Nl4d(Nl4dOptions::default());
    let frame_layout = layout();
    let clip = frames(13);
    let tuned_options = options(algorithm, TuningMode::Auto, false);
    let request = TuneRequest {
        budget: BudgetRequest::Full,
        force: false,
    };
    let report = tune_planes(&tuned_options, frame_layout, &clip, &request);
    assert_eq!(report.outcome, TuneOutcome::Tuned, "{report:?}");

    let deterministic_options = options(algorithm, TuningMode::Cached, true);
    let baseline_options = options(algorithm, TuningMode::Off, false);
    let deterministic = denoise(&deterministic_options, &clip);
    let baseline = denoise(&baseline_options, &clip);

    assert_eq!(deterministic.len(), clip.len());
    assert!(planes_equal(&deterministic, &baseline));
}

#[test]
fn tuning_off_output_is_unchanged_by_a_stored_result() {
    let cache = tempfile::tempdir().expect("a temporary directory");
    install_compilation_cache_at(cache.path()).expect("the cache installs");

    let frame_layout = layout();
    let clip = frames(13);
    let tuned_options = options(Algorithm::default(), TuningMode::Auto, false);
    let request = TuneRequest {
        budget: BudgetRequest::Full,
        force: false,
    };
    tune_planes(&tuned_options, frame_layout, &clip, &request);
    assert!(is_tuned(&tuned_options, frame_layout));

    let cached_options = options(Algorithm::default(), TuningMode::Cached, false);
    let off_options = options(Algorithm::default(), TuningMode::Off, false);
    let cached = denoise(&cached_options, &clip);
    let off = denoise(&off_options, &clip);

    // NLMeans has only bit-identical candidates, so a tuned plan cannot change its output.
    assert_eq!(cached.len(), clip.len());
    assert!(planes_equal(&cached, &off));
}
