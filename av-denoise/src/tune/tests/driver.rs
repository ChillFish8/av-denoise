use std::path::Path;
use std::time::{Duration, Instant};

use av_denoise_core::DenoisingMode;
use av_denoise_core::tune::{
    Budget,
    BudgetRequest,
    KeyValue,
    PROFILE_MIN_BLOCKS,
    PipelineProfile,
    SiteKey,
    TuneLedger,
    choose_profile,
    profile_wins,
};

use crate::accelerate::Accelerator;
use crate::planar::{
    ChannelIntent,
    FrameLayout,
    PlanarDenoiser,
    PlaneOptions,
    Planes,
    Subsampling,
    plane_profile_keys,
};
use crate::tune::driver::{FrameCycle, alternative_profiles_for, tune_keys};
use crate::tune::setup::device_priority;
use crate::tune::{TuneOutcome, TuneRequest, TuneStore, TuningMode, is_tuned, selected_device, tune_planes};
use crate::{Algorithm, Depth, Device, Nl4dOptions, install_compilation_cache_at};

const WIDTH: u32 = 64;
const HEIGHT: u32 = 48;

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
                    let ramp = (index % WIDTH) * 3;
                    let grain = index.wrapping_mul(2_654_435_761).wrapping_add(frame * 97) >> 28;
                    let value = 40 + ramp + grain;
                    value.min(255) as u8
                })
                .collect();
            let chroma: Vec<u8> = (0..chroma_pixels)
                .map(|index| {
                    let grain = index.wrapping_mul(40_503).wrapping_add(frame) >> 29;
                    (120 + grain) as u8
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

fn options(algorithm: Algorithm, tuning: TuningMode) -> PlaneOptions {
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
        deterministic: false,
    }
}

fn full_request() -> TuneRequest {
    TuneRequest {
        budget: BudgetRequest::Full,
        force: false,
    }
}

/// A budget, and so a compile ceiling, long enough for a cold nl4d pass to compile every candidate.
fn cold_nl4d_request() -> TuneRequest {
    TuneRequest {
        budget: BudgetRequest::Seconds(1200.0),
        force: false,
    }
}

/// How many launch sites the store holds for the device `plane_options` selects.
fn stored_site_count(plane_options: &PlaneOptions) -> usize {
    let device = selected_device(&plane_options.accelerators, &plane_options.device).expect("a device");
    let store = TuneStore::open_default().expect("the cache is installed");
    let stored = store.load(&device.identity);

    stored.sites.len()
}

fn install_cache(dir: &Path) {
    install_compilation_cache_at(dir).expect("the cache installs");
}

#[test]
fn a_full_pass_tunes_and_saves_and_a_second_finds_it_done() {
    let cache = tempfile::tempdir().unwrap();
    install_cache(cache.path());
    let algorithm = Algorithm::default();
    let plane_options = options(algorithm, TuningMode::Auto);
    let frame_layout = layout();
    let clip = frames(12);
    let request = full_request();

    assert!(!is_tuned(&plane_options, frame_layout));

    let first = tune_planes(&plane_options, frame_layout, &clip, &request);
    let second = tune_planes(&plane_options, frame_layout, &clip, &request);

    assert_eq!(first.outcome, TuneOutcome::Tuned, "{first:?}");
    assert!(first.tuned_keys >= 1, "{first:?}");
    assert_eq!(first.tuned_keys, stored_site_count(&plane_options), "{first:?}");
    assert!(first.saved);
    assert_eq!(second.outcome, TuneOutcome::AlreadyTuned, "{second:?}");
    assert_eq!(second.tuned_keys, 0);
    assert!(first.compile > Duration::ZERO, "{first:?}");
    assert!(first.compile < first.elapsed, "{first:?}");
    assert_eq!(second.compile, Duration::ZERO);
    assert!(is_tuned(&plane_options, frame_layout));

    let store_dir = cache.path().join(crate::KERNEL_HASH).join("tune");
    let entries = std::fs::read_dir(store_dir).unwrap();
    let files: Vec<_> = entries.flatten().collect();
    assert_eq!(files.len(), 1, "one store file for one device");
}

#[test]
fn forcing_tunes_again() {
    let cache = tempfile::tempdir().unwrap();
    install_cache(cache.path());
    let algorithm = Algorithm::default();
    let plane_options = options(algorithm, TuningMode::Auto);
    let frame_layout = layout();
    let clip = frames(12);
    let request = full_request();
    let forced = TuneRequest {
        force: true,
        ..request
    };

    tune_planes(&plane_options, frame_layout, &clip, &request);
    let again = tune_planes(&plane_options, frame_layout, &clip, &forced);

    assert_eq!(again.outcome, TuneOutcome::Tuned);
    assert!(again.tuned_keys >= 1);
}

#[test]
fn a_clip_too_short_to_earn_back_tuning_is_skipped() {
    let cache = tempfile::tempdir().unwrap();
    install_cache(cache.path());
    let algorithm = Algorithm::default();
    let plane_options = options(algorithm, TuningMode::Auto);
    let frame_layout = layout();
    let clip = frames(12);
    let request = TuneRequest {
        budget: BudgetRequest::Automatic { frames: Some(12) },
        force: false,
    };

    let report = tune_planes(&plane_options, frame_layout, &clip, &request);

    assert!(matches!(report.outcome, TuneOutcome::Skipped(_)), "{report:?}");
    assert!(!report.saved);
    assert!(!is_tuned(&plane_options, frame_layout));
}

/// The first pass compiles the kernels, so the second pass is mostly warm-up.
#[test]
fn a_tiny_budget_ends_the_warm_up_early() {
    let cache = tempfile::tempdir().unwrap();
    install_cache(cache.path());
    let algorithm = Algorithm::default();
    let plane_options = options(algorithm, TuningMode::Auto);
    let frame_layout = layout();
    let clip = frames(12);
    let budget_seconds = 0.5;
    let request = TuneRequest {
        budget: BudgetRequest::Seconds(budget_seconds),
        force: false,
    };

    tune_planes(&plane_options, frame_layout, &clip, &request);
    let report = tune_planes(&plane_options, frame_layout, &clip, &request);

    let budget = Duration::from_secs_f64(budget_seconds);
    let frame_allowance = Duration::from_secs(1);

    assert!(matches!(report.outcome, TuneOutcome::Skipped(_)), "{report:?}");
    assert!(!report.saved);
    assert!(report.elapsed <= budget + frame_allowance, "{report:?}");
}

#[test]
fn a_denoiser_built_after_tuning_carries_no_ledger() {
    let algorithm = Algorithm::default();
    let tuning = TuningMode::Auto.after_tuning();
    let plane_options = options(algorithm, tuning);
    let frame_layout = layout();

    let planar = PlanarDenoiser::create(&plane_options, frame_layout).expect("the denoiser builds");

    assert!(planar.ledger().is_none());
}

#[test]
fn a_spent_compile_ceiling_stops_before_the_next_key() {
    let algorithm = Algorithm::default();
    let plane_options = options(algorithm, TuningMode::Auto);
    let frame_layout = layout();
    let clip = frames(2);
    let mut planar = PlanarDenoiser::create(&plane_options, frame_layout).expect("the denoiser builds");
    let mut cycle = FrameCycle::new(&clip);
    let ledger = TuneLedger::new(false);
    let budget = Budget {
        total: Duration::from_secs(600),
        full: true,
    };
    let order = [KeyValue {
        key: SiteKey::builder("test", 64, 64).build(),
        per_frame: Duration::from_millis(1),
        candidates: 2,
    }];

    ledger.begin_compile_pass(Duration::ZERO);
    let tuned = tune_keys(&mut planar, &ledger, &mut cycle, &order, &budget, Instant::now());

    assert!(tuned.frame_error.is_none());
    assert_eq!(cycle.next, 0, "a frame was pushed for the key");
}

#[test]
fn tuning_off_never_tunes() {
    let algorithm = Algorithm::default();
    let plane_options = options(algorithm, TuningMode::Off);
    let frame_layout = layout();
    let clip = frames(4);
    let request = full_request();

    let report = tune_planes(&plane_options, frame_layout, &clip, &request);

    assert!(matches!(report.outcome, TuneOutcome::Skipped(_)));
}

/// Pushes `clip` through `planar` and returns every frame it emits, in order.
fn denoise_clip(planar: &mut PlanarDenoiser, clip: &[Planes]) -> Vec<Planes> {
    let mut emitted = Vec::new();

    for frame in clip {
        let pushed = planar.push(frame);
        if crate::push_needs_retry(pushed).expect("the push succeeds") {
            let received = planar.recv().expect("a frame is received");
            emitted.extend(received);
            planar.push(frame).expect("the retried push succeeds");
        }
    }

    planar
        .flush(|planes| emitted.push(planes))
        .expect("the flush succeeds");
    emitted
}

/// The fresh denoiser builds from the result the pass remembered, so both run the same plan.
#[test]
fn a_tuned_denoiser_matches_a_fresh_one_built_from_its_result() {
    let cache = tempfile::tempdir().unwrap();
    install_cache(cache.path());
    let algorithm = Algorithm::default();
    let plane_options = options(algorithm, TuningMode::Auto);
    let frame_layout = layout();
    let clip = frames(12);
    let request = full_request();
    let mut planar = PlanarDenoiser::create(&plane_options, frame_layout).expect("the denoiser builds");

    let report = planar.tune(&clip, &request);
    assert_eq!(report.outcome, TuneOutcome::Tuned, "{report:?}");

    let cached_algorithm = Algorithm::default();
    let cached_options = options(cached_algorithm, TuningMode::Cached);
    let mut fresh = PlanarDenoiser::create(&cached_options, frame_layout).expect("the fresh denoiser builds");
    let tuned_output = denoise_clip(&mut planar, &clip);
    let fresh_output = denoise_clip(&mut fresh, &clip);

    assert_eq!(tuned_output.len(), clip.len());
    assert_eq!(fresh_output.len(), clip.len());

    let frame_pairs = tuned_output.iter().zip(&fresh_output);
    for (index, (tuned_frame, fresh_frame)) in frame_pairs.enumerate() {
        assert_eq!(tuned_frame.y, fresh_frame.y, "luma of frame {index}");
        assert_eq!(tuned_frame.u, fresh_frame.u, "u of frame {index}");
        assert_eq!(tuned_frame.v, fresh_frame.v, "v of frame {index}");
    }
}

#[test]
fn a_tuned_grain_export_denoiser_drains_only_the_real_streams_grain() {
    let cache = tempfile::tempdir().unwrap();
    install_cache(cache.path());
    let nl4d = Nl4dOptions {
        grain_export: true,
        ..Nl4dOptions::default()
    };
    let plane_options = options(Algorithm::Nl4d(nl4d), TuningMode::Auto);
    let frame_layout = layout();
    let clip = frames(13);
    let request = full_request();
    let mut planar = PlanarDenoiser::create(&plane_options, frame_layout).expect("the denoiser builds");
    let mut fresh = PlanarDenoiser::create(&plane_options, frame_layout).expect("the fresh denoiser builds");

    planar.tune(&clip, &request);
    denoise_clip(&mut planar, &clip);
    denoise_clip(&mut fresh, &clip);
    let tuned_chunks = planar.drain_grain_chunks().expect("the tuned grain drains");
    let fresh_chunks = fresh.drain_grain_chunks().expect("the fresh grain drains");

    let tuned_frames: u32 = tuned_chunks.iter().map(|chunk| chunk.frames).sum();
    let fresh_frames: u32 = fresh_chunks.iter().map(|chunk| chunk.frames).sum();
    assert!(fresh_frames > 0, "the clip measures some grain");
    assert_eq!(tuned_chunks.len(), fresh_chunks.len());
    assert_eq!(tuned_frames, fresh_frames);
}

#[test]
fn an_nl4d_pass_records_a_ranking_for_every_plane() {
    let cache = tempfile::tempdir().unwrap();
    install_cache(cache.path());
    let nl4d = Nl4dOptions::default();
    let plane_options = options(Algorithm::Nl4d(nl4d), TuningMode::Auto);
    let frame_layout = layout();
    let clip = frames(13);
    let request = cold_nl4d_request();

    let report = tune_planes(&plane_options, frame_layout, &clip, &request);

    assert_eq!(report.outcome, TuneOutcome::Tuned, "{report:?}");
    assert_eq!(report.partial_keys, 0, "{report:?}");
    assert_eq!(report.tuned_keys, stored_site_count(&plane_options), "{report:?}");
    assert!(is_tuned(&plane_options, frame_layout), "{report:?}");

    let device = selected_device(&plane_options.accelerators, &plane_options.device).expect("a device");
    let store = TuneStore::open_default().expect("the cache is installed");
    let stored = store.load(&device.identity);
    let table = device_priority(&device);
    let alternatives = alternative_profiles_for(true, &device, table);
    let profile_keys = plane_profile_keys(&plane_options, frame_layout);

    for profile_key in &profile_keys {
        let ranking = stored
            .profiles
            .get(profile_key)
            .expect("every plane stores its profiles");
        let records = &ranking.profiles;
        let baseline = &records[0];
        let compared: Vec<PipelineProfile> = records[1..].iter().map(|record| record.profile).collect();
        let pooled_blocks = PROFILE_MIN_BLOCKS * compared.len();
        let reached = &alternatives[..compared.len()];

        assert_eq!(
            baseline.profile,
            PipelineProfile::Baseline,
            "{profile_key}: {records:?}"
        );
        assert_eq!(
            baseline.blocks as usize, pooled_blocks,
            "{profile_key}: {records:?}"
        );
        assert!(!compared.is_empty(), "{profile_key}: {records:?}");
        assert_eq!(compared, reached, "{profile_key}: {records:?}");

        let fastest = records[1..]
            .iter()
            .min_by_key(|record| record.median_frame_ns)
            .expect("an alternative was compared");
        let baseline_median = Duration::from_nanos(baseline.median_frame_ns);
        let fastest_median = Duration::from_nanos(fastest.median_frame_ns);
        let kept = if profile_wins(baseline_median, fastest_median) {
            fastest.profile
        } else {
            PipelineProfile::Baseline
        };
        let chosen = choose_profile(ranking, false);

        assert_eq!(chosen, kept, "{profile_key}: {records:?}");
    }
}
