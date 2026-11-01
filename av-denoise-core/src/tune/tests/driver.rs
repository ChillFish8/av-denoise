use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::engine::{DevicePlane, PassGeometry, PassSpec, SampleFormat};
use crate::nl4d::Nl4dOptions;
use crate::nlmeans::{ChannelMode, DenoisingMode, NlmeansAlgorithm, NlmeansOptions};
use crate::tune::driver::pass::{FrameCycle, tune_keys};
use crate::tune::driver::profiles::alternative_profiles_for;
use crate::tune::driver::{TuneOutcome, TuneReport, TuneRequest, is_tuned, tune_frames, tune_set};
use crate::tune::engine_set::EngineSet;
use crate::tune::setup::{KernelSetup, device_priority};
use crate::tune::{
    Budget,
    BudgetRequest,
    CacheHook,
    CachePhase,
    DeviceLimits,
    DeviceProfile,
    KeyValue,
    MIN_SAMPLES,
    PROFILE_MIN_BLOCKS,
    PipelineProfile,
    SearchMode,
    SiteKey,
    TuneLedger,
    TuneStore,
    TuningConfig,
    TuningFrame,
    TuningMode,
    choose_profile,
    describe,
    profile_wins,
    tuning_frame_count,
};

const WIDTH: u32 = 64;
const HEIGHT: u32 = 48;
const CHROMA_WIDTH: u32 = WIDTH / 2;
const CHROMA_HEIGHT: u32 = HEIGHT / 2;

/// A budget, and so a compile ceiling, long enough for a cold nl4d pass to compile every candidate.
const COLD_NL4D_SECONDS: f64 = 1200.0;

/// The read-back planes of one emitted frame.
type FramePlanes = Vec<Vec<u8>>;

fn geometry(width: u32, height: u32, channels: ChannelMode) -> PassGeometry {
    PassGeometry {
        width,
        height,
        channels,
        input: SampleFormat::U8,
        output: SampleFormat::U8,
    }
}

fn nlmeans_spec(width: u32, height: u32, channels: ChannelMode) -> PassSpec {
    let options = NlmeansOptions {
        mode: DenoisingMode::Temporal { radius: 1 },
        ..NlmeansOptions::default()
    };

    PassSpec::Nlmeans {
        algorithm: NlmeansAlgorithm::Fast(options),
        geometry: geometry(width, height, channels),
    }
}

fn nl4d_spec(options: Nl4dOptions, width: u32, height: u32, channels: ChannelMode) -> PassSpec {
    let options = Nl4dOptions {
        temporal_radius: 1,
        ..options
    };

    PassSpec::Nl4d {
        options,
        geometry: geometry(width, height, channels),
    }
}

/// The luma and chroma engines of a 64x48 4:2:0 frame.
fn nlmeans_specs() -> Vec<PassSpec> {
    let luma = nlmeans_spec(WIDTH, HEIGHT, ChannelMode::Luma);
    let chroma = nlmeans_spec(CHROMA_WIDTH, CHROMA_HEIGHT, ChannelMode::Chroma);

    vec![luma, chroma]
}

fn nl4d_specs(options: Nl4dOptions) -> Vec<PassSpec> {
    let luma = nl4d_spec(options, WIDTH, HEIGHT, ChannelMode::Luma);
    let chroma = nl4d_spec(options, CHROMA_WIDTH, CHROMA_HEIGHT, ChannelMode::Chroma);

    vec![luma, chroma]
}

/// One plane of frame `frame`, a ramp plus hashed grain for luma and flat grain for chroma.
fn plane_bytes(channels: ChannelMode, frame: u32, width: u32, height: u32) -> Vec<u8> {
    let pixels = width * height;

    match channels {
        ChannelMode::Luma => (0..pixels)
            .map(|index| {
                let ramp = (index % width) * 3;
                let grain = index.wrapping_mul(2_654_435_761).wrapping_add(frame * 97) >> 28;
                let value = 40 + ramp + grain;
                value.min(255) as u8
            })
            .collect(),
        _ => (0..pixels)
            .map(|index| {
                let grain = index.wrapping_mul(40_503).wrapping_add(frame) >> 29;
                (120 + grain) as u8
            })
            .collect(),
    }
}

/// Frame `frame` for `specs`, uploaded to `client`, one list of planes per engine.
fn frame_planes(client: &Client, specs: &[PassSpec], frame: u32) -> Vec<Vec<Handle>> {
    let mut planes = Vec::with_capacity(specs.len());

    for spec in specs {
        let geometry = spec.geometry();
        let channels = geometry.channels.count();
        let bytes = plane_bytes(geometry.channels, frame, geometry.width, geometry.height);
        let handles = (0..channels).map(|_| client.create_from_slice(&bytes)).collect();
        planes.push(handles);
    }

    planes
}

/// `count` frames for `specs`, uploaded to `client`.
fn frames(client: &Client, specs: &[PassSpec], count: usize) -> Vec<TuningFrame> {
    let mut frames = Vec::with_capacity(count);

    for frame in 0..count as u32 {
        let planes = frame_planes(client, specs, frame);
        frames.push(TuningFrame::new(planes));
    }

    frames
}

fn no_store_config() -> TuningConfig {
    TuningConfig {
        mode: TuningMode::Auto,
        store: None,
        budget: BudgetRequest::Full,
        ..TuningConfig::default()
    }
}

fn auto_config(store: &Path, budget: BudgetRequest) -> TuningConfig {
    TuningConfig {
        mode: TuningMode::Auto,
        store: Some(store.to_path_buf()),
        budget,
        ..TuningConfig::default()
    }
}

fn request() -> TuneRequest {
    TuneRequest {
        force: false,
        mode: SearchMode::Families,
    }
}

fn device_profile(device: &cubecl::Device, client: &Client) -> DeviceProfile {
    let identity = describe(device, client);
    let limits = DeviceLimits::from_client(client);

    DeviceProfile { identity, limits }
}

/// How many launch sites the store at `store` holds for `device`.
fn stored_site_count(device: &cubecl::Device, client: &Client, store: &Path) -> usize {
    let identity = describe(device, client);
    let stored = TuneStore::at(store).load(&identity);

    stored.sites.len()
}

/// Tunes `specs` on fresh frames, returning the report.
fn tune(specs: &[PassSpec], frame_count: usize, config: &TuningConfig, request: &TuneRequest) -> TuneReport {
    let device = crate::testing::test_device();
    let client = device.client();
    let clip = frames(&client, specs, frame_count);

    tune_frames(&device, &client, specs, &clip, config, request)
}

fn tuning_set(
    device: &cubecl::Device,
    client: &Client,
    specs: &[PassSpec],
    config: &TuningConfig,
) -> EngineSet {
    let ledger = TuneLedger::new(config.deterministic);
    let setup = KernelSetup {
        ledger: Some(ledger),
        profile_override: None,
    };

    EngineSet::build(device, client, specs, config, &setup).expect("the set builds")
}

fn read_planes(client: &Client, handles: &[Handle]) -> FramePlanes {
    handles
        .iter()
        .map(|handle| {
            let read = client.read_one(handle.clone()).expect("the plane reads");
            read.to_vec()
        })
        .collect()
}

/// Emits `count` frames from engine `index` of `set` and reads each back.
fn emit_frames(
    set: &mut EngineSet,
    client: &Client,
    spec: &PassSpec,
    index: usize,
    count: usize,
) -> Vec<FramePlanes> {
    let geometry = spec.geometry();
    let bytes = geometry.output.plane_bytes(geometry.pixels()) as usize;
    let channels = geometry.channels.count();
    let mut emitted = Vec::with_capacity(count);

    for _ in 0..count {
        let handles: Vec<Handle> = (0..channels).map(|_| client.empty(bytes)).collect();
        let planes: Vec<DevicePlane<'_>> = handles
            .iter()
            .map(|handle| DevicePlane::new(handle, geometry.width, geometry.height))
            .collect();

        let engine = &mut set.engines_mut()[index];
        engine.emit_into(&planes).expect("the frame emits");

        let read = read_planes(client, &handles);
        emitted.push(read);
    }

    emitted
}

/// Pushes `clip` through every engine of `set` and returns each engine's frames, in order.
fn denoise_clip(
    set: &mut EngineSet,
    client: &Client,
    specs: &[PassSpec],
    clip: &[TuningFrame],
) -> Vec<Vec<FramePlanes>> {
    let mut outputs = vec![Vec::new(); specs.len()];

    for frame in clip {
        for (index, spec) in specs.iter().enumerate() {
            let planes = frame.planes_for(index, spec.geometry());
            let engine = &mut set.engines_mut()[index];
            let ready = engine.push(&planes).expect("the push succeeds");
            let emitted = emit_frames(set, client, spec, index, ready);
            outputs[index].extend(emitted);
        }
    }

    for (index, spec) in specs.iter().enumerate() {
        let engine = &mut set.engines_mut()[index];
        let tail = engine.finish().expect("the stream finishes");
        let emitted = emit_frames(set, client, spec, index, tail);
        outputs[index].extend(emitted);
    }

    outputs
}

#[test]
fn tune_frames_saves_a_complete_result_then_reports_already_tuned() {
    let store = tempfile::tempdir().unwrap();
    let budget = BudgetRequest::Seconds(600.0);
    let config = auto_config(store.path(), budget);
    let device = crate::testing::test_device();
    let client = device.client();
    let specs = [nlmeans_spec(256, 256, ChannelMode::Luma)];
    let frame_count = tuning_frame_count(&specs);
    let clip = frames(&client, &specs, frame_count);

    let first = tune_frames(&device, &client, &specs, &clip, &config, &request());

    assert_eq!(first.outcome, TuneOutcome::Tuned, "{first:?}");
    assert!(first.saved, "{first:?}");
    assert!(is_tuned(&device, &client, &specs, &config));

    let second = tune_frames(&device, &client, &specs, &clip, &config, &request());

    assert_eq!(second.outcome, TuneOutcome::AlreadyTuned, "{second:?}");
}

#[test]
fn off_skips_tuning() {
    let store = tempfile::tempdir().unwrap();
    let config = TuningConfig {
        mode: TuningMode::Off,
        ..auto_config(store.path(), BudgetRequest::Full)
    };
    let specs = nlmeans_specs();

    let report = tune(&specs, 4, &config, &request());

    let skipped = TuneOutcome::Skipped("tuning is off".to_string());
    assert_eq!(report.outcome, skipped);
    assert!(!report.saved);
}

#[test]
fn no_frames_skips_tuning() {
    let store = tempfile::tempdir().unwrap();
    let config = auto_config(store.path(), BudgetRequest::Full);
    let specs = nlmeans_specs();

    let report = tune(&specs, 0, &config, &request());

    let skipped = TuneOutcome::Skipped("there are no frames to tune on".to_string());
    assert_eq!(report.outcome, skipped);
    assert!(!report.saved);
}

#[test]
fn a_pass_that_compiles_calls_the_kernels_compiled_hook() {
    let store = tempfile::tempdir().unwrap();
    let compiled = Arc::new(AtomicUsize::new(0));
    let counter = compiled.clone();
    let hook: CacheHook = Arc::new(move |phase| {
        if phase == CachePhase::KernelsCompiled {
            counter.fetch_add(1, Ordering::SeqCst);
        }
    });
    let config = TuningConfig {
        cache_hook: Some(hook),
        ..auto_config(store.path(), BudgetRequest::Full)
    };
    let device = crate::testing::test_device();
    let client = device.client();
    let specs = nlmeans_specs();
    let clip = frames(&client, &specs, 12);

    let first = tune_frames(&device, &client, &specs, &clip, &config, &request());

    assert_eq!(first.outcome, TuneOutcome::Tuned, "{first:?}");
    assert!(first.compile > Duration::ZERO, "{first:?}");
    assert_eq!(compiled.load(Ordering::SeqCst), 1);

    let second = tune_frames(&device, &client, &specs, &clip, &config, &request());

    assert_eq!(second.outcome, TuneOutcome::AlreadyTuned, "{second:?}");
    assert_eq!(compiled.load(Ordering::SeqCst), 1);
}

#[test]
fn without_a_store_a_result_lasts_for_the_process_and_takes_no_lock() {
    let locks = Arc::new(AtomicUsize::new(0));
    let counter = locks.clone();
    let hook: CacheHook = Arc::new(move |phase| {
        if phase == CachePhase::LockTaken {
            counter.fetch_add(1, Ordering::SeqCst);
        }
    });
    let config = TuningConfig {
        cache_hook: Some(hook),
        ..no_store_config()
    };
    let device = crate::testing::test_device();
    let client = device.client();
    let specs = nlmeans_specs();
    let clip = frames(&client, &specs, 12);

    let first = tune_frames(&device, &client, &specs, &clip, &config, &request());

    assert_eq!(first.outcome, TuneOutcome::Tuned, "{first:?}");
    assert!(!first.saved, "{first:?}");
    assert!(is_tuned(&device, &client, &specs, &config));

    let second = tune_frames(&device, &client, &specs, &clip, &config, &request());

    assert_eq!(second.outcome, TuneOutcome::AlreadyTuned, "{second:?}");
    assert_eq!(
        locks.load(Ordering::SeqCst),
        0,
        "no lock is taken without a store"
    );
}

#[test]
fn without_a_store_a_second_group_keeps_the_first_groups_result() {
    let config = no_store_config();
    let device = crate::testing::test_device();
    let client = device.client();
    let first_specs = nlmeans_specs();
    let second_specs = [nlmeans_spec(256, 256, ChannelMode::Luma)];
    let first_clip = frames(&client, &first_specs, 12);
    let second_clip = frames(&client, &second_specs, 12);

    let first = tune_frames(&device, &client, &first_specs, &first_clip, &config, &request());
    let second = tune_frames(&device, &client, &second_specs, &second_clip, &config, &request());

    assert_eq!(first.outcome, TuneOutcome::Tuned, "{first:?}");
    assert_eq!(second.outcome, TuneOutcome::Tuned, "{second:?}");
    assert!(is_tuned(&device, &client, &first_specs, &config));
    assert!(is_tuned(&device, &client, &second_specs, &config));
}

#[test]
fn an_unwritable_store_still_remembers_the_result() {
    let dir = tempfile::tempdir().unwrap();
    let store = dir.path().join("not-a-directory");
    std::fs::write(&store, b"a file").unwrap();
    let config = auto_config(&store, BudgetRequest::Full);
    let device = crate::testing::test_device();
    let client = device.client();
    let specs = nlmeans_specs();
    let clip = frames(&client, &specs, 12);

    let report = tune_frames(&device, &client, &specs, &clip, &config, &request());

    assert_eq!(report.outcome, TuneOutcome::Tuned, "{report:?}");
    assert!(!report.saved, "{report:?}");
    assert!(is_tuned(&device, &client, &specs, &config));
}

#[test]
fn a_full_pass_tunes_and_saves_and_a_second_finds_it_done() {
    let store = tempfile::tempdir().unwrap();
    let config = auto_config(store.path(), BudgetRequest::Full);
    let device = crate::testing::test_device();
    let client = device.client();
    let specs = nlmeans_specs();
    let clip = frames(&client, &specs, 12);

    assert!(!is_tuned(&device, &client, &specs, &config));

    let first = tune_frames(&device, &client, &specs, &clip, &config, &request());
    let second = tune_frames(&device, &client, &specs, &clip, &config, &request());

    assert_eq!(first.outcome, TuneOutcome::Tuned, "{first:?}");
    assert!(first.tuned_keys >= 1, "{first:?}");
    assert_eq!(
        first.tuned_keys,
        stored_site_count(&device, &client, store.path()),
        "{first:?}"
    );
    assert!(first.saved);
    assert_eq!(second.outcome, TuneOutcome::AlreadyTuned, "{second:?}");
    assert_eq!(second.tuned_keys, 0);
    assert!(first.compile > Duration::ZERO, "{first:?}");
    assert!(first.compile < first.elapsed, "{first:?}");
    assert_eq!(second.compile, Duration::ZERO);
    assert!(is_tuned(&device, &client, &specs, &config));

    let entries = std::fs::read_dir(store.path()).unwrap();
    let files: Vec<_> = entries
        .flatten()
        .filter(|entry| {
            entry
                .path()
                .extension()
                .is_some_and(|extension| extension == "json")
        })
        .collect();
    assert_eq!(files.len(), 1, "one store file for one device");
}

#[test]
fn forcing_tunes_again() {
    let store = tempfile::tempdir().unwrap();
    let config = auto_config(store.path(), BudgetRequest::Full);
    let specs = nlmeans_specs();
    let forced = TuneRequest {
        force: true,
        ..request()
    };

    tune(&specs, 12, &config, &request());
    let again = tune(&specs, 12, &config, &forced);

    assert_eq!(again.outcome, TuneOutcome::Tuned);
    assert!(again.tuned_keys >= 1);
}

#[test]
fn a_clip_too_short_to_earn_back_tuning_is_skipped() {
    let store = tempfile::tempdir().unwrap();
    let budget = BudgetRequest::Automatic { frames: Some(12) };
    let config = auto_config(store.path(), budget);
    let device = crate::testing::test_device();
    let client = device.client();
    let specs = nlmeans_specs();
    let clip = frames(&client, &specs, 12);

    let report = tune_frames(&device, &client, &specs, &clip, &config, &request());

    assert!(matches!(report.outcome, TuneOutcome::Skipped(_)), "{report:?}");
    assert!(!report.saved);
    assert!(!is_tuned(&device, &client, &specs, &config));
}

/// The first pass compiles the kernels, so the second pass is mostly warm-up.
#[test]
fn a_tiny_budget_ends_the_warm_up_early() {
    let store = tempfile::tempdir().unwrap();
    let budget_seconds = 0.5;
    let budget = BudgetRequest::Seconds(budget_seconds);
    let config = auto_config(store.path(), budget);
    let specs = nlmeans_specs();

    tune(&specs, 12, &config, &request());
    let report = tune(&specs, 12, &config, &request());

    let budget = Duration::from_secs_f64(budget_seconds);
    let frame_allowance = Duration::from_secs(1);

    assert!(matches!(report.outcome, TuneOutcome::Skipped(_)), "{report:?}");
    assert!(!report.saved);
    assert!(report.elapsed <= budget + frame_allowance, "{report:?}");
}

#[test]
fn a_frame_with_a_short_handle_is_skipped() {
    let store = tempfile::tempdir().unwrap();
    let config = auto_config(store.path(), BudgetRequest::Full);
    let device = crate::testing::test_device();
    let client = device.client();
    let specs = nlmeans_specs();
    let mut planes = frame_planes(&client, &specs, 0);
    planes[0][0] = client.create_from_slice(&[0u8; 16]);
    let clip = vec![TuningFrame::new(planes)];

    let report = tune_frames(&device, &client, &specs, &clip, &config, &request());

    let TuneOutcome::Skipped(reason) = &report.outcome else {
        panic!("a short handle never tunes, got {report:?}");
    };
    assert!(reason.contains("plane mismatch"), "{reason}");
    assert!(!report.saved);
}

#[test]
fn a_frame_for_too_few_engines_is_skipped() {
    let store = tempfile::tempdir().unwrap();
    let config = auto_config(store.path(), BudgetRequest::Full);
    let device = crate::testing::test_device();
    let client = device.client();
    let specs = nlmeans_specs();
    let mut planes = frame_planes(&client, &specs, 0);
    planes.pop();
    let clip = vec![TuningFrame::new(planes)];

    let report = tune_frames(&device, &client, &specs, &clip, &config, &request());

    let TuneOutcome::Skipped(reason) = &report.outcome else {
        panic!("a frame missing an engine never tunes, got {report:?}");
    };
    assert!(reason.contains("plane mismatch"), "{reason}");
}

#[test]
fn a_set_without_a_ledger_is_skipped() {
    let store = tempfile::tempdir().unwrap();
    let config = auto_config(store.path(), BudgetRequest::Full);
    let device = crate::testing::test_device();
    let client = device.client();
    let specs = nlmeans_specs();
    let clip = frames(&client, &specs, 4);
    let setup = KernelSetup::default();
    let mut set = EngineSet::build(&device, &client, &specs, &config, &setup).expect("the set builds");

    let report = tune_set(&mut set, &clip, &config, &request());

    assert!(matches!(report.outcome, TuneOutcome::Skipped(_)), "{report:?}");
}

#[test]
fn a_spent_compile_ceiling_stops_before_the_next_key() {
    let store = tempfile::tempdir().unwrap();
    let config = auto_config(store.path(), BudgetRequest::Full);
    let device = crate::testing::test_device();
    let client = device.client();
    let specs = nlmeans_specs();
    let clip = frames(&client, &specs, 2);
    let mut set = tuning_set(&device, &client, &specs, &config);
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
    let tuned = tune_keys(&mut set, &ledger, &mut cycle, &order, &budget, Instant::now());

    assert!(tuned.frame_error.is_none());
    assert_eq!(cycle.next, 0, "a frame was pushed for the key");
}

/// The fresh set builds from the result the pass remembered, so both run the same plan.
#[test]
fn a_tuned_set_matches_a_fresh_one_built_from_its_result() {
    let store = tempfile::tempdir().unwrap();
    let config = auto_config(store.path(), BudgetRequest::Full);
    let device = crate::testing::test_device();
    let client = device.client();
    let specs = nlmeans_specs();
    let clip = frames(&client, &specs, 12);
    let mut set = tuning_set(&device, &client, &specs, &config);

    let report = tune_set(&mut set, &clip, &config, &request());
    assert_eq!(report.outcome, TuneOutcome::Tuned, "{report:?}");

    let cached_config = TuningConfig {
        mode: TuningMode::Cached,
        ..config.clone()
    };
    let setup = KernelSetup::default();
    let mut fresh =
        EngineSet::build(&device, &client, &specs, &cached_config, &setup).expect("the fresh set builds");
    let tuned_output = denoise_clip(&mut set, &client, &specs, &clip);
    let fresh_output = denoise_clip(&mut fresh, &client, &specs, &clip);

    for (engine, (tuned_frames, fresh_frames)) in tuned_output.iter().zip(&fresh_output).enumerate() {
        assert_eq!(tuned_frames.len(), clip.len(), "engine {engine}");
        assert_eq!(fresh_frames.len(), clip.len(), "engine {engine}");

        let frame_pairs = tuned_frames.iter().zip(fresh_frames);
        for (index, (tuned_frame, fresh_frame)) in frame_pairs.enumerate() {
            assert_eq!(tuned_frame, fresh_frame, "engine {engine}, frame {index}");
        }
    }
}

#[test]
fn a_tuned_grain_export_set_drains_only_the_real_streams_grain() {
    let store = tempfile::tempdir().unwrap();
    let config = auto_config(store.path(), BudgetRequest::Full);
    let device = crate::testing::test_device();
    let client = device.client();
    let nl4d = Nl4dOptions {
        grain_export: true,
        ..Nl4dOptions::default()
    };
    let specs = nl4d_specs(nl4d);
    let clip = frames(&client, &specs, 13);
    let mut set = tuning_set(&device, &client, &specs, &config);
    let mut fresh = tuning_set(&device, &client, &specs, &config);

    tune_set(&mut set, &clip, &config, &request());
    denoise_clip(&mut set, &client, &specs, &clip);
    denoise_clip(&mut fresh, &client, &specs, &clip);

    for index in 0..specs.len() {
        let tuned_engine = &mut set.engines_mut()[index];
        let tuned_chunks = tuned_engine.drain_grain_chunks().expect("the tuned grain drains");
        let fresh_engine = &mut fresh.engines_mut()[index];
        let fresh_chunks = fresh_engine.drain_grain_chunks().expect("the fresh grain drains");

        let tuned_frames: u32 = tuned_chunks.iter().map(|chunk| chunk.frames).sum();
        let fresh_frames: u32 = fresh_chunks.iter().map(|chunk| chunk.frames).sum();

        if index == 0 {
            assert!(fresh_frames > 0, "the clip measures some grain");
        }

        assert_eq!(tuned_chunks.len(), fresh_chunks.len(), "engine {index}");
        assert_eq!(tuned_frames, fresh_frames, "engine {index}");
    }
}

#[test]
fn an_nl4d_pass_records_a_ranking_for_every_plane() {
    let store = tempfile::tempdir().unwrap();
    let budget = BudgetRequest::Seconds(COLD_NL4D_SECONDS);
    let config = auto_config(store.path(), budget);
    let device = crate::testing::test_device();
    let client = device.client();
    let specs = nl4d_specs(Nl4dOptions::default());
    let clip = frames(&client, &specs, 13);

    let report = tune_frames(&device, &client, &specs, &clip, &config, &request());

    assert_eq!(report.outcome, TuneOutcome::Tuned, "{report:?}");
    assert_eq!(report.partial_keys, 0, "{report:?}");
    assert_eq!(
        report.tuned_keys,
        stored_site_count(&device, &client, store.path()),
        "{report:?}"
    );
    assert!(is_tuned(&device, &client, &specs, &config), "{report:?}");

    let profile = device_profile(&device, &client);
    let stored = TuneStore::at(store.path()).load(&profile.identity);
    let table = device_priority(&profile);
    let alternatives = alternative_profiles_for(true, &profile, table);
    let profile_keys: Vec<String> = specs.iter().map(crate::tune::profile_key).collect();

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

#[test]
fn a_report_run_overwrites_a_stored_key() {
    let store_dir = tempfile::tempdir().unwrap();
    let config = auto_config(store_dir.path(), BudgetRequest::Full);
    let device = crate::testing::test_device();
    let client = device.client();
    let specs = nlmeans_specs();
    let clip = frames(&client, &specs, 12);

    let first = tune_frames(&device, &client, &specs, &clip, &config, &request());
    assert_eq!(first.outcome, TuneOutcome::Tuned, "{first:?}");
    assert!(first.summary.is_none(), "{first:?}");

    let identity = describe(&device, &client);
    let store = TuneStore::at(store_dir.path());
    let mut stored = store.load(&identity);
    let (key, ranking) = stored
        .sites
        .iter_mut()
        .find(|(_, ranking)| ranking.candidates.len() > 1)
        .expect("a key timed more than its baseline");
    let key = key.clone();
    for record in &mut ranking.candidates {
        record.median_ns = 1;
        record.samples = 1;
    }

    store.save(&identity, &stored).expect("the planted result saves");

    let report_request = TuneRequest {
        force: true,
        mode: SearchMode::Exhaustive,
    };
    let report = tune_frames(&device, &client, &specs, &clip, &config, &report_request);
    assert_eq!(report.outcome, TuneOutcome::Tuned, "{report:?}");

    let reloaded = store.load(&identity);
    let overwritten = reloaded.sites.get(&key).expect("the key is still stored");
    let fully_sampled = overwritten
        .candidates
        .iter()
        .all(|record| record.samples as usize >= MIN_SAMPLES);
    assert!(fully_sampled, "{overwritten:?}");

    let summary = report.summary.expect("an exhaustive pass sums up its search");
    assert!(summary.contains(key.as_str()), "{summary}");
}
