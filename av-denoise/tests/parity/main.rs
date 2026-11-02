//! Checks denoised output against hashes recorded on this machine
//!
//! Setting `AVD_PARITY_RECORD=1` records the fixture instead of checking it.

#![cfg(feature = "vulkan")]

mod clips;
mod configs;
mod fixture;

use av_denoise::{
    ChannelIntent,
    ChromaLayout,
    DenoiserError,
    DenoiserOptions,
    HostDenoiser,
    PlanarDenoiser,
    Planes,
    ReseedWindow,
    SceneGrain,
    build_table,
    push_needs_retry,
};

use self::configs::{ParityConfig, Reseed};
use self::fixture::Entries;

fn record_planes(entries: &mut Entries, config: &str, prefix: &str, planes: &Planes) {
    for (name, bytes) in [("y", &planes.y), ("u", &planes.u), ("v", &planes.v)] {
        let config_name = config.to_string();
        let label = format!("{prefix}.{name}");
        let key = (config_name, label);
        let hash = fixture::hash(bytes);
        let value = format!("{hash:016x}");
        entries.insert(key, value);
    }
}

/// How many frames `config` runs.
///
/// A reseeding config needs a frame on each side of its reseed window, so the window sits
/// mid-clip.
fn clip_length(config: &ParityConfig, denoiser: &PlanarDenoiser) -> usize {
    if config.reseed == Reseed::None {
        return config.frames;
    }

    let span = denoiser.window_span();
    let reseed_frames = span.frame_count() + 2;
    config.frames.max(reseed_frames)
}

fn run(config: &ParityConfig, entries: &mut Entries) {
    if config.greyscale {
        run_greyscale(config, entries);
        return;
    }

    let mut denoiser = PlanarDenoiser::create(&config.options, config.layout).expect("create");
    let frame_count = clip_length(config, &denoiser);
    let clip = clips::clip(config.clip, config.layout, frame_count);
    let mut outputs = Vec::new();

    for planes in &clip {
        loop {
            let pushed = denoiser.push(planes);
            let retry = push_needs_retry(pushed).expect("push");
            if !retry {
                break;
            }

            let received = denoiser.recv().expect("recv");
            outputs.extend(received);
        }

        while let Some(denoised) = denoiser.recv().expect("recv") {
            outputs.push(denoised);
        }
    }

    denoiser.flush(|planes| outputs.push(planes)).expect("flush");

    for (index, planes) in outputs.iter().enumerate() {
        let label = format!("f{index}");
        record_planes(entries, config.name, &label, planes);
    }

    let chunks = denoiser.drain_grain_chunks().expect("grain");
    if !chunks.is_empty() {
        let scene = SceneGrain {
            first_frame: 0,
            chunks,
        };
        let table = build_table(&[scene], (24, 1));
        let config_name = config.name.to_string();
        let label = "grain".to_string();
        let key = (config_name, label);
        let table_bytes = table.as_bytes();
        let hash = fixture::hash(table_bytes);
        let value = format!("{hash:016x}");
        entries.insert(key, value);
    }

    run_reseed(config, &mut denoiser, &clip, &outputs, entries);
}

/// Runs the config's reseed and records it.
///
/// A mid-clip reseed must reproduce the streamed output of its target frame byte for byte.
fn run_reseed(
    config: &ParityConfig,
    denoiser: &mut PlanarDenoiser,
    clip: &[Planes],
    streamed: &[Planes],
    entries: &mut Entries,
) {
    let span = denoiser.window_span();

    match config.reseed {
        Reseed::None => {},
        Reseed::Middle => {
            let start = 1;
            let end = start + span.frame_count();
            let window = &clip[start..end];
            let reseeded = denoiser.reseed(window).expect("reseed");
            let target = &streamed[start + span.behind];
            let matches = reseeded.y == target.y && reseeded.u == target.u && reseeded.v == target.v;
            assert!(
                matches,
                "{}: the reseed differs from its streamed frame",
                config.name
            );
            record_planes(entries, config.name, "reseed", &reseeded);
        },
        Reseed::ClipStart => {
            let frames = &clip[..=span.ahead];
            let request = ReseedWindow {
                frames,
                target: 0,
                at_clip_start: true,
                at_clip_end: false,
            };
            let reseeded = denoiser.reseed_window(request).expect("reseed_window");
            record_reseeded(entries, config.name, &reseeded);
        },
        Reseed::ClipEnd => {
            let frames = &clip[clip.len() - 1 - span.behind..];
            let request = ReseedWindow {
                frames,
                target: span.behind,
                at_clip_start: false,
                at_clip_end: true,
            };
            let reseeded = denoiser.reseed_window(request).expect("reseed_window");
            record_reseeded(entries, config.name, &reseeded);
        },
    }
}

fn record_reseeded(entries: &mut Entries, config: &str, reseeded: &[Planes]) {
    for (index, planes) in reseeded.iter().enumerate() {
        let label = format!("reseed{index}");
        record_planes(entries, config, &label, planes);
    }
}

fn run_greyscale(config: &ParityConfig, entries: &mut Entries) {
    let options = DenoiserOptions::builder()
        .intent(ChannelIntent::Luma)
        .chroma(ChromaLayout::Gray)
        .mode(config.options.mode)
        .algorithm(config.options.algorithm)
        .depth(config.layout.depth)
        .tuning(config.options.tuning)
        .deterministic(config.options.deterministic)
        .build();
    let accelerator = av_denoise::testing::test_accelerator();
    let device = av_denoise::testing::test_device();
    let (width, height) = (config.layout.width, config.layout.height);
    let mut denoiser = HostDenoiser::create(&[accelerator], &device, width, height, options).expect("create");
    let clip = clips::clip(config.clip, config.layout, config.frames);
    let mut outputs: Vec<Vec<Vec<u8>>> = Vec::new();

    for planes in &clip {
        loop {
            let luma: &[&[u8]] = &[&planes.y];
            match denoiser.push(luma) {
                Ok(()) => break,
                Err(DenoiserError::QueueFull) => {
                    let received = denoiser.recv().expect("recv");
                    outputs.extend(received);
                },
                Err(other) => panic!("push failed: {other}"),
            }
        }

        while let Some(denoised) = denoiser.recv().expect("recv") {
            outputs.push(denoised);
        }
    }

    denoiser.flush(|planes| outputs.push(planes)).expect("flush");

    for (index, planes) in outputs.iter().enumerate() {
        assert_eq!(planes.len(), 1, "a luma frame has one plane");
        let key = (config.name.to_string(), format!("f{index}.y"));
        let hash = fixture::hash(&planes[0]);
        let value = format!("{hash:016x}");
        entries.insert(key, value);
    }
}

#[test]
#[ignore = "exact hashes only hold on the GPU and driver they were recorded on"]
fn outputs_match_the_recorded_fixture() {
    let recording = std::env::var_os("AVD_PARITY_RECORD").is_some();
    let expected = (!recording).then(fixture::read);
    if let Some(fixture) = &expected {
        fixture::check_adapters(&fixture.adapters);
    }

    let mut entries = Entries::new();
    for config in configs::all() {
        run(&config, &mut entries);
    }

    let Some(expected) = expected else {
        fixture::write(&entries);
        return;
    };

    let expected = expected.entries;
    assert_eq!(entries.len(), expected.len(), "entry count changed");

    for (key, hash) in &entries {
        let recorded = expected
            .get(key)
            .unwrap_or_else(|| panic!("no recorded entry for {key:?}"));
        assert_eq!(hash, recorded, "output changed for {key:?}");
    }
}
