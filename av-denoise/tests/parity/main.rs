//! Checks denoised output against hashes recorded on this machine
//!
//! Setting `AVD_PARITY_RECORD=1` records the fixture instead of checking it.

mod clips;
mod configs;
mod fixture;

use av_denoise::{PlanarDenoiser, Planes, SceneGrain, build_table, push_needs_retry};

use self::configs::ParityConfig;
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

fn run(config: &ParityConfig, entries: &mut Entries) {
    let clip = clips::clip(config.layout, config.frames);
    let mut denoiser = PlanarDenoiser::create(&config.options, config.layout).expect("create");
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

        while let Some(planes) = denoiser.recv().expect("recv") {
            outputs.push(planes);
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

    if config.reseed {
        let span = denoiser.window_span();
        let start = 1;
        let end = start + span.frame_count();
        let window = &clip[start..end];
        let reseeded = denoiser.reseed(window).expect("reseed");
        record_planes(entries, config.name, "reseed", &reseeded);
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
