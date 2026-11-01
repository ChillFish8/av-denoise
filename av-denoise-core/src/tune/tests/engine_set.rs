use cubecl::prelude::*;

use crate::engine::{DevicePlane, PassGeometry, PassSpec, SampleFormat};
use crate::nlmeans::{ChannelMode, DenoisingMode, NlmeansAlgorithm, NlmeansOptions};
use crate::tune::engine_set::EngineSet;
use crate::tune::setup::KernelSetup;
use crate::tune::{TuningConfig, TuningFrame, TuningMode};

const WIDTH: u32 = 64;
const HEIGHT: u32 = 48;
const PLANE_BYTES: usize = (WIDTH * HEIGHT) as usize;

#[cube(launch_unchecked)]
fn fill_words(words: &mut [u32], value: u32) {
    words[ABSOLUTE_POS_X as usize] = value;
}

fn spec(mode: DenoisingMode) -> PassSpec {
    let geometry = PassGeometry {
        width: WIDTH,
        height: HEIGHT,
        channels: ChannelMode::Luma,
        input: SampleFormat::U8,
        output: SampleFormat::U8,
    };
    let options = NlmeansOptions {
        mode,
        ..NlmeansOptions::default()
    };

    PassSpec::Nlmeans {
        algorithm: NlmeansAlgorithm::Fast(options),
        geometry,
    }
}

fn spatial() -> PassSpec {
    spec(DenoisingMode::Spacial)
}

fn temporal() -> PassSpec {
    spec(DenoisingMode::Temporal { radius: 1 })
}

fn off() -> TuningConfig {
    TuningConfig {
        mode: TuningMode::Off,
        ..TuningConfig::default()
    }
}

fn build_set(specs: &[PassSpec]) -> (EngineSet, Client) {
    let device = crate::testing::test_device();
    let client = device.client();
    let setup = KernelSetup::default();
    let set = EngineSet::build(&device, &client, specs, &off(), &setup).expect("the set builds");

    (set, client)
}

fn flat_frame(client: &Client, engines: usize) -> TuningFrame {
    let bytes = vec![100u8; PLANE_BYTES];
    let planes = (0..engines)
        .map(|_| vec![client.create_from_slice(&bytes)])
        .collect();

    TuningFrame::new(planes)
}

#[test]
fn a_spatial_set_times_every_frame() {
    let (mut set, client) = build_set(&[spatial()]);
    let frame = flat_frame(&client, 1);

    for _ in 0..3 {
        let timed = set.timed_frame(&frame).expect("the frame runs");

        assert!(timed.is_some());
    }
}

#[test]
fn a_temporal_set_times_nothing_until_its_window_fills() {
    let (mut set, client) = build_set(&[temporal()]);
    let frame = flat_frame(&client, 1);

    let first = set.timed_frame(&frame).expect("the first frame runs");
    let second = set.timed_frame(&frame).expect("the second frame runs");

    assert!(first.is_none());
    assert!(second.is_some());
}

#[test]
fn copied_frames_are_independent_of_their_source() {
    let client = crate::testing::test_client();
    let specs = [spatial()];
    let bytes = vec![7u8; PLANE_BYTES];
    let source = client.create_from_slice(&bytes);
    let plane = DevicePlane::new(&source, WIDTH, HEIGHT);
    let planes = [plane];

    let frame = TuningFrame::copy_from(&client, &specs, &[&planes]).expect("the frame copies");

    let words = PLANE_BYTES / 4;
    unsafe {
        fill_words::launch_unchecked(
            &client,
            CubeCount::new_1d((words / 64) as u32),
            CubeDim::new_1d(64),
            BufferArg::from_raw_parts(source.clone(), words),
            0u32,
        );
    }

    let copied = frame.planes_for(0, specs[0].geometry());
    let read = client
        .read_one(copied[0].handle().clone())
        .expect("the copy reads");

    assert_eq!(read.to_vec(), bytes);
}

#[test]
fn engines_out_of_step_are_an_error() {
    let (mut set, client) = build_set(&[spatial(), temporal()]);
    let frame = flat_frame(&client, 2);

    let result = set.timed_frame(&frame);

    assert!(result.is_err());
}

#[test]
fn copying_the_wrong_number_of_planes_is_an_error() {
    let client = crate::testing::test_client();
    let specs = [spatial()];
    let none: [&[DevicePlane<'_>]; 0] = [];
    let empty: [DevicePlane<'_>; 0] = [];

    let outer = TuningFrame::copy_from(&client, &specs, &none);
    let inner = TuningFrame::copy_from(&client, &specs, &[&empty]);

    assert!(matches!(outer, Err(crate::Error::PlaneMismatch(_))));
    assert!(matches!(inner, Err(crate::Error::PlaneMismatch(_))));
}

#[test]
fn an_empty_set_is_an_error() {
    let device = crate::testing::test_device();
    let client = device.client();
    let setup = KernelSetup::default();

    let result = EngineSet::build(&device, &client, &[], &off(), &setup);

    assert!(matches!(result, Err(crate::Error::InvalidOptions(_))));
}

#[test]
fn copies_of_an_odd_sized_plane_match_their_source_byte_for_byte() {
    let client = crate::testing::test_client();
    let geometry = PassGeometry {
        width: 61,
        height: 37,
        channels: ChannelMode::Luma,
        input: SampleFormat::U8,
        output: SampleFormat::U8,
    };
    let specs = [PassSpec::Nlmeans {
        algorithm: NlmeansAlgorithm::Fast(NlmeansOptions::default()),
        geometry,
    }];
    let bytes = geometry.input.plane_bytes(geometry.pixels()) as usize;
    let pattern: Vec<u8> = (0..bytes).map(|index| (index * 7 + index / 251) as u8).collect();
    let source = client.create_from_slice(&pattern);
    let planes = [DevicePlane::new(&source, 61, 37)];

    let frame = TuningFrame::copy_from(&client, &specs, &[&planes]).expect("the frame copies");

    let copied = frame.planes_for(0, geometry);
    let read = client
        .read_one(copied[0].handle().clone())
        .expect("the copy reads");

    assert_eq!(read.to_vec(), pattern);
}
