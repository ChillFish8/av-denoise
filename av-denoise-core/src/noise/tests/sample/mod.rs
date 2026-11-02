mod channels;
mod gating;
mod layout;
mod moved;
mod recorded;

use cubecl::prelude::*;

use super::records::{BlockSpec, frame_records, hashed_unit, static_block, ulp_gap};
use crate::engine::align::StorageAlign;
use crate::noise::NOISE_CHANNELS;
use crate::noise::reading::ReadingRecord;
use crate::noise::select::{RankRule, SelectQuery};
use crate::noise::spatial::{RESULTS_PER_SLOT, sigma_block_p25_from_partials};
use crate::noise::state::{BLOCK_ACCEPTED, NoiseShape, NoiseState, ReadingSettings};
use crate::noise::temporal::{
    STATIC_GATE,
    TemporalNoiseSample,
    accepted_static_blocks,
    temporal_noise_reading,
    temporal_stats_blocks,
};
use crate::tune::tests::runtime::test_client;

const MAX_ULPS: u32 = 2;
const RESULTS_ROW: [f32; RESULTS_PER_SLOT] = [0.0; RESULTS_PER_SLOT];

/// The geometry of one plane a test reads.
#[derive(Debug, Clone, Copy)]
struct Plane {
    width: u32,
    height: u32,
    channels: u32,
}

const LUMA_1080P: Plane = Plane {
    width: 1920,
    height: 1080,
    channels: 1,
};

const CHROMA_1080P: Plane = Plane {
    width: 960,
    height: 540,
    channels: 2,
};

impl Plane {
    fn luma(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            channels: 1,
        }
    }

    fn chroma(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            channels: 2,
        }
    }

    fn blocks(self) -> usize {
        let (blocks_x, blocks_y) = temporal_stats_blocks(self.width, self.height);
        (blocks_x * blocks_y) as usize
    }

    fn records(self, blocks: &[BlockSpec]) -> Vec<f32> {
        frame_records(self.width, self.height, self.channels, blocks)
    }
}

/// What the host reference reads from one slot's stats.
struct HostReading {
    immerkaer_low: [f32; NOISE_CHANNELS as usize],
    sample: Option<TemporalNoiseSample>,
    accepted: Vec<usize>,
}

fn host_reading(plane: Plane, partials: &[f32], records: &[f32]) -> HostReading {
    let immerkaer_low = sigma_block_p25_from_partials(partials, plane.channels, plane.width, plane.height);
    let reading = temporal_noise_reading(records, plane.channels, plane.width, plane.height, false);
    let accepted_blocks = accepted_static_blocks(records, plane.channels, plane.width, plane.height);
    let accepted = accepted_blocks
        .unwrap_or_default()
        .iter()
        .map(|block| block.index)
        .collect();

    HostReading {
        immerkaer_low,
        sample: reading.sample,
        accepted,
    }
}

fn shape_for(client: &Client, plane: Plane, temporal: bool) -> NoiseShape {
    NoiseShape {
        width: plane.width,
        height: plane.height,
        channels: plane.channels,
        frames: 3,
        ring_radius: 1,
        temporal,
        align: StorageAlign::from_client(client),
    }
}

fn state_for(
    client: &Client,
    plane: Plane,
    temporal: bool,
    settings: ReadingSettings,
    partials: &[f32],
    records: &[f32],
) -> NoiseState {
    let shape = shape_for(client, plane, temporal);

    NoiseState::from_stats_for_test(client, shape, settings, partials, RESULTS_ROW, records)
}

/// The reading of `slot` and, with temporal stats, the indices of the blocks the last reading
/// accepted.
fn read_back(state: &NoiseState, slot: u32) -> (ReadingRecord, Vec<usize>) {
    let reading = state.read_reading(slot);
    let flags = state.read_block_flags();
    let accepted = accepted_indices(&flags);

    (reading, accepted)
}

/// The GPU reading of slot 0 and the indices of the blocks it accepted.
fn gpu_reading(
    client: &Client,
    plane: Plane,
    temporal: bool,
    partials: &[f32],
    records: &[f32],
) -> (ReadingRecord, Vec<usize>) {
    let settings = ReadingSettings::default();
    let state = state_for(client, plane, temporal, settings, partials, records);
    state.queue_reading(0).expect("the reading queues");

    read_back(&state, 0)
}

fn accepted_indices(flags: &[u32]) -> Vec<usize> {
    flags
        .iter()
        .enumerate()
        .filter(|(_, flag)| **flag & BLOCK_ACCEPTED != 0)
        .map(|(index, _)| index)
        .collect()
}

fn assert_close(gpu: f32, host: f32, max_ulps: u32, what: &str) {
    let gap = ulp_gap(gpu, host);
    assert!(gap <= max_ulps, "{what}: gpu {gpu} host {host}, {gap} ulp apart");
}

fn assert_immerkaer_matches(gpu: &ReadingRecord, host: &HostReading) {
    for channel in 0..NOISE_CHANNELS as usize {
        let what = format!("immerkaer_low[{channel}]");
        assert_close(
            gpu.immerkaer_low[channel],
            host.immerkaer_low[channel],
            MAX_ULPS,
            &what,
        );
    }
}

fn assert_sample_matches(gpu: Option<TemporalNoiseSample>, host: Option<TemporalNoiseSample>, max_ulps: u32) {
    let (gpu, host) = match (gpu, host) {
        (Some(gpu), Some(host)) => (gpu, host),
        (None, None) => return,
        (gpu, host) => panic!("the gpu sample {gpu:?} and the host sample {host:?} disagree"),
    };

    assert_eq!(gpu.valid, host.valid, "the channels with a sample");

    for channel in 0..NOISE_CHANNELS as usize {
        let sigma_what = format!("sigma[{channel}]");
        let low_what = format!("sigma_low[{channel}]");
        assert_close(gpu.sigma[channel], host.sigma[channel], max_ulps, &sigma_what);
        assert_close(
            gpu.sigma_low[channel],
            host.sigma_low[channel],
            max_ulps,
            &low_what,
        );
    }

    assert_close(gpu.rho, host.rho, max_ulps, "rho");
    assert_close(
        gpu.static_fraction,
        host.static_fraction,
        max_ulps,
        "static_fraction",
    );
}

fn assert_reading_matches(gpu: &ReadingRecord, gpu_accepted: &[usize], host: &HostReading) {
    let host_accepted = host.accepted.len() as u32;

    assert_immerkaer_matches(gpu, host);
    assert_sample_matches(gpu.sample, host.sample, MAX_ULPS);
    assert_eq!(gpu.accepted, host_accepted, "accepted count");
    let host_blocks = host.accepted.as_slice();
    let has_curve = gpu.curve.is_some();
    assert_eq!(gpu_accepted, host_blocks, "accepted blocks");
    assert!(!has_curve);
}

fn assert_sampled(gpu: &ReadingRecord, host: &HostReading, expected: bool) {
    let gpu_sampled = gpu.sample.is_some();
    let host_sampled = host.sample.is_some();

    assert_eq!(gpu_sampled, expected, "gpu sample");
    assert_eq!(host_sampled, expected, "host sample");
}

/// Runs one plane on the GPU and on the host and checks they agree.
fn check_plane(plane: Plane, partials: &[f32], records: &[f32]) -> (ReadingRecord, HostReading) {
    let client = test_client();
    let (gpu, gpu_accepted) = gpu_reading(&client, plane, true, partials, records);
    let host = host_reading(plane, partials, records);

    assert_reading_matches(&gpu, &gpu_accepted, &host);

    (gpu, host)
}

/// A block whose mean residual is far past the static gate.
fn moving_block(sigma: f32) -> BlockSpec {
    BlockSpec::uniform(4.0 * STATIC_GATE, sigma, 0.2)
}

fn static_blocks(plane: Plane, seed: u32) -> Vec<BlockSpec> {
    let count = plane.blocks() as u32;
    (0..count).map(|index| static_block(seed, index)).collect()
}

/// Asserts `queries` holds `tag` with each of `rules` in turn, starting at `first`.
fn assert_query_run(queries: &[SelectQuery], first: u32, tag: u32, rules: &[RankRule]) {
    for (offset, &rule) in rules.iter().enumerate() {
        let index = first as usize + offset;
        let expected = SelectQuery { tag, rule };

        assert_eq!(queries[index], expected, "query {index}");
    }
}

/// Static blocks where two in three moved, and every fifth carries a high texture at a raised sigma.
fn moved_blocks(plane: Plane, seed: u32) -> Vec<BlockSpec> {
    let mut blocks = static_blocks(plane, seed);

    for (index, block) in blocks.iter_mut().enumerate() {
        let unit = hashed_unit(seed + 7, index as u32);
        let textured = index % 5 == 0;
        block.moved = index % 3 != 0;
        let texture = if textured { 1000.0 + unit } else { unit };
        block.texture = [texture; NOISE_CHANNELS as usize];

        if textured {
            block.sigma = block.sigma.map(|sigma| 1.8 * sigma);
        }
    }

    blocks
}
