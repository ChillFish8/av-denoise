use cubecl::prelude::*;

use super::records::{
    BlockSpec,
    QuarterSpec,
    frame_records,
    hashed_partials,
    hashed_unit,
    static_block,
    ulp_gap,
};
use crate::collab::geometry::strength_map_dims;
use crate::engine::MAX_GRID_1D;
use crate::engine::align::StorageAlign;
use crate::noise::curve::NOISE_CURVE_BINS;
use crate::noise::state::{
    CLASS_FLAT,
    CLASS_LINE,
    CLASS_PRESENT,
    NoiseShape,
    NoiseState,
    ReadingSettings,
    SlotStats,
    StrengthMapKind,
    quarter_grid,
};
use crate::noise::strength_map::{QuarterClass, QuarterClasses, QuarterSettings, StrengthMapParams};
use crate::noise::temporal::{
    QUARTER_LUMA_SUM,
    QUARTER_TENSOR_XX,
    QUARTER_TENSOR_XY,
    QUARTER_TENSOR_YY,
    TEMPORAL_NOISE_BLOCK,
    TEMPORAL_QUARTER_BASE,
    TEMPORAL_QUARTER_FIELDS,
    TEMPORAL_QUARTER_SIZE,
    UNMEASURED_FLATNESS,
    temporal_noise_reading,
    temporal_stats_blocks,
    temporal_stats_record_len,
};
use crate::tune::tests::runtime::test_client;

const MAX_ULPS: u32 = 2;
const RESULTS_ROW: [f32; 4] = [0.0; 4];
/// Launches of a luma reading with the curve and the map.
const MAP_LAUNCHES: u32 = 15;
/// Launches of a luma reading without the luma quarter fields.
const SAMPLE_LAUNCHES: u32 = 9;
/// The odd curve bins the pattern spreads its quarters over, one per band of four columns.
const BANDS: u32 = 7;
const LUMA_PARAMS: StrengthMapParams = StrengthMapParams {
    flat_boost: 1.5,
    shadow_soften: 0.65,
};
const CHROMA_BOOST: f32 = 1.5;
/// A tensor component well above any pooled grain tensor.
const TENSOR: f32 = 1.0e-4;

/// What a pattern quarter is built to be.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Flat,
    Textured,
    Clipped,
    Motion,
}

/// The kind of the quarter at `col`, `row`, in a repeating eight-quarter pattern.
fn kind_at(col: u32, row: u32) -> Kind {
    match (col + 2 * row) % 8 {
        0..=3 => Kind::Flat,
        4 | 5 => Kind::Textured,
        6 => Kind::Clipped,
        _ => Kind::Motion,
    }
}

/// The geometry of one plane a test reads.
#[derive(Debug, Clone, Copy)]
struct Plane {
    width: u32,
    height: u32,
    channels: u32,
    stored_ch: u32,
}

impl Plane {
    fn luma(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            channels: 1,
            stored_ch: 1,
        }
    }

    fn chroma(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            channels: 2,
            stored_ch: 2,
        }
    }
}

/// A frame of static blocks whose quarters follow [kind_at], with a tensor per map quarter.
struct MapFrame {
    plane: Plane,
    cols: u32,
    rows: u32,
    blocks: Vec<BlockSpec>,
    /// The `xx`, `yy` and `xy` tensor sums of each map quarter, row-major.
    tensors: Vec<[f32; 3]>,
}

impl MapFrame {
    fn patterned(plane: Plane, seed: u32) -> Self {
        let (blocks_x, blocks_y) = temporal_stats_blocks(plane.width, plane.height);
        let (cols, rows) = strength_map_dims(plane.width, plane.height);
        let count = blocks_x * blocks_y;

        let mut blocks = Vec::with_capacity(count as usize);
        for index in 0..count {
            let block = static_block(seed, index);
            blocks.push(block);
        }

        let tensors = vec![[0.0f32; 3]; (cols * rows) as usize];
        let mut frame = Self {
            plane,
            cols,
            rows,
            blocks,
            tensors,
        };

        for row in 0..rows {
            for col in 0..cols {
                let quarter = frame.pattern_quarter(seed, col, row);
                *frame.quarter_mut(col, row) = quarter;
            }
        }

        frame
    }

    fn pattern_quarter(&self, seed: u32, col: u32, row: u32) -> QuarterSpec {
        let index = row * self.cols + col;
        let bin = 2 * ((col / 4) % BANDS) + 1;
        let luma_unit = hashed_unit(seed + 50, index);
        let sigma_unit = hashed_unit(seed + 60, index);
        let luma = (bin as f32 + 0.25 + 0.5 * luma_unit) / NOISE_CURVE_BINS as f32;
        let sigma = (3.0 + 0.5 * bin as f32 + 2.0 * sigma_unit) / 255.0;
        let flat = QuarterSpec {
            mean: 0.0,
            sigma,
            luma,
            flatness: 0.0,
            luma_min: luma - 0.01,
            luma_max: luma + 0.01,
        };

        let quarter = match kind_at(col, row) {
            Kind::Flat => flat,
            Kind::Textured => QuarterSpec {
                flatness: 0.8 * sigma * sigma,
                ..flat
            },
            Kind::Clipped => QuarterSpec {
                luma_max: 0.995,
                ..flat
            },
            Kind::Motion => QuarterSpec {
                sigma: 4.0 * sigma,
                ..flat
            },
        };

        let pixels = self.quarter_pixels(col, row);
        let full_pixels = TEMPORAL_QUARTER_SIZE * TEMPORAL_QUARTER_SIZE;
        if pixels == full_pixels {
            return quarter;
        }

        // A quarter smaller than 8x8 reads the unmeasured flatness, as the stats kernel writes it.
        QuarterSpec {
            flatness: UNMEASURED_FLATNESS,
            ..quarter
        }
    }

    /// The block index and quarter index of the map quarter at `col`, `row`.
    fn block_quarter(&self, col: u32, row: u32) -> (usize, usize) {
        let (blocks_x, _) = temporal_stats_blocks(self.plane.width, self.plane.height);
        let block = (row / 2) * blocks_x + col / 2;
        let quarter = (row % 2) * 2 + col % 2;

        (block as usize, quarter as usize)
    }

    fn quarter_pixels(&self, col: u32, row: u32) -> u32 {
        let block_x = col / 2;
        let block_y = row / 2;
        let block_width = TEMPORAL_NOISE_BLOCK.min(self.plane.width - block_x * TEMPORAL_NOISE_BLOCK);
        let block_height = TEMPORAL_NOISE_BLOCK.min(self.plane.height - block_y * TEMPORAL_NOISE_BLOCK);
        let offset_x = (col % 2) * TEMPORAL_QUARTER_SIZE;
        let offset_y = (row % 2) * TEMPORAL_QUARTER_SIZE;
        let quarter_width = block_width.saturating_sub(offset_x).min(TEMPORAL_QUARTER_SIZE);
        let quarter_height = block_height.saturating_sub(offset_y).min(TEMPORAL_QUARTER_SIZE);

        quarter_width * quarter_height
    }

    fn quarter_mut(&mut self, col: u32, row: u32) -> &mut QuarterSpec {
        let (block, quarter) = self.block_quarter(col, row);
        &mut self.blocks[block].quarters[quarter]
    }

    fn set_tensor(&mut self, col: u32, row: u32, tensor: [f32; 3]) {
        let index = (row * self.cols + col) as usize;
        self.tensors[index] = tensor;
    }

    /// Turns the quarter at `col`, `row` into a strong line with a wide luma range.
    fn place_line(&mut self, col: u32, row: u32) {
        let quarter = self.quarter_mut(col, row);
        quarter.flatness = 1.0;
        quarter.luma_min = quarter.luma - 0.08;
        quarter.luma_max = quarter.luma + 0.08;

        let trace = 49.0 * 0.25;
        self.set_tensor(col, row, [trace, 0.0, 0.0]);
    }

    /// The first record word of the map quarter at `col`, `row`.
    fn field_start(&self, col: u32, row: u32) -> usize {
        let stored_ch = self.plane.stored_ch;
        let record_len = temporal_stats_record_len(stored_ch) as usize;
        let quarters_base = (2 * stored_ch + TEMPORAL_QUARTER_BASE) as usize;
        let (block, quarter) = self.block_quarter(col, row);
        let quarter_start = quarter * TEMPORAL_QUARTER_FIELDS as usize;

        block * record_len + quarters_base + quarter_start
    }

    fn records(&self) -> Vec<f32> {
        let plane = self.plane;
        let mut records = frame_records(plane.width, plane.height, plane.stored_ch, &self.blocks);

        for row in 0..self.rows {
            for col in 0..self.cols {
                let start = self.field_start(col, row);
                let [xx, yy, xy] = self.tensors[(row * self.cols + col) as usize];

                records[start + QUARTER_TENSOR_XX as usize] = xx;
                records[start + QUARTER_TENSOR_YY as usize] = yy;
                records[start + QUARTER_TENSOR_XY as usize] = xy;
            }
        }

        records
    }

    fn shape(&self, client: &Client) -> NoiseShape {
        NoiseShape {
            width: self.plane.width,
            height: self.plane.height,
            channels: self.plane.channels,
            stored_ch: self.plane.stored_ch,
            frames: 3,
            temporal: true,
            align: StorageAlign::from_client(client),
        }
    }
}

fn map_settings(map: StrengthMapKind, quarter: QuarterSettings) -> ReadingSettings {
    ReadingSettings {
        luma_fields: true,
        quarter,
        map,
    }
}

fn luma_settings(quarter: QuarterSettings) -> ReadingSettings {
    let map = StrengthMapKind::Luma(LUMA_PARAMS);
    map_settings(map, quarter)
}

/// The GPU's map and quarter flags of one slot.
struct GpuMap {
    map: Vec<f32>,
    flags: Vec<u32>,
    lumas: Vec<f32>,
    launches: u32,
}

/// Runs `frame` through a poisoned state's slot 0 reading.
fn gpu_map(client: &Client, frame: &MapFrame, settings: ReadingSettings) -> GpuMap {
    let shape = frame.shape(client);
    let records = frame.records();
    let partials = hashed_partials(frame.plane.width, frame.plane.height, 7);
    let mut state =
        NoiseState::from_stats_for_test(client, shape, settings, &partials, RESULTS_ROW, &records);
    state.poison_for_test();

    let launches = state.queue_reading(0).expect("the reading queues");
    let map = state.read_slot_map(0);
    let flags = state.read_quarter_flags();
    let lumas = state.read_quarter_lumas();

    GpuMap {
        map,
        flags,
        lumas,
        launches,
    }
}

fn host_classes(frame: &MapFrame, settings: ReadingSettings) -> Option<QuarterClasses> {
    let plane = frame.plane;
    let records = frame.records();
    let reading = temporal_noise_reading(
        &records,
        plane.channels,
        plane.stored_ch,
        plane.width,
        plane.height,
        true,
        settings.quarter,
    );

    reading.classes
}

/// The host's multipliers for `classes`, or ones without classes.
fn host_map(frame: &MapFrame, classes: Option<&QuarterClasses>, map: StrengthMapKind) -> Vec<f32> {
    let quarters = (frame.cols * frame.rows) as usize;
    let Some(classes) = classes else {
        return vec![1.0; quarters];
    };

    match map {
        StrengthMapKind::Luma(params) => classes.luma_multipliers(params),
        StrengthMapKind::Chroma { flat_boost } => classes.chroma_multipliers(flat_boost),
        StrengthMapKind::Off => vec![1.0; quarters],
    }
}

/// Checks every multiplier is within [MAX_ULPS] of the host's and prints the largest gap.
fn assert_map_close(gpu: &[f32], host: &[f32], what: &str) {
    assert_eq!(gpu.len(), host.len(), "{what}: map length");

    let mut largest = 0;
    for (index, (&gpu_value, &host_value)) in gpu.iter().zip(host).enumerate() {
        let gap = ulp_gap(gpu_value, host_value);
        assert!(
            gap <= MAX_ULPS,
            "{what}: quarter {index} gpu {gpu_value} host {host_value}, {gap} ulp apart"
        );
        largest = largest.max(gap);
    }

    println!("{what}: largest multiplier gap {largest} ulp");
}

fn has_flag(flags: u32, flag: u32) -> bool {
    flags & flag != 0
}

/// Runs `frame` on the GPU and the host and checks the maps agree.
fn check_map(frame: &MapFrame, settings: ReadingSettings, what: &str) -> (GpuMap, Vec<f32>) {
    let client = test_client();
    let gpu = gpu_map(&client, frame, settings);
    let classes = host_classes(frame, settings);
    let has_classes = classes.is_some();
    let host = host_map(frame, classes.as_ref(), settings.map);

    assert!(has_classes, "{what}: the host has classes");
    assert_eq!(gpu.launches, MAP_LAUNCHES, "{what}: launches");
    assert_map_close(&gpu.map, &host, what);

    (gpu, host)
}

#[test]
fn the_luma_map_matches_the_host_multipliers() {
    for seed in [3, 5] {
        let plane = Plane::luma(256, 192);
        let frame = MapFrame::patterned(plane, seed);
        let settings = luma_settings(QuarterSettings::default());
        let (gpu, host) = check_map(&frame, settings, "luma");

        let flat_boost = LUMA_PARAMS.flat_boost;
        for (index, &flags) in gpu.flags.iter().enumerate() {
            let gpu_flat = has_flag(flags, CLASS_FLAT);
            let host_flat = host[index] == flat_boost;
            let present = has_flag(flags, CLASS_PRESENT);
            assert_eq!(gpu_flat, host_flat, "flat flag of quarter {index}");
            assert!(present, "quarter {index} is present");

            let col = index as u32 % frame.cols;
            let row = index as u32 / frame.cols;
            let kind = kind_at(col, row);
            let expected_flat = kind == Kind::Flat;
            assert_eq!(gpu_flat, expected_flat, "quarter {index} is {kind:?}");
        }

        // Flat quarters, then dark, mid and bright quarters that are not flat.
        let mut counts = [0usize; 4];
        for &multiplier in &host {
            let bucket = if multiplier == flat_boost {
                0
            } else if multiplier == LUMA_PARAMS.shadow_soften {
                1
            } else if multiplier == 1.0 {
                3
            } else {
                2
            };
            counts[bucket] += 1;
        }

        let [flat, dark, mid, bright] = counts;
        assert!(flat > 0 && dark > 0 && mid > 0 && bright > 0, "{counts:?}");
    }
}

#[test]
fn the_chroma_map_boosts_only_flat_quarters() {
    let plane = Plane::chroma(256, 192);
    let frame = MapFrame::patterned(plane, 7);
    let map = StrengthMapKind::Chroma {
        flat_boost: CHROMA_BOOST,
    };
    let settings = map_settings(map, QuarterSettings::default());
    let (gpu, host) = check_map(&frame, settings, "chroma");

    assert_eq!(gpu.map, host);

    let mut boosted = 0;
    for (index, &multiplier) in gpu.map.iter().enumerate() {
        let flat = has_flag(gpu.flags[index], CLASS_FLAT);
        let expected = if flat { CHROMA_BOOST } else { 1.0 };
        assert_eq!(multiplier, expected, "quarter {index}");
        boosted += usize::from(flat);
    }

    assert!(boosted > 0);
}

/// A frame whose left half holds one-directional tensors and whose right half holds isotropic ones,
/// with a line in the left half.
fn veto_frame() -> MapFrame {
    let plane = Plane::luma(256, 192);
    let mut frame = MapFrame::patterned(plane, 11);

    for row in 0..frame.rows {
        for col in 0..frame.cols {
            let oriented = col < frame.cols / 2;
            let tensor = if oriented {
                [TENSOR, 0.0, 0.0]
            } else {
                [TENSOR, TENSOR, 0.0]
            };
            frame.set_tensor(col, row, tensor);
        }
    }

    frame.place_line(5, 12);
    frame
}

#[test]
fn the_texture_cut_vetoes_like_the_host() {
    let frame = veto_frame();
    let quarter = QuarterSettings {
        texture_cut: Some(0.21),
        line_ring: Some(1),
    };
    let settings = luma_settings(quarter);
    let (gpu, host) = check_map(&frame, settings, "veto");

    let flat_boost = LUMA_PARAMS.flat_boost;
    let mut vetoed = 0;
    let mut kept = 0;
    let mut exempt = 0;
    for (index, &flags) in gpu.flags.iter().enumerate() {
        if !has_flag(flags, CLASS_FLAT) {
            continue;
        }

        let col = index as u32 % frame.cols;
        let row = index as u32 / frame.cols;
        let boosted = host[index] == flat_boost;
        let near_line = col.abs_diff(5) <= 1 && row.abs_diff(12) <= 1;
        let inside_oriented = col + 1 < frame.cols / 2;
        let inside_isotropic = col > frame.cols / 2;

        if near_line {
            assert!(boosted, "quarter {index} lies in the ring");
            exempt += 1;
        } else if inside_oriented {
            assert!(!boosted, "quarter {index} lies in the oriented half");
            vetoed += 1;
        } else if inside_isotropic {
            assert!(boosted, "quarter {index} lies in the isotropic half");
            kept += 1;
        }
    }

    assert!(vetoed > 0 && kept > 0 && exempt > 0, "{vetoed} {kept} {exempt}");
}

#[test]
fn the_texture_cut_pools_only_present_quarters() {
    // 264x200 leaves whole 8x8 quarters beside the absent last column and row.
    let plane = Plane::luma(264, 200);
    let mut frame = MapFrame::patterned(plane, 43);
    let last_col = frame.cols - 1;
    let last_row = frame.rows - 1;

    // Present quarters are isotropic, and the absent ones carry a strong one-directional tensor.
    for row in 0..frame.rows {
        for col in 0..frame.cols {
            let absent = col == last_col || row == last_row;
            let tensor = if absent {
                [1.0, 0.0, 0.0]
            } else {
                [TENSOR, TENSOR, 0.0]
            };
            frame.set_tensor(col, row, tensor);
        }
    }

    let quarter = QuarterSettings {
        texture_cut: Some(0.21),
        line_ring: None,
    };
    let settings = luma_settings(quarter);
    let (gpu, host) = check_map(&frame, settings, "present pool");
    let records = frame.records();

    let flat_boost = LUMA_PARAMS.flat_boost;
    let mut edge_flats = 0;
    for (index, &flags) in gpu.flags.iter().enumerate() {
        let col = index as u32 % frame.cols;
        let row = index as u32 / frame.cols;
        let pixels = frame.quarter_pixels(col, row);
        if pixels == 0 {
            continue;
        }

        let start = frame.field_start(col, row);
        let host_luma = records[start + QUARTER_LUMA_SUM as usize] / pixels as f32;
        assert_eq!(gpu.lumas[index], host_luma, "luma of quarter {index}");

        let beside_absent = col + 1 == last_col || row + 1 == last_row;
        let flat = has_flag(flags, CLASS_FLAT);
        if beside_absent && flat {
            assert_eq!(host[index], flat_boost, "quarter {index} beside the edge");
            edge_flats += 1;
        }
    }

    assert!(edge_flats > 0);
}

#[test]
fn a_map_without_luma_fields_queues_no_classes() {
    let client = test_client();
    let frame = MapFrame::patterned(Plane::luma(256, 192), 47);
    let settings = ReadingSettings {
        luma_fields: false,
        ..luma_settings(QuarterSettings::default())
    };

    // The poisoned map keeps its garbage, since nothing classifies without the curve.
    let gpu = gpu_map(&client, &frame, settings);
    let untouched = gpu.map.iter().all(|&multiplier| multiplier == 7.5);
    assert_eq!(gpu.launches, SAMPLE_LAUNCHES);
    assert!(untouched);

    let shape = frame.shape(&client);
    let records = frame.records();
    let partials = hashed_partials(256, 192, 7);
    let state = NoiseState::from_stats_for_test(&client, shape, settings, &partials, RESULTS_ROW, &records);
    let launches = state.queue_reading(0).expect("the reading queues");
    let map = state.read_slot_map(0);
    let all_ones = map.iter().all(|&multiplier| multiplier == 1.0);
    assert_eq!(launches, SAMPLE_LAUNCHES);
    assert!(all_ones);
}

#[test]
fn the_line_ring_matches_the_host_dilation() {
    let plane = Plane::luma(256, 192);
    let mut frame = MapFrame::patterned(plane, 13);
    let corner = (0, 0);
    let edge = (17, frame.rows - 1);
    frame.place_line(corner.0, corner.1);
    frame.place_line(edge.0, edge.1);

    let quarter = QuarterSettings {
        texture_cut: None,
        line_ring: Some(2),
    };
    let settings = luma_settings(quarter);
    let (gpu, host) = check_map(&frame, settings, "ring");

    let unringed_settings = luma_settings(QuarterSettings::default());
    let unringed_classes = host_classes(&frame, unringed_settings);
    let unringed = host_map(&frame, unringed_classes.as_ref(), unringed_settings.map);

    let mut lifted = 0;
    for (index, &flags) in gpu.flags.iter().enumerate() {
        let col = index as u32 % frame.cols;
        let row = index as u32 / frame.cols;
        let line = has_flag(flags, CLASS_LINE);
        let placed = (col, row) == corner || (col, row) == edge;
        assert_eq!(line, placed, "line flag of quarter {index}");

        let near_corner = col.abs_diff(corner.0) <= 2 && row.abs_diff(corner.1) <= 2;
        let near_edge = col.abs_diff(edge.0) <= 2 && row.abs_diff(edge.1) <= 2;
        let in_ring = near_corner || near_edge;
        if !in_ring {
            assert_eq!(
                host[index], unringed[index],
                "quarter {index} lies outside the ring"
            );
            continue;
        }

        let flat = has_flag(flags, CLASS_FLAT);
        let expected = if flat { LUMA_PARAMS.flat_boost } else { 1.0 };
        assert_eq!(host[index], expected, "quarter {index} lies in the ring");
        lifted += usize::from(unringed[index] != expected);
    }

    assert!(lifted > 0);
}

#[test]
fn a_quarter_past_the_frame_edge_gets_one() {
    let client = test_client();
    let settings = luma_settings(QuarterSettings::default());

    // 70x54 holds too few quarters for a curve, so every quarter gets one.
    let small = MapFrame::patterned(Plane::luma(70, 54), 17);
    let small_classes = host_classes(&small, settings);
    let small_gpu = gpu_map(&client, &small, settings);
    let small_has_classes = small_classes.is_some();
    let all_ones = small_gpu.map.iter().all(|&multiplier| multiplier == 1.0);
    assert!(!small_has_classes);
    assert!(all_ones);
    assert_eq!(small_gpu.map.len(), 80);

    // A ragged frame large enough for a curve, whose 6x8 quarters divide by 48.
    let frame = MapFrame::patterned(Plane::luma(262, 198), 19);
    let gpu = gpu_map(&client, &frame, settings);
    let classes = host_classes(&frame, settings);
    let has_classes = classes.is_some();
    let host = host_map(&frame, classes.as_ref(), settings.map);
    assert!(has_classes);
    assert_eq!(gpu.launches, MAP_LAUNCHES);

    let records = frame.records();
    let flat_boost = LUMA_PARAMS.flat_boost;
    let mut past_edge = 0;
    let mut ragged = 0;
    let mut largest_luma_gap = 0;
    let mut gpu_classes = Vec::with_capacity(gpu.map.len());
    for (index, &multiplier) in gpu.map.iter().enumerate() {
        let col = index as u32 % frame.cols;
        let row = index as u32 / frame.cols;
        let pixels = frame.quarter_pixels(col, row);
        let flags = gpu.flags[index];
        let present = has_flag(flags, CLASS_PRESENT);
        assert_eq!(present, pixels > 0, "quarter {index}, {pixels} pixels");

        if pixels == 0 {
            assert_eq!(multiplier, 1.0, "quarter {index} lies past the edge");
            gpu_classes.push(None);
            past_edge += 1;
            continue;
        }

        let start = frame.field_start(col, row);
        let host_luma = records[start + QUARTER_LUMA_SUM as usize] / pixels as f32;
        let gpu_luma = gpu.lumas[index];
        let luma_gap = ulp_gap(gpu_luma, host_luma);
        let exact = pixels.is_power_of_two();
        if !exact {
            ragged += 1;
        }

        let luma_limit = if exact { 0 } else { MAX_ULPS };
        assert!(
            luma_gap <= luma_limit,
            "luma of quarter {index}, {pixels} pixels: gpu {gpu_luma} host {host_luma}, {luma_gap} ulp apart"
        );
        largest_luma_gap = largest_luma_gap.max(luma_gap);

        let flat = has_flag(flags, CLASS_FLAT);
        let host_flat = host[index] == flat_boost;
        assert_eq!(flat, host_flat, "flat flag of quarter {index}");

        let class = QuarterClass { flat, luma: gpu_luma };
        gpu_classes.push(Some(class));
    }

    println!("ragged: largest luma gap {largest_luma_gap} ulp");
    assert_eq!(past_edge, frame.cols + frame.rows - 1);
    assert!(ragged > 0);

    // The host formula at the GPU's own lumas, so the map is checked apart from the luma division.
    let cols = frame.cols as usize;
    let rows = frame.rows as usize;
    let at_gpu_luma = QuarterClasses::from_classes(cols, rows, gpu_classes);
    let expected = at_gpu_luma.luma_multipliers(LUMA_PARAMS);
    assert_map_close(&gpu.map, &expected, "ragged");
}

#[test]
fn a_slot_without_a_curve_gets_a_map_of_ones() {
    let client = test_client();
    let plane = Plane::luma(256, 192);
    let settings = luma_settings(QuarterSettings::default());

    // Only the first two bands stay flat, so two bins populate and no curve forms.
    let mut flat_less = MapFrame::patterned(plane, 23);
    for row in 0..flat_less.rows {
        for col in 0..flat_less.cols {
            let band = (col / 4) % BANDS;
            if band < 2 {
                continue;
            }

            let quarter = flat_less.quarter_mut(col, row);
            quarter.flatness = 1.0;
        }
    }

    let curved = MapFrame::patterned(plane, 29);
    let flat_less_records = flat_less.records();
    let curved_records = curved.records();
    let flat_less_partials = hashed_partials(256, 192, 31);
    let curved_partials = hashed_partials(256, 192, 37);
    let slots = [
        SlotStats {
            partials: &flat_less_partials,
            results_row: RESULTS_ROW,
            records: &flat_less_records,
        },
        SlotStats {
            partials: &curved_partials,
            results_row: RESULTS_ROW,
            records: &curved_records,
        },
    ];
    let shape = flat_less.shape(&client);
    let mut state = NoiseState::from_slots_for_test(&client, shape, settings, &slots);
    state.poison_for_test();

    state.queue_reading(1).expect("the curved reading queues");
    state.queue_reading(0).expect("the flat-less reading queues");

    let flat_less_reading = state.read_reading(0);
    let flat_less_map = state.read_slot_map(0);
    let curved_map = state.read_slot_map(1);
    let has_sample = flat_less_reading.sample.is_some();
    let has_curve = flat_less_reading.curve.is_some();
    let host_has_classes = host_classes(&flat_less, settings).is_some();
    let all_ones = flat_less_map.iter().all(|&multiplier| multiplier == 1.0);
    assert!(has_sample);
    assert!(!has_curve);
    assert!(!host_has_classes);
    assert!(all_ones);

    let curved_classes = host_classes(&curved, settings);
    let curved_host = host_map(&curved, curved_classes.as_ref(), settings.map);
    assert_map_close(&curved_map, &curved_host, "second slot");
}

#[test]
fn a_4k_luma_plane_classifies_within_the_grid_limit() {
    let client = test_client();
    let frame = MapFrame::patterned(Plane::luma(3840, 2160), 41);
    let shape = frame.shape(&client);
    let grid = quarter_grid(shape);
    assert!(grid <= MAX_GRID_1D, "{grid} cubes exceed the 1D grid limit");

    let quarter = QuarterSettings {
        texture_cut: Some(0.21),
        line_ring: Some(2),
    };
    let settings = luma_settings(quarter);
    let (gpu, _) = check_map(&frame, settings, "4k");
    assert_eq!(gpu.map.len(), 480 * 270);
}

#[test]
fn the_map_is_identical_across_runs() {
    let client = test_client();
    let frame = veto_frame();
    let quarter = QuarterSettings {
        texture_cut: Some(0.21),
        line_ring: Some(2),
    };
    let settings = luma_settings(quarter);

    let first = gpu_map(&client, &frame, settings);
    let second = gpu_map(&client, &frame, settings);
    let first_bits: Vec<u32> = first.map.iter().map(|value| value.to_bits()).collect();
    let second_bits: Vec<u32> = second.map.iter().map(|value| value.to_bits()).collect();

    assert_eq!(first_bits, second_bits);
    assert_eq!(first.flags, second.flags);
}
