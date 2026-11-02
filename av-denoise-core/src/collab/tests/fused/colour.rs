use std::f32::consts::LN_2;

use super::noise_curve::{assert_columns_identical, assert_identical, columns_differ};
use super::{Aggregated, Setup, SetupColour, chroma_cross_frame_setup, run_fused};
use crate::collab::tests::helpers::noisy_flat_field;
use crate::nlmeans::ChannelMode;
use crate::noise::NOISE_CURVE_BINS;
use crate::noise::colour::{COLOUR_CELLS, COLOUR_CELLS_PER_BRIGHTNESS, COLOUR_MEAN_FIELDS};

/// The side of the square frame the luma tests filter.
const SIDE: u32 = 64;

/// Columns either side of the colour seam left out of each half's measurement.
const SEAM_MARGIN: u32 = 8;

/// A threshold low enough that scaling it still changes which noise coefficients survive.
const LAMBDA: f32 = 1.0;

/// The side of a block of the colour means grid.
const BLOCK: u32 = 16;

/// The Cb of the coloured half, far past the last saturation centre.
const COLOURED_CB: f32 = 60.0;

/// The block means of a `width` by `height` plane, luma `luma` throughout, with Cb `left_cb` in
/// the blocks left of the middle and `right_cb` in the rest.
fn split_means(width: u32, height: u32, luma: f32, left_cb: f32, right_cb: f32) -> Vec<f32> {
    let blocks_x = width.div_ceil(BLOCK);
    let blocks_y = height.div_ceil(BLOCK);
    let mut means = Vec::with_capacity((blocks_x * blocks_y * COLOUR_MEAN_FIELDS) as usize);
    for _ in 0..blocks_y {
        for block_x in 0..blocks_x {
            let left = block_x < blocks_x / 2;
            let cb = if left { left_cb } else { right_cb };
            means.extend_from_slice(&[luma, cb, 0.0]);
        }
    }

    means
}

/// `planes` log gain tables, every cell at `log_gain`.
fn uniform_tables(planes: usize, log_gain: f32) -> Vec<f32> {
    vec![log_gain; planes * COLOUR_CELLS as usize]
}

/// A luma table at log 0 in every neutral cell and `log_gain` in every coloured cell.
fn coloured_table(log_gain: f32) -> Vec<f32> {
    let per_brightness = COLOUR_CELLS_PER_BRIGHTNESS as usize;

    (0..COLOUR_CELLS as usize)
        .map(|cell| {
            let neutral = cell % per_brightness == 0;
            if neutral { 0.0 } else { log_gain }
        })
        .collect()
}

/// A noisy flat luma frame at 0.4.
fn luma_frame() -> Vec<f32> {
    noisy_flat_field(SIDE, SIDE, 0.4, 0.02)
}

fn run_luma(
    lambda_ht: f32,
    curve: Option<[f32; NOISE_CURVE_BINS]>,
    colour: Option<SetupColour>,
) -> Aggregated {
    let mut setup = Setup::spatial_only(luma_frame(), SIDE, SIDE);
    setup.lambda_ht = lambda_ht;
    setup.noise_curve = curve;
    setup.colour = colour;

    run_fused(&setup)
}

fn luma_colour(log_gains: Vec<f32>, left_cb: f32, right_cb: f32) -> SetupColour {
    SetupColour {
        log_gains,
        means: split_means(SIDE, SIDE, 102.0, left_cb, right_cb),
        scale: 255.0,
    }
}

#[test]
fn luma_threshold_scales_by_the_colour_gain() {
    let table = coloured_table(LN_2);
    let colour = luma_colour(table, COLOURED_CB, 0.0);
    let gained = run_luma(LAMBDA, None, Some(colour));

    let plain = run_luma(LAMBDA, None, None);
    let doubled = run_luma(2.0 * LAMBDA, None, None);

    let half = SIDE / 2;
    let left_end = half - SEAM_MARGIN;
    let right_start = half + SEAM_MARGIN;
    assert_columns_identical("coloured half", &gained, &doubled, SIDE, 0, left_end);
    assert_columns_identical("neutral half", &gained, &plain, SIDE, right_start, SIDE);

    let changed = columns_differ(&doubled, &plain, SIDE, 0, left_end);
    assert!(changed, "doubling lambda changed nothing in the coloured half");
}

#[test]
fn a_luma_gain_multiplies_the_curve_ratio() {
    let curve = Some([1.5; NOISE_CURVE_BINS]);
    let table = uniform_tables(1, LN_2);
    let colour = luma_colour(table, 0.0, 0.0);
    let gained = run_luma(LAMBDA, curve, Some(colour));

    let scaled = run_luma(3.0 * LAMBDA, None, None);

    assert_identical("curve 1.5 by gain 2", &gained, &scaled);
}

#[test]
fn the_luma_curve_and_gain_clamp_at_six() {
    let curve = Some([3.0; NOISE_CURVE_BINS]);
    let table = uniform_tables(1, 3.0f32.ln());
    let colour = luma_colour(table, 0.0, 0.0);
    let gained = run_luma(LAMBDA, curve, Some(colour));

    let ceiling = run_luma(6.0 * LAMBDA, None, None);
    let old_ceiling = run_luma(3.0 * LAMBDA, None, None);

    assert_identical("curve 3 by gain 3", &gained, &ceiling);
    assert_ne!(
        ceiling.accum, old_ceiling.accum,
        "the wider ceiling changed nothing"
    );
}

#[test]
fn unit_gains_change_nothing() {
    let table = uniform_tables(1, 0.0);
    let colour = luma_colour(table, COLOURED_CB, 0.0);
    let gained = run_luma(LAMBDA, None, Some(colour));

    let plain = run_luma(LAMBDA, None, None);

    assert_identical("unit gains", &gained, &plain);
}

/// Channel `channel`'s accumulator values, every region of the ring.
fn channel_accum(got: &Aggregated, channel: usize) -> Vec<i32> {
    got.accum.iter().skip(channel).step_by(2).copied().collect()
}

fn run_chroma(lambda_ht: f32, log_gains: Option<[f32; 2]>) -> Aggregated {
    let mut setup = chroma_cross_frame_setup(SIDE, SIDE, 2);
    setup.lambda_ht = lambda_ht;
    setup.colour = log_gains.map(|[cb_gain, cr_gain]| {
        let mut tables = uniform_tables(1, cb_gain);
        let cr_table = uniform_tables(1, cr_gain);
        tables.extend(cr_table);

        SetupColour {
            log_gains: tables,
            means: split_means(SIDE, SIDE, 120.0, 0.0, 0.0),
            scale: 255.0,
        }
    });

    run_fused(&setup)
}

#[test]
fn each_chroma_channel_reads_its_own_gain() {
    let plain = run_chroma(LAMBDA, None);
    let doubled = run_chroma(2.0 * LAMBDA, None);

    let both = run_chroma(LAMBDA, Some([LN_2, LN_2]));
    assert_identical("both channels at gain 2", &both, &doubled);

    let cb_only = run_chroma(LAMBDA, Some([LN_2, 0.0]));
    assert_eq!(
        channel_accum(&cb_only, 0),
        channel_accum(&doubled, 0),
        "Cb at gain 2"
    );
    assert_eq!(cb_only.wsum, doubled.wsum, "weight sums with Cb at gain 2");
    assert_ne!(
        channel_accum(&cb_only, 1),
        channel_accum(&doubled, 1),
        "Cr read Cb's gain"
    );

    let cr_only = run_chroma(LAMBDA, Some([0.0, LN_2]));
    assert_eq!(
        channel_accum(&cr_only, 0),
        channel_accum(&plain, 0),
        "Cb at gain 1"
    );
    assert_eq!(cr_only.wsum, plain.wsum, "weight sums with Cb at gain 1");
    assert_ne!(
        channel_accum(&cr_only, 1),
        channel_accum(&plain, 1),
        "Cr ignored its own gain"
    );
}

/// A noisy two-channel chroma frame, Cb 0.75 and Cr 0.5 left of the middle and neutral at 0.5 and
/// 0.5 to the right.
fn half_coloured_chroma() -> Vec<f32> {
    let noise = noisy_flat_field(2 * SIDE, SIDE, 0.0, 0.02);
    let mut frame = Vec::with_capacity(noise.len());
    for (index, sample) in noise.iter().enumerate() {
        let pixel = index as u32 / 2;
        let x = pixel % SIDE;
        let cb_channel = index % 2 == 0;
        let base = if x < SIDE / 2 && cb_channel { 0.75 } else { 0.5 };
        frame.push(base + sample);
    }

    frame
}

fn run_half_coloured_chroma(lambda_ht: f32, colour: Option<SetupColour>) -> Aggregated {
    let placeholder = vec![0.0f32; (SIDE * SIDE) as usize];
    let mut setup = Setup::spatial_only(placeholder, SIDE, SIDE);
    setup.ring = half_coloured_chroma();
    setup.channel_mode = ChannelMode::Chroma;
    setup.lambda_ht = lambda_ht;
    setup.colour = colour;

    run_fused(&setup)
}

/// Asserts both channels of every pixel in columns `x_start..x_end` match exactly.
fn assert_chroma_columns_identical(
    label: &str,
    got: &Aggregated,
    want: &Aggregated,
    x_start: u32,
    x_end: u32,
) {
    for y in 0..SIDE {
        for x in x_start..x_end {
            let pixel = (y * SIDE + x) as usize;
            for channel in 0..2 {
                let index = 2 * pixel + channel;
                assert_eq!(
                    got.accum[index], want.accum[index],
                    "{label}: channel {channel} differs at ({x}, {y})",
                );
            }

            assert_eq!(
                got.wsum[pixel], want.wsum[pixel],
                "{label}: weight sum differs at ({x}, {y})",
            );
        }
    }
}

#[test]
fn chroma_gains_follow_the_patch_colour() {
    // The block means stay neutral, so only the patch's own chroma can place it in a coloured cell.
    let mut tables = coloured_table(LN_2);
    let cr_table = coloured_table(LN_2);
    tables.extend(cr_table);
    let colour = SetupColour {
        log_gains: tables,
        means: split_means(SIDE, SIDE, 120.0, 0.0, 0.0),
        scale: 255.0,
    };
    let gained = run_half_coloured_chroma(LAMBDA, Some(colour));

    let plain = run_half_coloured_chroma(LAMBDA, None);
    let doubled = run_half_coloured_chroma(2.0 * LAMBDA, None);

    let half = SIDE / 2;
    let left_end = half - SEAM_MARGIN;
    let right_start = half + SEAM_MARGIN;
    assert_chroma_columns_identical("coloured half", &gained, &doubled, 0, left_end);
    assert_chroma_columns_identical("neutral half", &gained, &plain, right_start, SIDE);
    assert_ne!(doubled.accum, plain.accum, "doubling lambda changed nothing");
}
