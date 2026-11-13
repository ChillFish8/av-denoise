use std::collections::HashMap;
use std::f64::consts::PI;

use crate::nl4d::grain::consts::AR_COEFFS;
use crate::nl4d::grain::seeds::FGSE_SEEDS;
use crate::nl4d::grain::template::{LUMA_TEMPLATE_COLS, luma_template};

const FIELD_SIZE: usize = 512;
const BLOCK_SIZE: usize = 32;
const BLOCKS_PER_SIDE: usize = FIELD_SIZE / BLOCK_SIZE;
const MAX_LAG: usize = 200;
const MIN_LAG_DISTANCE: f64 = 12.0;
const TAIL_QUANTILE: f64 = 0.999;

/// The template offsets an AV1 decoder picks for the blocks of one row of 32x32 blocks.
pub(crate) fn block_offsets(seed: u16, row: u32, blocks: usize) -> Vec<(usize, usize)> {
    let mut state = seed as u32;
    state ^= ((row * 37 + 178) & 255) << 8;
    state ^= (row * 173 + 105) & 255;

    let mut offsets = Vec::with_capacity(blocks);
    for _ in 0..blocks {
        let bit = (state ^ (state >> 1) ^ (state >> 3) ^ (state >> 12)) & 1;
        state = ((state >> 1) | (bit << 15)) & 0xFFFF;

        let random = (state >> 8) & 255;
        let x = 9 + (random >> 4) * 2;
        let y = 9 + (random & 15) * 2;
        offsets.push((x as usize, y as usize));
    }

    offsets
}

/// An in-place radix-2 FFT. The inverse divides by the length.
pub(crate) fn fft(real: &mut [f64], imaginary: &mut [f64], inverse: bool) {
    let length = real.len();
    debug_assert!(length.is_power_of_two());
    debug_assert_eq!(length, imaginary.len());

    let mut target = 0;
    for source in 1..length {
        let mut bit = length >> 1;
        while target & bit != 0 {
            target ^= bit;
            bit >>= 1;
        }
        target |= bit;

        if source < target {
            real.swap(source, target);
            imaginary.swap(source, target);
        }
    }

    let direction = if inverse { 1.0 } else { -1.0 };
    let mut half = 1;
    while half < length {
        let angle = direction * PI / half as f64;
        let twiddles: Vec<(f64, f64)> = (0..half)
            .map(|index| {
                let phase = angle * index as f64;
                (phase.cos(), phase.sin())
            })
            .collect();

        for start in (0..length).step_by(half * 2) {
            for (index, &(cos, sin)) in twiddles.iter().enumerate() {
                let even = start + index;
                let odd = even + half;
                let odd_real = real[odd] * cos - imaginary[odd] * sin;
                let odd_imaginary = real[odd] * sin + imaginary[odd] * cos;

                real[odd] = real[even] - odd_real;
                imaginary[odd] = imaginary[even] - odd_imaginary;
                real[even] += odd_real;
                imaginary[even] += odd_imaginary;
            }
        }

        half *= 2;
    }

    if inverse {
        let scale = 1.0 / length as f64;
        real.iter_mut().for_each(|value| *value *= scale);
        imaginary.iter_mut().for_each(|value| *value *= scale);
    }
}

fn fft_2d(real: &mut [f64], imaginary: &mut [f64], inverse: bool) {
    for row in 0..FIELD_SIZE {
        let span = row * FIELD_SIZE..(row + 1) * FIELD_SIZE;
        fft(&mut real[span.clone()], &mut imaginary[span], inverse);
    }

    let mut real_column = vec![0.0; FIELD_SIZE];
    let mut imaginary_column = vec![0.0; FIELD_SIZE];
    for column in 0..FIELD_SIZE {
        for row in 0..FIELD_SIZE {
            real_column[row] = real[row * FIELD_SIZE + column];
            imaginary_column[row] = imaginary[row * FIELD_SIZE + column];
        }

        fft(&mut real_column, &mut imaginary_column, inverse);

        for row in 0..FIELD_SIZE {
            real[row * FIELD_SIZE + column] = real_column[row];
            imaginary[row * FIELD_SIZE + column] = imaginary_column[row];
        }
    }
}

fn wrapped_offset(index: usize) -> i64 {
    if index < FIELD_SIZE / 2 {
        index as i64
    } else {
        index as i64 - FIELD_SIZE as i64
    }
}

/// The autocorrelation indices that lie far enough from zero lag to count as a repeat.
pub(crate) fn kept_lags() -> Vec<usize> {
    let mut kept = Vec::new();
    for row in 0..FIELD_SIZE {
        let dy = wrapped_offset(row);
        for column in 0..FIELD_SIZE {
            let dx = wrapped_offset(column);
            let distance = ((dy * dy + dx * dx) as f64).sqrt();
            let in_range = dy.unsigned_abs() < MAX_LAG as u64 && dx.unsigned_abs() < MAX_LAG as u64;
            if in_range && distance > MIN_LAG_DISTANCE {
                kept.push(row * FIELD_SIZE + column);
            }
        }
    }

    kept
}

fn build_field(template: &[i32], seed: u16) -> Vec<f64> {
    let mut field = vec![0.0; FIELD_SIZE * FIELD_SIZE];
    for block_row in 0..BLOCKS_PER_SIDE {
        let offsets = block_offsets(seed, block_row as u32, BLOCKS_PER_SIDE);
        for (block_column, &(x, y)) in offsets.iter().enumerate() {
            for line in 0..BLOCK_SIZE {
                let source = (y + line) * LUMA_TEMPLATE_COLS + x;
                let target = (block_row * BLOCK_SIZE + line) * FIELD_SIZE + block_column * BLOCK_SIZE;
                for offset in 0..BLOCK_SIZE {
                    field[target + offset] = template[source + offset] as f64;
                }
            }
        }
    }

    field
}

/// How strongly a frame of grain built from one seed repeats itself.
///
/// A frame is made of 32x32 blocks cut from the seed's template. This is the largest
/// autocorrelation at the `kept` lags plus twice their 99.9th percentile. Lower is calmer.
pub(crate) fn repeat_score(coeffs: &[i32; AR_COEFFS], ar_shift: u32, seed: u16, kept: &[usize]) -> f64 {
    let template = luma_template(coeffs, ar_shift, seed);
    let mut real = build_field(&template, seed);
    let mean = real.iter().sum::<f64>() / real.len() as f64;
    real.iter_mut().for_each(|value| *value -= mean);
    let mut imaginary = vec![0.0; real.len()];

    fft_2d(&mut real, &mut imaginary, false);

    for (real_value, imaginary_value) in real.iter_mut().zip(imaginary.iter_mut()) {
        *real_value = *real_value * *real_value + *imaginary_value * *imaginary_value;
        *imaginary_value = 0.0;
    }

    fft_2d(&mut real, &mut imaginary, true);

    let zero_lag = real[0];
    let mut values: Vec<f64> = kept.iter().map(|&index| real[index] / zero_lag).collect();
    values.sort_by(f64::total_cmp);

    let tail_index = (TAIL_QUANTILE * (values.len() - 1) as f64) as usize;
    let largest = values[values.len() - 1];

    largest + 2.0 * values[tail_index]
}

/// Quantised AR weights and their shift.
type Weights = ([i32; AR_COEFFS], u32);

/// A cache of every listed seed's repeat score, kept per set of AR weights.
#[derive(Debug, Default)]
pub(crate) struct SeedScores {
    /// Each listed seed's score, in list order.
    scores: HashMap<Weights, Vec<f64>>,
}

impl SeedScores {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// The listed seed with the lowest repeat score that is not in `taken`.
    ///
    /// The earlier seed in the list wins a tie. Every seed is scored the first time a set of
    /// weights is asked for.
    pub(crate) fn calmest(&mut self, coeffs: &[i32; AR_COEFFS], ar_shift: u32, taken: &[u16]) -> Option<u16> {
        let all_taken = FGSE_SEEDS.iter().all(|seed| taken.contains(seed));
        if all_taken {
            return None;
        }

        let key = (*coeffs, ar_shift);
        let scores = self
            .scores
            .entry(key)
            .or_insert_with(|| list_scores(coeffs, ar_shift));

        let mut best: Option<(f64, u16)> = None;
        let candidates = FGSE_SEEDS
            .iter()
            .zip(scores.iter())
            .filter(|(seed, _)| !taken.contains(seed));
        for (&seed, &score) in candidates {
            if best.is_none_or(|(best_score, _)| score < best_score) {
                best = Some((score, seed));
            }
        }

        best.map(|(_, seed)| seed)
    }
}

/// The repeat score of every listed seed, in list order.
///
/// The seeds are scored in parallel across the available cores.
fn list_scores(coeffs: &[i32; AR_COEFFS], ar_shift: u32) -> Vec<f64> {
    let kept = kept_lags();
    let workers = std::thread::available_parallelism().map_or(1, usize::from);
    let chunk_size = FGSE_SEEDS.len().div_ceil(workers).max(1);
    let kept = kept.as_slice();

    std::thread::scope(|scope| {
        let handles: Vec<_> = FGSE_SEEDS
            .chunks(chunk_size)
            .map(|seeds| {
                scope.spawn(move || {
                    seeds
                        .iter()
                        .map(|&seed| repeat_score(coeffs, ar_shift, seed, kept))
                        .collect::<Vec<f64>>()
                })
            })
            .collect();

        let mut scores = Vec::with_capacity(FGSE_SEEDS.len());
        for handle in handles {
            let chunk_scores = handle.join().expect("a seed scoring worker panicked");
            scores.extend(chunk_scores);
        }

        scores
    })
}
