use cubecl::prelude::*;

use super::helpers::{correlated_noisy_frame, noisy_field_over};
use crate::engine::{DevicePlane, SampleFormat};
use crate::motion::{MotionCompensationMode, PairModelProducer, SharedMotion};
use crate::tune::tests::fixtures::upload;

/// Model ring slots, enough for every test front's look-ahead and window.
const RING_DEPTH: u32 = 32;

/// The sigma of the texture [pan_clip] slides.
const TEXTURE_SIGMA: f32 = 24.0 / 255.0;

/// Fits the camera models of a luma stream into a ring that seeded fronts read.
///
/// It numbers frames from 0 as a stream starts, as the engine does.
pub(crate) struct ModelFitter {
    client: Client,
    producer: PairModelProducer,
    width: u32,
    height: u32,
    next_frame: u32,
}

impl ModelFitter {
    pub(crate) fn new(client: &Client, width: u32, height: u32, mode: MotionCompensationMode) -> Self {
        let built = PairModelProducer::new(client, width, height, &mode, SampleFormat::F32, RING_DEPTH);
        let producer = built.expect("a searching mode builds a producer");

        Self {
            client: client.clone(),
            producer,
            width,
            height,
            next_frame: 0,
        }
    }

    /// The models as a pass whose pixels are `scale_x` by `scale_y` luma pixels reads them.
    pub(crate) fn motion(&self, scale_x: f32, scale_y: f32) -> SharedMotion {
        let ring = self.producer.ring().clone();

        SharedMotion {
            ring,
            scale_x,
            scale_y,
        }
    }

    /// Fits the model of the pair ending at the next stream frame, whose luma plane is `luma`.
    pub(crate) fn fit(&mut self, luma: &[f32]) {
        let handle = upload(&self.client, luma);
        let plane = DevicePlane::new(&handle, self.width, self.height);
        let frame = self.next_frame;

        self.producer.push(&plane, frame).expect("the pair model fits");
        self.next_frame += 1;
    }
}

/// One frame of a [pan_clip], its luma plane and its interleaved chroma planes.
pub(crate) struct PanFrame {
    pub(crate) luma: Vec<f32>,
    pub(crate) chroma: Vec<f32>,
}

/// The shape of a [pan_clip], in luma pixels.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PanShape {
    pub(crate) width: u32,
    pub(crate) height: u32,
    /// Luma pixels per chroma pixel on each axis.
    pub(crate) chroma_scale: (u32, u32),
    /// How far the content moves left and up each frame, in luma pixels.
    pub(crate) shift: (u32, u32),
    pub(crate) sigma: f32,
}

impl PanShape {
    pub(crate) fn chroma_dims(&self) -> (u32, u32) {
        let (scale_x, scale_y) = self.chroma_scale;

        (self.width / scale_x, self.height / scale_y)
    }
}

/// A noisy textured pan of `count` frames, seeded by `seed`.
///
/// The chroma planes sample the luma texture at every chroma pixel's luma position, so they move
/// by the luma shift over the chroma scale.
pub(crate) fn pan_clip(shape: PanShape, count: usize, seed: u32) -> Vec<PanFrame> {
    let (shift_x, shift_y) = shape.shift;
    let travel = count as u32 - 1;
    let texture_width = shape.width + shift_x * travel;
    let texture_height = shape.height + shift_y * travel;
    // The blur scales the variance by 0.375, as [correlated_noisy_frame] documents.
    let texture_sigma_pre = TEXTURE_SIGMA / 0.375f32.sqrt();
    let texture = correlated_noisy_frame(texture_width, texture_height, 0.5, texture_sigma_pre, seed);
    let (chroma_width, chroma_height) = shape.chroma_dims();
    let (scale_x, scale_y) = shape.chroma_scale;

    let sample = |x: u32, y: u32, index: u32| {
        let texture_x = x + shift_x * index;
        let texture_y = y + shift_y * index;
        texture[(texture_y * texture_width + texture_x) as usize]
    };

    let mut frames = Vec::with_capacity(count);
    for index in 0..count as u32 {
        let mut clean_luma = Vec::with_capacity((shape.width * shape.height) as usize);
        for y in 0..shape.height {
            for x in 0..shape.width {
                let value = sample(x, y, index);
                clean_luma.push(value);
            }
        }

        let mut clean_chroma = Vec::with_capacity((chroma_width * chroma_height) as usize);
        for y in 0..chroma_height {
            for x in 0..chroma_width {
                let value = sample(x * scale_x, y * scale_y, index);
                clean_chroma.push(value);
            }
        }

        let luma_seed = seed * 1000 + 2 * index;
        let chroma_seed = luma_seed + 1;
        let luma = noisy_field_over(&clean_luma, shape.width, shape.height, shape.sigma, luma_seed);
        let u_plane = noisy_field_over(
            &clean_chroma,
            chroma_width,
            chroma_height,
            shape.sigma,
            chroma_seed,
        );
        let v_plane = noisy_field_over(
            &clean_chroma,
            chroma_width,
            chroma_height,
            shape.sigma,
            chroma_seed + 500,
        );
        let chroma = u_plane
            .iter()
            .zip(&v_plane)
            .flat_map(|(&u_value, &v_value)| [u_value, v_value])
            .collect();

        frames.push(PanFrame { luma, chroma });
    }

    frames
}

/// The mean of `values` at the blocks `keep` accepts, by block column and row.
pub(crate) fn mean_over_blocks(
    values: &[f32],
    blocks_x: u32,
    blocks_y: u32,
    mut keep: impl FnMut(u32, u32) -> bool,
) -> f32 {
    let mut sum = 0.0f64;
    let mut count = 0usize;
    for row in 0..blocks_y {
        for col in 0..blocks_x {
            if !keep(col, row) {
                continue;
            }

            let value = values[(row * blocks_x + col) as usize];
            sum += f64::from(value);
            count += 1;
        }
    }

    assert!(count > 0, "no block lies inside both frames");
    (sum / count as f64) as f32
}

/// Whether a block at `origin` spanning `size` pixels, moved by `shift`, stays `margin` pixels
/// inside a plane `extent` pixels long.
pub(crate) fn stays_inside(origin: u32, size: u32, shift: i32, extent: u32, margin: u32) -> bool {
    let start = origin as i32 + shift;
    let end = start + size as i32;
    let low = margin as i32;
    let high = extent as i32 - margin as i32;

    start >= low && end <= high
}

/// How well one neighbour's field follows the pan, over the blocks inside both frames.
#[derive(Debug, Clone, Copy)]
pub(crate) struct NeighbourFit {
    /// The mean block confidence.
    pub(crate) confidence: f32,
    /// The share of blocks whose vector lands within a pixel of the pan.
    pub(crate) matched: f32,
}

/// The confidence every neighbour of a followed pan stays above.
const CONFIDENT: f32 = 0.8;

/// The share of blocks a followed pan matches, and a missed one stays below.
const FOLLOWED: f32 = 0.9;
const MISSED: f32 = 0.5;

/// 1 for each block whose vector is within a pixel of `expected`, otherwise 0.
pub(crate) fn matched_flags(vectors: &[[i32; 2]], expected: [i32; 2]) -> Vec<f32> {
    vectors
        .iter()
        .map(|vector| {
            let close_x = vector[0].abs_diff(expected[0]) <= 1;
            let close_y = vector[1].abs_diff(expected[1]) <= 1;
            f32::from(u8::from(close_x && close_y))
        })
        .collect()
}

pub(crate) fn assert_follows(fits: &[NeighbourFit], label: &str) {
    for (neighbour, fit) in fits.iter().enumerate() {
        assert!(
            fit.matched > FOLLOWED && fit.confidence > CONFIDENT,
            "{label}: neighbour {neighbour} fits {fit:?}, all {fits:?}"
        );
    }
}

pub(crate) fn assert_premise_misses(fits: &[NeighbourFit], label: &str) {
    let missed = fits.iter().any(|fit| fit.matched < MISSED);
    assert!(
        missed,
        "test premise: an unseeded {label} search misses the pan, got {fits:?}"
    );
}
