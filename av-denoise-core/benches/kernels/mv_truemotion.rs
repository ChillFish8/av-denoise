use av_denoise_core::bench_api::tune_sites::{
    TruemotionCandidate,
    TruemotionOutputs,
    TruemotionSite,
    TruemotionValues,
};
use av_denoise_core::bench_api::{DERIVED_SAD_FLOOR, DerivedValues};
use av_denoise_core::tune::{SiteCandidate, VariantSite};
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;

use super::{HEIGHT, WIDTH, block_sync, make_synthetic_frame, neighbour_bench_name, shapes_with_channels};

const BLKSIZE: u32 = 16;
const STEP: u32 = 8;
const THSAD_PIXEL: f32 = 0.02;
const FIELD_LAMBDA: f32 = 1.0;
const SURCHARGE: f32 = 0.125;
const FADE_SCALE: f32 = 2.0;

/// The first pass of the truemotion stage over every neighbour at 1080p in one launch.
///
/// It runs the shipped candidates, seeds, camera model and refine, and flags its changes.
pub struct MvTruemotionBench {
    pub client: Client,
    pub candidate: TruemotionCandidate,
    /// Neighbours scored in one launch, each against the centre frame.
    pub neighbours: u32,
    /// Whether the fields are all zero, so every candidate but the model repeats.
    pub static_field: bool,
}

impl Benchmark for MvTruemotionBench {
    type Input = (TruemotionSite, TruemotionOutputs);
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let blocks_x = WIDTH.div_ceil(STEP);
        let blocks_y = HEIGHT.div_ceil(STEP);
        let blocks = (blocks_x * blocks_y) as usize;
        let neighbours = self.neighbours as usize;
        let frame_pixels = (WIDTH * HEIGHT) as usize;

        // Every neighbour takes slots `0..neighbours` and the centre takes the last slot.
        let mut frames = Vec::with_capacity((neighbours + 1) * frame_pixels);
        for _ in 0..=self.neighbours {
            let frame = make_synthetic_frame(WIDTH, HEIGHT, 1);
            frames.extend_from_slice(&frame);
        }

        let frame_bytes = f32::as_bytes(&frames);
        let pyramid = self.client.create_from_slice(frame_bytes);
        let table: Vec<u32> = (0..self.neighbours).collect();
        let table_bytes = u32::as_bytes(&table);
        let neighbour_slots = self.client.create_from_slice(table_bytes);

        let mv_stride = 2 * blocks;
        let conf_stride = blocks;
        let mut field: Vec<i32> = (0..neighbours * mv_stride)
            .map(|index| (index % 7) as i32 - 3)
            .collect();
        let mut seed_field: Vec<i32> = (0..neighbours * mv_stride)
            .map(|index| (index % 5) as i32 - 2)
            .collect();
        if self.static_field {
            field.fill(0);
            seed_field.fill(0);
        }

        let field_bytes = i32::as_bytes(&field);
        let mv_in = self.client.create_from_slice(field_bytes);
        let seed_bytes = i32::as_bytes(&seed_field);
        let seeds = self.client.create_from_slice(seed_bytes);

        let model = [1.0f32, 0.0, 0.0, 1.0, 6.0, -2.0];
        let prediction_words: Vec<f32> = model.iter().copied().cycle().take(6 * neighbours).collect();
        let prediction_bytes = f32::as_bytes(&prediction_words);
        let predictions = self.client.create_from_slice(prediction_bytes);

        let vectors_len = neighbours * mv_stride;
        let confidence_len = neighbours * conf_stride;
        let vectors = self.client.empty(vectors_len * size_of::<i32>());
        let confidence = self.client.empty(confidence_len * size_of::<f32>());
        let changes = self.client.empty(confidence_len * size_of::<u32>());
        let outputs = TruemotionOutputs {
            vectors,
            vectors_len,
            confidence,
            confidence_len,
            writes_confidence: true,
            sad: None,
            changes: Some(changes),
            blocks,
            neighbours: self.neighbours,
            mv_stride: mv_stride as u32,
            conf_stride: conf_stride as u32,
        };

        let block_area = (BLKSIZE * BLKSIZE) as f32;
        let thsad = block_area * THSAD_PIXEL;
        let values = TruemotionValues {
            lambda_base: FIELD_LAMBDA * thsad,
            lambda_noise: FIELD_LAMBDA,
            surcharge: SURCHARGE,
            fade_level: FADE_SCALE * thsad,
            thsad,
            sad_field: DERIVED_SAD_FLOOR,
        };
        let noise_words = DerivedValues::default().to_words();
        let noise_bytes = f32::as_bytes(&noise_words);
        let noise = self.client.create_from_slice(noise_bytes);

        let site = TruemotionSite {
            client: self.client.clone(),
            pyramid,
            pyramid_len: frames.len(),
            centre_slot: self.neighbours,
            neighbour_slots,
            slot_pixels: frame_pixels as u32,
            mv_in,
            seeds: Some(seeds),
            predictions: Some(predictions),
            changes_in: None,
            outputs: outputs.clone(),
            values,
            noise,
            reads_floor: true,
            refine_radius: 1,
            width: WIDTH,
            height: HEIGHT,
            blksize: BLKSIZE,
            step: STEP,
            blocks_x,
            blocks_y,
        };

        (site, outputs)
    }

    fn execute(&self, input: Self::Input) -> Result<(), String> {
        let (site, outputs) = input;
        site.launch(self.candidate, &outputs);

        Ok(())
    }

    fn name(&self) -> String {
        let content = if self.static_field { "_static" } else { "" };
        let base = format!(
            "{}/{}/1080p_luma{content}",
            TruemotionCandidate::SITE,
            self.candidate.name()
        );

        neighbour_bench_name(&base, self.neighbours)
    }

    fn sync(&self) {
        block_sync(&self.client);
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        shapes_with_channels(1)
    }
}
