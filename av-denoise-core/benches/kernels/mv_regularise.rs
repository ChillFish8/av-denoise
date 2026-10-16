use av_denoise_core::bench_api::tune_sites::{RegulariseCandidate, RegulariseOutputs, RegulariseSite};
use av_denoise_core::tune::{SiteCandidate, VariantSite};
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;

use super::{HEIGHT, WIDTH, block_sync, make_synthetic_frame, neighbour_bench_name, shapes_with_channels};

const BLKSIZE: u32 = 16;
const STEP: u32 = 8;
const THSAD_PIXEL: f32 = 0.02;
const FIELD_LAMBDA: f32 = 1.0;

/// The nl4d field regularisation pass over every neighbour at 1080p in one launch.
pub struct MvRegulariseBench<R: Runtime> {
    pub client: ComputeClient<R>,
    pub candidate: RegulariseCandidate,
    /// Neighbours regularised in one launch, each against the centre frame.
    pub neighbours: u32,
    /// Whether the motion field is all zero, so six of every block's seven candidates repeat.
    pub static_field: bool,
}

impl<R: Runtime> Benchmark for MvRegulariseBench<R> {
    type Input = (RegulariseSite<R>, RegulariseOutputs);
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
        if self.static_field {
            field.fill(0);
        }

        let field_bytes = i32::as_bytes(&field);
        let mv_in = self.client.create_from_slice(field_bytes);
        let vectors = self.client.empty(neighbours * mv_stride * size_of::<i32>());
        let confidence = self.client.empty(neighbours * conf_stride * size_of::<f32>());
        let outputs = RegulariseOutputs {
            vectors,
            confidence,
            blocks,
            neighbours: self.neighbours,
            mv_stride: mv_stride as u32,
            conf_stride: conf_stride as u32,
        };

        let block_area = (BLKSIZE * BLKSIZE) as f32;
        let lambda_pixel = FIELD_LAMBDA * block_area * THSAD_PIXEL;
        let thsad = block_area * THSAD_PIXEL;

        let site = RegulariseSite {
            client: self.client.clone(),
            pyramid,
            pyramid_len: frames.len(),
            centre_slot: self.neighbours,
            neighbour_slots,
            slot_pixels: frame_pixels as u32,
            mv_in,
            outputs: outputs.clone(),
            lambda_pixel,
            sad_noise_floor: 0.0,
            thsad,
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
            RegulariseCandidate::SITE,
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
