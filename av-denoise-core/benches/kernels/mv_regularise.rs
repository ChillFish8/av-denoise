use av_denoise_core::bench_api::tune_sites::{RegulariseCandidate, RegulariseOutputs, RegulariseSite};
use av_denoise_core::tune::{SiteCandidate, VariantSite};
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;

use super::{HEIGHT, WIDTH, block_sync, make_synthetic_frame, shapes_with_channels};

const BLKSIZE: u32 = 16;
const STEP: u32 = 8;
const THSAD_PIXEL: f32 = 0.02;
const FIELD_LAMBDA: f32 = 1.0;

/// The nl4d field regularisation pass over one neighbour at 1080p.
pub struct MvRegulariseBench<R: Runtime> {
    pub client: ComputeClient<R>,
    pub candidate: RegulariseCandidate,
}

impl<R: Runtime> Benchmark for MvRegulariseBench<R> {
    type Input = (RegulariseSite<R>, RegulariseOutputs);
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let blocks_x = WIDTH.div_ceil(STEP);
        let blocks_y = HEIGHT.div_ceil(STEP);
        let blocks = (blocks_x * blocks_y) as usize;

        let centre_frame = make_synthetic_frame(WIDTH, HEIGHT, 1);
        let centre_bytes = f32::as_bytes(&centre_frame);
        let centre = self.client.create_from_slice(centre_bytes);
        let neighbour_frame = make_synthetic_frame(WIDTH, HEIGHT, 1);
        let neighbour_bytes = f32::as_bytes(&neighbour_frame);
        let neighbour = self.client.create_from_slice(neighbour_bytes);

        let field: Vec<i32> = (0..2 * blocks).map(|index| (index % 7) as i32 - 3).collect();
        let field_bytes = i32::as_bytes(&field);
        let mv_in = self.client.create_from_slice(field_bytes);
        let vectors = self.client.empty(2 * blocks * size_of::<i32>());
        let confidence = self.client.empty(blocks * size_of::<f32>());
        let outputs = RegulariseOutputs {
            vectors,
            confidence,
            blocks,
        };

        let block_area = (BLKSIZE * BLKSIZE) as f32;
        let lambda_pixel = FIELD_LAMBDA * block_area * THSAD_PIXEL;
        let thsad = block_area * THSAD_PIXEL;

        let site = RegulariseSite {
            client: self.client.clone(),
            centre,
            neighbour,
            level_len: (WIDTH * HEIGHT) as usize,
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
        format!(
            "{}/{}/1080p_luma",
            RegulariseCandidate::SITE,
            self.candidate.name()
        )
    }

    fn sync(&self) {
        block_sync(&self.client);
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        shapes_with_channels(1)
    }
}
