use av_denoise_core::bench_api::tune_sites::{
    AccumOutputs,
    AccumulateCandidate,
    AccumulateSite,
    allocate_accum,
};
use av_denoise_core::tune::{SiteCandidate, VariantSite};
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;

use super::{HEIGHT, Q_X, Q_Y, WIDTH, block_sync, make_synthetic_frame, shapes_with_channels};

pub struct AccumulateBench {
    pub client: Client,
    pub channels: u32,
    pub channel_name: &'static str,
    pub candidate: AccumulateCandidate,
}

impl Benchmark for AccumulateBench {
    type Input = (AccumulateSite, AccumOutputs);
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let pixels = (WIDTH * HEIGHT) as usize;
        let channels = self.channels;
        let frame = make_synthetic_frame(WIDTH, HEIGHT, channels);
        let frame_bytes = f32::as_bytes(&frame);
        let input = self.client.create_from_slice(frame_bytes);

        let weights_data = vec![0.5f32; pixels];
        let weights_bytes = f32::as_bytes(&weights_data);
        let weights = self.client.create_from_slice(weights_bytes);
        let outputs = allocate_accum(&self.client, pixels * channels as usize, pixels);

        let site = AccumulateSite {
            client: self.client.clone(),
            input,
            ring_len: frame.len(),
            weights,
            outputs: outputs.clone(),
            frame: 0,
            q_x: Q_X,
            q_y: Q_Y,
            width: WIDTH,
            height: HEIGHT,
            channels,
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
            "{}/{}/1080p_{}",
            AccumulateCandidate::SITE,
            self.candidate.name(),
            self.channel_name,
        )
    }

    fn sync(&self) {
        block_sync(&self.client);
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        shapes_with_channels(self.channels)
    }
}
