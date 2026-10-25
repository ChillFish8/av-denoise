use av_denoise_core::bench_api::DERIVED_H2_INV_NORM;
use av_denoise_core::bench_api::tune_sites::{
    AccumOutputs,
    ConfidenceBinding,
    WindowCandidate,
    WindowPass,
    WindowSite,
    allocate_accum,
};
use av_denoise_core::tune::{SiteCandidate, VariantSite};
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;

use super::{
    HEIGHT,
    PATCH_RADIUS,
    SEARCH_RADIUS,
    WIDTH,
    block_sync,
    make_padded_frame,
    shapes_with_channels,
    stored_channels,
    window_noise,
};

/// Which windowed kernel a row runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowKernel {
    Single,
    Pair,
    SingleRef,
    PairRef,
}

impl WindowKernel {
    pub const ALL: [WindowKernel; 4] = [
        WindowKernel::Single,
        WindowKernel::Pair,
        WindowKernel::SingleRef,
        WindowKernel::PairRef,
    ];

    fn label(self) -> &'static str {
        match self {
            WindowKernel::Single => "single",
            WindowKernel::Pair => "pair",
            WindowKernel::SingleRef => "single_ref",
            WindowKernel::PairRef => "pair_ref",
        }
    }

    fn is_pair(self) -> bool {
        matches!(self, WindowKernel::Pair | WindowKernel::PairRef)
    }

    fn has_reference(self) -> bool {
        matches!(self, WindowKernel::SingleRef | WindowKernel::PairRef)
    }
}

/// One windowed kernel at one candidate's cube shape over a 1080p frame.
pub struct WindowBench<R: Runtime> {
    pub client: ComputeClient<R>,
    pub channels: u32,
    pub channel_name: &'static str,
    pub kernel: WindowKernel,
    pub candidate: WindowCandidate,
}

impl<R: Runtime> Benchmark for WindowBench<R> {
    type Input = (WindowSite<R>, AccumOutputs);
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let pixels = (WIDTH * HEIGHT) as usize;
        let stored_ch = stored_channels(self.channels);
        let frame = make_padded_frame(WIDTH, HEIGHT, self.channels);
        let frame_bytes = f32::as_bytes(&frame);
        let input = self.client.create_from_slice(frame_bytes);
        let reference = self
            .kernel
            .has_reference()
            .then(|| self.client.create_from_slice(frame_bytes));
        let outputs = allocate_accum(&self.client, pixels * stored_ch as usize, pixels);
        let noise = window_noise(&self.client);

        let pass = if self.kernel.is_pair() {
            let placeholder = self.client.empty(size_of::<f32>());
            let confidence = ConfidenceBinding {
                enabled: false,
                forward: placeholder.clone(),
                backward: placeholder,
                len: 1,
                step: 1,
                blocks_x: 1,
                blocks_y: 1,
            };

            WindowPass::Pair {
                frame_fwd: 0,
                frame_bwd: 0,
                confidence,
            }
        } else {
            let side = (2 * SEARCH_RADIUS + 1) as usize;
            let offsets_host = vec![0.0f32; side * side];
            let offsets_bytes = f32::as_bytes(&offsets_host);
            let offsets = self.client.create_from_slice(offsets_bytes);

            WindowPass::Single {
                offsets,
                offsets_len: offsets_host.len(),
            }
        };

        let site = WindowSite {
            client: self.client.clone(),
            pass,
            input,
            reference,
            ring_len: frame.len(),
            outputs: outputs.clone(),
            frame_t: 0,
            noise,
            h2_field: DERIVED_H2_INV_NORM,
            width: WIDTH,
            height: HEIGHT,
            channels: self.channels,
            stored_ch,
            patch_radius: PATCH_RADIUS,
            search_radius: SEARCH_RADIUS,
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
            "{}/{}/{}_1080p_{}",
            WindowCandidate::SITE,
            self.candidate.name(),
            self.kernel.label(),
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
