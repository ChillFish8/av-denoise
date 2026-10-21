use av_denoise_core::bench_api::tune_sites::{
    AccumOutputs,
    ConfidenceBinding,
    WindowCandidate,
    WindowCodes,
    WindowPass,
    WindowSite,
    allocate_accum,
};
use av_denoise_core::tune::{IntegerCodes, SiteCandidate, VariantSite};
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;

use super::{
    HEIGHT,
    PATCH_RADIUS,
    SEARCH_RADIUS,
    WIDTH,
    block_sync,
    h2_inv_norm,
    make_padded_frame,
    shapes_with_channels,
    stored_channels,
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
///
/// With `codes` the site also reads the frame rounded to codes of that depth.
pub struct WindowBench<R: Runtime> {
    pub client: ComputeClient<R>,
    pub channels: u32,
    pub channel_name: &'static str,
    pub kernel: WindowKernel,
    pub candidate: WindowCandidate,
    pub codes: Option<IntegerCodes>,
}

impl<R: Runtime> WindowBench<R> {
    /// `frame` rounded to `codes` and uploaded as the u16 code ring.
    fn code_ring(&self, frame: &[f32], codes: IntegerCodes) -> WindowCodes {
        let max_value = codes.max_code() as f32;
        let code_values: Vec<u16> = frame
            .iter()
            .map(|&value| {
                let clamped = value.clamp(0.0, 1.0);
                let code = (clamped * max_value).round();
                code as u16
            })
            .collect();
        let code_bytes = u16::as_bytes(&code_values);
        let ring = self.client.create_from_slice(code_bytes);
        let inv_scale2 = codes.inv_scale2(self.channels);

        WindowCodes {
            ring,
            codes,
            inv_scale2,
        }
    }
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
        let codes = self.codes.map(|codes| self.code_ring(&frame, codes));

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
                noise_offset: 0.0,
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
            codes,
            ring_len: frame.len(),
            outputs: outputs.clone(),
            frame_t: 0,
            h2_inv_norm: h2_inv_norm(),
            width: WIDTH,
            height: HEIGHT,
            channels: self.channels,
            stored_ch,
            patch_radius: PATCH_RADIUS,
            search_radius: SEARCH_RADIUS,
            matrix_plane: 0,
        };

        (site, outputs)
    }

    fn execute(&self, input: Self::Input) -> Result<(), String> {
        let (site, outputs) = input;
        site.launch(self.candidate, &outputs);

        Ok(())
    }

    /// `<site>/<candidate>/<kernel>_1080p_<channels>`, with `_10bit` after the channels for 10-bit
    /// codes.
    fn name(&self) -> String {
        let ten_bit = self.codes.is_some_and(|codes| codes.depth == 10);
        let suffix = if ten_bit { "_10bit" } else { "" };

        format!(
            "{}/{}/{}_1080p_{}{suffix}",
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
