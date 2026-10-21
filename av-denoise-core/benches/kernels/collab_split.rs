use std::marker::PhantomData;

use av_denoise_core::bench_api::MAX_GRID_1D;
use av_denoise_core::bench_api::collab::geometry::ref_count;
use av_denoise_core::bench_api::kernels::gpu_cast_codes;
use av_denoise_core::bench_api::tune_sites::{
    CollabFilterSite,
    CollabInputs,
    CollabSearchSite,
    CollabShape,
    CollabVariant,
    FilterCandidate,
    SearchOutputs,
    SearchVariant,
};
use av_denoise_core::tune::{
    DeviceLimits,
    IntegerCodes,
    SiteCandidate,
    VariantSite,
    fit_candidates,
    int8_matrix_plane_width,
    matrix_plane_width,
};
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::collab_fused::{CollabFusedBench, CollabFusedInput};
use super::{BLOCK_1D, HEIGHT, WIDTH, block_sync, shapes_with_channels};

/// Which half of the split collab pass a bench times.
#[derive(Debug, Clone, Copy)]
pub enum SplitKernel {
    Search(SearchVariant),
    Filter(FilterCandidate),
}

/// One half of the split collab pass over the fused bench's 1080p ring and geometry.
///
/// Each bench fills its own member buffer once with the baseline search, so the filter reads real
/// groups. With `codes` the search site also reads the ring cast to integer codes of that depth.
pub struct CollabSplitBench<R: Runtime> {
    pub client: ComputeClient<R>,
    pub channels: u32,
    pub channel_name: &'static str,
    pub kernel: SplitKernel,
    pub codes: Option<IntegerCodes>,
}

#[derive(Clone)]
pub struct CollabSplitInput {
    pub fused: CollabFusedInput,
    pub members: Handle,
    /// The ring cast to u16 codes, when the bench has codes.
    pub code_ring: Option<Handle>,
}

/// The search variants and filter candidates this device runs for a bench of `channels`, baseline
/// first.
pub fn fitted_split_kernels<R: Runtime>(
    client: &ComputeClient<R>,
    channels: u32,
    channel_name: &'static str,
) -> Vec<SplitKernel> {
    let first = SearchVariant::Float(CollabVariant::BASELINE);
    let probe = CollabSplitBench {
        client: client.clone(),
        channels,
        channel_name,
        kernel: SplitKernel::Search(first),
        codes: None,
    };
    let input = probe.input();
    let (search, filter) = probe.sites(&input);
    let limits = DeviceLimits::from_client(client);

    let searches = fit_candidates(&search, &limits).expect("the search baseline fits the bench device");
    let filters = fit_candidates(&filter, &limits).expect("the filter baseline fits the bench device");
    let search_kernels = searches.into_iter().map(SplitKernel::Search);
    let filter_kernels = filters.into_iter().map(SplitKernel::Filter);

    search_kernels.chain(filter_kernels).collect()
}

/// The integer search variants this device runs for a bench of `channels` over `codes`.
pub fn fitted_integer_kernels<R: Runtime>(
    client: &ComputeClient<R>,
    channels: u32,
    channel_name: &'static str,
    codes: IntegerCodes,
) -> Vec<SplitKernel> {
    let first = SearchVariant::Float(CollabVariant::BASELINE);
    let probe = CollabSplitBench {
        client: client.clone(),
        channels,
        channel_name,
        kernel: SplitKernel::Search(first),
        codes: Some(codes),
    };
    let input = probe.input();
    let (search, _) = probe.sites(&input);
    let limits = DeviceLimits::from_client(client);

    let searches = fit_candidates(&search, &limits).expect("the search baseline fits the bench device");
    let integer_searches = searches
        .into_iter()
        .filter(|candidate| matches!(candidate, SearchVariant::Integer(_)));

    integer_searches.map(SplitKernel::Search).collect()
}

impl<R: Runtime> CollabSplitBench<R> {
    /// The fused bench's buffers, a zeroed member buffer and the code ring when the bench has codes.
    fn input(&self) -> CollabSplitInput {
        let fused_bench = self.fused();
        let fused = fused_bench.prepare();
        let refs = ref_count(WIDTH, HEIGHT);
        let members = SearchOutputs::zeroed(&self.client, refs);
        let code_ring = self.codes.map(|codes| self.cast_ring(&fused, codes));

        CollabSplitInput {
            fused,
            members: members.members,
            code_ring,
        }
    }

    /// The fused bench's ring cast to u16 codes of `codes`' depth.
    fn cast_ring(&self, fused: &CollabFusedInput, codes: IntegerCodes) -> Handle {
        let ring_len = fused.ring_len;
        let code_ring = self.client.empty(ring_len * size_of::<u16>());
        let grid = (ring_len as u32).div_ceil(BLOCK_1D).min(MAX_GRID_1D);
        let total_threads = grid * BLOCK_1D;
        let max_code = codes.max_code() as f32;
        let cube_count = CubeCount::new_1d(grid);
        let cube_dim = CubeDim::new_1d(BLOCK_1D);
        let source_handle = fused.ring.clone();
        let codes_handle = code_ring.clone();

        unsafe {
            let source_arg = ArrayArg::from_raw_parts(source_handle, ring_len);
            let codes_arg = ArrayArg::from_raw_parts(codes_handle, ring_len);

            gpu_cast_codes::launch_unchecked::<u16, R>(
                &self.client,
                cube_count,
                cube_dim,
                source_arg,
                codes_arg,
                max_code,
                0,
                ring_len as u32,
                total_threads,
            );
        }

        code_ring
    }

    fn fused(&self) -> CollabFusedBench<R> {
        CollabFusedBench {
            client: self.client.clone(),
            channels: self.channels,
            channel_name: self.channel_name,
            split_mv: false,
            noise_curve: false,
            strength_map: false,
            pooled: false,
            packed: false,
            candidate: CollabVariant::BASELINE,
        }
    }

    /// The search and filter sites over `input`, sharing its member buffer.
    ///
    /// With codes the search shape names them, the device's matrix planes and its int8 product, and
    /// reads the code ring.
    fn sites(&self, input: &CollabSplitInput) -> (CollabSearchSite<R, f32>, CollabFilterSite<R>) {
        let fused_bench = self.fused();
        let fused = fused_bench.site(&input.fused);
        let refs = ref_count(WIDTH, HEIGHT);
        let outputs = SearchOutputs {
            members: input.members.clone(),
            refs,
        };

        let (search_inputs, search_shape) = match &input.code_ring {
            Some(code_ring) => {
                let limits = DeviceLimits::from_client(&self.client);
                let matrix_plane = matrix_plane_width(&limits).unwrap_or(0);
                let int8_plane = int8_matrix_plane_width(&limits).unwrap_or(0);
                let inputs = CollabInputs {
                    codes: code_ring.clone(),
                    codes_len: input.fused.ring_len,
                    ..fused.inputs.clone()
                };
                let shape = CollabShape {
                    codes: self.codes,
                    matrix_plane,
                    int8: limits.int8_matrix,
                    int8_plane,
                    ..fused.shape
                };
                (inputs, shape)
            },
            None => (fused.inputs.clone(), fused.shape),
        };

        let search = CollabSearchSite {
            client: self.client.clone(),
            inputs: search_inputs,
            values: fused.values,
            shape: search_shape,
            outputs,
            search: PhantomData,
        };
        let filter = CollabFilterSite {
            client: self.client.clone(),
            inputs: fused.inputs,
            members: input.members.clone(),
            values: fused.values,
            shape: fused.shape,
            outputs: fused.outputs,
        };

        (search, filter)
    }
}

impl<R: Runtime> Benchmark for CollabSplitBench<R> {
    type Input = CollabSplitInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let input = self.input();

        let (search, _) = self.sites(&input);
        let real_outputs = search.real_outputs();
        let baseline = search.baseline();
        search.launch(baseline, &real_outputs);
        block_sync(&self.client);

        input
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let (search, filter) = self.sites(&args);

        match self.kernel {
            SplitKernel::Search(variant) => {
                let outputs = search.real_outputs();
                search.launch(variant, &outputs);
            },
            SplitKernel::Filter(candidate) => {
                let outputs = filter.real_outputs();
                filter.launch(candidate, &outputs);
            },
        }

        Ok(())
    }

    /// `<site>/<candidate>/1080p_<channels>`, with `_10bit` after the channels for 10-bit codes.
    fn name(&self) -> String {
        let (site, candidate) = match self.kernel {
            SplitKernel::Search(variant) => (SearchVariant::SITE, variant.name()),
            SplitKernel::Filter(candidate) => (FilterCandidate::SITE, candidate.name()),
        };
        let ten_bit = self.codes.is_some_and(|codes| codes.depth == 10);
        let suffix = if ten_bit { "_10bit" } else { "" };

        format!("{site}/{candidate}/1080p_{}{suffix}", self.channel_name)
    }

    fn sync(&self) {
        block_sync(&self.client);
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        shapes_with_channels(self.channels)
    }
}
