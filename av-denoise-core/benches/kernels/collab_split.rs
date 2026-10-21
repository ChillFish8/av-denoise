use std::marker::PhantomData;

use av_denoise_core::bench_api::collab::geometry::ref_count;
use av_denoise_core::bench_api::tune_sites::{
    CollabFilterSite,
    CollabSearchSite,
    CollabVariant,
    FilterCandidate,
    SearchOutputs,
    SearchVariant,
};
use av_denoise_core::tune::{DeviceLimits, SiteCandidate, VariantSite, fit_candidates};
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::collab_fused::{CollabFusedBench, CollabFusedInput};
use super::{HEIGHT, WIDTH, block_sync, shapes_with_channels};

/// Which half of the split collab pass a bench times.
#[derive(Debug, Clone, Copy)]
pub enum SplitKernel {
    Search(SearchVariant),
    Filter(FilterCandidate),
}

/// One half of the split collab pass over the fused bench's 1080p ring and geometry.
///
/// Each bench fills its own member buffer once with the baseline search, so the filter reads real
/// groups.
pub struct CollabSplitBench<R: Runtime> {
    pub client: ComputeClient<R>,
    pub channels: u32,
    pub channel_name: &'static str,
    pub kernel: SplitKernel,
}

#[derive(Clone)]
pub struct CollabSplitInput {
    pub fused: CollabFusedInput,
    pub members: Handle,
}

/// The search variants and filter candidates this device runs for a bench of `channels`, baseline
/// first.
pub fn fitted_split_kernels<R: Runtime>(
    client: &ComputeClient<R>,
    channels: u32,
    channel_name: &'static str,
) -> Vec<SplitKernel> {
    let first = SearchVariant(CollabVariant::BASELINE);
    let probe = CollabSplitBench {
        client: client.clone(),
        channels,
        channel_name,
        kernel: SplitKernel::Search(first),
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

impl<R: Runtime> CollabSplitBench<R> {
    /// The fused bench's buffers and a zeroed member buffer.
    fn input(&self) -> CollabSplitInput {
        let fused_bench = self.fused();
        let fused = fused_bench.prepare();
        let refs = ref_count(WIDTH, HEIGHT);
        let members = SearchOutputs::zeroed(&self.client, refs);

        CollabSplitInput {
            fused,
            members: members.members,
        }
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
            candidate: CollabVariant::BASELINE,
        }
    }

    /// The search and filter sites over `input`, sharing its member buffer.
    fn sites(&self, input: &CollabSplitInput) -> (CollabSearchSite<R, f32>, CollabFilterSite<R>) {
        let fused_bench = self.fused();
        let fused = fused_bench.site(&input.fused);
        let refs = ref_count(WIDTH, HEIGHT);
        let outputs = SearchOutputs {
            members: input.members.clone(),
            refs,
        };

        let search = CollabSearchSite {
            client: self.client.clone(),
            inputs: fused.inputs.clone(),
            values: fused.values,
            shape: fused.shape,
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

    fn name(&self) -> String {
        let (site, candidate) = match self.kernel {
            SplitKernel::Search(variant) => (SearchVariant::SITE, variant.name()),
            SplitKernel::Filter(candidate) => (FilterCandidate::SITE, candidate.name()),
        };

        format!("{site}/{candidate}/1080p_{}", self.channel_name)
    }

    fn sync(&self) {
        block_sync(&self.client);
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        shapes_with_channels(self.channels)
    }
}
