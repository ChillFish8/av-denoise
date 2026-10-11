mod accumulate;
mod collab;
mod regularise;
mod window;

use cubecl::prelude::*;
use cubecl::server::Handle;

pub use self::accumulate::{AccumulateCandidate, AccumulateSite};
pub use self::collab::{CollabCandidate, CollabInputs, CollabOutputs, CollabShape, CollabSite, CollabValues};
#[cfg(test)]
pub(crate) use self::regularise::compare_fields;
pub use self::regularise::{
    COOP_MAX_ABS,
    COOP_MAX_FLIPPED_SHARE,
    COOP_MIN_PSNR,
    RegulariseCandidate,
    RegulariseLayout,
    RegulariseOutputs,
    RegulariseSite,
};
pub use self::window::{ConfidenceBinding, WindowCandidate, WindowPass, WindowSite};
use crate::tune::fidelity::{CompareResult, Fidelity, diff_plane};
use crate::tune::gpu::{read_f32, zero_accumulators};

/// The fused windowed NLM kernels, one launch per temporal offset.
pub const NLM_WINDOW: &str = "nlm_window";
/// The per-pixel accumulation on the separable NLM path.
pub const NLM_ACCUMULATE: &str = "nlm_accumulate";
/// The nl4d grouping, filtering and scatter kernel.
pub const COLLAB_FUSED: &str = "collab_fused";
/// The nl4d motion field regularisation pass.
pub const NL4D_MV_REGULARISE: &str = "nl4d_mv_regularise";

/// The NLM accumulators a windowed or accumulate launch adds into.
#[derive(Debug, Clone)]
pub struct AccumOutputs {
    pub accum: Handle,
    pub weight_sum: Handle,
    pub max_weight: Handle,
    /// Elements in `accum`, one per stored channel per pixel.
    pub accum_len: usize,
    /// Elements in `weight_sum` and in `max_weight`.
    pub pixels: usize,
}

/// Allocates zeroed accumulators for `pixels` pixels and `accum_len` accumulator lanes.
pub fn allocate_accum<R: Runtime>(
    client: &ComputeClient<R>,
    accum_len: usize,
    pixels: usize,
) -> AccumOutputs {
    let accum = client.empty(accum_len * size_of::<f32>());
    let weight_sum = client.empty(pixels * size_of::<f32>());
    let max_weight = client.empty(pixels * size_of::<f32>());
    let outputs = AccumOutputs {
        accum,
        weight_sum,
        max_weight,
        accum_len,
        pixels,
    };

    clear_accum(client, &outputs);

    outputs
}

pub fn clear_accum<R: Runtime>(client: &ComputeClient<R>, outputs: &AccumOutputs) {
    zero_accumulators(
        client,
        &outputs.accum,
        &outputs.weight_sum,
        &outputs.max_weight,
        outputs.accum_len,
        outputs.pixels,
    );
}

/// Compares two sets of accumulators plane by plane under `fidelity`.
pub fn compare_accum<R: Runtime>(
    client: &ComputeClient<R>,
    baseline: &AccumOutputs,
    candidate: &AccumOutputs,
    fidelity: Fidelity,
) -> Result<CompareResult, anyhow::Error> {
    let planes = [
        (&baseline.accum, &candidate.accum, baseline.accum_len),
        (&baseline.weight_sum, &candidate.weight_sum, baseline.pixels),
        (&baseline.max_weight, &candidate.max_weight, baseline.pixels),
    ];
    let mut diffs = Vec::with_capacity(planes.len());

    for (expected_handle, actual_handle, len) in planes {
        let expected = read_f32(client, expected_handle, len)?;
        let actual = read_f32(client, actual_handle, len)?;
        let diff = diff_plane(&expected, &actual);
        diffs.push(diff);
    }

    let verdict = fidelity.judge(&diffs);

    Ok(verdict)
}
