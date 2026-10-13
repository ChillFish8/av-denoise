use std::marker::PhantomData;
use std::sync::LazyLock;

use cubecl::prelude::*;
use cubecl::server::Handle;

use super::COLLAB_FUSED;
use crate::collab::geometry::{BASELINE_GROUPS, fused_cubes_x};
use crate::collab::kernels::fused::{Centre, CollabVariant, GROUP_LANES, Walk, collab_fused};
use crate::collab::{PATCH_SIZE, needs_warp_uniform_search};
use crate::nlmeans::NOISE_CURVE_BINS;
use crate::tune::DeviceLimits;
use crate::tune::fidelity::{CompareResult, Fidelity, diff_integers, diff_plane};
use crate::tune::gpu::{bind, read_f32, read_i32, zero_words};
use crate::tune::key::SiteKey;
use crate::tune::site::{LaunchShape, SiteCandidate, VariantSite};

/// Group counts per cube, the baseline's first.
const GROUP_OPTIONS: [u32; 3] = [BASELINE_GROUPS, 4, 16];

/// Walks, the clipped baseline first.
const WALK_OPTIONS: [Walk; 2] = [Walk::Clipped, Walk::Uniform];

/// Centre sources, the global baseline first.
const CENTRE_OPTIONS: [Centre; 2] = [Centre::Global, Centre::Tiled];

/// Batched centre sums off, the baseline, then on.
const BATCHED_OPTIONS: [bool; 2] = [false, true];

const VARIANT_COUNT: usize =
    GROUP_OPTIONS.len() * WALK_OPTIONS.len() * CENTRE_OPTIONS.len() * BATCHED_OPTIONS.len();

/// Every listed variant, baseline first, with the batched flag varying fastest.
const VARIANTS: [CollabVariant; VARIANT_COUNT] = listed_variants();

// The tuner and the store read the first listed variant as the site's baseline.
const _: () = assert!(
    VARIANTS[0].groups == CollabVariant::BASELINE.groups
        && VARIANTS[0].walk as u8 == CollabVariant::BASELINE.walk as u8
        && VARIANTS[0].centre as u8 == CollabVariant::BASELINE.centre as u8
        && VARIANTS[0].batched == CollabVariant::BASELINE.batched,
    "the first listed collab variant must be CollabVariant::BASELINE"
);

/// Every listed variant's stored name, in [VARIANTS] order.
static NAMES: LazyLock<Vec<&'static str>> = LazyLock::new(|| {
    VARIANTS
        .iter()
        .map(|&variant| {
            let name = variant_name(variant);
            let leaked: &'static str = name.leak();
            leaked
        })
        .collect()
});

const fn listed_variants() -> [CollabVariant; VARIANT_COUNT] {
    let mut variants = [CollabVariant::BASELINE; VARIANT_COUNT];
    let batched_options = BATCHED_OPTIONS.len();
    let centres = CENTRE_OPTIONS.len();
    let walks = WALK_OPTIONS.len();
    let mut index = 0;

    while index < VARIANT_COUNT {
        let batched = BATCHED_OPTIONS[index % batched_options];
        let centre = CENTRE_OPTIONS[(index / batched_options) % centres];
        let walk = WALK_OPTIONS[(index / (batched_options * centres)) % walks];
        let groups = GROUP_OPTIONS[index / (batched_options * centres * walks)];
        variants[index] = CollabVariant {
            groups,
            walk,
            centre,
            batched,
        };
        index += 1;
    }

    variants
}

/// `groups<g>_<walk>` with `_tiled` and `_batched` where they apply, for example
/// `groups8_uniform_tiled_batched`.
fn variant_name(variant: CollabVariant) -> String {
    let walk = variant.walk.name();
    let centre = variant.centre.name_suffix();
    let batched = if variant.batched { "_batched" } else { "" };
    format!("groups{}_{walk}{centre}{batched}", variant.groups)
}

impl SiteCandidate for CollabVariant {
    const SITE: &'static str = COLLAB_FUSED;
    const ALL: &'static [Self] = &VARIANTS;

    fn name(self) -> &'static str {
        let position = VARIANTS.iter().position(|&variant| variant == self);
        let index = position.expect("every collab candidate is a listed variant");
        NAMES[index]
    }

    fn fidelity(self) -> Fidelity {
        Fidelity::Exact
    }
}

/// The buffers `collab_fused` reads, with the element count it binds for each.
#[derive(Debug, Clone)]
pub struct CollabInputs {
    pub ring: Handle,
    pub ring_len: usize,
    /// The f16 copy of the ring when the f16 search runs, otherwise a placeholder never read.
    pub search_ring: Handle,
    pub search_len: usize,
    pub mv_field: Handle,
    pub mv_len: usize,
    pub confidence: Handle,
    pub conf_len: usize,
    pub neighbour_slots: Handle,
    pub neighbour_slots_len: usize,
    /// One value per stored channel.
    pub sigma: Handle,
    /// [NOISE_CURVE_BINS](crate::nlmeans::NOISE_CURVE_BINS) ratios.
    pub noise_curve: Handle,
    pub strength_map: Handle,
    pub map_len: usize,
    /// Eight profile values.
    pub dct_profile: Handle,
    /// [PATCH_SIZE](crate::collab::PATCH_SIZE) taps.
    pub kaiser: Handle,
}

/// The runtime values `collab_fused` reads.
#[derive(Debug, Clone, Copy)]
pub struct CollabValues {
    pub centre_slot: u32,
    pub c_min: f32,
    pub lambda_ht: f32,
    pub curve_valid: u32,
    pub map_mode: u32,
    pub weight_scale: f32,
    pub accum_scale: f32,
    pub pool_ratio: f32,
}

/// The compile-time geometry and switches `collab_fused` is built for.
#[derive(Debug, Clone, Copy)]
pub struct CollabShape {
    pub f16_search: bool,
    pub radius: u32,
    pub grid_frames: u32,
    pub refine: u32,
    pub mv_stride: u32,
    pub conf_stride: u32,
    pub blk_step: u32,
    pub blksize: u32,
    pub blocks_x: u32,
    pub blocks_y: u32,
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    pub k_max: u32,
    pub stored_ch: u32,
    pub spatial_radius: u32,
    pub refs_x: u32,
    pub refs_y: u32,
    pub map_cols: u32,
    pub map_rows: u32,
    pub pooled: bool,
}

/// The fixed-point accumulator rings and group weights `collab_fused` writes.
#[derive(Debug, Clone)]
pub struct CollabOutputs {
    pub accum: Handle,
    pub wsum: Handle,
    pub group_weight: Handle,
    pub accum_len: usize,
    pub wsum_len: usize,
    /// Reference patches, one group weight each.
    pub refs: usize,
}

/// One `collab_fused` pass, with its candidate distances read as `S`.
#[derive(Clone)]
pub struct CollabSite<R: Runtime, S: Float> {
    pub client: ComputeClient<R>,
    pub inputs: CollabInputs,
    pub values: CollabValues,
    pub shape: CollabShape,
    pub outputs: CollabOutputs,
    pub search: PhantomData<S>,
}

impl<R: Runtime, S: Float> VariantSite<R> for CollabSite<R, S> {
    type Candidate = CollabVariant;
    type Outputs = CollabOutputs;

    const PRESCREEN: bool = true;
    const STAGES: usize = 2;

    fn client(&self) -> &ComputeClient<R> {
        &self.client
    }

    fn key(&self) -> SiteKey {
        let shape = &self.shape;

        SiteKey::builder(COLLAB_FUSED, shape.width, shape.height)
            .param("ch", shape.channels)
            .param("stored", shape.stored_ch)
            .param("radius", shape.radius)
            .param("grid", shape.grid_frames)
            .param("refine", shape.refine)
            .param("blksize", shape.blksize)
            .param("step", shape.blk_step)
            .param("kmax", shape.k_max)
            .param("spatial", shape.spatial_radius)
            .param("pooled", u8::from(shape.pooled))
            .param("f16", u8::from(shape.f16_search))
            .build()
    }

    fn shape(&self, candidate: CollabVariant) -> LaunchShape {
        let lanes = candidate.lanes();
        LaunchShape { x: lanes, y: 1 }
    }

    fn shared_bytes(&self, candidate: CollabVariant) -> usize {
        let search_bytes = size_of::<S>();
        candidate.shared_bytes(self.shape.spatial_radius, self.shape.stored_ch, search_bytes)
    }

    fn min_plane_size(&self, _candidate: CollabVariant) -> u32 {
        GROUP_LANES
    }

    /// The first stage varies every search dimension at the seed's group count, and the second
    /// varies only the group count.
    fn in_stage(&self, stage: usize, seed: CollabVariant, candidate: CollabVariant) -> bool {
        if stage == 0 {
            return candidate.groups == seed.groups;
        }

        let regrouped = CollabVariant {
            groups: seed.groups,
            ..candidate
        };

        regrouped == seed
    }

    fn launch(&self, candidate: CollabVariant, outputs: &CollabOutputs) {
        let groups = candidate.groups;
        let lanes = candidate.lanes();
        let inputs = &self.inputs;
        let values = &self.values;
        let shape = &self.shape;
        let cubes_x = fused_cubes_x(shape.width, groups);
        let cube_count = CubeCount::new_2d(cubes_x, shape.refs_y);
        let cube_dim = CubeDim::new_1d(lanes);

        let ring = bind::<R>(&inputs.ring, inputs.ring_len);
        let search_ring = bind::<R>(&inputs.search_ring, inputs.search_len);
        let mv_field = bind::<R>(&inputs.mv_field, inputs.mv_len);
        let confidence = bind::<R>(&inputs.confidence, inputs.conf_len);
        let neighbour_slots = bind::<R>(&inputs.neighbour_slots, inputs.neighbour_slots_len);
        let sigma = bind::<R>(&inputs.sigma, shape.stored_ch as usize);
        let noise_curve = bind::<R>(&inputs.noise_curve, NOISE_CURVE_BINS);
        let strength_map = bind::<R>(&inputs.strength_map, inputs.map_len);
        let dct_profile = bind::<R>(&inputs.dct_profile, 8);
        let kaiser = bind::<R>(&inputs.kaiser, PATCH_SIZE as usize);
        let accum = bind::<R>(&outputs.accum, outputs.accum_len);
        let wsum = bind::<R>(&outputs.wsum, outputs.wsum_len);
        let group_weight = bind::<R>(&outputs.group_weight, outputs.refs);

        unsafe {
            collab_fused::launch_unchecked::<S, R>(
                &self.client,
                cube_count,
                cube_dim,
                shape.stored_ch as usize,
                ring,
                search_ring,
                mv_field,
                confidence,
                neighbour_slots,
                sigma,
                noise_curve,
                strength_map,
                dct_profile,
                kaiser,
                accum,
                wsum,
                group_weight,
                values.centre_slot,
                values.c_min,
                values.lambda_ht,
                values.curve_valid,
                values.map_mode,
                values.weight_scale,
                values.accum_scale,
                shape.f16_search,
                shape.radius,
                shape.grid_frames,
                shape.refine,
                shape.mv_stride,
                shape.conf_stride,
                shape.blk_step,
                shape.blksize,
                shape.blocks_x,
                shape.blocks_y,
                shape.width,
                shape.height,
                shape.channels,
                shape.k_max,
                shape.stored_ch,
                shape.spatial_radius,
                shape.refs_x,
                shape.map_cols,
                shape.map_rows,
                values.pool_ratio,
                shape.pooled,
                candidate,
            );
        }
    }

    fn real_outputs(&self) -> CollabOutputs {
        self.outputs.clone()
    }

    fn scratch_outputs(&self) -> CollabOutputs {
        let accum = self.client.empty(self.outputs.accum_len * size_of::<i32>());
        let wsum = self.client.empty(self.outputs.wsum_len * size_of::<i32>());
        let group_weight = self.client.empty(self.outputs.refs * size_of::<f32>());
        let scratch = CollabOutputs {
            accum,
            wsum,
            group_weight,
            ..self.outputs.clone()
        };

        self.clear_outputs(&scratch);
        scratch
    }

    fn clear_outputs(&self, outputs: &CollabOutputs) {
        zero_words(&self.client, &outputs.accum, outputs.accum_len);
        zero_words(&self.client, &outputs.wsum, outputs.wsum_len);
        zero_words(&self.client, &outputs.group_weight, outputs.refs);
    }

    fn compare(
        &self,
        baseline: &CollabOutputs,
        candidate: &CollabOutputs,
        fidelity: Fidelity,
    ) -> Result<CompareResult, anyhow::Error> {
        let expected_accum = read_i32(&self.client, &baseline.accum, baseline.accum_len)?;
        let actual_accum = read_i32(&self.client, &candidate.accum, candidate.accum_len)?;
        let expected_wsum = read_i32(&self.client, &baseline.wsum, baseline.wsum_len)?;
        let actual_wsum = read_i32(&self.client, &candidate.wsum, candidate.wsum_len)?;
        let expected_weight = read_f32(&self.client, &baseline.group_weight, baseline.refs)?;
        let actual_weight = read_f32(&self.client, &candidate.group_weight, candidate.refs)?;

        let accum_diff = diff_integers(&expected_accum, &actual_accum);
        let wsum_diff = diff_integers(&expected_wsum, &actual_wsum);
        let weight_diff = diff_plane(&expected_weight, &actual_weight);
        let verdict = fidelity.judge(&[accum_diff, wsum_diff, weight_diff]);

        Ok(verdict)
    }

    fn baseline(&self) -> CollabVariant {
        CollabVariant::for_client(&self.client)
    }

    fn legal(&self, candidate: CollabVariant, _limits: &DeviceLimits) -> bool {
        let uniform_only = needs_warp_uniform_search(&self.client);
        candidate.runs_where(uniform_only)
    }
}
