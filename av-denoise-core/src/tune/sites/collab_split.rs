use std::marker::PhantomData;

use cubecl::prelude::*;
use cubecl::server::Handle;

use super::collab::{
    CollabInputs,
    CollabOutputs,
    CollabShape,
    CollabValues,
    VARIANT_COUNT,
    VARIANTS,
    blank_collab_inputs,
    blank_collab_outputs,
    clear_collab_outputs,
    collab_key_builder,
    compare_collab_outputs,
    scratch_collab_outputs,
};
use crate::collab::PATCH_SIZE;
use crate::collab::geometry::fused_cubes_x;
use crate::collab::kernels::fused::{
    CollabVariant,
    GROUP_LANES,
    MEMBER_WORDS,
    TRANSPOSE_REGION_FLOATS,
    collab_filter,
    collab_search,
};
use crate::noise::{DERIVED_WORDS, NOISE_CURVE_BINS};
use crate::tune::fidelity::{CompareResult, Fidelity, diff_integers};
use crate::tune::gpu::{bind, read_i32, zero_words, zeroed_words};
use crate::tune::key::SiteKey;
use crate::tune::site::{LaunchShape, SiteCandidate, VariantSite};
use crate::tune::{DeviceLimits, Legality};

/// The collab search kernel.
pub const COLLAB_SEARCH: &str = "collab_search";
/// The collab filter kernel.
pub const COLLAB_FILTER: &str = "collab_filter";

/// How many search variants exist, the collab variants with 8 groups per cube.
pub const SEARCH_VARIANT_COUNT: usize = VARIANT_COUNT / 3;

/// The search's group count, which every search variant shares.
const SEARCH_GROUPS: u32 = 8;

const fn search_variants() -> [SearchVariant; SEARCH_VARIANT_COUNT] {
    let mut variants = [SearchVariant(VARIANTS[0]); SEARCH_VARIANT_COUNT];
    let mut index = 0;

    while index < SEARCH_VARIANT_COUNT {
        variants[index] = SearchVariant(VARIANTS[index]);
        index += 1;
    }

    variants
}

const fn leading_variants_have_search_groups() -> bool {
    let mut index = 0;

    while index < SEARCH_VARIANT_COUNT {
        if VARIANTS[index].groups != SEARCH_GROUPS {
            return false;
        }

        index += 1;
    }

    true
}

// The collab listing varies the group count outermost, so its first third is every 8-group variant.
const _: () = assert!(
    leading_variants_have_search_groups(),
    "the first SEARCH_VARIANT_COUNT collab variants must all run 8 groups per cube"
);

/// Every search variant, baseline first.
pub const SEARCH_VARIANTS: [SearchVariant; SEARCH_VARIANT_COUNT] = search_variants();

/// One way to run the search kernel, a collab variant at 8 groups per cube.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SearchVariant(pub CollabVariant);

impl SiteCandidate for SearchVariant {
    const SITE: &'static str = COLLAB_SEARCH;
    const ALL: &'static [Self] = &SEARCH_VARIANTS;

    fn name(self) -> &'static str {
        self.0.name()
    }

    fn fidelity(self) -> Fidelity {
        self.0.fidelity()
    }
}

/// How many reference patches one filter cube handles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterCandidate {
    Groups8,
    Groups4,
    Groups16,
}

impl FilterCandidate {
    pub fn groups(self) -> u32 {
        match self {
            FilterCandidate::Groups8 => 8,
            FilterCandidate::Groups4 => 4,
            FilterCandidate::Groups16 => 16,
        }
    }
}

const FILTER_CANDIDATES: [FilterCandidate; 3] = [
    FilterCandidate::Groups8,
    FilterCandidate::Groups4,
    FilterCandidate::Groups16,
];

impl SiteCandidate for FilterCandidate {
    const SITE: &'static str = COLLAB_FILTER;
    const ALL: &'static [Self] = &FILTER_CANDIDATES;

    fn name(self) -> &'static str {
        match self {
            FilterCandidate::Groups8 => "groups8",
            FilterCandidate::Groups4 => "groups4",
            FilterCandidate::Groups16 => "groups16",
        }
    }

    fn fidelity(self) -> Fidelity {
        Fidelity::Exact
    }
}

/// The member buffer the search writes and the filter reads.
#[derive(Debug, Clone)]
pub struct SearchOutputs {
    pub members: Handle,
    /// Reference patches, [MEMBER_WORDS](crate::collab::kernels::fused::MEMBER_WORDS) words
    /// each.
    pub refs: usize,
}

impl SearchOutputs {
    /// A member buffer for `refs` references, zeroed on the GPU.
    pub fn zeroed<R: Runtime>(client: &ComputeClient<R>, refs: usize) -> SearchOutputs {
        let words = refs * MEMBER_WORDS as usize;
        let members = client.empty(words * size_of::<u32>());
        zero_words(client, &members, words);

        SearchOutputs { members, refs }
    }

    pub fn words(&self) -> usize {
        self.refs * MEMBER_WORDS as usize
    }
}

/// One `collab_search` pass, with its candidate distances read as `S`.
#[derive(Clone)]
pub struct CollabSearchSite<R: Runtime, S: Float> {
    pub client: ComputeClient<R>,
    pub inputs: CollabInputs,
    pub values: CollabValues,
    pub shape: CollabShape,
    pub outputs: SearchOutputs,
    pub search: PhantomData<S>,
}

impl<R: Runtime, S: Float> VariantSite<R> for CollabSearchSite<R, S> {
    type Candidate = SearchVariant;
    type Outputs = SearchOutputs;

    const PRESCREEN: bool = true;
    const COPIES: bool = true;

    fn client(&self) -> &ComputeClient<R> {
        &self.client
    }

    fn key(&self) -> SiteKey {
        let shape = &self.shape;
        let builder = collab_key_builder(COLLAB_SEARCH, shape);

        builder.param("f16", u8::from(shape.f16_search)).build()
    }

    /// The exact compile-time shape and the search type, which the bucketed key leaves out.
    ///
    /// The strength map and pooling are left out because the search never reads them.
    fn compile_identity(&self) -> String {
        let search = std::any::type_name::<S>();
        let shape = CollabShape {
            pooled: false,
            map_cols: 0,
            map_rows: 0,
            ..self.shape
        };

        format!("{shape:?}|{search}")
    }

    fn shape(&self, candidate: SearchVariant) -> LaunchShape {
        let lanes = candidate.0.lanes();
        LaunchShape { x: lanes, y: 1 }
    }

    /// The centre tile, the only shared buffer the search allocates.
    fn shared_bytes(&self, candidate: SearchVariant) -> usize {
        let shape = &self.shape;
        let tile_elements = candidate.0.tile_elements(shape.spatial_radius, shape.stored_ch);
        tile_elements as usize * size_of::<S>()
    }

    fn min_plane_size(&self, _candidate: SearchVariant) -> u32 {
        GROUP_LANES
    }

    fn launch(&self, candidate: SearchVariant, outputs: &SearchOutputs) {
        let cubes_x = fused_cubes_x(self.shape.width, SEARCH_GROUPS);
        let cube_count = CubeCount::new_2d(cubes_x, self.shape.refs_y);

        self.dispatch(candidate, outputs, cube_count);
    }

    /// A launch with no cubes, which on wgpu compiles the kernel without running it.
    fn compile_only(&self, candidate: SearchVariant, outputs: &SearchOutputs) {
        let no_cubes = CubeCount::Static(0, 0, 1);
        self.dispatch(candidate, outputs, no_cubes);
    }

    fn real_outputs(&self) -> SearchOutputs {
        self.outputs.clone()
    }

    fn scratch_outputs(&self) -> SearchOutputs {
        SearchOutputs::zeroed(&self.client, self.outputs.refs)
    }

    fn clear_outputs(&self, outputs: &SearchOutputs) {
        let words = outputs.words();
        zero_words(&self.client, &outputs.members, words);
    }

    /// The copy's frame-sized buffers hold one word each, which [VariantSite::compile_only] never
    /// reads.
    fn on_client(&self, client: &ComputeClient<R>) -> Option<Self> {
        let inputs = blank_collab_inputs(client);
        let members = zeroed_words(client, 1);
        let outputs = SearchOutputs { members, refs: 0 };

        Some(Self {
            client: client.clone(),
            inputs,
            outputs,
            ..self.clone()
        })
    }

    /// The members must match word for word.
    fn compare(
        &self,
        baseline: &SearchOutputs,
        candidate: &SearchOutputs,
        fidelity: Fidelity,
    ) -> Result<CompareResult, anyhow::Error> {
        let expected = read_i32(&self.client, &baseline.members, baseline.words())?;
        let actual = read_i32(&self.client, &candidate.members, candidate.words())?;
        let diff = diff_integers(&expected, &actual);
        let verdict = fidelity.judge(&[diff]);

        Ok(verdict)
    }

    fn baseline(&self) -> SearchVariant {
        let variant = CollabVariant::baseline_for(self.shape.uniform_walk_only);
        SearchVariant(variant)
    }

    fn legal(&self, candidate: SearchVariant, _limits: &DeviceLimits, legality: &Legality) -> bool {
        let uniform_only = self.shape.uniform_walk_only || legality.uniform_walk_only;
        candidate.0.runs_where(uniform_only)
    }
}

impl<R: Runtime, S: Float> CollabSearchSite<R, S> {
    /// Launches `candidate` over `cube_count` cubes.
    fn dispatch(&self, candidate: SearchVariant, outputs: &SearchOutputs, cube_count: CubeCount) {
        let lanes = candidate.0.lanes();
        let inputs = &self.inputs;
        let values = &self.values;
        let shape = &self.shape;
        let cube_dim = CubeDim::new_1d(lanes);
        let words = outputs.words().max(1);

        let ring = bind::<R>(&inputs.ring, inputs.ring_len);
        let search_ring = bind::<R>(&inputs.search_ring, inputs.search_len);
        let mv_field = bind::<R>(&inputs.mv_field, inputs.mv_len);
        let confidence = bind::<R>(&inputs.confidence, inputs.conf_len);
        let neighbour_slots = bind::<R>(&inputs.neighbour_slots, inputs.neighbour_slots_len);
        let members = bind::<R>(&outputs.members, words);

        unsafe {
            collab_search::launch_unchecked::<S, R>(
                &self.client,
                cube_count,
                cube_dim,
                shape.stored_ch as usize,
                ring,
                search_ring,
                mv_field,
                confidence,
                neighbour_slots,
                members,
                values.centre_slot,
                values.c_min,
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
                candidate.0,
            );
        }
    }
}

/// One `collab_filter` pass over the groups a `collab_search` pass wrote to `members`.
#[derive(Clone)]
pub struct CollabFilterSite<R: Runtime> {
    pub client: ComputeClient<R>,
    pub inputs: CollabInputs,
    /// [MEMBER_WORDS](crate::collab::kernels::fused::MEMBER_WORDS) words per reference.
    pub members: Handle,
    pub values: CollabValues,
    pub shape: CollabShape,
    pub outputs: CollabOutputs,
}

impl<R: Runtime> VariantSite<R> for CollabFilterSite<R> {
    type Candidate = FilterCandidate;
    type Outputs = CollabOutputs;

    const COPIES: bool = true;

    fn client(&self) -> &ComputeClient<R> {
        &self.client
    }

    fn key(&self) -> SiteKey {
        let shape = &self.shape;
        let builder = collab_key_builder(COLLAB_FILTER, shape);

        builder.param("pooled", u8::from(shape.pooled)).build()
    }

    /// The exact compile-time shape, which the bucketed key leaves out.
    fn compile_identity(&self) -> String {
        format!("{:?}", self.shape)
    }

    fn shape(&self, candidate: FilterCandidate) -> LaunchShape {
        let lanes = GROUP_LANES * candidate.groups();
        LaunchShape { x: lanes, y: 1 }
    }

    /// The transpose regions, one per group.
    fn shared_bytes(&self, candidate: FilterCandidate) -> usize {
        let transpose_floats = candidate.groups() * TRANSPOSE_REGION_FLOATS;
        transpose_floats as usize * size_of::<f32>()
    }

    fn min_plane_size(&self, _candidate: FilterCandidate) -> u32 {
        GROUP_LANES
    }

    fn launch(&self, candidate: FilterCandidate, outputs: &CollabOutputs) {
        let cubes_x = fused_cubes_x(self.shape.width, candidate.groups());
        let cube_count = CubeCount::new_2d(cubes_x, self.shape.refs_y);

        self.dispatch(candidate, outputs, cube_count);
    }

    /// A launch with no cubes, which on wgpu compiles the kernel without running it.
    fn compile_only(&self, candidate: FilterCandidate, outputs: &CollabOutputs) {
        let no_cubes = CubeCount::Static(0, 0, 1);
        self.dispatch(candidate, outputs, no_cubes);
    }

    fn real_outputs(&self) -> CollabOutputs {
        self.outputs.clone()
    }

    fn scratch_outputs(&self) -> CollabOutputs {
        scratch_collab_outputs(&self.client, &self.outputs)
    }

    fn clear_outputs(&self, outputs: &CollabOutputs) {
        clear_collab_outputs(&self.client, outputs);
    }

    /// The copy's frame-sized buffers hold one word each, which [VariantSite::compile_only] never
    /// reads. The small fixed-size tables keep their sizes.
    fn on_client(&self, client: &ComputeClient<R>) -> Option<Self> {
        let inputs = blank_collab_inputs(client);
        let members = zeroed_words(client, MEMBER_WORDS as usize);
        let outputs = blank_collab_outputs(client, false);

        Some(Self {
            client: client.clone(),
            inputs,
            members,
            outputs,
            ..self.clone()
        })
    }

    fn compare(
        &self,
        baseline: &CollabOutputs,
        candidate: &CollabOutputs,
        fidelity: Fidelity,
    ) -> Result<CompareResult, anyhow::Error> {
        compare_collab_outputs(&self.client, baseline, candidate, fidelity)
    }
}

impl<R: Runtime> CollabFilterSite<R> {
    /// Launches `candidate` over `cube_count` cubes.
    fn dispatch(&self, candidate: FilterCandidate, outputs: &CollabOutputs, cube_count: CubeCount) {
        let groups = candidate.groups();
        let lanes = GROUP_LANES * groups;
        let inputs = &self.inputs;
        let values = &self.values;
        let shape = &self.shape;
        let cube_dim = CubeDim::new_1d(lanes);
        let member_words = (outputs.refs * MEMBER_WORDS as usize).max(1);

        let ring = bind::<R>(&inputs.ring, inputs.ring_len);
        let ring_scalar = bind::<R>(&inputs.ring, inputs.ring_len);
        let members = bind::<R>(&self.members, member_words);
        let neighbour_slots = bind::<R>(&inputs.neighbour_slots, inputs.neighbour_slots_len);
        let noise = bind::<R>(&inputs.noise, DERIVED_WORDS as usize);
        let noise_curve = bind::<R>(&inputs.noise_curve, NOISE_CURVE_BINS);
        let strength_map = bind::<R>(&inputs.strength_map, inputs.map_len);
        let dct_profile = bind::<R>(&inputs.dct_profile, 8);
        let kaiser = bind::<R>(&inputs.kaiser, PATCH_SIZE as usize);
        let accum = bind::<R>(&outputs.accum, outputs.accum_len);
        let wsum = bind::<R>(&outputs.wsum, outputs.wsum_len);
        let group_weight = bind::<R>(&outputs.group_weight, outputs.refs);

        unsafe {
            collab_filter::launch_unchecked::<i32, R>(
                &self.client,
                cube_count,
                cube_dim,
                shape.stored_ch as usize,
                ring,
                ring_scalar,
                members,
                neighbour_slots,
                noise,
                noise_curve,
                strength_map,
                dct_profile,
                kaiser,
                accum,
                wsum,
                group_weight,
                values.centre_slot,
                values.lambda_ht,
                values.accum_scale,
                values.pool_ratio,
                shape.grid_frames,
                shape.width,
                shape.height,
                shape.channels,
                shape.stored_ch,
                shape.refs_x,
                shape.map_cols,
                shape.map_rows,
                shape.k_max,
                shape.pooled,
                groups,
                false,
            );
        }
    }
}
