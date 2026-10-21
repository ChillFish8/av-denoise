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
use super::search_bytes::{search_candidate_fits, search_candidate_legal, search_shared_bytes};
use crate::collab::PATCH_SIZE;
use crate::collab::geometry::fused_cubes_x;
use crate::collab::integer::{CodeView, tie_bound_verdict};
use crate::collab::kernels::fused::{
    CollabVariant,
    GROUP_LANES,
    MEMBER_WORDS,
    TRANSPOSE_REGION_FLOATS,
    collab_filter,
    collab_search,
};
use crate::collab::kernels::integer::neighbour::NEIGHBOUR_GROUPS;
use crate::collab::kernels::integer::{
    INTEGER_VARIANT_COUNT,
    INTEGER_VARIANTS,
    IntegerVariant,
    builds_volume_grid,
    collab_centre_sliding,
    collab_neighbour_int,
    collab_search_int,
};
use crate::nlmeans::NOISE_CURVE_BINS;
use crate::tune::fidelity::{CompareResult, Fidelity, IntegerBound, diff_integers};
use crate::tune::gpu::{bind, read_i32, read_u16, zero_words, zeroed_words};
use crate::tune::key::SiteKey;
use crate::tune::site::{LaunchShape, SiteCandidate, VariantSite};
use crate::tune::{DeviceLimits, Legality};

/// The collab search kernel.
pub const COLLAB_SEARCH: &str = "collab_search";
/// The collab filter kernel.
pub const COLLAB_FILTER: &str = "collab_filter";

/// How many float search variants exist, the collab variants with 8 groups per cube.
pub const FLOAT_SEARCH_COUNT: usize = VARIANT_COUNT / 3;

/// How many search variants exist, the float ones and then the integer ones.
pub const SEARCH_VARIANT_COUNT: usize = FLOAT_SEARCH_COUNT + INTEGER_VARIANT_COUNT;

/// The float search's group count, which every float search variant shares.
const SEARCH_GROUPS: u32 = 8;

/// The cubes of one search launch, the centre kernel's and the neighbour pass's.
#[derive(Debug, Clone)]
struct SearchCubes {
    centre: CubeCount,
    neighbour: CubeCount,
}

const fn search_variants() -> [SearchVariant; SEARCH_VARIANT_COUNT] {
    let mut variants = [SearchVariant::Float(VARIANTS[0]); SEARCH_VARIANT_COUNT];
    let mut index = 0;

    while index < FLOAT_SEARCH_COUNT {
        variants[index] = SearchVariant::Float(VARIANTS[index]);
        index += 1;
    }

    let mut integer_index = 0;

    while integer_index < INTEGER_VARIANT_COUNT {
        let integer = INTEGER_VARIANTS[integer_index];
        variants[FLOAT_SEARCH_COUNT + integer_index] = SearchVariant::Integer(integer);
        integer_index += 1;
    }

    variants
}

const fn leading_variants_have_search_groups() -> bool {
    let mut index = 0;

    while index < FLOAT_SEARCH_COUNT {
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
    "the first FLOAT_SEARCH_COUNT collab variants must all run 8 groups per cube"
);

/// Every search variant, baseline first.
pub const SEARCH_VARIANTS: [SearchVariant; SEARCH_VARIANT_COUNT] = search_variants();

/// One way to run the search, a float collab variant at 8 groups per cube or an integer variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchVariant {
    /// `collab_search` over the f32 or f16 ring.
    Float(CollabVariant),
    /// `collab_search_int` over the integer code ring.
    Integer(IntegerVariant),
}

impl SearchVariant {
    /// Threads in one cube.
    pub fn lanes(self) -> u32 {
        match self {
            SearchVariant::Float(variant) => variant.lanes(),
            SearchVariant::Integer(variant) => variant.lanes(),
        }
    }
}

impl SiteCandidate for SearchVariant {
    const SITE: &'static str = COLLAB_SEARCH;
    const ALL: &'static [Self] = &SEARCH_VARIANTS;

    fn name(self) -> &'static str {
        match self {
            SearchVariant::Float(variant) => variant.name(),
            SearchVariant::Integer(variant) => variant.name(),
        }
    }

    fn fidelity(self) -> Fidelity {
        match self {
            SearchVariant::Float(variant) => variant.fidelity(),
            SearchVariant::Integer(_) => Fidelity::ExactInteger {
                reference: IntegerBound::EighthBest,
            },
        }
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

    /// Only a key with codes names their depth, so every float key keeps its stored text.
    fn key(&self) -> SiteKey {
        let shape = &self.shape;
        let builder = collab_key_builder(COLLAB_SEARCH, shape);
        let f16_flag = u8::from(shape.f16_search);
        let float_builder = builder.param("f16", f16_flag);

        if let Some(codes) = shape.codes {
            let integer_builder = float_builder.param("int", codes.depth);
            return integer_builder.build();
        }

        float_builder.build()
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
        let lanes = candidate.lanes();
        LaunchShape { x: lanes, y: 1 }
    }

    fn shared_bytes(&self, candidate: SearchVariant) -> usize {
        let search_bytes = size_of::<S>();
        search_shared_bytes(candidate, &self.shape, search_bytes)
    }

    fn min_plane_size(&self, _candidate: SearchVariant) -> u32 {
        GROUP_LANES
    }

    /// A sliding variant launches its centre kernel and then, when the search builds volume grids,
    /// the neighbour pass.
    fn launch(&self, candidate: SearchVariant, outputs: &SearchOutputs) {
        let shape = &self.shape;
        let centre = match candidate {
            SearchVariant::Float(_) => {
                let cubes_x = fused_cubes_x(shape.width, SEARCH_GROUPS);
                CubeCount::new_2d(cubes_x, shape.refs_y)
            },
            SearchVariant::Integer(variant) => {
                let cubes_x = fused_cubes_x(shape.width, variant.cube_refs_x);
                let cubes_y = shape.refs_y.div_ceil(variant.ref_rows);
                CubeCount::new_2d(cubes_x, cubes_y)
            },
        };
        let neighbour_x = fused_cubes_x(shape.width, NEIGHBOUR_GROUPS);
        let neighbour = CubeCount::new_2d(neighbour_x, shape.refs_y);
        let cubes = SearchCubes { centre, neighbour };

        self.dispatch(candidate, outputs, &cubes);
    }

    /// A launch with no cubes, which on wgpu compiles every kernel the candidate launches without
    /// running it.
    fn compile_only(&self, candidate: SearchVariant, outputs: &SearchOutputs) {
        let cubes = SearchCubes {
            centre: CubeCount::Static(0, 0, 1),
            neighbour: CubeCount::Static(0, 0, 1),
        };

        self.dispatch(candidate, outputs, &cubes);
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
        let inputs = blank_collab_inputs(client, self.shape.stored_ch);
        let members = zeroed_words(client, 1);
        let outputs = SearchOutputs { members, refs: 0 };

        Some(Self {
            client: client.clone(),
            inputs,
            outputs,
            ..self.clone()
        })
    }

    /// Under the eighth-best bound the candidate's members are checked against the baseline's with
    /// [tie_bound_verdict](crate::collab::integer::tie_bound_verdict). Otherwise the members must
    /// match word for word.
    fn compare(
        &self,
        baseline: &SearchOutputs,
        candidate: &SearchOutputs,
        fidelity: Fidelity,
    ) -> Result<CompareResult, anyhow::Error> {
        let expected = read_i32(&self.client, &baseline.members, baseline.words())?;
        let actual = read_i32(&self.client, &candidate.members, candidate.words())?;

        let eighth_best = Fidelity::ExactInteger {
            reference: IntegerBound::EighthBest,
        };
        if fidelity == eighth_best {
            return self.tie_bound_compare(&expected, &actual);
        }

        let diff = diff_integers(&expected, &actual);
        let verdict = fidelity.judge(&[diff]);

        Ok(verdict)
    }

    fn baseline(&self) -> SearchVariant {
        let variant = CollabVariant::baseline_for(self.shape.uniform_walk_only);
        SearchVariant::Float(variant)
    }

    /// The scalar integer variant at the baseline's cube shape, when the shape has codes.
    ///
    /// It takes its walk from the shape's flag, as the baseline does.
    fn integer_reference(&self) -> Option<SearchVariant> {
        self.shape.codes?;

        let reference = IntegerVariant::reference_for(self.shape.uniform_walk_only);
        Some(SearchVariant::Integer(reference))
    }

    fn legal(&self, candidate: SearchVariant, _limits: &DeviceLimits, legality: &Legality) -> bool {
        search_candidate_legal(candidate, &self.shape, legality)
    }

    fn fits(&self, candidate: SearchVariant, limits: &DeviceLimits, legality: &Legality) -> bool {
        let search_bytes = size_of::<S>();
        search_candidate_fits(candidate, &self.shape, limits, legality, search_bytes)
    }
}

impl<R: Runtime, S: Float> CollabSearchSite<R, S> {
    /// Checks the candidate's members against the baseline's under the eighth-best tie bound.
    ///
    /// The whole code ring is read back, and the neighbour slots are the first `2 * radius` of the
    /// slot buffer, the real slot list without its padding.
    fn tie_bound_compare(&self, baseline: &[i32], candidate: &[i32]) -> Result<CompareResult, anyhow::Error> {
        let shape = &self.shape;
        let inputs = &self.inputs;

        if shape.codes.is_none() {
            let reason = String::from("the eighth-best bound needs a code ring, and this search has none");
            return Ok(CompareResult::Fail(reason));
        }

        let codes = read_u16(&self.client, &inputs.codes, inputs.codes_len)?;
        let widened_codes: Vec<u32> = codes.iter().map(|&code| u32::from(code)).collect();
        let slot_words = read_i32(&self.client, &inputs.neighbour_slots, inputs.neighbour_slots_len)?;
        let neighbours = (2 * shape.radius) as usize;
        let neighbour_slots: Vec<u32> = slot_words
            .iter()
            .take(neighbours)
            .map(|&slot| slot as u32)
            .collect();

        let view = CodeView {
            codes: &widened_codes,
            width: shape.width,
            height: shape.height,
            stored_ch: shape.stored_ch,
            channels: shape.channels,
            centre_slot: self.values.centre_slot,
            neighbour_slots: &neighbour_slots,
        };
        let baseline_words: Vec<u32> = baseline.iter().map(|&word| word as u32).collect();
        let candidate_words: Vec<u32> = candidate.iter().map(|&word| word as u32).collect();
        let verdict = tie_bound_verdict(
            &view,
            shape.refs_x,
            shape.refs_y,
            shape.grid_frames,
            &baseline_words,
            &candidate_words,
        );

        Ok(verdict)
    }

    /// Launches `candidate` over `cubes`.
    fn dispatch(&self, candidate: SearchVariant, outputs: &SearchOutputs, cubes: &SearchCubes) {
        match candidate {
            SearchVariant::Float(variant) => {
                let centre = cubes.centre.clone();
                self.dispatch_float(variant, outputs, centre);
            },
            SearchVariant::Integer(variant) if variant.is_sliding() => {
                self.dispatch_sliding(variant, outputs, cubes);
            },
            SearchVariant::Integer(variant) => {
                let centre = cubes.centre.clone();
                self.dispatch_integer(variant, outputs, centre);
            },
        }
    }

    /// Launches `collab_centre_sliding` with `variant`, then `collab_neighbour_int` when the search
    /// builds volume grids, both reading u16 codes.
    fn dispatch_sliding(&self, variant: IntegerVariant, outputs: &SearchOutputs, cubes: &SearchCubes) {
        let inputs = &self.inputs;
        let values = &self.values;
        let shape = &self.shape;
        let centre_dim = CubeDim::new_1d(variant.lanes());
        let words = outputs.words().max(1);
        let centre_cubes = cubes.centre.clone();

        let centre_codes = bind::<R>(&inputs.codes, inputs.codes_len);
        let centre_members = bind::<R>(&outputs.members, words);

        unsafe {
            collab_centre_sliding::launch_unchecked::<u16, R>(
                &self.client,
                centre_cubes,
                centre_dim,
                shape.stored_ch as usize,
                centre_codes,
                centre_members,
                values.centre_slot,
                shape.width,
                shape.height,
                shape.channels,
                shape.k_max,
                shape.spatial_radius,
                shape.refs_x,
                variant,
            );
        }

        if !builds_volume_grid(shape.grid_frames, shape.k_max) {
            return;
        }

        let neighbour_lanes = GROUP_LANES * NEIGHBOUR_GROUPS;
        let neighbour_dim = CubeDim::new_1d(neighbour_lanes);
        let neighbour_cubes = cubes.neighbour.clone();

        let codes = bind::<R>(&inputs.codes, inputs.codes_len);
        let mv_field = bind::<R>(&inputs.mv_field, inputs.mv_len);
        let confidence = bind::<R>(&inputs.confidence, inputs.conf_len);
        let neighbour_slots = bind::<R>(&inputs.neighbour_slots, inputs.neighbour_slots_len);
        let members = bind::<R>(&outputs.members, words);

        unsafe {
            collab_neighbour_int::launch_unchecked::<u16, R>(
                &self.client,
                neighbour_cubes,
                neighbour_dim,
                shape.stored_ch as usize,
                codes,
                mv_field,
                confidence,
                neighbour_slots,
                members,
                values.centre_slot,
                values.c_min,
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
                variant.walk,
            );
        }
    }

    /// Launches `collab_search_int` with `variant` over `cube_count` cubes, reading u16 codes.
    fn dispatch_integer(&self, variant: IntegerVariant, outputs: &SearchOutputs, cube_count: CubeCount) {
        let lanes = variant.lanes();
        let inputs = &self.inputs;
        let values = &self.values;
        let shape = &self.shape;
        let cube_dim = CubeDim::new_1d(lanes);
        let words = outputs.words().max(1);

        let codes = bind::<R>(&inputs.codes, inputs.codes_len);
        let mv_field = bind::<R>(&inputs.mv_field, inputs.mv_len);
        let confidence = bind::<R>(&inputs.confidence, inputs.conf_len);
        let neighbour_slots = bind::<R>(&inputs.neighbour_slots, inputs.neighbour_slots_len);
        let members = bind::<R>(&outputs.members, words);

        unsafe {
            collab_search_int::launch_unchecked::<u16, R>(
                &self.client,
                cube_count,
                cube_dim,
                shape.stored_ch as usize,
                codes,
                mv_field,
                confidence,
                neighbour_slots,
                members,
                values.centre_slot,
                values.c_min,
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
                variant,
            );
        }
    }

    /// Launches `collab_search` with `variant` over `cube_count` cubes.
    fn dispatch_float(&self, variant: CollabVariant, outputs: &SearchOutputs, cube_count: CubeCount) {
        let lanes = variant.lanes();
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
                variant,
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
    ///
    /// The codes are left out because the filter never reads them, so every profile shares one filter
    /// identity.
    fn compile_identity(&self) -> String {
        let shape = CollabShape {
            codes: None,
            ..self.shape
        };

        format!("{shape:?}")
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
        let inputs = blank_collab_inputs(client, self.shape.stored_ch);
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
        let sigma = bind::<R>(&inputs.sigma, shape.stored_ch as usize);
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
                sigma,
                noise_curve,
                strength_map,
                dct_profile,
                kaiser,
                accum,
                wsum,
                group_weight,
                values.centre_slot,
                values.lambda_ht,
                values.curve_valid,
                values.map_mode,
                values.weight_scale,
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
