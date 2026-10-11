use std::marker::PhantomData;

use cubecl::prelude::*;
use cubecl::server::Handle;

use super::COLLAB_FUSED;
use crate::collab::PATCH_SIZE;
use crate::collab::geometry::fused_cubes_x;
use crate::collab::kernels::fused::collab_fused;
use crate::nlmeans::NOISE_CURVE_BINS;
use crate::tune::fidelity::{CompareResult, Fidelity, diff_integers, diff_plane};
use crate::tune::gpu::{bind, read_f32, read_i32, zero_words};
use crate::tune::key::SiteKey;
use crate::tune::site::{LaunchShape, SiteCandidate, VariantSite};

/// Floats in each group's transpose region, one past 64 so lanes never collide on a bank.
const TRANSPOSE_REGION_FLOATS: u32 = 65;

/// Lanes in one group, one per patch column.
const GROUP_LANES: u32 = 8;

/// How many reference patches one `collab_fused` cube handles.
///
/// The scatter adds fixed-point integers, so the output is the same for every group count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollabCandidate {
    Groups8,
    Groups4,
    Groups16,
}

impl CollabCandidate {
    /// Reference patches, and so 8-lane groups, per cube.
    pub fn groups(self) -> u32 {
        match self {
            CollabCandidate::Groups8 => 8,
            CollabCandidate::Groups4 => 4,
            CollabCandidate::Groups16 => 16,
        }
    }
}

impl SiteCandidate for CollabCandidate {
    const SITE: &'static str = COLLAB_FUSED;
    const ALL: &'static [Self] = &[
        CollabCandidate::Groups8,
        CollabCandidate::Groups4,
        CollabCandidate::Groups16,
    ];

    fn name(self) -> &'static str {
        match self {
            CollabCandidate::Groups8 => "groups8",
            CollabCandidate::Groups4 => "groups4",
            CollabCandidate::Groups16 => "groups16",
        }
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
    pub warp_uniform: bool,
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
    type Candidate = CollabCandidate;
    type Outputs = CollabOutputs;

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
            .param("warp", u8::from(shape.warp_uniform))
            .param("f16", u8::from(shape.f16_search))
            .build()
    }

    fn shape(&self, candidate: CollabCandidate) -> LaunchShape {
        let lanes = GROUP_LANES * candidate.groups();
        LaunchShape { x: lanes, y: 1 }
    }

    fn shared_bytes(&self, candidate: CollabCandidate) -> usize {
        let floats = candidate.groups() * TRANSPOSE_REGION_FLOATS;
        floats as usize * size_of::<f32>()
    }

    fn min_plane_size(&self, _candidate: CollabCandidate) -> u32 {
        GROUP_LANES
    }

    fn launch(&self, candidate: CollabCandidate, outputs: &CollabOutputs) {
        let groups = candidate.groups();
        let inputs = &self.inputs;
        let values = &self.values;
        let shape = &self.shape;
        let cubes_x = fused_cubes_x(shape.width, groups);
        let cube_count = CubeCount::new_2d(cubes_x, shape.refs_y);
        let cube_dim = CubeDim::new_1d(GROUP_LANES * groups);

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
                shape.warp_uniform,
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
                groups,
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
}
