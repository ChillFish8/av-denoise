use cubecl::AutotuneKey;
use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::tune::{LocalTuner, Tunable, TunableSet, local_tuner};
use serde::{Deserialize, Serialize};

use super::{TuneId, alternatives, fits, zeroed};
use crate::collab::geometry::fused_cubes_x_for;
use crate::collab::kernels::fused::collab_fused;
use crate::collab::{COLLAB_GROUPS, PATCH_SIZE};
use crate::nlmeans::NOISE_CURVE_BINS;

/// Every groups-per-cube count the tuner tries. The first is the default launch.
pub const COLLAB_CANDIDATES: [u32; 3] = [COLLAB_GROUPS, 4, 16];

pub fn candidate_label(index: usize) -> String {
    let groups = COLLAB_CANDIDATES[index];
    format!("c{index}_{groups}groups")
}

#[derive(AutotuneKey, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(crate) struct CollabKey {
    channels: u32,
    stored_ch: u32,
    spatial_radius: u32,
    grid_frames: u32,
    k_max: u32,
    f16_search: bool,
    warp_uniform: bool,
    pooled: bool,
    #[autotune(anchor)]
    width: usize,
    #[autotune(anchor)]
    height: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct CollabParams {
    pub stored_ch: u32,
    pub centre_slot: u32,
    pub c_min: f32,
    pub lambda_ht: f32,
    pub curve_valid: u32,
    pub map_mode: u32,
    pub weight_scale: f32,
    pub accum_scale: f32,
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
    pub spatial_radius: u32,
    pub refs_x: u32,
    pub refs_y: u32,
    pub map_cols: u32,
    pub map_rows: u32,
    pub pool_ratio: f32,
    pub pooled: bool,
}

/// One `collab_fused` pass, with everything a candidate needs to run it.
#[derive(Clone)]
pub struct CollabLaunch<R: Runtime> {
    pub client: ComputeClient<R>,
    pub ring: Handle,
    pub ring_len: usize,
    pub search_ring: Handle,
    pub search_len: usize,
    pub mv_field: Handle,
    pub mv_len: usize,
    pub confidence: Handle,
    pub conf_len: usize,
    pub neighbour_slots: Handle,
    pub neighbour_slots_len: usize,
    pub sigma: Handle,
    pub noise_curve: Handle,
    pub strength_map: Handle,
    pub map_len: usize,
    pub dct_profile: Handle,
    pub kaiser: Handle,
    pub accum: Handle,
    pub accum_len: usize,
    pub wsum: Handle,
    pub wsum_len: usize,
    pub group_weight: Handle,
    pub refs: usize,
    pub params: CollabParams,
}

impl<R: Runtime> CollabLaunch<R> {
    pub(crate) fn key(&self) -> CollabKey {
        let params = self.params;

        CollabKey::new(
            params.channels,
            params.stored_ch,
            params.spatial_radius,
            params.grid_frames,
            params.k_max,
            params.f16_search,
            params.warp_uniform,
            params.pooled,
            params.width as usize,
            params.height as usize,
        )
    }

    /// A copy whose accumulators and group weights are fresh zeroed scratch.
    pub(crate) fn with_scratch(&self) -> Self {
        let accum_bytes = self.accum_len * size_of::<i32>();
        let wsum_bytes = self.wsum_len * size_of::<i32>();
        let weight_bytes = self.refs * size_of::<f32>();

        Self {
            accum: zeroed(&self.client, accum_bytes),
            wsum: zeroed(&self.client, wsum_bytes),
            group_weight: zeroed(&self.client, weight_bytes),
            ..self.clone()
        }
    }

    pub fn launch_candidate(&self, index: usize) -> Result<(), String> {
        let groups = COLLAB_CANDIDATES[index];
        let threads = groups * 8;
        let shared_bytes = (groups * 65) as usize * size_of::<f32>();
        let hardware = &self.client.properties().hardware;
        if index != 0 && !fits(hardware, threads, shared_bytes) {
            let label = candidate_label(index);
            return Err(format!("{label} does not fit this device"));
        }

        match self.params.f16_search {
            true => self.launch_with::<half::f16>(groups),
            false => self.launch_with::<f32>(groups),
        }

        Ok(())
    }

    fn launch_with<S: Float>(&self, groups: u32) {
        let params = self.params;
        let cubes_x = fused_cubes_x_for(params.width, groups);
        let grid = CubeCount::new_2d(cubes_x, params.refs_y);
        let dim = CubeDim::new_1d(groups * 8);

        unsafe {
            collab_fused::launch_unchecked::<S, R>(
                &self.client,
                grid,
                dim,
                params.stored_ch as usize,
                ArrayArg::from_raw_parts(self.ring.clone(), self.ring_len),
                ArrayArg::from_raw_parts(self.search_ring.clone(), self.search_len),
                ArrayArg::from_raw_parts(self.mv_field.clone(), self.mv_len),
                ArrayArg::from_raw_parts(self.confidence.clone(), self.conf_len),
                ArrayArg::from_raw_parts(self.neighbour_slots.clone(), self.neighbour_slots_len),
                ArrayArg::from_raw_parts(self.sigma.clone(), params.stored_ch as usize),
                ArrayArg::from_raw_parts(self.noise_curve.clone(), NOISE_CURVE_BINS),
                ArrayArg::from_raw_parts(self.strength_map.clone(), self.map_len),
                ArrayArg::from_raw_parts(self.dct_profile.clone(), 8),
                ArrayArg::from_raw_parts(self.kaiser.clone(), PATCH_SIZE as usize),
                ArrayArg::from_raw_parts(self.accum.clone(), self.accum_len),
                ArrayArg::from_raw_parts(self.wsum.clone(), self.wsum_len),
                ArrayArg::from_raw_parts(self.group_weight.clone(), self.refs),
                params.centre_slot,
                params.c_min,
                params.lambda_ht,
                params.curve_valid,
                params.map_mode,
                params.weight_scale,
                params.accum_scale,
                params.warp_uniform,
                params.f16_search,
                params.radius,
                params.grid_frames,
                params.refine,
                params.mv_stride,
                params.conf_stride,
                params.blk_step,
                params.blksize,
                params.blocks_x,
                params.blocks_y,
                params.width,
                params.height,
                params.channels,
                params.k_max,
                params.stored_ch,
                params.spatial_radius,
                params.refs_x,
                params.map_cols,
                params.map_rows,
                params.pool_ratio,
                params.pooled,
                groups,
            );
        }
    }
}

/// Runs the fastest collab candidate for this device, tuning on the first call for a key.
pub(crate) fn launch<R: Runtime>(collab: CollabLaunch<R>) {
    static TUNER: LocalTuner<CollabKey, TuneId> = local_tuner!("collab");

    let tunables = TUNER.init(|| {
        let group = alternatives();
        let mut set = TunableSet::new(
            |collab: &CollabLaunch<R>| collab.key(),
            |_key: &CollabKey, collab: &CollabLaunch<R>| collab.with_scratch(),
        );

        for index in 0..COLLAB_CANDIDATES.len() {
            let label = candidate_label(index);
            let tunable = Tunable::new(&label, move |collab: CollabLaunch<R>| {
                collab.launch_candidate(index)
            });

            set = match index {
                0 => set.with(tunable),
                _ => set.with(tunable.group(&group, |_key| 0)),
            };
        }

        set
    });

    let client = collab.client.clone();
    let tune_id = TuneId::new(&client);
    TUNER.execute(&tune_id, &client, tunables, collab);
}
