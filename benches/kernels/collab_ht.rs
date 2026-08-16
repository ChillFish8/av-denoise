use av_denoise::collab::geometry::{filtered_buf_len, member_buf_len, ref_count, refs_along};
use av_denoise::collab::kernels::filter_ht::collab_filter_ht;
use av_denoise::collab::kernels::group::collab_group_spatial;
use av_denoise::collab::kernels::transforms::dct_noise_profile;
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::{H, W, block_sync, make_synthetic_frame};

const SPATIAL_RADIUS: u32 = 9;
const K_MAX: u32 = 8;
const LAMBDA_HT: f32 = 2.7;
const SIGMA: f32 = 0.02;

// Duplicated for the same reason `collab_group.rs` duplicates them:
// bench code can't reach crate-internal items, and `CollabParams` alone
// doesn't give a ready-to-use noise floor or tau.
const NOISE_FLOOR: f32 = 0.0;
const TAU_ADMIT: f32 = 1e-3;

/// The hard-threshold shrinkage kernel at the library's default search
/// geometry, over a 1080p luma frame.
///
/// Grouping runs once in `prepare`, so `execute` measures only the
/// filter launch, one cube per reference patch, an 8x8 window of
/// threads.
pub struct CollabHtBench<R: Runtime> {
    pub client: ComputeClient<R>,
}

#[derive(Clone)]
pub struct CollabHtInput {
    pub reference: Handle,
    pub member_pos: Handle,
    pub member_count: Handle,
    pub member_sig2_dummy: Handle,
    pub filtered: Handle,
    pub group_weight: Handle,
    pub sigma: Handle,
    pub dct_profile: Handle,
}

impl<R: Runtime> Benchmark for CollabHtBench<R> {
    type Input = CollabHtInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let frame = make_synthetic_frame(W, H, 1);
        let reference = self.client.create_from_slice(f32::as_bytes(&frame));

        let pos_len = member_buf_len(W, H, K_MAX);
        let refs = ref_count(W, H);
        let filt_len = filtered_buf_len(W, H);

        let member_pos = self.client.empty(pos_len * size_of::<u32>());
        let member_count = self.client.empty(refs * size_of::<u32>());

        let refs_x = refs_along(W);
        let refs_y = refs_along(H);
        let grid = CubeCount::new_2d(refs_x, refs_y);
        let dim = CubeDim::new_2d(8, 8);

        unsafe {
            collab_group_spatial::launch_unchecked::<R>(
                &self.client,
                grid,
                dim,
                1usize,
                ArrayArg::from_raw_parts(reference.clone(), (W * H) as usize),
                ArrayArg::from_raw_parts(member_pos.clone(), pos_len),
                ArrayArg::from_raw_parts(member_count.clone(), refs),
                0u32,
                NOISE_FLOOR,
                TAU_ADMIT,
                W,
                H,
                1u32,
                K_MAX,
                SPATIAL_RADIUS,
                refs_x,
            );
        }
        block_sync(&self.client);

        let member_sig2_dummy = self.client.empty(size_of::<f32>());
        let filtered = self.client.empty(filt_len * size_of::<f32>());
        let group_weight = self.client.empty(refs * size_of::<f32>());
        let sigma = self.client.create_from_slice(f32::as_bytes(&[SIGMA]));
        let dct_profile = self
            .client
            .create_from_slice(f32::as_bytes(&dct_noise_profile(0.0)));

        CollabHtInput {
            reference,
            member_pos,
            member_count,
            member_sig2_dummy,
            filtered,
            group_weight,
            sigma,
            dct_profile,
        }
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let frame_len = (W * H) as usize;
        let pos_len = member_buf_len(W, H, K_MAX);
        let refs = ref_count(W, H);
        let filt_len = filtered_buf_len(W, H);
        let refs_x = refs_along(W);
        let refs_y = refs_along(H);

        let grid = CubeCount::new_2d(refs_x, refs_y);
        let dim = CubeDim::new_2d(8, 8);

        unsafe {
            collab_filter_ht::launch_unchecked::<R>(
                &self.client,
                grid,
                dim,
                1usize,
                ArrayArg::from_raw_parts(args.reference.clone(), frame_len),
                ArrayArg::from_raw_parts(args.member_pos.clone(), pos_len),
                ArrayArg::from_raw_parts(args.member_count.clone(), refs),
                ArrayArg::from_raw_parts(args.member_sig2_dummy.clone(), 1),
                ArrayArg::from_raw_parts(args.filtered.clone(), filt_len),
                ArrayArg::from_raw_parts(args.group_weight.clone(), refs),
                0u32,
                ArrayArg::from_raw_parts(args.sigma.clone(), 1),
                ArrayArg::from_raw_parts(args.dct_profile.clone(), 8),
                LAMBDA_HT,
                false,
                W,
                H,
                1u32,
                K_MAX,
                refs_x,
            );
        }
        Ok(())
    }

    fn name(&self) -> String {
        "collab_ht_1080p_luma".to_string()
    }

    fn sync(&self) {
        block_sync(&self.client);
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![vec![W as usize, H as usize, 1]]
    }
}
