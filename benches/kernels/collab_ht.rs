use av_denoise::collab::geometry::{
    member_buf_len,
    member_frame_buf_len,
    member_sig2_buf_len,
    ref_count,
    refs_along,
};
use av_denoise::collab::kernels::aggregate::{ACCUM_SCALE, weight_scale};
use av_denoise::collab::kernels::filter_ht::collab_filter_ht;
use av_denoise::collab::kernels::group_temporal::collab_group_temporal;
use av_denoise::collab::kernels::transforms::dct_noise_profile;
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::{H, W, block_sync, make_synthetic_frame};

const SPATIAL_RADIUS: u32 = 9;
const K_MAX: u32 = 8;
const LAMBDA_HT: f32 = 2.7;
const SIGMA: f32 = 0.02;

// The channel-scaled distance two independent noisy copies of the same
// content are expected to show at this sigma, which is what the search
// subtracts before it ranks candidates.
const NOISE_FLOOR: f32 = 2.0 * 3.0 * SIGMA * SIGMA * 64.0;

/// The hard-threshold shrinkage kernel at the library's default search
/// geometry, over a 1080p luma frame.
///
/// Grouping runs once in `prepare`, at `radius = 0` so the search stays
/// spatial over a one-frame ring, and fills every group to `K_MAX`. That
/// is the group size the filter actually runs at, and nearly everything
/// the filter does scales with it. `execute` then measures only the
/// filter launch, one cube per reference patch, an 8x8 window of
/// threads.
pub struct CollabHtBench<R: Runtime> {
    pub client: ComputeClient<R>,
}

#[derive(Clone)]
pub struct CollabHtInput {
    pub reference: Handle,
    pub member_pos: Handle,
    // Never read: `temporal` is false below, so the kernel indexes the
    // frame ring by `frame` alone and a 1-element dummy buffer is valid
    // here.
    pub member_frame_dummy: Handle,
    pub member_count: Handle,
    pub member_sig2_dummy: Handle,
    pub accum: Handle,
    pub wsum: Handle,
    pub filtered_dummy: Handle,
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
        let pixels = (W * H) as usize;

        let frame_out_len = member_frame_buf_len(W, H, K_MAX);
        let sig2_out_len = member_sig2_buf_len(W, H, K_MAX);

        let member_pos = self.client.empty(pos_len * size_of::<u32>());
        let member_count = self.client.empty(refs * size_of::<u32>());
        let member_frame_out = self.client.empty(frame_out_len * size_of::<u32>());
        let member_sig2_out = self.client.empty(sig2_out_len * size_of::<f32>());
        let mv_dummy = self.client.create_from_slice(i32::as_bytes(&[0i32, 0i32]));
        let conf_dummy = self.client.create_from_slice(f32::as_bytes(&[1.0f32]));
        let slots_dummy = self.client.create_from_slice(u32::as_bytes(&[0u32]));

        let refs_x = refs_along(W);
        let refs_y = refs_along(H);
        let grid = CubeCount::new_2d(refs_x, refs_y);
        let dim = CubeDim::new_2d(8, 8);

        unsafe {
            collab_group_temporal::launch_unchecked::<R>(
                &self.client,
                grid,
                dim,
                1usize,
                ArrayArg::from_raw_parts(reference.clone(), (W * H) as usize),
                ArrayArg::from_raw_parts(mv_dummy, 2),
                ArrayArg::from_raw_parts(conf_dummy, 1),
                ArrayArg::from_raw_parts(member_pos.clone(), pos_len),
                ArrayArg::from_raw_parts(member_frame_out, frame_out_len),
                ArrayArg::from_raw_parts(member_count.clone(), refs),
                ArrayArg::from_raw_parts(member_sig2_out, sig2_out_len),
                0u32,
                ArrayArg::from_raw_parts(slots_dummy, 1),
                NOISE_FLOOR,
                0.0f32,
                1.0f32,
                0u32,
                0u32,
                2u32,
                1u32,
                8u32,
                8u32,
                1u32,
                1u32,
                W,
                H,
                1u32,
                K_MAX,
                SPATIAL_RADIUS,
                refs_x,
            );
        }
        block_sync(&self.client);

        let member_frame_dummy = self.client.empty(size_of::<u32>());
        let member_sig2_dummy = self.client.empty(size_of::<f32>());
        let accum = self.client.empty(pixels * size_of::<i32>());
        let wsum = self.client.empty(pixels * size_of::<i32>());
        // A denoiser never asks for the filtered patches themselves, it
        // scatters straight into the accumulators, so this binds the same
        // one-element placeholder a real caller does.
        let filtered_dummy = self.client.empty(size_of::<f32>());
        let group_weight = self.client.empty(refs * size_of::<f32>());
        let sigma = self.client.create_from_slice(f32::as_bytes(&[SIGMA]));
        let dct_profile = self
            .client
            .create_from_slice(f32::as_bytes(&dct_noise_profile(0.0)));

        CollabHtInput {
            reference,
            member_pos,
            member_frame_dummy,
            member_count,
            member_sig2_dummy,
            accum,
            wsum,
            filtered_dummy,
            group_weight,
            sigma,
            dct_profile,
        }
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let frame_len = (W * H) as usize;
        let pos_len = member_buf_len(W, H, K_MAX);
        let refs = ref_count(W, H);
        let pixels = (W * H) as usize;
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
                ArrayArg::from_raw_parts(args.member_frame_dummy.clone(), 1),
                ArrayArg::from_raw_parts(args.member_count.clone(), refs),
                ArrayArg::from_raw_parts(args.member_sig2_dummy.clone(), 1),
                ArrayArg::from_raw_parts(args.accum.clone(), pixels),
                ArrayArg::from_raw_parts(args.wsum.clone(), pixels),
                ArrayArg::from_raw_parts(args.filtered_dummy.clone(), 1),
                ArrayArg::from_raw_parts(args.group_weight.clone(), refs),
                0u32,
                ArrayArg::from_raw_parts(args.sigma.clone(), 1),
                ArrayArg::from_raw_parts(args.dct_profile.clone(), 8),
                LAMBDA_HT,
                weight_scale(SIGMA, &dct_noise_profile(0.0)),
                ACCUM_SCALE,
                false,
                false,
                false,
                false,
                W,
                H,
                1u32,
                K_MAX,
                1u32,
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
