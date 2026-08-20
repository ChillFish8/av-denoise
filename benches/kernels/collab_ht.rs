use av_denoise::collab::geometry::{
    member_buf_len,
    member_frame_buf_len,
    member_sig2_buf_len,
    ref_count,
    refs_along,
};
use av_denoise::collab::kernels::aggregate::{cross_frame_accum_scale, weight_scale};
use av_denoise::collab::kernels::filter_ht::collab_filter_ht;
use av_denoise::collab::kernels::group_temporal::collab_group_temporal;
use av_denoise::collab::kernels::transforms::dct_noise_profile;
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::nl4d_geometry::{
    BLK_STEP,
    BLKSIZE,
    CENTRE_SLOT,
    CONFIDENCE_VARIANCE,
    K_MAX,
    LAMBDA_HT,
    N_FRAMES,
    NEIGHBOUR_SLOTS,
    RADIUS,
    REFINE,
    SIGMA,
    SPATIAL_RADIUS,
    THSAD,
};
use super::{H, W, block_sync, make_padded_frame, shapes_with_ch, stored_channels};

/// The hard-threshold shrinkage kernel as `nl4d` runs it, over a 1080p
/// frame ring.
///
/// Each combination of the kernel's `#[comptime]` flags compiles a
/// separate program, so this bench sets the two the denoiser sets.
/// `temporal` indexes members by the ring slot they were matched in, and
/// `use_member_sigma` folds each member's mismatch variance into its own
/// threshold. Measuring with either one off would report a program the
/// denoiser never runs.
///
/// Grouping runs once in `prepare`, in the same temporal mode, so
/// `member_frame` and `member_sig2` hold real values rather than whatever
/// a dummy buffer happened to contain.
///
/// Confidence is uniformly 1.0 and every group fills to `K_MAX`, the
/// group size the filter runs at and what nearly all of its work scales
/// with. `execute` measures only the filter launch, one cube per
/// reference patch.
pub struct CollabHtBench<R: Runtime> {
    pub client: ComputeClient<R>,
    pub ch: u32,
    pub ch_name: &'static str,
}

#[derive(Clone)]
pub struct CollabHtInput {
    pub ring: Handle,
    pub member_pos: Handle,
    pub member_frame: Handle,
    pub member_count: Handle,
    pub member_sig2: Handle,
    pub accum: Handle,
    pub wsum: Handle,
    pub filtered_dummy: Handle,
    pub group_weight: Handle,
    pub sigma: Handle,
    pub dct_profile: Handle,
    pub ring_len: usize,
}

/// Motion-field block counts, shared by the grouping launch and the
/// buffers it reads.
fn block_grid() -> (u32, u32, u32, u32) {
    let blocks_x = W.div_ceil(BLK_STEP);
    let blocks_y = H.div_ceil(BLK_STEP);
    (blocks_x, blocks_y, blocks_x * blocks_y * 2, blocks_x * blocks_y)
}

impl<R: Runtime> Benchmark for CollabHtBench<R> {
    type Input = CollabHtInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let stored_ch = stored_channels(self.ch);
        let pixels = (W * H) as usize;
        let frame_len = pixels * stored_ch as usize;

        let mut ring_data = Vec::new();
        for _ in 0..N_FRAMES {
            ring_data.extend(make_padded_frame(W, H, self.ch));
        }
        let ring = self.client.create_from_slice(f32::as_bytes(&ring_data));

        let (blocks_x, blocks_y, mv_stride, conf_stride) = block_grid();
        let mv_data = vec![0i32; (2 * RADIUS * mv_stride) as usize];
        let mv_field = self.client.create_from_slice(i32::as_bytes(&mv_data));
        let conf_data = vec![1.0f32; (2 * RADIUS * conf_stride) as usize];
        let confidence = self.client.create_from_slice(f32::as_bytes(&conf_data));
        let neighbour_slots = self.client.create_from_slice(u32::as_bytes(&NEIGHBOUR_SLOTS));

        let pos_len = member_buf_len(W, H, K_MAX);
        let member_frame_len = member_frame_buf_len(W, H, K_MAX);
        let sig2_len = member_sig2_buf_len(W, H, K_MAX);
        let refs = ref_count(W, H);

        let member_pos = self.client.empty(pos_len * size_of::<u32>());
        let member_frame = self.client.empty(member_frame_len * size_of::<u32>());
        let member_count = self.client.empty(refs * size_of::<u32>());
        let member_sig2 = self.client.empty(sig2_len * size_of::<f32>());

        let refs_x = refs_along(W);
        let refs_y = refs_along(H);

        unsafe {
            collab_group_temporal::launch_unchecked::<R>(
                &self.client,
                CubeCount::new_2d(refs_x, refs_y),
                CubeDim::new_2d(8, 8),
                stored_ch as usize,
                ArrayArg::from_raw_parts(ring.clone(), ring_data.len()),
                ArrayArg::from_raw_parts(mv_field, (2 * RADIUS * mv_stride) as usize),
                ArrayArg::from_raw_parts(confidence, (2 * RADIUS * conf_stride) as usize),
                ArrayArg::from_raw_parts(member_pos.clone(), pos_len),
                ArrayArg::from_raw_parts(member_frame.clone(), member_frame_len),
                ArrayArg::from_raw_parts(member_count.clone(), refs),
                ArrayArg::from_raw_parts(member_sig2.clone(), sig2_len),
                CENTRE_SLOT,
                ArrayArg::from_raw_parts(neighbour_slots, NEIGHBOUR_SLOTS.len()),
                0.0f32,
                0.0f32,
                THSAD,
                RADIUS,
                REFINE,
                mv_stride,
                conf_stride,
                BLK_STEP,
                BLKSIZE,
                blocks_x,
                blocks_y,
                W,
                H,
                self.ch,
                K_MAX,
                SPATIAL_RADIUS,
                refs_x,
            );
        }
        block_sync(&self.client);

        // One accumulator region per ring slot, the same shape
        // `Nl4dDenoiser` allocates, so the filter scatters across the
        // same address range it does in the pipeline.
        let accum = self
            .client
            .empty(frame_len * N_FRAMES as usize * size_of::<i32>());
        let wsum = self.client.empty(pixels * N_FRAMES as usize * size_of::<i32>());
        // A denoiser never asks for the filtered patches themselves, it
        // scatters straight into the accumulators, so this binds the same
        // one-element placeholder a real caller does.
        let filtered_dummy = self.client.empty(size_of::<f32>());
        let group_weight = self.client.empty(refs * size_of::<f32>());

        // Sized for the stored lane count and filled for the logical
        // ones, matching what `Nl4dDenoiser` uploads each pass.
        let mut sigma_host = vec![0.0f32; stored_ch as usize];
        sigma_host[..self.ch as usize].fill(SIGMA);
        let sigma = self.client.create_from_slice(f32::as_bytes(&sigma_host));
        let dct_profile = self
            .client
            .create_from_slice(f32::as_bytes(&dct_noise_profile(0.0)));

        CollabHtInput {
            ring,
            member_pos,
            member_frame,
            member_count,
            member_sig2,
            accum,
            wsum,
            filtered_dummy,
            group_weight,
            sigma,
            dct_profile,
            ring_len: ring_data.len(),
        }
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let stored_ch = stored_channels(self.ch);
        let pixels = (W * H) as usize;
        let frame_len = pixels * stored_ch as usize;

        let pos_len = member_buf_len(W, H, K_MAX);
        let member_frame_len = member_frame_buf_len(W, H, K_MAX);
        let sig2_len = member_sig2_buf_len(W, H, K_MAX);
        let refs = ref_count(W, H);
        let refs_x = refs_along(W);
        let refs_y = refs_along(H);

        unsafe {
            collab_filter_ht::launch_unchecked::<R>(
                &self.client,
                CubeCount::new_2d(refs_x, refs_y),
                CubeDim::new_2d(8, 8),
                stored_ch as usize,
                ArrayArg::from_raw_parts(args.ring.clone(), args.ring_len),
                ArrayArg::from_raw_parts(args.member_pos.clone(), pos_len),
                ArrayArg::from_raw_parts(args.member_frame.clone(), member_frame_len),
                ArrayArg::from_raw_parts(args.member_count.clone(), refs),
                ArrayArg::from_raw_parts(args.member_sig2.clone(), sig2_len),
                ArrayArg::from_raw_parts(args.accum.clone(), frame_len * N_FRAMES as usize),
                ArrayArg::from_raw_parts(args.wsum.clone(), pixels * N_FRAMES as usize),
                ArrayArg::from_raw_parts(args.filtered_dummy.clone(), 1),
                ArrayArg::from_raw_parts(args.group_weight.clone(), refs),
                CENTRE_SLOT,
                ArrayArg::from_raw_parts(args.sigma.clone(), stored_ch as usize),
                ArrayArg::from_raw_parts(args.dct_profile.clone(), 8),
                LAMBDA_HT,
                weight_scale(SIGMA, &dct_noise_profile(0.0)),
                cross_frame_accum_scale(SPATIAL_RADIUS, RADIUS),
                CONFIDENCE_VARIANCE,
                false,
                false,
                true,
                W,
                H,
                self.ch,
                K_MAX,
                stored_ch,
                refs_x,
            );
        }
        Ok(())
    }

    fn name(&self) -> String {
        format!("collab_ht_1080p_{}", self.ch_name)
    }

    fn sync(&self) {
        block_sync(&self.client);
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        shapes_with_ch(self.ch)
    }
}
