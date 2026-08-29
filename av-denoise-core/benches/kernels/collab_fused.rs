use av_denoise_core::collab::geometry::{fused_cubes_x, ref_count, refs_along};
use av_denoise_core::collab::kernels::aggregate::{cross_frame_accum_scale, weight_scale};
use av_denoise_core::collab::kernels::fused::collab_fused;
use av_denoise_core::collab::kernels::transforms::dct_noise_profile;
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
    conf_stride,
    mv_stride,
};
use super::{H, W, block_sync, make_padded_frame, shapes_with_ch, stored_channels};

/// The fused collaborative kernel at the library's default search
/// geometry, over a 1080p frame ring. One 64-lane cube carries eight
/// reference patches, eight lanes each, so the grid is an eighth as wide
/// along x as the reference count.
///
/// This is the whole collaborative stage in one launch, matching,
/// filtering and scatter.
///
/// Confidence is uniformly 1.0, so no neighbour block is gated and every
/// candidate the kernel finds runs the full patch comparison. Gating a
/// block skips its comparisons entirely, so leaving it always open here
/// measures the worst case. A bench that gates freely would report a
/// time well under the real one.
pub struct CollabFusedBench<R: Runtime> {
    pub client: ComputeClient<R>,
    pub ch: u32,
    pub ch_name: &'static str,
}

#[derive(Clone)]
pub struct CollabFusedInput {
    pub ring: Handle,
    pub mv_field: Handle,
    pub confidence: Handle,
    pub neighbour_slots: Handle,
    pub sigma: Handle,
    pub dct_profile: Handle,
    pub accum: Handle,
    pub wsum: Handle,
    pub group_weight: Handle,
    pub ring_len: usize,
}

impl<R: Runtime> Benchmark for CollabFusedBench<R> {
    type Input = CollabFusedInput;
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

        let blocks_x = W.div_ceil(BLK_STEP);
        let blocks_y = H.div_ceil(BLK_STEP);
        let align = self.client.properties().memory.alignment;
        let mv_stride = mv_stride(blocks_x, blocks_y, align);
        let conf_stride = conf_stride(blocks_x, blocks_y, align);

        let mv_data = vec![0i32; (2 * RADIUS * mv_stride) as usize];
        let mv_field = self.client.create_from_slice(i32::as_bytes(&mv_data));
        let conf_data = vec![1.0f32; (2 * RADIUS * conf_stride) as usize];
        let confidence = self.client.create_from_slice(f32::as_bytes(&conf_data));
        let neighbour_slots = self.client.create_from_slice(u32::as_bytes(&NEIGHBOUR_SLOTS));

        // Sized for the stored lane count and filled for the logical
        // ones, matching what `Nl4dDenoiser` uploads each pass.
        let mut sigma_host = vec![0.0f32; stored_ch as usize];
        sigma_host[..self.ch as usize].fill(SIGMA);
        let sigma = self.client.create_from_slice(f32::as_bytes(&sigma_host));
        let dct_profile = self
            .client
            .create_from_slice(f32::as_bytes(&dct_noise_profile(0.0)));

        // One accumulator region per ring slot, the same shape
        // `Nl4dDenoiser` allocates, so the scatter crosses the same
        // address range it does in the pipeline.
        let accum = self
            .client
            .empty(frame_len * N_FRAMES as usize * size_of::<i32>());
        let wsum = self.client.empty(pixels * N_FRAMES as usize * size_of::<i32>());
        let group_weight = self.client.empty(ref_count(W, H) * size_of::<f32>());

        CollabFusedInput {
            ring,
            mv_field,
            confidence,
            neighbour_slots,
            sigma,
            dct_profile,
            accum,
            wsum,
            group_weight,
            ring_len: ring_data.len(),
        }
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let stored_ch = stored_channels(self.ch);
        let pixels = (W * H) as usize;
        let frame_len = pixels * stored_ch as usize;
        let refs = ref_count(W, H);
        let refs_x = refs_along(W);
        let refs_y = refs_along(H);

        let blocks_x = W.div_ceil(BLK_STEP);
        let blocks_y = H.div_ceil(BLK_STEP);
        let align = self.client.properties().memory.alignment;
        let mv_stride = mv_stride(blocks_x, blocks_y, align);
        let conf_stride = conf_stride(blocks_x, blocks_y, align);

        let grid = CubeCount::new_2d(fused_cubes_x(W), refs_y);
        let dim = CubeDim::new_1d(64);

        unsafe {
            collab_fused::launch_unchecked::<R>(
                &self.client,
                grid,
                dim,
                stored_ch as usize,
                ArrayArg::from_raw_parts(args.ring.clone(), args.ring_len),
                ArrayArg::from_raw_parts(args.mv_field.clone(), (2 * RADIUS * mv_stride) as usize),
                ArrayArg::from_raw_parts(args.confidence.clone(), (2 * RADIUS * conf_stride) as usize),
                ArrayArg::from_raw_parts(args.neighbour_slots.clone(), NEIGHBOUR_SLOTS.len()),
                ArrayArg::from_raw_parts(args.sigma.clone(), stored_ch as usize),
                ArrayArg::from_raw_parts(args.dct_profile.clone(), 8),
                ArrayArg::from_raw_parts(args.accum.clone(), frame_len * N_FRAMES as usize),
                ArrayArg::from_raw_parts(args.wsum.clone(), pixels * N_FRAMES as usize),
                ArrayArg::from_raw_parts(args.group_weight.clone(), refs),
                CENTRE_SLOT,
                0.0f32,
                0.0f32,
                THSAD,
                LAMBDA_HT,
                weight_scale(SIGMA, &dct_noise_profile(0.0)),
                cross_frame_accum_scale(SPATIAL_RADIUS, RADIUS),
                CONFIDENCE_VARIANCE,
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
                stored_ch,
                SPATIAL_RADIUS,
                refs_x,
            );
        }
        Ok(())
    }

    fn name(&self) -> String {
        format!("collab_fused_1080p_{}", self.ch_name)
    }

    fn sync(&self) {
        block_sync(&self.client);
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        shapes_with_ch(self.ch)
    }
}
