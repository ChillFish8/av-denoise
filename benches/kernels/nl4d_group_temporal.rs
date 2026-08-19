use av_denoise::collab::geometry::{member_buf_len, member_frame_buf_len, ref_count, refs_along};
use av_denoise::collab::kernels::group_temporal::collab_group_temporal;
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::{H, W, block_sync, make_padded_frame, shapes_with_ch, stored_channels};

const RADIUS: u32 = 2;
const REFINE: u32 = 2;
const SPATIAL_RADIUS: u32 = 9;
const K_MAX: u32 = 8;
const BLK_STEP: u32 = 8;
// The library's own default motion block side length
// (`MotionCompensationMode::Mvtools`'s `blksize`), distinct from
// `BLK_STEP` above, which this bench keeps at `PATCH_SIZE` so a block
// boundary lines up with a patch boundary.
const BLKSIZE: u32 = 16;
// `thsad(BLKSIZE, 1.0)` in normalised SAD units (block_area *
// THSAD_PIXEL, see `crate::nlmeans::motion::thsad`), hand-computed here
// since that function is crate-private.
const THSAD: f32 = (BLKSIZE * BLKSIZE) as f32 * 0.02;
const MISMATCH_SCALE: f32 = 1.0;
const N_FRAMES: u32 = 2 * RADIUS + 1;
const CENTRE_SLOT: u32 = RADIUS;

// The centre slot is skipped, and physical slots run 0..N_FRAMES, so
// `neighbour_slots[t]` for the neighbour at temporal offset `k` is
// `k + RADIUS`, laid out in the same `neighbour_idx_for_k` order
// `crate::nlmeans::motion::chain` uses: ascending k on the negative
// side first, then ascending k on the positive side.
const NEIGHBOUR_SLOTS: [u32; (2 * RADIUS) as usize] = [0, 1, 3, 4];

/// The temporal grouping kernel at the library's default search
/// geometry, over a 1080p frame ring. One cube per reference patch, its
/// 64 threads score the 19x19 centre-frame window plus a 5x5 refine
/// window for each of the 4 neighbour frames.
///
/// Confidence is uniformly 1.0, so no neighbour block is gated and every
/// candidate the kernel finds runs the full patch comparison. Gating a
/// block skips its comparisons entirely, so leaving it always open here
/// measures the worst case. A bench that gates freely would report a
/// time well under the real one.
pub struct CollabGroupTemporalBench<R: Runtime> {
    pub client: ComputeClient<R>,
    pub ch: u32,
    pub ch_name: &'static str,
}

#[derive(Clone)]
pub struct CollabGroupTemporalInput {
    pub ring: Handle,
    pub mv_field: Handle,
    pub confidence: Handle,
    pub neighbour_slots: Handle,
    pub member_pos: Handle,
    pub member_frame: Handle,
    pub member_count: Handle,
    pub member_sig2: Handle,
    pub ring_len: usize,
}

impl<R: Runtime> Benchmark for CollabGroupTemporalBench<R> {
    type Input = CollabGroupTemporalInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let mut ring_data = Vec::new();
        for _ in 0..N_FRAMES {
            ring_data.extend(make_padded_frame(W, H, self.ch));
        }
        let ring = self.client.create_from_slice(f32::as_bytes(&ring_data));

        let blocks_x = W.div_ceil(BLK_STEP);
        let blocks_y = H.div_ceil(BLK_STEP);
        let mv_stride = blocks_x * blocks_y * 2;
        let conf_stride = blocks_x * blocks_y;

        let mv_data = vec![0i32; (2 * RADIUS * mv_stride) as usize];
        let mv_field = self.client.create_from_slice(i32::as_bytes(&mv_data));
        let conf_data = vec![1.0f32; (2 * RADIUS * conf_stride) as usize];
        let confidence = self.client.create_from_slice(f32::as_bytes(&conf_data));
        let neighbour_slots = self.client.create_from_slice(u32::as_bytes(&NEIGHBOUR_SLOTS));

        let pos_len = member_buf_len(W, H, K_MAX);
        let frame_len = member_frame_buf_len(W, H, K_MAX);
        let refs = ref_count(W, H);
        let member_pos = self.client.empty(pos_len * size_of::<u32>());
        let member_frame = self.client.empty(frame_len * size_of::<u32>());
        let member_count = self.client.empty(refs * size_of::<u32>());
        let member_sig2 = self.client.empty(pos_len * size_of::<f32>());

        CollabGroupTemporalInput {
            ring,
            mv_field,
            confidence,
            neighbour_slots,
            member_pos,
            member_frame,
            member_count,
            member_sig2,
            ring_len: ring_data.len(),
        }
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let pos_len = member_buf_len(W, H, K_MAX);
        let frame_len = member_frame_buf_len(W, H, K_MAX);
        let refs = ref_count(W, H);
        let refs_x = refs_along(W);
        let refs_y = refs_along(H);

        let blocks_x = W.div_ceil(BLK_STEP);
        let blocks_y = H.div_ceil(BLK_STEP);
        let mv_stride = blocks_x * blocks_y * 2;
        let conf_stride = blocks_x * blocks_y;

        let grid = CubeCount::new_2d(refs_x, refs_y);
        let dim = CubeDim::new_2d(8, 8);

        unsafe {
            collab_group_temporal::launch_unchecked::<R>(
                &self.client,
                grid,
                dim,
                stored_channels(self.ch) as usize,
                ArrayArg::from_raw_parts(args.ring.clone(), args.ring_len),
                ArrayArg::from_raw_parts(args.mv_field.clone(), (2 * RADIUS * mv_stride) as usize),
                ArrayArg::from_raw_parts(args.confidence.clone(), (2 * RADIUS * conf_stride) as usize),
                ArrayArg::from_raw_parts(args.member_pos.clone(), pos_len),
                ArrayArg::from_raw_parts(args.member_frame.clone(), frame_len),
                ArrayArg::from_raw_parts(args.member_count.clone(), refs),
                ArrayArg::from_raw_parts(args.member_sig2.clone(), pos_len),
                CENTRE_SLOT,
                ArrayArg::from_raw_parts(args.neighbour_slots.clone(), NEIGHBOUR_SLOTS.len()),
                0.0f32,
                0.0f32,
                THSAD,
                MISMATCH_SCALE,
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
        Ok(())
    }

    fn name(&self) -> String {
        format!("nl4d_group_temporal_1080p_{}", self.ch_name)
    }

    fn sync(&self) {
        block_sync(&self.client);
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        shapes_with_ch(self.ch)
    }
}
