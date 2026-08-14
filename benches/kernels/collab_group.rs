use av_denoise::collab::geometry::{member_buf_len, ref_count, refs_along};
use av_denoise::collab::kernels::group::collab_group_spatial;
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::{H, W, block_sync, make_synthetic_frame};

const SPATIAL_RADIUS: u32 = 9;
const K_MAX: u32 = 8;

// Mirrors `collab::CollabParams::default().tau_match`-shaped admission
// values, duplicated here since bench code can't reach crate-internal
// items and `CollabParams`'s fields alone don't give a ready-to-use
// noise floor or tau. `mc_confidence.rs` duplicates `THSAD_PIXEL` for
// the same reason.
const NOISE_FLOOR: f32 = 0.0;
const TAU_ADMIT: f32 = 1e-3;

/// The spatial grouping kernel at the library's default search geometry,
/// over a 1080p luma frame. One cube per reference patch, an 8x8 window
/// of threads scoring every candidate in a 19x19 search window.
pub struct CollabGroupBench<R: Runtime> {
    pub client: ComputeClient<R>,
}

#[derive(Clone)]
pub struct CollabGroupInput {
    pub reference: Handle,
    pub member_pos: Handle,
    pub member_count: Handle,
}

impl<R: Runtime> Benchmark for CollabGroupBench<R> {
    type Input = CollabGroupInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let frame = make_synthetic_frame(W, H, 1);
        let reference = self.client.create_from_slice(f32::as_bytes(&frame));

        let pos_len = member_buf_len(W, H, K_MAX);
        let refs = ref_count(W, H);
        let member_pos = self.client.empty(pos_len * size_of::<u32>());
        let member_count = self.client.empty(refs * size_of::<u32>());

        CollabGroupInput {
            reference,
            member_pos,
            member_count,
        }
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let frame_len = (W * H) as usize;
        let pos_len = member_buf_len(W, H, K_MAX);
        let refs = ref_count(W, H);
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
                ArrayArg::from_raw_parts(args.reference.clone(), frame_len),
                ArrayArg::from_raw_parts(args.member_pos.clone(), pos_len),
                ArrayArg::from_raw_parts(args.member_count.clone(), refs),
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
        Ok(())
    }

    fn name(&self) -> String {
        "collab_group_1080p_luma".to_string()
    }

    fn sync(&self) {
        block_sync(&self.client);
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![vec![W as usize, H as usize, 1]]
    }
}
