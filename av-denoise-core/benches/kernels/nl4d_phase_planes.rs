use av_denoise_core::nl4d::HALF_PEL_TAPS;
use av_denoise_core::nl4d::kernels::nl4d_phase_planes;
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::{H, W, block_sync, make_synthetic_frame, shapes_with_ch};

/// The nl4d phase-plane rebuild for one ring slot at 1080p.
pub struct PhasePlanesBench<R: Runtime> {
    pub client: ComputeClient<R>,
}

#[derive(Clone)]
pub struct PhasePlanesInput {
    pub ring: Handle,
    pub phase_ring: Handle,
    pub taps: Handle,
}

impl<R: Runtime> Benchmark for PhasePlanesBench<R> {
    type Input = PhasePlanesInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let ring = self
            .client
            .create_from_slice(f32::as_bytes(&make_synthetic_frame(W, H, 1)));
        let phase_ring = self.client.empty(4 * (W * H) as usize * size_of::<f32>());
        let taps = self.client.create_from_slice(f32::as_bytes(&HALF_PEL_TAPS));
        PhasePlanesInput {
            ring,
            phase_ring,
            taps,
        }
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let pixels = (W * H) as usize;
        unsafe {
            nl4d_phase_planes::launch_unchecked::<R>(
                &self.client,
                CubeCount::new_2d(W.div_ceil(super::BLOCK_X), H.div_ceil(super::BLOCK_Y)),
                CubeDim::new_2d(super::BLOCK_X, super::BLOCK_Y),
                1,
                ArrayArg::from_raw_parts(args.ring.clone(), pixels),
                ArrayArg::from_raw_parts(args.phase_ring.clone(), 4 * pixels),
                ArrayArg::from_raw_parts(args.taps.clone(), 8),
                0,
                W,
                H,
            );
        }
        Ok(())
    }

    fn name(&self) -> String {
        "nl4d_phase_planes_1080p_luma".to_string()
    }

    fn sync(&self) {
        block_sync(&self.client);
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        shapes_with_ch(1)
    }
}
