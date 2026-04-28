use cubecl::prelude::*;
use cubecl::{cube, terminate};

use crate::models::Accelerator;

/// Attempt to launch a test kernel, selecting the first accelerator that
/// the host hardware can use.
pub fn sniff_best_accelerator(enable: &[Accelerator]) -> Option<Accelerator> {
    for accelerator in enable {
        let is_enabled = match accelerator {
            #[cfg(feature = "cuda")]
            Accelerator::Cuda => run_test_kernel::<cubecl::cuda::CudaRuntime>("CUDA"),
            #[cfg(feature = "rocm")]
            Accelerator::Rocm => run_test_kernel::<cubecl::hip::HipRuntime>("ROCM"),
            #[cfg(feature = "vulkan")]
            Accelerator::Vulkan => run_test_kernel::<cubecl::wgpu::WgpuRuntime>("VULKAN"),
            #[cfg(feature = "metal")]
            Accelerator::Metal => run_test_kernel::<cubecl::wgpu::WgpuRuntime>("METAL"),
            #[cfg(feature = "cpu")]
            Accelerator::Cpu => run_test_kernel::<cubecl::cpu::CpuRuntime>("CPU"),
        };

        if is_enabled {
            return Some(*accelerator);
        }
    }

    None
}

fn run_test_kernel<R: Runtime>(name: &'static str) -> bool {
    let device = <R::Device as Default>::default();
    let client = R::client(&device);

    let result = basic_test_kernel::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        ScalarArg::new(10.0),
        ScalarArg::new(15.0),
    );

    if let Err(err) = result {
        tracing::debug!(err = ?err, "could not use {name} runtime due to an error launching the kernel");
        false
    } else {
        true
    }
}

#[cube(launch)]
fn basic_test_kernel(width: f32, height: f32) {
    let value = width * height;
    if value == 0.0 {
        terminate!();
    }
}
