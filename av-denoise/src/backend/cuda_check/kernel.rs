use std::f32::consts::E;

use cubecl::prelude::*;

use super::CudaUnavailable;

#[cube(launch_unchecked)]
fn libdevice_probe(input: &[f32], output: &mut [f32]) {
    output[0] = f32::exp(input[0]);
}

/// Compiles and runs a kernel that needs libdevice, then checks the result.
///
/// `exp` has no NVPTX instruction, so compiling it links `libdevice.10.bc`. A kernel that fails to
/// compile marks its outputs, which makes the readback an error.
pub(super) fn run_libdevice_probe(client: &Client) -> Result<(), CudaUnavailable> {
    let input_values = [1.0f32];
    let input_bytes = f32::as_bytes(&input_values);
    let input = client.create_from_slice(input_bytes);
    let output = client.empty(size_of::<f32>());

    // SAFETY: both buffers hold one `f32`, which is all the kernel touches.
    unsafe {
        libdevice_probe::launch_unchecked(
            client,
            CubeCount::new_single(),
            CubeDim::new_1d(1),
            BufferArg::from_raw_parts(input, 1),
            BufferArg::from_raw_parts(output.clone(), 1),
        );
    }

    let read = client.read_one(output);
    let bytes = match read {
        Ok(bytes) => bytes,
        Err(error) => {
            let message = format!("{error:?}");
            return Err(CudaUnavailable::TestKernel(message));
        },
    };

    let values = f32::from_bytes(&bytes);
    let value = values[0];
    if (value - E).abs() < 1e-4 {
        return Ok(());
    }

    let message = format!("exp(1.0) returned {value}");
    Err(CudaUnavailable::TestKernel(message))
}
