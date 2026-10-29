use av_denoise_core::testing::test_client;
use cubecl::CompilationError;
use cubecl::prelude::*;
use cubecl::server::{LaunchError, ServerError};

#[cube(launch_unchecked)]
fn failing_expansion(output: &mut [f32]) {
    if comptime!(true) {
        panic!("expansion fails on purpose");
    }

    output[0] = 1.0;
}

#[cube(launch_unchecked)]
fn copy_first(input: &[f32], output: &mut [f32]) {
    output[0] = input[0];
}

#[test]
fn a_failed_kernel_poisons_what_reads_it() {
    let client = test_client();
    let intermediate = client.empty(size_of::<f32>());
    let output = client.empty(size_of::<f32>());

    // SAFETY: every buffer holds one `f32`, which is all the kernels touch.
    unsafe {
        failing_expansion::launch_unchecked(
            &client,
            CubeCount::new_single(),
            CubeDim::new_1d(1),
            BufferArg::from_raw_parts(intermediate.clone(), 1),
        );
        copy_first::launch_unchecked(
            &client,
            CubeCount::new_single(),
            CubeDim::new_1d(1),
            BufferArg::from_raw_parts(intermediate, 1),
            BufferArg::from_raw_parts(output.clone(), 1),
        );
    }

    let read = client.read_one(output);
    let error = read.expect_err("readback of a dependent buffer succeeded");
    let ServerError::Several { errors, .. } = error else {
        panic!("expected a Several error, got {error:?}");
    };
    assert_eq!(errors.len(), 1, "expected one failure, got {errors:?}");

    let first = errors.into_iter().next().expect("one failure");
    let ServerError::Unwritten { root, chain, .. } = first else {
        panic!("expected an Unwritten error, got {first:?}");
    };

    let from_expansion = matches!(
        *root,
        ServerError::Launch(LaunchError::CompilationError(
            CompilationError::ExpansionPanicked { .. }
        ))
    );
    assert!(from_expansion, "the root was not the expansion panic: {root:?}");
    assert!(!chain.is_empty(), "the skip chain is empty");
}
