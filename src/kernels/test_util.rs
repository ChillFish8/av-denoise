use std::mem::size_of;

use cubecl::Runtime;
use cubecl::prelude::*;
use cubecl::server::Allocation;

pub(crate) fn cpu_client() -> ComputeClient<cubecl::cpu::CpuRuntime> {
    let device = <cubecl::cpu::CpuRuntime as Runtime>::Device::default();
    cubecl::cpu::CpuRuntime::client(&device)
}

pub(crate) fn tensor_arg_1d_f32<'a>(
    allocation: &'a Allocation,
    shape: &'a [usize],
) -> TensorArg<'a, cubecl::cpu::CpuRuntime> {
    tensor_arg_f32(allocation, shape)
}

pub(crate) fn tensor_arg_f32<'a>(
    allocation: &'a Allocation,
    shape: &'a [usize],
) -> TensorArg<'a, cubecl::cpu::CpuRuntime> {
    unsafe {
        TensorArg::from_raw_parts::<f32>(
            &allocation.handle,
            &allocation.strides,
            shape,
            1,
        )
    }
}

pub(crate) fn read_1d_f32_allocation(
    client: &ComputeClient<cubecl::cpu::CpuRuntime>,
    allocation: &Allocation,
    len: usize,
) -> Vec<f32> {
    let shape = [len];
    read_f32_allocation(client, allocation, &shape)
}

pub(crate) fn read_f32_allocation(
    client: &ComputeClient<cubecl::cpu::CpuRuntime>,
    allocation: &Allocation,
    shape: &[usize],
) -> Vec<f32> {
    let descriptor =
        allocation
            .handle
            .copy_descriptor(shape, &allocation.strides, size_of::<f32>());

    bytes_to_f32_vec(client.read_one_tensor(descriptor).as_ref())
}

pub(crate) fn f32_as_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

pub(crate) fn bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(size_of::<f32>())
        .map(|chunk| {
            f32::from_ne_bytes(chunk.try_into().expect("chunk should be 4 bytes"))
        })
        .collect()
}

pub(crate) fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");

    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        let delta = (actual - expected).abs();
        assert!(
            delta <= tolerance,
            "value mismatch at index {index}: actual={actual}, expected={expected}, delta={delta}, tolerance={tolerance}"
        );
    }
}
