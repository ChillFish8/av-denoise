use av_denoise::nlmeans::kernels::{
    nlm_fused_pair_accumulate_window,
    nlm_fused_pair_accumulate_window_ref,
    nlm_fused_single_window,
    nlm_fused_single_window_ref,
};
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::{
    BLOCK_X,
    BLOCK_Y,
    H,
    PATCH_RADIUS,
    SEARCH_RADIUS,
    W,
    block_sync,
    cube_count_2d,
    cube_dim_2d,
    h2_inv_norm,
    make_padded_frame,
    shapes_with_ch,
    stored_channels,
};

/// Zero-filled spatial-offset LUT for `SEARCH_RADIUS`. Zero everywhere
/// reproduces the old flat `noise_offset = 0.0` the single-window
/// benches measured before the LUT replaced that scalar, so the
/// timing stays comparable.
fn zero_spatial_offset_lut<R: Runtime>(client: &ComputeClient<R>) -> (Handle, usize) {
    let side = (2 * SEARCH_RADIUS + 1) as usize;
    let lut = vec![0.0f32; side * side];
    let handle = client.create_from_slice(f32::as_bytes(&lut));
    (handle, lut.len())
}

#[derive(Clone)]
pub struct WindowInput {
    input: Handle,
    accum: Handle,
    weight_sum: Handle,
    max_weight: Handle,
    weight_sq_sum_dummy: Handle,
    confidence_dummy: Handle,
    spatial_offset_lut: Handle,
    spatial_offset_lut_len: usize,
    frame_len: usize,
}

#[derive(Clone)]
pub struct WindowRefInput {
    input: Handle,
    reference: Handle,
    accum: Handle,
    weight_sum: Handle,
    max_weight: Handle,
    weight_sq_sum_dummy: Handle,
    confidence_dummy: Handle,
    spatial_offset_lut: Handle,
    spatial_offset_lut_len: usize,
    frame_len: usize,
}

fn prepare_window<R: Runtime>(client: &ComputeClient<R>, ch: u32) -> WindowInput {
    let pixels = (W * H) as usize;
    let stored = stored_channels(ch) as usize;
    let frame = make_padded_frame(W, H, ch);
    let input = client.create_from_slice(f32::as_bytes(&frame));
    let accum = client.empty(pixels * stored * size_of::<f32>());
    let weight_sum = client.empty(pixels * size_of::<f32>());
    let max_weight = client.empty(pixels * size_of::<f32>());
    let weight_sq_sum_dummy = client.empty(size_of::<f32>());
    let confidence_dummy = client.empty(size_of::<f32>());
    let (spatial_offset_lut, spatial_offset_lut_len) = zero_spatial_offset_lut(client);
    WindowInput {
        input,
        accum,
        weight_sum,
        max_weight,
        weight_sq_sum_dummy,
        confidence_dummy,
        spatial_offset_lut,
        spatial_offset_lut_len,
        frame_len: frame.len(),
    }
}

fn prepare_window_ref<R: Runtime>(client: &ComputeClient<R>, ch: u32) -> WindowRefInput {
    let pixels = (W * H) as usize;
    let stored = stored_channels(ch) as usize;
    let frame = make_padded_frame(W, H, ch);
    let input = client.create_from_slice(f32::as_bytes(&frame));
    let reference = client.create_from_slice(f32::as_bytes(&frame));
    let accum = client.empty(pixels * stored * size_of::<f32>());
    let weight_sum = client.empty(pixels * size_of::<f32>());
    let max_weight = client.empty(pixels * size_of::<f32>());
    let weight_sq_sum_dummy = client.empty(size_of::<f32>());
    let confidence_dummy = client.empty(size_of::<f32>());
    let (spatial_offset_lut, spatial_offset_lut_len) = zero_spatial_offset_lut(client);
    WindowRefInput {
        input,
        reference,
        accum,
        weight_sum,
        max_weight,
        weight_sq_sum_dummy,
        confidence_dummy,
        spatial_offset_lut,
        spatial_offset_lut_len,
        frame_len: frame.len(),
    }
}

pub struct FusedPairWindowBench<R: Runtime> {
    pub client: ComputeClient<R>,
    pub ch: u32,
    pub ch_name: &'static str,
}

impl<R: Runtime> Benchmark for FusedPairWindowBench<R> {
    type Input = WindowInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        prepare_window(&self.client, self.ch)
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let pixels = (W * H) as usize;
        let stored = stored_channels(self.ch) as usize;
        unsafe {
            nlm_fused_pair_accumulate_window::launch_unchecked::<R>(
                &self.client,
                cube_count_2d(),
                cube_dim_2d(),
                stored,
                ArrayArg::from_raw_parts(args.input.clone(), args.frame_len),
                ArrayArg::from_raw_parts(args.accum.clone(), pixels * stored),
                ArrayArg::from_raw_parts(args.weight_sum.clone(), pixels),
                ArrayArg::from_raw_parts(args.max_weight.clone(), pixels),
                ArrayArg::from_raw_parts(args.weight_sq_sum_dummy.clone(), 1),
                ArrayArg::from_raw_parts(args.confidence_dummy.clone(), 1),
                ArrayArg::from_raw_parts(args.confidence_dummy.clone(), 1),
                false,
                false,
                0u32,
                0u32,
                0u32,
                h2_inv_norm(),
                0.0f32,
                W,
                H,
                self.ch,
                PATCH_RADIUS,
                SEARCH_RADIUS,
                BLOCK_X,
                BLOCK_Y,
                1u32,
                1u32,
                1u32,
            );
        }
        Ok(())
    }

    fn name(&self) -> String {
        format!("fused_pair_accumulate_window_1080p_{}", self.ch_name)
    }
    fn sync(&self) {
        block_sync(&self.client);
    }
    fn shapes(&self) -> Vec<Vec<usize>> {
        shapes_with_ch(self.ch)
    }
}

pub struct FusedSingleWindowBench<R: Runtime> {
    pub client: ComputeClient<R>,
    pub ch: u32,
    pub ch_name: &'static str,
}

impl<R: Runtime> Benchmark for FusedSingleWindowBench<R> {
    type Input = WindowInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        prepare_window(&self.client, self.ch)
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let pixels = (W * H) as usize;
        let stored = stored_channels(self.ch) as usize;
        unsafe {
            nlm_fused_single_window::launch_unchecked::<R>(
                &self.client,
                cube_count_2d(),
                cube_dim_2d(),
                stored,
                ArrayArg::from_raw_parts(args.input.clone(), args.frame_len),
                ArrayArg::from_raw_parts(args.accum.clone(), pixels * stored),
                ArrayArg::from_raw_parts(args.weight_sum.clone(), pixels),
                ArrayArg::from_raw_parts(args.max_weight.clone(), pixels),
                ArrayArg::from_raw_parts(args.weight_sq_sum_dummy.clone(), 1),
                false,
                0u32,
                h2_inv_norm(),
                ArrayArg::from_raw_parts(args.spatial_offset_lut.clone(), args.spatial_offset_lut_len),
                W,
                H,
                self.ch,
                PATCH_RADIUS,
                SEARCH_RADIUS,
                BLOCK_X,
                BLOCK_Y,
            );
        }
        Ok(())
    }

    fn name(&self) -> String {
        format!("fused_single_window_1080p_{}", self.ch_name)
    }
    fn sync(&self) {
        block_sync(&self.client);
    }
    fn shapes(&self) -> Vec<Vec<usize>> {
        shapes_with_ch(self.ch)
    }
}

pub struct FusedPairWindowRefBench<R: Runtime> {
    pub client: ComputeClient<R>,
    pub ch: u32,
    pub ch_name: &'static str,
}

impl<R: Runtime> Benchmark for FusedPairWindowRefBench<R> {
    type Input = WindowRefInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        prepare_window_ref(&self.client, self.ch)
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let pixels = (W * H) as usize;
        let stored = stored_channels(self.ch) as usize;
        unsafe {
            nlm_fused_pair_accumulate_window_ref::launch_unchecked::<R>(
                &self.client,
                cube_count_2d(),
                cube_dim_2d(),
                stored,
                ArrayArg::from_raw_parts(args.input.clone(), args.frame_len),
                ArrayArg::from_raw_parts(args.reference.clone(), args.frame_len),
                ArrayArg::from_raw_parts(args.accum.clone(), pixels * stored),
                ArrayArg::from_raw_parts(args.weight_sum.clone(), pixels),
                ArrayArg::from_raw_parts(args.max_weight.clone(), pixels),
                ArrayArg::from_raw_parts(args.weight_sq_sum_dummy.clone(), 1),
                ArrayArg::from_raw_parts(args.confidence_dummy.clone(), 1),
                ArrayArg::from_raw_parts(args.confidence_dummy.clone(), 1),
                false,
                false,
                0u32,
                0u32,
                0u32,
                h2_inv_norm(),
                0.0f32,
                W,
                H,
                self.ch,
                PATCH_RADIUS,
                SEARCH_RADIUS,
                BLOCK_X,
                BLOCK_Y,
                1u32,
                1u32,
                1u32,
            );
        }
        Ok(())
    }

    fn name(&self) -> String {
        format!("fused_pair_accumulate_window_ref_1080p_{}", self.ch_name)
    }
    fn sync(&self) {
        block_sync(&self.client);
    }
    fn shapes(&self) -> Vec<Vec<usize>> {
        shapes_with_ch(self.ch)
    }
}

pub struct FusedSingleWindowRefBench<R: Runtime> {
    pub client: ComputeClient<R>,
    pub ch: u32,
    pub ch_name: &'static str,
}

impl<R: Runtime> Benchmark for FusedSingleWindowRefBench<R> {
    type Input = WindowRefInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        prepare_window_ref(&self.client, self.ch)
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let pixels = (W * H) as usize;
        let stored = stored_channels(self.ch) as usize;
        unsafe {
            nlm_fused_single_window_ref::launch_unchecked::<R>(
                &self.client,
                cube_count_2d(),
                cube_dim_2d(),
                stored,
                ArrayArg::from_raw_parts(args.input.clone(), args.frame_len),
                ArrayArg::from_raw_parts(args.reference.clone(), args.frame_len),
                ArrayArg::from_raw_parts(args.accum.clone(), pixels * stored),
                ArrayArg::from_raw_parts(args.weight_sum.clone(), pixels),
                ArrayArg::from_raw_parts(args.max_weight.clone(), pixels),
                ArrayArg::from_raw_parts(args.weight_sq_sum_dummy.clone(), 1),
                false,
                0u32,
                h2_inv_norm(),
                ArrayArg::from_raw_parts(args.spatial_offset_lut.clone(), args.spatial_offset_lut_len),
                W,
                H,
                self.ch,
                PATCH_RADIUS,
                SEARCH_RADIUS,
                BLOCK_X,
                BLOCK_Y,
            );
        }
        Ok(())
    }

    fn name(&self) -> String {
        format!("fused_single_window_ref_1080p_{}", self.ch_name)
    }
    fn sync(&self) {
        block_sync(&self.client);
    }
    fn shapes(&self) -> Vec<Vec<usize>> {
        shapes_with_ch(self.ch)
    }
}
