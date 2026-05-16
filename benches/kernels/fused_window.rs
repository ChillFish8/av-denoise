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

#[derive(Clone)]
pub struct WindowInput {
    input: Handle,
    accum: Handle,
    weight_sum: Handle,
    max_weight: Handle,
    frame_len: usize,
}

#[derive(Clone)]
pub struct WindowRefInput {
    input: Handle,
    reference: Handle,
    accum: Handle,
    weight_sum: Handle,
    max_weight: Handle,
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
    WindowInput {
        input,
        accum,
        weight_sum,
        max_weight,
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
    WindowRefInput {
        input,
        reference,
        accum,
        weight_sum,
        max_weight,
        frame_len: frame.len(),
    }
}

// ---- nlm_fused_pair_accumulate_window (temporal k≠0 windowed) ----

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
                0u32,
                0u32,
                0u32,
                h2_inv_norm(),
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
        format!("fused_pair_accumulate_window_1080p_{}", self.ch_name)
    }
    fn sync(&self) {
        block_sync(&self.client);
    }
    fn shapes(&self) -> Vec<Vec<usize>> {
        shapes_with_ch(self.ch)
    }
}

// ---- nlm_fused_single_window (spatial k=0 windowed) ----

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
                0u32,
                h2_inv_norm(),
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

// ---- nlm_fused_pair_accumulate_window_ref ----

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
                0u32,
                0u32,
                0u32,
                h2_inv_norm(),
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
        format!("fused_pair_accumulate_window_ref_1080p_{}", self.ch_name)
    }
    fn sync(&self) {
        block_sync(&self.client);
    }
    fn shapes(&self) -> Vec<Vec<usize>> {
        shapes_with_ch(self.ch)
    }
}

// ---- nlm_fused_single_window_ref ----

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
                0u32,
                h2_inv_norm(),
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
