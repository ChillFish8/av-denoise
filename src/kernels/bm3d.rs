use cubecl::cube;
use cubecl::prelude::*;

#[cube(launch)]
/// Produces the basic estimate denoising via hard thresholding.
///
/// For each reference block (on a grid with stride n_step):
/// 1. Find similar blocks within a (2·n_s + 1)² search window.
/// 2. Stack them (with Kaiser windowing) and apply the 3-D transform.
/// 3. Hard-threshold every coefficient whose |value| ≤ λ₃D · σ.
/// 4. Apply the inverse 3-D transform.
/// 5. Aggregate filtered blocks back into the output image,
///
/// weighted by 1 / n_nonzero (fewer surviving coefficients → lower weight,
/// as the estimate is noisier).
pub fn bm3d_stage1(
    frame: &Tensor<f32>,
    kaiser: &Tensor<f32>,
    result: &mut Tensor<f32>,
    sigma: f32,
    lambda_3d: f32,
    #[comptime] k: usize,
    #[comptime] n_s: usize,
    #[comptime] n_step: usize,
    #[comptime] n_max: usize,
    #[comptime] tau_match: u32,
) {
    let row = CUBE_POS_Y;
    let col = CUBE_POS_X;

    todo!()
}
