use av_denoise_core::collab::geometry::{fused_cubes_x, ref_count, refs_along, strength_map_dims};
use av_denoise_core::collab::kernels::aggregate::{cross_frame_accum_scale, kaiser_window, weight_scale};
use av_denoise_core::collab::kernels::fused::{STRENGTH_MAP_LUMA, STRENGTH_MAP_OFF, collab_fused};
use av_denoise_core::collab::kernels::transforms::dct_noise_profile;
use av_denoise_core::collab::{PATCH_SIZE, grid_frames, needs_warp_uniform_search};
use av_denoise_core::nlmeans::NOISE_CURVE_BINS;
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::nl4d_geometry::{
    BLK_STEP,
    BLKSIZE,
    CENTRE_SLOT,
    K_MAX,
    LAMBDA_HT,
    N_FRAMES,
    NEIGHBOUR_SLOTS,
    RADIUS,
    REFINE,
    SIGMA,
    SPATIAL_RADIUS,
    conf_stride,
    mv_stride,
};
use super::{H, W, block_sync, make_padded_frame, shapes_with_ch, stored_channels};

/// The fused collaborative kernel at the library's default search
/// geometry, over a 1080p frame ring. One 64-lane cube carries eight
/// reference patches, eight lanes each, so the grid is an eighth as wide
/// along x as the reference count.
///
/// This is the whole collaborative stage in one launch, matching,
/// filtering and scatter.
///
/// Confidence is uniformly 1.0, so no neighbour block is gated and every
/// candidate the kernel finds runs the full patch comparison. Gating a
/// block skips its comparisons entirely, so leaving it always open here
/// measures the worst case. A bench that gates freely would report a
/// time well under the real one.
///
/// `split_mv` picks which of the two motion fields the arm runs on, and
/// the two bracket the real cost of the covering-block search.
///
/// `false` gives a zeroed field. Every block covering a patch then
/// predicts the same position, all four rectangles coincide, three of
/// them are dropped by the duplicate check and no extra pixel
/// comparison runs. That arm measures the duplicate check on its own.
///
/// `true` gives each block a vector from its own grid parity, spaced
/// eight pixels apart, which is further than the refine window is wide.
/// The four rectangles covering a patch are then disjoint, nothing
/// deduplicates, and the neighbour search scores four times the
/// positions. That arm is the worst case, and a real motion field lands
/// between the two.
pub struct CollabFusedBench<R: Runtime> {
    pub client: ComputeClient<R>,
    pub ch: u32,
    pub ch_name: &'static str,
    pub split_mv: bool,
    /// Launches with a stepped noise curve and `curve_valid = 1`.
    pub noise_curve: bool,
    /// Launches with a varied strength map in [STRENGTH_MAP_LUMA] mode.
    pub strength_map: bool,
    /// Launches with the pooled threshold on.
    pub pooled: bool,
}

/// The luma pool ratio at the default lambda. The kernel's cost does not depend on its value.
const POOL_RATIO: f32 = 2.2 / 3.78;

/// A curve that doubles the luma threshold in the darker half and halves it
/// in the brighter one.
fn stepped_curve() -> [f32; NOISE_CURVE_BINS] {
    let mut curve = [2.0f32; NOISE_CURVE_BINS];
    curve[NOISE_CURVE_BINS / 2..].fill(0.5);
    curve
}

/// How far apart two neighbouring blocks' vectors sit in the split
/// field, in pixels.
///
/// `REFINE` is the rectangle's half-width, so two rectangles stay
/// disjoint once their centres are more than `2 * REFINE` apart. Eight
/// clears that with room and keeps every predicted position well inside
/// a 1080p frame.
const SPLIT_MV_SPACING: i32 = 8;

#[derive(Clone)]
pub struct CollabFusedInput {
    pub ring: Handle,
    pub mv_field: Handle,
    pub confidence: Handle,
    pub neighbour_slots: Handle,
    pub sigma: Handle,
    pub dct_profile: Handle,
    /// The uniform aggregation window. The bench measures the kernel's
    /// own cost, and a taper changes none of the work it does.
    pub kaiser: Handle,
    pub accum: Handle,
    pub wsum: Handle,
    pub group_weight: Handle,
    pub ring_len: usize,
    /// The noise curve, all zeros unless the arm runs one.
    pub noise_curve: Handle,
    pub strength_map: Handle,
    pub map_len: usize,
}

impl<R: Runtime> Benchmark for CollabFusedBench<R> {
    type Input = CollabFusedInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let stored_ch = stored_channels(self.ch);
        let pixels = (W * H) as usize;
        let frame_len = pixels * stored_ch as usize;

        let mut ring_data = Vec::new();
        for _ in 0..N_FRAMES {
            ring_data.extend(make_padded_frame(W, H, self.ch));
        }
        let ring = self.client.create_from_slice(f32::as_bytes(&ring_data));

        let blocks_x = W.div_ceil(BLK_STEP);
        let blocks_y = H.div_ceil(BLK_STEP);
        let align = self.client.properties().memory.alignment;
        let mv_stride = mv_stride(blocks_x, blocks_y, align);
        let conf_stride = conf_stride(blocks_x, blocks_y, align);

        let mut mv_data = vec![0i32; (2 * RADIUS * mv_stride) as usize];
        if self.split_mv {
            for t in 0..2 * RADIUS {
                for by in 0..blocks_y {
                    for bx in 0..blocks_x {
                        let base = (t * mv_stride + (by * blocks_x + bx) * 2) as usize;
                        mv_data[base] = (bx % 2) as i32 * SPLIT_MV_SPACING;
                        mv_data[base + 1] = (by % 2) as i32 * SPLIT_MV_SPACING;
                    }
                }
            }
        }
        let mv_field = self.client.create_from_slice(i32::as_bytes(&mv_data));
        let conf_data = vec![1.0f32; (2 * RADIUS * conf_stride) as usize];
        let confidence = self.client.create_from_slice(f32::as_bytes(&conf_data));
        let neighbour_slots = self.client.create_from_slice(u32::as_bytes(&NEIGHBOUR_SLOTS));

        // Sized for the stored lane count and filled for the logical
        // ones, matching what `Nl4dDenoiser` uploads each pass.
        let mut sigma_host = vec![0.0f32; stored_ch as usize];
        sigma_host[..self.ch as usize].fill(SIGMA);
        let sigma = self.client.create_from_slice(f32::as_bytes(&sigma_host));
        let dct_profile = self
            .client
            .create_from_slice(f32::as_bytes(&dct_noise_profile(0.0)));
        let kaiser = self.client.create_from_slice(f32::as_bytes(&kaiser_window(0.0)));

        // One accumulator region per ring slot, the same shape
        // `Nl4dDenoiser` allocates, so the scatter crosses the same
        // address range it does in the pipeline.
        let accum = self
            .client
            .empty(frame_len * N_FRAMES as usize * size_of::<i32>());
        let wsum = self.client.empty(pixels * N_FRAMES as usize * size_of::<i32>());
        let group_weight = self.client.empty(ref_count(W, H) * size_of::<f32>());
        let curve_host = if self.noise_curve {
            stepped_curve()
        } else {
            [0.0f32; NOISE_CURVE_BINS]
        };
        let noise_curve = self.client.create_from_slice(f32::as_bytes(&curve_host));

        let (map_cols, map_rows) = strength_map_dims(W, H);
        let map_len = (map_cols * map_rows) as usize;
        let map_host: Vec<f32> = if self.strength_map {
            (0..map_len)
                .map(|index| if index % 3 == 0 { 1.5 } else { 0.65 })
                .collect()
        } else {
            vec![1.0f32; map_len]
        };
        let strength_map = self.client.create_from_slice(f32::as_bytes(&map_host));

        CollabFusedInput {
            ring,
            mv_field,
            confidence,
            neighbour_slots,
            sigma,
            dct_profile,
            kaiser,
            accum,
            wsum,
            group_weight,
            ring_len: ring_data.len(),
            noise_curve,
            strength_map,
            map_len,
        }
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let stored_ch = stored_channels(self.ch);
        let pixels = (W * H) as usize;
        let frame_len = pixels * stored_ch as usize;
        let refs = ref_count(W, H);
        let refs_x = refs_along(W);
        let refs_y = refs_along(H);

        let blocks_x = W.div_ceil(BLK_STEP);
        let blocks_y = H.div_ceil(BLK_STEP);
        let align = self.client.properties().memory.alignment;
        let mv_stride = mv_stride(blocks_x, blocks_y, align);
        let conf_stride = conf_stride(blocks_x, blocks_y, align);

        let (map_cols, map_rows) = strength_map_dims(W, H);
        let map_mode = if self.strength_map {
            STRENGTH_MAP_LUMA
        } else {
            STRENGTH_MAP_OFF
        };

        let grid = CubeCount::new_2d(fused_cubes_x(W), refs_y);
        let dim = CubeDim::new_1d(64);

        unsafe {
            collab_fused::launch_unchecked::<R>(
                &self.client,
                grid,
                dim,
                stored_ch as usize,
                ArrayArg::from_raw_parts(args.ring.clone(), args.ring_len),
                ArrayArg::from_raw_parts(args.mv_field.clone(), (2 * RADIUS * mv_stride) as usize),
                ArrayArg::from_raw_parts(args.confidence.clone(), (2 * RADIUS * conf_stride) as usize),
                ArrayArg::from_raw_parts(args.neighbour_slots.clone(), NEIGHBOUR_SLOTS.len()),
                ArrayArg::from_raw_parts(args.sigma.clone(), stored_ch as usize),
                ArrayArg::from_raw_parts(args.noise_curve.clone(), NOISE_CURVE_BINS),
                ArrayArg::from_raw_parts(args.strength_map.clone(), args.map_len),
                ArrayArg::from_raw_parts(args.dct_profile.clone(), 8),
                ArrayArg::from_raw_parts(args.kaiser.clone(), PATCH_SIZE as usize),
                ArrayArg::from_raw_parts(args.accum.clone(), frame_len * N_FRAMES as usize),
                ArrayArg::from_raw_parts(args.wsum.clone(), pixels * N_FRAMES as usize),
                ArrayArg::from_raw_parts(args.group_weight.clone(), refs),
                CENTRE_SLOT,
                0.0f32,
                LAMBDA_HT,
                u32::from(self.noise_curve),
                map_mode,
                weight_scale(SIGMA, &dct_noise_profile(0.0)),
                cross_frame_accum_scale(SPATIAL_RADIUS, RADIUS),
                needs_warp_uniform_search(&self.client),
                RADIUS,
                grid_frames(RADIUS),
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
                stored_ch,
                SPATIAL_RADIUS,
                refs_x,
                map_cols,
                map_rows,
                POOL_RATIO,
                self.pooled,
            );
        }
        Ok(())
    }

    fn name(&self) -> String {
        let field = if self.split_mv { "_split_mv" } else { "" };
        let curve = if self.noise_curve { "_noise_curve" } else { "" };
        let map = if self.strength_map { "_strength_map" } else { "" };
        let pool = if self.pooled { "_pooled" } else { "" };
        format!("collab_fused_1080p_{}{field}{curve}{map}{pool}", self.ch_name)
    }

    fn sync(&self) {
        block_sync(&self.client);
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        shapes_with_ch(self.ch)
    }
}
