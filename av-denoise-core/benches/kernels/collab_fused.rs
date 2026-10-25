use std::marker::PhantomData;

use av_denoise_core::bench_api::collab::geometry::{ref_count, refs_along, strength_map_dims};
use av_denoise_core::bench_api::collab::kernels::aggregate::{
    cross_frame_accum_scale,
    kaiser_window,
    weight_scale,
};
use av_denoise_core::bench_api::collab::kernels::transforms::dct_noise_profile;
use av_denoise_core::bench_api::collab::{grid_frames, needs_warp_uniform_search};
use av_denoise_core::bench_api::tune_sites::{
    CollabInputs,
    CollabOutputs,
    CollabShape,
    CollabSite,
    CollabValues,
    CollabVariant,
};
use av_denoise_core::bench_api::{DerivedValues, NOISE_CURVE_BINS, STRENGTH_MAP_LUMA, STRENGTH_MAP_OFF};
use av_denoise_core::tune::{DeviceLimits, SiteCandidate, VariantSite, fit_candidates};
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
use super::{HEIGHT, WIDTH, block_sync, make_padded_frame, shapes_with_channels, stored_channels};

/// The fused collaborative kernel at the library's default search geometry over a 1080p frame ring.
///
/// This is the whole collaborative stage in one launch, matching, filtering and scatter. One 64-lane
/// cube carries eight reference patches of eight lanes each, so the grid is an eighth as wide along
/// x as the reference count.
///
/// Confidence is uniformly 1.0, so no neighbour block is gated and every candidate runs the full
/// patch comparison. Gating a block skips its comparisons entirely, so this measures the worst case
/// and a bench that gates freely would report a time well under the real one.
///
/// `split_mv` picks one of two motion fields that bracket the real cost of the covering-block
/// search. `false` gives a zeroed field. Every block covering a patch then predicts the same
/// position, the duplicate check drops three of the four rectangles and no extra pixel comparison
/// runs, so that arm measures the duplicate check on its own. `true` gives each block a vector from
/// its own grid parity, eight pixels apart, which is further than the refine window is wide. The
/// four rectangles are then disjoint, nothing deduplicates and the neighbour search scores four
/// times the positions. That arm is the worst case, and a real motion field lands between the two.
pub struct CollabFusedBench<R: Runtime> {
    pub client: ComputeClient<R>,
    pub channels: u32,
    pub channel_name: &'static str,
    pub split_mv: bool,
    /// Launches with a stepped noise curve and the curve flag set.
    pub noise_curve: bool,
    /// Launches with a varied strength map in [STRENGTH_MAP_LUMA] mode.
    pub strength_map: bool,
    /// Launches with the pooled threshold on.
    pub pooled: bool,
    /// Scatters into the packed 64-bit accumulator layout.
    pub packed: bool,
    pub candidate: CollabVariant,
}

/// The element counts of `accum` and `wsum` over the bench ring, in the packed layout or the split
/// one.
fn accumulator_lens(stored_ch: u32, packed: bool) -> (usize, usize) {
    let pixels = (WIDTH * HEIGHT) as usize;
    let frames = N_FRAMES as usize;
    let wsum_len = pixels * frames;

    if packed {
        let rest_len = pixels * (stored_ch as usize - 1) * frames;
        return (rest_len.max(1), wsum_len);
    }

    let accum_len = pixels * stored_ch as usize * frames;
    (accum_len, wsum_len)
}

/// The variants this device runs for a bench of `channels`, baseline first.
pub fn fitted_variants<R: Runtime>(
    client: &ComputeClient<R>,
    channels: u32,
    channel_name: &'static str,
) -> Vec<CollabVariant> {
    let bench = CollabFusedBench {
        client: client.clone(),
        channels,
        channel_name,
        split_mv: false,
        noise_curve: false,
        strength_map: false,
        pooled: false,
        packed: false,
        candidate: CollabVariant::BASELINE,
    };
    let input = bench.prepare();
    let site = bench.site(&input);
    let limits = DeviceLimits::from_client(client);
    let fitted = fit_candidates(&site, &limits);

    fitted.expect("the baseline fits the bench device")
}

/// The luma pool ratio at the default lambda. The kernel's cost does not depend on its value.
const POOL_RATIO: f32 = 2.2 / 3.78;

/// A curve that doubles the luma threshold in the darker half and halves it in the brighter one.
fn stepped_curve() -> [f32; NOISE_CURVE_BINS] {
    let mut curve = [2.0f32; NOISE_CURVE_BINS];
    curve[NOISE_CURVE_BINS / 2..].fill(0.5);
    curve
}

/// How far apart two neighbouring blocks' vectors sit in the split field, in pixels.
///
/// `REFINE` is the rectangle's half-width, so two rectangles stay disjoint once their centres are
/// more than `2 * REFINE` apart. Eight clears that with room and keeps every predicted position
/// well inside a 1080p frame.
const SPLIT_MV_SPACING: i32 = 8;

#[derive(Clone)]
pub struct CollabFusedInput {
    pub ring: Handle,
    pub mv_field: Handle,
    pub confidence: Handle,
    pub neighbour_slots: Handle,
    /// The derived noise values, with every live channel at `SIGMA`.
    pub noise: Handle,
    pub dct_profile: Handle,
    /// The uniform aggregation window.
    ///
    /// The bench measures the kernel's own cost, and a taper changes none of the work it does.
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

impl<R: Runtime> CollabFusedBench<R> {
    /// The tunable site over `args`, with the library's default search geometry.
    pub fn site(&self, args: &CollabFusedInput) -> CollabSite<R, f32> {
        let stored_ch = stored_channels(self.channels);
        let (accum_len, wsum_len) = accumulator_lens(stored_ch, self.packed);
        let refs = ref_count(WIDTH, HEIGHT);
        let refs_x = refs_along(WIDTH);
        let refs_y = refs_along(HEIGHT);

        let blocks_x = WIDTH.div_ceil(BLK_STEP);
        let blocks_y = HEIGHT.div_ceil(BLK_STEP);
        let align = self.client.properties().memory.alignment;
        let neighbour_mv_stride = mv_stride(blocks_x, blocks_y, align);
        let neighbour_conf_stride = conf_stride(blocks_x, blocks_y, align);

        let (map_cols, map_rows) = strength_map_dims(WIDTH, HEIGHT);
        let accum_scale = cross_frame_accum_scale(SPATIAL_RADIUS, RADIUS);
        let frames_per_volume = grid_frames(RADIUS);

        let inputs = CollabInputs {
            ring: args.ring.clone(),
            ring_len: args.ring_len,
            search_ring: args.ring.clone(),
            search_len: stored_ch as usize,
            mv_field: args.mv_field.clone(),
            mv_len: (2 * RADIUS * neighbour_mv_stride) as usize,
            confidence: args.confidence.clone(),
            conf_len: (2 * RADIUS * neighbour_conf_stride) as usize,
            neighbour_slots: args.neighbour_slots.clone(),
            neighbour_slots_len: NEIGHBOUR_SLOTS.len(),
            noise: args.noise.clone(),
            noise_curve: args.noise_curve.clone(),
            strength_map: args.strength_map.clone(),
            map_len: args.map_len,
            dct_profile: args.dct_profile.clone(),
            kaiser: args.kaiser.clone(),
        };
        let values = CollabValues {
            centre_slot: CENTRE_SLOT,
            c_min: 0.0,
            lambda_ht: LAMBDA_HT,
            accum_scale,
            pool_ratio: POOL_RATIO,
        };
        let uniform_walk_only = needs_warp_uniform_search(&self.client);
        let shape = CollabShape {
            f16_search: false,
            radius: RADIUS,
            grid_frames: frames_per_volume,
            refine: REFINE,
            mv_stride: neighbour_mv_stride,
            conf_stride: neighbour_conf_stride,
            blk_step: BLK_STEP,
            blksize: BLKSIZE,
            blocks_x,
            blocks_y,
            width: WIDTH,
            height: HEIGHT,
            channels: self.channels,
            k_max: K_MAX,
            stored_ch,
            spatial_radius: SPATIAL_RADIUS,
            refs_x,
            refs_y,
            map_cols,
            map_rows,
            pooled: self.pooled,
            uniform_walk_only,
            packed: self.packed,
        };
        let outputs = CollabOutputs {
            accum: args.accum.clone(),
            wsum: args.wsum.clone(),
            group_weight: args.group_weight.clone(),
            accum_len,
            wsum_len,
            refs,
            packed: self.packed,
        };
        CollabSite::<R, f32> {
            client: self.client.clone(),
            inputs,
            values,
            shape,
            outputs,
            search: PhantomData,
        }
    }
}

impl<R: Runtime> Benchmark for CollabFusedBench<R> {
    type Input = CollabFusedInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let stored_ch = stored_channels(self.channels);

        let mut ring_data = Vec::new();
        for _ in 0..N_FRAMES {
            let frame = make_padded_frame(WIDTH, HEIGHT, self.channels);
            ring_data.extend(frame);
        }

        let ring_bytes = f32::as_bytes(&ring_data);
        let ring = self.client.create_from_slice(ring_bytes);

        let blocks_x = WIDTH.div_ceil(BLK_STEP);
        let blocks_y = HEIGHT.div_ceil(BLK_STEP);
        let align = self.client.properties().memory.alignment;
        let neighbour_mv_stride = mv_stride(blocks_x, blocks_y, align);
        let neighbour_conf_stride = conf_stride(blocks_x, blocks_y, align);

        let mut mv_data = vec![0i32; (2 * RADIUS * neighbour_mv_stride) as usize];
        if self.split_mv {
            for slot in 0..2 * RADIUS {
                for block_y in 0..blocks_y {
                    for block_x in 0..blocks_x {
                        let base = (slot * neighbour_mv_stride + (block_y * blocks_x + block_x) * 2) as usize;
                        mv_data[base] = (block_x % 2) as i32 * SPLIT_MV_SPACING;
                        mv_data[base + 1] = (block_y % 2) as i32 * SPLIT_MV_SPACING;
                    }
                }
            }
        }

        let mv_bytes = i32::as_bytes(&mv_data);
        let mv_field = self.client.create_from_slice(mv_bytes);
        let conf_data = vec![1.0f32; (2 * RADIUS * neighbour_conf_stride) as usize];
        let conf_bytes = f32::as_bytes(&conf_data);
        let confidence = self.client.create_from_slice(conf_bytes);
        let slots_bytes = u32::as_bytes(&NEIGHBOUR_SLOTS);
        let neighbour_slots = self.client.create_from_slice(slots_bytes);

        let profile_host = dct_noise_profile(0.0);
        let map_mode = if self.strength_map {
            STRENGTH_MAP_LUMA
        } else {
            STRENGTH_MAP_OFF
        };

        let mut sigma = [0.0f32; 3];
        sigma[..self.channels as usize].fill(SIGMA);
        let group_weight_scale = weight_scale(SIGMA, &profile_host);
        let derived = DerivedValues {
            sigma,
            weight_scale: group_weight_scale,
            curve_valid: self.noise_curve,
            map_mode,
            ..DerivedValues::default()
        };
        let noise_words = derived.to_words();
        let noise_bytes = f32::as_bytes(&noise_words);
        let noise = self.client.create_from_slice(noise_bytes);

        let profile_bytes = f32::as_bytes(&profile_host);
        let dct_profile = self.client.create_from_slice(profile_bytes);
        let kaiser_host = kaiser_window(0.0);
        let kaiser_bytes = f32::as_bytes(&kaiser_host);
        let kaiser = self.client.create_from_slice(kaiser_bytes);

        // One accumulator region per ring slot, the same shape `Nl4dDenoiser` allocates, so the
        // scatter crosses the same address range it does in the pipeline.
        let (accum_len, wsum_len) = accumulator_lens(stored_ch, self.packed);
        let wsum_element_bytes = if self.packed {
            size_of::<i64>()
        } else {
            size_of::<i32>()
        };
        let accum = self.client.empty(accum_len * size_of::<i32>());
        let wsum = self.client.empty(wsum_len * wsum_element_bytes);
        let refs = ref_count(WIDTH, HEIGHT);
        let group_weight = self.client.empty(refs * size_of::<f32>());
        let curve_host = if self.noise_curve {
            stepped_curve()
        } else {
            [0.0f32; NOISE_CURVE_BINS]
        };
        let curve_bytes = f32::as_bytes(&curve_host);
        let noise_curve = self.client.create_from_slice(curve_bytes);

        let (map_cols, map_rows) = strength_map_dims(WIDTH, HEIGHT);
        let map_len = (map_cols * map_rows) as usize;
        let map_host: Vec<f32> = if self.strength_map {
            (0..map_len)
                .map(|index| if index % 3 == 0 { 1.5 } else { 0.65 })
                .collect()
        } else {
            vec![1.0f32; map_len]
        };
        let map_bytes = f32::as_bytes(&map_host);
        let strength_map = self.client.create_from_slice(map_bytes);

        CollabFusedInput {
            ring,
            mv_field,
            confidence,
            neighbour_slots,
            noise,
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
        let site = self.site(&args);

        let real_outputs = site.real_outputs();
        site.launch(self.candidate, &real_outputs);

        Ok(())
    }

    fn name(&self) -> String {
        let field = if self.split_mv { "_split_mv" } else { "" };
        let curve = if self.noise_curve { "_noise_curve" } else { "" };
        let map = if self.strength_map { "_strength_map" } else { "" };
        let pool = if self.pooled { "_pooled" } else { "" };
        let packed = if self.packed { "_packed" } else { "" };

        format!(
            "{}/{}/1080p_{}{field}{curve}{map}{pool}{packed}",
            CollabVariant::SITE,
            self.candidate.name(),
            self.channel_name,
        )
    }

    fn sync(&self) {
        block_sync(&self.client);
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        shapes_with_channels(self.channels)
    }
}
