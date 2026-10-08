mod behaviour;
mod noise_curve;
mod pooled;
mod recorded;
mod strength_map;
mod walks;

use cubecl::prelude::*;
use cubecl::server::Handle;

use super::helpers::{R, make_client, make_unique_frame, noisy_flat_field};
use crate::collab::geometry::{fused_cubes_x, ref_count, ref_pos, refs_along, strength_map_dims};
use crate::collab::kernels::aggregate::{WEIGHT_GAIN, cross_frame_accum_scale, kaiser_window, weight_scale};
use crate::collab::kernels::fused::{STRENGTH_MAP_OFF, collab_fused};
use crate::collab::kernels::transforms::dct_noise_profile;
use crate::collab::{PATCH_SIZE, grid_frames, needs_warp_uniform_search};
use crate::nlmeans::{ChannelMode, NOISE_CURVE_BINS};

/// The spatial search radius most runs use.
///
/// A reference patch away from the frame edge scores a 9x9 window, well past the eight members a
/// group keeps, and the whole sweep stays quick.
const SPATIAL_RADIUS: u32 = 4;

/// The group size most runs use.
///
/// The fused kernel carries one member per lane of an 8-lane group, so this is the size it is built
/// for.
const K_MAX: u32 = 8;

/// Motion-block side length.
const BLKSIZE: u32 = 16;

/// Motion-block stride, equal to `PATCH_SIZE` so a block boundary lines up with a patch boundary.
const BLK_STEP: u32 = 8;

/// The noise level the filter shrinks against.
///
/// Against content between 0 and 1 the threshold keeps a spread of coefficients rather than
/// everything or nothing, so both sides of the keep decision are exercised.
const SIGMA: f32 = 0.02;

/// A hard-threshold multiplier pinned independently of the shipped `lambda_ht` default.
///
/// Several tests recorded their expected output at this value.
const LAMBDA_HT: f32 = 5.3;

/// [make_unique_frame] rescaled to between 0 and 1.
///
/// The raw ramp reaches ten times the frame width. Values in the hundreds saturate
/// [ACCUM_CLAMP](crate::collab::kernels::aggregate::ACCUM_CLAMP) and put every coefficient far above
/// the noise threshold, so the threshold would stop being tested. Dividing by a constant keeps every
/// 8x8 window exactly as distinct as before.
pub(super) fn unique_frame(width: u32, height: u32) -> Vec<f32> {
    let raw = make_unique_frame(width, height);
    let peak = raw.iter().copied().fold(0.0f32, f32::max);
    raw.into_iter().map(|value| value / peak).collect()
}

/// Everything one launch of [collab_fused] takes.
pub(super) struct Setup {
    pub(super) ring: Vec<f32>,
    pub(super) mv_field: Vec<i32>,
    pub(super) confidence: Vec<f32>,
    pub(super) neighbour_slots: Vec<u32>,
    pub(super) centre_slot: u32,
    pub(super) c_min: f32,
    pub(super) radius: u32,
    pub(super) refine: u32,
    pub(super) spatial_radius: u32,
    pub(super) mv_stride: u32,
    pub(super) conf_stride: u32,
    pub(super) blocks_x: u32,
    pub(super) blocks_y: u32,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) k_max: u32,
    pub(super) sigma: f32,
    pub(super) lambda_ht: f32,
    /// Residual correlation the noise profile is built for. `0.0` gives the all-ones profile.
    pub(super) rho: f32,
    /// A profile supplied outright, bypassing [dct_noise_profile].
    ///
    /// The weight scale still follows whichever profile is in force.
    pub(super) profile_override: Option<[f32; 8]>,
    /// The aggregation window's beta. `0.0` is uniform aggregation.
    pub(super) kaiser_beta: f32,
    /// The frame's noise curve. `None` launches with `curve_valid = 0` and a zeroed buffer.
    pub(super) noise_curve: Option<[f32; NOISE_CURVE_BINS]>,
    /// A strength map and the mode it applies in. `None` launches a unit map with the map off.
    pub(super) strength_map: Option<(Vec<f32>, u32)>,
    /// The pooled threshold's ratio to lambda. `None` launches the plain per-coefficient test.
    pub(super) pooled: Option<f32>,
    /// The channels the ring interleaves per pixel, laid out at
    /// [ChannelMode::storage_count](crate::nlmeans::ChannelMode::storage_count) floats each.
    pub(super) channel_mode: ChannelMode,
    /// Scores candidates from an f16 copy of the ring.
    pub(super) f16_search: bool,
}

impl Setup {
    /// A single-frame ring with no neighbours.
    ///
    /// Every candidate comes from the spatial window, and the motion, confidence and neighbour-slot
    /// buffers are dummies nothing reads.
    pub(super) fn spatial_only(frame: Vec<f32>, width: u32, height: u32) -> Self {
        assert_eq!(frame.len(), (width * height) as usize);

        Setup {
            ring: frame,
            mv_field: vec![0i32, 0i32],
            confidence: vec![1.0f32],
            neighbour_slots: vec![0u32],
            centre_slot: 0,
            c_min: 0.0,
            radius: 0,
            refine: 0,
            spatial_radius: SPATIAL_RADIUS,
            mv_stride: 2,
            conf_stride: 1,
            blocks_x: 1,
            blocks_y: 1,
            width,
            height,
            k_max: K_MAX,
            sigma: SIGMA,
            lambda_ht: LAMBDA_HT,
            rho: 0.0,
            profile_override: None,
            kaiser_beta: 0.0,
            noise_curve: None,
            strength_map: None,
            pooled: None,
            channel_mode: ChannelMode::Luma,
            f16_search: false,
        }
    }

    /// Slots in the frame ring, which is also how many regions the accumulators carry.
    pub(super) fn frames(&self) -> u32 {
        let frame_len = self.width * self.height * self.stored_channels();
        self.ring.len() as u32 / frame_len
    }

    pub(super) fn stored_channels(&self) -> u32 {
        self.channel_mode.storage_count()
    }

    pub(super) fn pixels(&self) -> usize {
        (self.width * self.height) as usize
    }

    /// The fixed-point scale the scatter counts in.
    pub(super) fn accum_scale(&self) -> f32 {
        cross_frame_accum_scale(self.spatial_radius, self.radius)
    }

    /// The correlation profile this run's threshold reads.
    pub(super) fn profile(&self) -> [f32; 8] {
        self.profile_override
            .unwrap_or_else(|| dct_noise_profile(self.rho))
    }
}

/// One run's aggregated output, read back after its launch.
pub(super) struct Aggregated {
    pub(super) accum: Vec<i32>,
    pub(super) wsum: Vec<i32>,
    pub(super) group_weight: Vec<f32>,
    pixels: usize,
}

impl Aggregated {
    /// One finished pixel, the weighted mean of every filtered patch that covered it.
    ///
    /// This is what the caller sees, so a tolerance against it is in pixel values. Comparing the raw
    /// accumulator instead would fail on a group-weight difference the division cancels. A pixel no
    /// member covered reads zero.
    pub(super) fn pixel(&self, idx: usize) -> f64 {
        let weight_sum = self.wsum[idx];
        if weight_sum == 0 {
            0.0
        } else {
            // `wsum` counts at `WEIGHT_GAIN` times `accum`'s scale, the one factor that does not
            // cancel, so it is multiplied back out as `collab_normalise` does.
            self.accum[idx] as f64 * WEIGHT_GAIN as f64 / weight_sum as f64
        }
    }

    /// The total weight one ring slot's region received.
    pub(super) fn frame_weight_sum(&self, slot: usize) -> i64 {
        self.wsum[slot * self.pixels..(slot + 1) * self.pixels]
            .iter()
            .map(|&weight| weight as i64)
            .sum()
    }

    /// A summary of the whole run, small enough to record as literals and specific enough that a
    /// kernel writing nothing cannot reproduce it.
    fn digest(&self) -> Digest {
        // Luma stores one channel per pixel, so the two accumulators share an index.
        assert_eq!(self.accum.len(), self.wsum.len());
        let len = self.accum.len();
        let mut sum = 0.0f64;
        let mut sum_sq = 0.0f64;
        let mut covered = 0usize;
        for idx in 0..len {
            let pixel = self.pixel(idx);
            sum += pixel;
            sum_sq += pixel * pixel;
            if self.wsum[idx] != 0 {
                covered += 1;
            }
        }

        let weight_total = self.group_weight.iter().map(|&weight| weight as f64).sum::<f64>();
        let weight_mean = weight_total / self.group_weight.len() as f64;

        let mut probes = [0.0f64; PROBE_COUNT];
        for (i, probe) in probes.iter_mut().enumerate() {
            let index = probe_index(i, len);
            *probe = self.pixel(index);
        }

        Digest {
            covered,
            pixel_mean: sum / len as f64,
            pixel_rms: (sum_sq / len as f64).sqrt(),
            weight_mean,
            probes,
        }
    }
}

/// How many individual pixels a [Digest] pins alongside its whole-run statistics.
const PROBE_COUNT: usize = 8;

/// The pixel a probe reads.
///
/// The odd stride spreads the eight probes over the buffer so no two land in one patch or one row.
pub(super) fn probe_index(probe: usize, len: usize) -> usize {
    (probe * 7919 + 1013) % len
}

/// One run's output, boiled down to numbers a test can carry as literals.
pub(super) struct Digest {
    /// Pixels whose weight sum is non-zero.
    pub(super) covered: usize,
    /// Mean normalised pixel over every slot of the accumulator ring.
    pub(super) pixel_mean: f64,
    /// Root mean square of the same pixels.
    pub(super) pixel_rms: f64,
    /// Mean of the per-reference group weight.
    pub(super) weight_mean: f64,
    /// Individual pixels at [probe_index] positions.
    pub(super) probes: [f64; PROBE_COUNT],
}

/// How far a recorded whole-run statistic may move, relative.
///
/// Each statistic sums thousands of values, so one coefficient crossing the hard threshold moves it
/// by around `1e-8`. The recorded literals carry a truncate-toward-zero bias of up to one fixed-point
/// unit per contribution, while [to_fixed](crate::collab::kernels::aggregate::to_fixed) rounds, so
/// live values sit about `1e-5` relative above them. That is the quantisation step moving, not the
/// filter, and no implementation can match across it more tightly. Re-recording from the fused
/// kernel would prove nothing, since the literals are a second implementation's answer. `2e-5` is
/// still vanishingly small next to what a kernel that stopped writing would produce.
const DIGEST_RELATIVE_TOLERANCE: f64 = 2.0e-5;

/// How far a recorded probe pixel may move, absolute.
///
/// A coefficient within float rounding of `lambda_ht * sigma` can fall either side of the hard
/// threshold and moves its group's reconstruction by its own magnitude. A probe reads one pixel
/// rather than an average, so it allows `1e-3`, a quarter of an 8-bit code level.
const PROBE_TOLERANCE: f64 = 1.0e-3;

/// Checks a run against values recorded from a second, known-good implementation.
///
/// The literals come from a two-kernel group-then-filter implementation, which agreed with the fused
/// kernel to `5e-9` on the whole-run statistics and `5e-7` on the worst probe when recorded.
/// Fixed literals keep this meaningful because a cubecl compiler bug can make a failing shader do
/// nothing at all, and a kernel compared against itself would then match zeros to zeros.
pub(super) fn assert_matches_recorded(label: &str, got: &Aggregated, want: &Digest) {
    let digest = got.digest();
    assert_eq!(
        digest.covered, want.covered,
        "{label}: {} pixels carry weight, recorded {}",
        digest.covered, want.covered
    );

    for (name, have, expect) in [
        ("pixel_mean", digest.pixel_mean, want.pixel_mean),
        ("pixel_rms", digest.pixel_rms, want.pixel_rms),
        ("weight_mean", digest.weight_mean, want.weight_mean),
    ] {
        let rel = (have - expect).abs() / expect.abs().max(1.0e-30);
        assert!(
            rel < DIGEST_RELATIVE_TOLERANCE,
            "{label}: {name} is {have}, recorded {expect}, relative error {rel}"
        );
    }

    for (i, (&have, &expect)) in digest.probes.iter().zip(want.probes.iter()).enumerate() {
        assert!(
            (have - expect).abs() < PROBE_TOLERANCE,
            "{label}: probe {i} is {have}, recorded {expect}"
        );
    }
}

/// The device-side buffers one launch needs.
pub(super) struct Buffers {
    client: ComputeClient<R>,
    ring: Handle,
    search_ring: Handle,
    mv_field: Handle,
    confidence: Handle,
    neighbour_slots: Handle,
    sigma: Handle,
    dct_profile: Handle,
    kaiser: Handle,
    accum: Handle,
    wsum: Handle,
    group_weight: Handle,
    accum_len: usize,
    wsum_len: usize,
    refs: usize,
    refs_x: u32,
    refs_y: u32,
}

pub(super) fn buffers(setup: &Setup) -> Buffers {
    let client = make_client();
    let refs_x = refs_along(setup.width);
    let refs_y = refs_along(setup.height);
    let refs = ref_count(setup.width, setup.height);
    let frames = setup.frames() as usize;
    let stored_channels = setup.stored_channels() as usize;
    let accum_len = setup.pixels() * stored_channels * frames;
    let wsum_len = setup.pixels() * frames;

    // Padding lanes past the live channels carry a zero sigma, as the denoiser uploads them.
    let mut sigma = vec![0.0f32; stored_channels];
    let live_channels = setup.channel_mode.count() as usize;
    sigma[..live_channels].fill(setup.sigma);

    let profile = setup.profile();
    let kaiser = kaiser_window(setup.kaiser_beta);
    // Zeroed here rather than by `collab_zero_accum`, since the scatter is the only writer in these runs.
    let zeroed_accum = vec![0i32; accum_len];
    let zeroed_wsum = vec![0i32; wsum_len];

    let ring_bytes = f32::as_bytes(&setup.ring);
    let mv_bytes = i32::as_bytes(&setup.mv_field);
    let conf_bytes = f32::as_bytes(&setup.confidence);
    let slots_bytes = u32::as_bytes(&setup.neighbour_slots);
    let sigma_bytes = f32::as_bytes(&sigma);
    let profile_bytes = f32::as_bytes(&profile);
    let kaiser_bytes = f32::as_bytes(&kaiser);
    let accum_bytes = i32::as_bytes(&zeroed_accum);
    let wsum_bytes = i32::as_bytes(&zeroed_wsum);
    let ring = client.create_from_slice(ring_bytes);
    let mv_field = client.create_from_slice(mv_bytes);
    let confidence = client.create_from_slice(conf_bytes);
    let neighbour_slots = client.create_from_slice(slots_bytes);
    let sigma = client.create_from_slice(sigma_bytes);
    let dct_profile = client.create_from_slice(profile_bytes);
    let kaiser = client.create_from_slice(kaiser_bytes);
    let accum = client.create_from_slice(accum_bytes);
    let wsum = client.create_from_slice(wsum_bytes);
    let group_weight = client.empty(refs * size_of::<f32>());

    let search_values: Vec<half::f16> = setup
        .ring
        .iter()
        .map(|value| half::f16::from_f32(*value))
        .collect();
    let search_bytes = half::f16::as_bytes(&search_values);
    let search_ring = client.create_from_slice(search_bytes);

    Buffers {
        ring,
        search_ring,
        mv_field,
        confidence,
        neighbour_slots,
        sigma,
        dct_profile,
        kaiser,
        accum,
        wsum,
        group_weight,
        accum_len,
        wsum_len,
        refs,
        refs_x,
        refs_y,
        client,
    }
}

pub(super) fn read_back(handles: Buffers, setup: &Setup) -> Aggregated {
    let accum_bytes = handles
        .client
        .read_one(handles.accum)
        .expect("accum readback failed");
    let wsum_bytes = handles
        .client
        .read_one(handles.wsum)
        .expect("wsum readback failed");
    let weight_bytes = handles
        .client
        .read_one(handles.group_weight)
        .expect("group_weight readback failed");

    Aggregated {
        accum: i32::from_bytes(&accum_bytes)[..handles.accum_len].to_vec(),
        wsum: i32::from_bytes(&wsum_bytes)[..handles.wsum_len].to_vec(),
        group_weight: f32::from_bytes(&weight_bytes)[..handles.refs].to_vec(),
        pixels: setup.pixels(),
    }
}

/// Launches [collab_fused] on its eight-references-per-cube grid and reads back what it aggregated.
///
/// The search walk is whichever one this runtime needs, matching what the shipping code launches.
pub(super) fn run_fused(setup: &Setup) -> Aggregated {
    run_fused_walk(setup, None)
}

/// [run_fused] with the search walk pinned rather than taken from the runtime.
pub(super) fn run_fused_walk(setup: &Setup, warp_uniform: Option<bool>) -> Aggregated {
    let handles = buffers(setup);
    let warp_uniform = warp_uniform.unwrap_or_else(|| needs_warp_uniform_search(&handles.client));

    // The f32 search launches with the f32 ring as its placeholder, as the shipping code does.
    if setup.f16_search {
        launch_fused::<half::f16>(setup, &handles, handles.search_ring.clone(), warp_uniform);
    } else {
        launch_fused::<f32>(setup, &handles, handles.ring.clone(), warp_uniform);
    }

    read_back(handles, setup)
}

/// Launches [collab_fused] with its search ring read as `S`.
fn launch_fused<S: Float>(setup: &Setup, handles: &Buffers, search_ring: Handle, warp_uniform: bool) {
    let profile = setup.profile();
    let curve = setup.noise_curve.unwrap_or([0.0f32; NOISE_CURVE_BINS]);
    let curve_bytes = f32::as_bytes(&curve);
    let curve_buf = handles.client.create_from_slice(curve_bytes);
    let curve_valid = u32::from(setup.noise_curve.is_some());

    let (map_cols, map_rows) = strength_map_dims(setup.width, setup.height);
    let map_len = (map_cols * map_rows) as usize;
    let (map_values, map_mode) = match &setup.strength_map {
        Some((values, mode)) => (values.clone(), *mode),
        None => (vec![1.0f32; map_len], STRENGTH_MAP_OFF),
    };
    assert_eq!(
        map_values.len(),
        map_len,
        "a strength map must cover the frame's quarter grid"
    );
    let map_bytes = f32::as_bytes(&map_values);
    let map_buf = handles.client.create_from_slice(map_bytes);
    let stored_channels = setup.stored_channels();

    let cubes_x = fused_cubes_x(setup.width);
    let grid = CubeCount::new_2d(cubes_x, handles.refs_y);
    let dim = CubeDim::new_1d(64);
    let scale = weight_scale(setup.sigma, &profile);
    let accum_scale = setup.accum_scale();
    let grid_frame_count = grid_frames(setup.radius);
    let pooled_ratio = setup.pooled.unwrap_or(0.0);

    unsafe {
        collab_fused::launch_unchecked::<S, R>(
            &handles.client,
            grid,
            dim,
            stored_channels as usize,
            ArrayArg::from_raw_parts(handles.ring.clone(), setup.ring.len()),
            ArrayArg::from_raw_parts(search_ring, setup.ring.len()),
            ArrayArg::from_raw_parts(handles.mv_field.clone(), setup.mv_field.len()),
            ArrayArg::from_raw_parts(handles.confidence.clone(), setup.confidence.len()),
            ArrayArg::from_raw_parts(handles.neighbour_slots.clone(), setup.neighbour_slots.len()),
            ArrayArg::from_raw_parts(handles.sigma.clone(), stored_channels as usize),
            ArrayArg::from_raw_parts(curve_buf, NOISE_CURVE_BINS),
            ArrayArg::from_raw_parts(map_buf, map_len),
            ArrayArg::from_raw_parts(handles.dct_profile.clone(), 8),
            ArrayArg::from_raw_parts(handles.kaiser.clone(), PATCH_SIZE as usize),
            ArrayArg::from_raw_parts(handles.accum.clone(), handles.accum_len),
            ArrayArg::from_raw_parts(handles.wsum.clone(), handles.wsum_len),
            ArrayArg::from_raw_parts(handles.group_weight.clone(), handles.refs),
            setup.centre_slot,
            setup.c_min,
            setup.lambda_ht,
            curve_valid,
            map_mode,
            scale,
            accum_scale,
            warp_uniform,
            setup.f16_search,
            setup.radius,
            grid_frame_count,
            setup.refine,
            setup.mv_stride,
            setup.conf_stride,
            BLK_STEP,
            BLKSIZE,
            setup.blocks_x,
            setup.blocks_y,
            setup.width,
            setup.height,
            setup.channel_mode.count(),
            setup.k_max,
            stored_channels,
            setup.spatial_radius,
            handles.refs_x,
            map_cols,
            map_rows,
            pooled_ratio,
            setup.pooled.is_some(),
        );
    }
}

/// A ring of `2 * radius + 1` frames of unique content, with motion and confidence fields that
/// vary by block.
///
/// The ring is frame-major, so one [unique_frame] over a `2 * radius + 1` times taller image gives
/// content no two 8x8 windows share, across frames as well as within one. The confidences run from
/// below `c_min` to 1.0, so some blocks skip their whole window and the rest are searched.
pub(super) fn cross_frame_setup(width: u32, height: u32, radius: u32) -> Setup {
    let frames = 2 * radius + 1;
    let blocks_x = width.div_ceil(BLK_STEP);
    let blocks_y = height.div_ceil(BLK_STEP);
    let conf_stride = blocks_x * blocks_y;
    let mv_stride = conf_stride * 2;

    let mut mv_field = vec![0i32; (2 * radius * mv_stride) as usize];
    let mut confidence = vec![0.0f32; (2 * radius * conf_stride) as usize];
    for t in 0..(2 * radius) {
        for block in 0..conf_stride {
            let mv_index = (t * mv_stride + block * 2) as usize;
            // A spread of shifts in both signs, some pushing the refine window off the frame so the
            // clip matters.
            mv_field[mv_index] = (block % 11) as i32 - 5 + t as i32;
            mv_field[mv_index + 1] = 4 - (block % 9) as i32 - t as i32;
            confidence[(t * conf_stride + block) as usize] = ((block * 7 + t * 3) % 11) as f32 / 10.0;
        }
    }

    // The centre sits in the middle of the ring, and the neighbours are the slots either side of it,
    // nearest first.
    let centre_slot = radius;
    let mut neighbour_slots = Vec::new();
    for t in 0..radius {
        neighbour_slots.push(radius - 1 - t);
        neighbour_slots.push(radius + 1 + t);
    }

    let ring = unique_frame(width, height * frames);
    let placeholder = vec![0.0f32; (width * height) as usize];

    Setup {
        ring,
        mv_field,
        confidence,
        neighbour_slots,
        centre_slot,
        c_min: 0.5,
        radius,
        refine: 2,
        mv_stride,
        conf_stride,
        blocks_x,
        blocks_y,
        ..Setup::spatial_only(placeholder, width, height)
    }
}

/// [cross_frame_setup] with a two-channel chroma ring.
///
/// Each pixel takes two neighbouring values of a [unique_frame] twice as wide, so no two 8x8
/// windows share content in either channel.
pub(super) fn chroma_cross_frame_setup(width: u32, height: u32, radius: u32) -> Setup {
    let frames = 2 * radius + 1;
    let mut setup = cross_frame_setup(width, height, radius);
    setup.ring = unique_frame(width * 2, height * frames);
    setup.channel_mode = ChannelMode::Chroma;

    setup
}

/// A three-frame ring whose neighbours hold an exact copy of the centre frame, at the position the
/// zero motion field predicts.
///
/// An exact copy scores distance zero, which every other candidate loses to. At radius 1 the group
/// is four volumes of two frames, and each volume's second frame is neighbour 0, which wins every
/// tie. `refine = 0` narrows each neighbour to that one predicted position, so a volume has nothing
/// else to pick.
pub(super) fn three_frame_ring_with_a_planted_match(width: u32, height: u32) -> Setup {
    let frame = unique_frame(width, height);
    let mut ring = Vec::with_capacity(frame.len() * 3);
    for _ in 0..3 {
        ring.extend_from_slice(&frame);
    }

    let blocks_x = width.div_ceil(BLK_STEP);
    let blocks_y = height.div_ceil(BLK_STEP);
    let conf_stride = blocks_x * blocks_y;
    let mv_stride = conf_stride * 2;
    let placeholder = vec![0.0f32; (width * height) as usize];

    Setup {
        ring,
        mv_field: vec![0i32; (2 * mv_stride) as usize],
        confidence: vec![1.0f32; (2 * conf_stride) as usize],
        neighbour_slots: vec![0u32, 2u32],
        centre_slot: 1,
        radius: 1,
        refine: 0,
        mv_stride,
        conf_stride,
        blocks_x,
        blocks_y,
        ..Setup::spatial_only(placeholder, width, height)
    }
}

/// A five-frame ring whose neighbours hold the centre frame plus a small jitter of their own.
///
/// The jitter differs per neighbour and per pixel, so which three of the four neighbours a volume
/// keeps varies from group to group, and across a frame every neighbour is kept somewhere. The
/// motion field is zero and `refine` is 0, so each neighbour offers exactly one position.
pub(super) fn five_frame_ring_with_jittered_copies(width: u32, height: u32) -> Setup {
    let radius = 2u32;
    let frame = unique_frame(width, height);
    let jitter = noisy_flat_field(width, height * 4, 0.5, 0.002);
    let pixels = (width * height) as usize;

    let mut ring = Vec::with_capacity(pixels * 5);
    for slot in 0..5usize {
        if slot == radius as usize {
            ring.extend_from_slice(&frame);
            continue;
        }

        let neighbour = if slot < radius as usize { slot } else { slot - 1 };
        let offsets = &jitter[neighbour * pixels..(neighbour + 1) * pixels];
        let copy = frame
            .iter()
            .zip(offsets)
            .map(|(&value, &offset)| value + offset - 0.5);
        ring.extend(copy);
    }

    let blocks_x = width.div_ceil(BLK_STEP);
    let blocks_y = height.div_ceil(BLK_STEP);
    let conf_stride = blocks_x * blocks_y;
    let mv_stride = conf_stride * 2;
    let placeholder = vec![0.0f32; pixels];

    Setup {
        ring,
        mv_field: vec![0i32; (2 * radius * mv_stride) as usize],
        confidence: vec![1.0f32; (2 * radius * conf_stride) as usize],
        neighbour_slots: vec![0u32, 1, 3, 4],
        centre_slot: radius,
        radius,
        refine: 0,
        mv_stride,
        conf_stride,
        blocks_x,
        blocks_y,
        ..Setup::spatial_only(placeholder, width, height)
    }
}

/// How many reference patches cover each pixel, on the same grid [ref_pos] lays out.
pub(super) fn reference_cover_counts(width: u32, height: u32) -> Vec<i64> {
    let mut counts = vec![0i64; (width * height) as usize];
    for ref_y in 0..refs_along(height) {
        for ref_x in 0..refs_along(width) {
            let left = ref_pos(ref_x, width);
            let top = ref_pos(ref_y, height);
            for row in 0..PATCH_SIZE {
                for col in 0..PATCH_SIZE {
                    counts[((top + row) * width + left + col) as usize] += 1;
                }
            }
        }
    }

    counts
}

pub(super) fn patch_pool_variance(frame: &[f32], width: u32, height: u32) -> f64 {
    let mut pool: Vec<f64> = Vec::new();
    for ref_y in 0..refs_along(height) {
        for ref_x in 0..refs_along(width) {
            let left = ref_pos(ref_x, width);
            let top = ref_pos(ref_y, height);
            for row in 0..PATCH_SIZE {
                for col in 0..PATCH_SIZE {
                    pool.push(frame[((top + row) * width + left + col) as usize] as f64);
                }
            }
        }
    }

    let mean = pool.iter().sum::<f64>() / pool.len() as f64;
    pool.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / pool.len() as f64
}

pub(super) fn output_variance(got: &Aggregated) -> f64 {
    let values: Vec<f64> = (0..got.accum.len()).map(|i| got.pixel(i)).collect();
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    values.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / values.len() as f64
}

/// A flat field carrying nothing but noise, at the settings a real caller would filter it with.
pub(super) fn flat_noise_setup(width: u32, height: u32, sigma: f32) -> Setup {
    let field = noisy_flat_field(width, height, 0.5, sigma);
    let mut setup = Setup::spatial_only(field, width, height);
    setup.spatial_radius = 9;
    setup.sigma = sigma;
    setup.lambda_ht = 2.7;

    setup
}
