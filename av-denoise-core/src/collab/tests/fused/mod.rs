mod behaviour;
mod noise_curve;
mod recorded;
mod strength_map;
mod walks;

use cubecl::prelude::*;
use cubecl::server::Handle;

use super::helpers::{R, make_client, make_unique_frame, noisy_field_over};
use crate::collab::geometry::{fused_cubes_x, ref_count, ref_pos, refs_along, strength_map_dims};
use crate::collab::kernels::aggregate::{WEIGHT_GAIN, cross_frame_accum_scale, kaiser_window, weight_scale};
use crate::collab::kernels::fused::{STRENGTH_MAP_OFF, collab_fused};
use crate::collab::kernels::transforms::dct_noise_profile;
use crate::collab::{PATCH_SIZE, grid_frames, needs_warp_uniform_search};
use crate::nlmeans::{ChannelMode, NOISE_CURVE_BINS};

/// The spatial search radius most runs below use.
///
/// Large enough that a reference patch away from the frame edge scores
/// a 9x9 window, which is well past the eight members a group keeps,
/// and small enough that the whole sweep stays quick. It is a [`Setup`]
/// field rather than a constant so one test can narrow it far enough to
/// shrink a group below `k_max`.
const SPATIAL_RADIUS: u32 = 4;

/// The group size most runs below use. The fused kernel carries one
/// member per lane of an 8-lane group, so this is the size it is built
/// for.
const K_MAX: u32 = 8;

/// Motion-block side length. The kernel searches every block whose
/// `blksize` span contains a patch.
const BLKSIZE: u32 = 16;

/// Motion-block stride. It stays at `PATCH_SIZE` so a block boundary
/// lines up with a patch boundary.
const BLK_STEP: u32 = 8;

/// The noise level the filter is told to shrink against.
///
/// Small enough against content in `[0, 1]` that the threshold keeps a
/// spread of coefficients rather than everything or nothing, so both
/// sides of the keep decision are exercised.
const SIGMA: f32 = 0.02;

/// A fixed hard-threshold multiplier, pinned independently of
/// `Nl4dParams::default().lambda_ht`.
///
/// Several tests in this file recorded their expected output at this
/// value, so it stays fixed even when the shipped default moves.
const LAMBDA_HT: f32 = 5.3;

/// [`make_unique_frame`] rescaled into `[0, 1]`.
///
/// That helper ramps to ten times the frame width, which suits a
/// matching test and breaks a filtering one. Everything downstream of
/// the match is defined over `[0, 1]`, the scatter clamps at
/// [`crate::collab::kernels::aggregate::ACCUM_CLAMP`], and a patch of
/// values in the hundreds both saturates that clamp and puts every
/// coefficient so far above the noise threshold that the threshold stops
/// being tested at all. Dividing by a constant leaves every 8x8 window
/// exactly as distinct as it was, so the tie-free property these runs
/// rely on is untouched.
pub(super) fn unique_frame(w: u32, h: u32) -> Vec<f32> {
    let raw = make_unique_frame(w, h);
    let peak = raw.iter().copied().fold(0.0f32, f32::max);
    raw.into_iter().map(|v| v / peak).collect()
}

/// Everything one launch of [`collab_fused`] takes, so a test reads as
/// the scenario it sets up rather than as an argument list.
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
    /// Residual correlation the noise profile is built for. `0.0` gives
    /// the all-ones profile most runs use.
    pub(super) rho: f32,
    /// A profile buffer supplied outright, bypassing
    /// [`dct_noise_profile`]. The weight scale still follows whatever
    /// profile is in force.
    pub(super) profile_override: Option<[f32; 8]>,
    /// The aggregation window's `beta`. `0.0`, what every run here uses
    /// unless it says otherwise, is uniform aggregation.
    pub(super) kaiser_beta: f32,
    /// The frame's noise curve. `None` launches with `curve_valid = 0`
    /// and a zeroed buffer.
    pub(super) noise_curve: Option<[f32; NOISE_CURVE_BINS]>,
    /// A strength map and the mode it applies in. `None` launches a unit map with the map off.
    pub(super) strength_map: Option<(Vec<f32>, u32)>,
    /// The channels the ring interleaves per pixel, laid out at
    /// [ChannelMode::storage_count](crate::nlmeans::ChannelMode::storage_count) floats each.
    pub(super) channel_mode: ChannelMode,
}

impl Setup {
    /// A single-frame ring with no neighbours, which leaves every
    /// candidate in the spatial window around the reference patch and
    /// the motion, confidence, and neighbour-slot buffers as dummies
    /// nothing reads.
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
            channel_mode: ChannelMode::Luma,
        }
    }

    /// Ring slots in this setup's frame ring, which is also how many
    /// regions the accumulators carry.
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
    /// One finished pixel, the weighted mean of every filtered patch
    /// that covered it.
    ///
    /// This is what [`crate::collab::kernels::aggregate::collab_normalise`]
    /// computes and what the caller actually sees, so a tolerance stated
    /// against it is a tolerance in pixel values. Comparing the raw
    /// accumulator instead would fail on a group-weight difference that
    /// the division cancels out.
    ///
    /// A pixel no member covered has a zero weight sum and reads zero.
    pub(super) fn pixel(&self, idx: usize) -> f64 {
        let w = self.wsum[idx];
        if w == 0 {
            0.0
        } else {
            // `wsum` counts at `WEIGHT_GAIN` times `accum`'s scale, the
            // one factor that does not cancel between the two, exactly as
            // `collab_normalise` multiplies it back out.
            self.accum[idx] as f64 * WEIGHT_GAIN as f64 / w as f64
        }
    }

    /// The total weight one ring slot's region received. A slot no
    /// member scattered into reads exactly zero.
    pub(super) fn frame_weight_sum(&self, slot: usize) -> i64 {
        self.wsum[slot * self.pixels..(slot + 1) * self.pixels]
            .iter()
            .map(|&v| v as i64)
            .sum()
    }

    /// A compact summary of the whole run, small enough to record as
    /// literals and specific enough that a kernel writing nothing cannot
    /// reproduce it.
    fn digest(&self) -> Digest {
        // Luma stores one channel per pixel across this file, so the two
        // accumulators hold one entry each per pixel and share an index.
        assert_eq!(self.accum.len(), self.wsum.len());
        let n = self.accum.len();
        let mut sum = 0.0f64;
        let mut sum_sq = 0.0f64;
        let mut covered = 0usize;
        for idx in 0..n {
            let v = self.pixel(idx);
            sum += v;
            sum_sq += v * v;
            if self.wsum[idx] != 0 {
                covered += 1;
            }
        }
        let weight_mean =
            self.group_weight.iter().map(|&w| w as f64).sum::<f64>() / self.group_weight.len() as f64;

        let mut probes = [0.0f64; PROBE_COUNT];
        for (i, probe) in probes.iter_mut().enumerate() {
            *probe = self.pixel(probe_index(i, n));
        }

        Digest {
            covered,
            pixel_mean: sum / n as f64,
            pixel_rms: (sum_sq / n as f64).sqrt(),
            weight_mean,
            probes,
        }
    }
}

/// How many individual pixels a [`Digest`] pins alongside its whole-run
/// statistics.
const PROBE_COUNT: usize = 8;

/// The pixel a probe reads. The odd stride spreads the eight probes over
/// the buffer so no two land in one patch or one row.
pub(super) fn probe_index(i: usize, len: usize) -> usize {
    (i * 7919 + 1013) % len
}

/// One run's output, boiled down to numbers a test can carry as
/// literals.
pub(super) struct Digest {
    /// Pixels whose weight sum is non-zero.
    pub(super) covered: usize,
    /// Mean normalised pixel over every slot of the accumulator ring.
    pub(super) pixel_mean: f64,
    /// Root mean square of the same pixels.
    pub(super) pixel_rms: f64,
    /// Mean of the per-reference group weight.
    pub(super) weight_mean: f64,
    /// Individual pixels at [`probe_index`] positions.
    pub(super) probes: [f64; PROBE_COUNT],
}

/// How far a recorded whole-run statistic may move, relative.
///
/// Each of these sums thousands of values, so a single coefficient
/// falling the other side of the hard threshold moves one by around
/// `1e-8`.
///
/// The literals below were recorded from an implementation that
/// truncated toward zero on the way into the accumulators, which biased
/// every contribution down by up to a fixed-point unit.
/// [`crate::collab::kernels::aggregate::to_fixed`] rounds instead, so the
/// values it produces sit about `1e-5` relative above the recorded ones.
/// That is the quantisation step itself moving, not the filter, and no
/// implementation can match across it more tightly than this. Re-recording
/// from the fused kernel would be worse than loosening, because these
/// literals are a second implementation's answer and matching the kernel
/// against itself would prove nothing.
///
/// `2e-5` is still vanishingly small next to the difference a kernel that
/// stopped writing would produce.
const DIGEST_RELATIVE_TOLERANCE: f64 = 2.0e-5;

/// How far a recorded probe pixel may move, absolute.
///
/// The hard threshold is a discontinuity, and a coefficient whose
/// magnitude sits within float rounding of `lambda_ht * sigma` can fall
/// either way. One such coefficient moves its group's reconstruction by
/// its own magnitude, and a probe reads one pixel rather than an
/// average, so this is the same `1e-3` (a quarter of an 8-bit code
/// level) the differential these literals were recorded from allowed.
const PROBE_TOLERANCE: f64 = 1.0e-3;

/// Checks a run against values recorded from a known-good
/// implementation.
///
/// Every expected value below was produced by
/// `collab_group_temporal` + `collab_filter_ht`, the two-kernel pair the
/// fused kernel replaces, on 2026-08-21, immediately before that pair
/// was deleted. The two agreed to `5e-9` on the whole-run statistics and
/// `5e-7` on the worst probe at the time of recording.
///
/// Fixed literals rather than a second kernel is what keeps this
/// meaningful. A cubecl 0.10 compiler bug makes a failing shader
/// compile silently do nothing at all, leaving the buffers untouched,
/// and a test that compared the fused kernel against itself would have
/// compared zeros to zeros. Zeros do not match these.
pub(super) fn assert_matches_recorded(label: &str, got: &Aggregated, want: &Digest) {
    let d = got.digest();
    assert_eq!(
        d.covered, want.covered,
        "{label}: {} pixels carry weight, recorded {}",
        d.covered, want.covered
    );

    for (name, have, expect) in [
        ("pixel_mean", d.pixel_mean, want.pixel_mean),
        ("pixel_rms", d.pixel_rms, want.pixel_rms),
        ("weight_mean", d.weight_mean, want.weight_mean),
    ] {
        let rel = (have - expect).abs() / expect.abs().max(1.0e-30);
        assert!(
            rel < DIGEST_RELATIVE_TOLERANCE,
            "{label}: {name} is {have}, recorded {expect}, relative error {rel}"
        );
    }

    for (i, (&have, &expect)) in d.probes.iter().zip(want.probes.iter()).enumerate() {
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

pub(super) fn buffers(s: &Setup) -> Buffers {
    let client = make_client();
    let refs_x = refs_along(s.width);
    let refs_y = refs_along(s.height);
    let refs = ref_count(s.width, s.height);
    let frames = s.frames() as usize;
    let stored_ch = s.stored_channels() as usize;
    let accum_len = s.pixels() * stored_ch * frames;
    let wsum_len = s.pixels() * frames;

    // Padding lanes past the live channels carry a zero sigma, as the
    // denoiser uploads them.
    let mut sigma = vec![0.0f32; stored_ch];
    let live_channels = s.channel_mode.count() as usize;
    sigma[..live_channels].fill(s.sigma);

    Buffers {
        ring: client.create_from_slice(f32::as_bytes(&s.ring)),
        mv_field: client.create_from_slice(i32::as_bytes(&s.mv_field)),
        confidence: client.create_from_slice(f32::as_bytes(&s.confidence)),
        neighbour_slots: client.create_from_slice(u32::as_bytes(&s.neighbour_slots)),
        sigma: client.create_from_slice(f32::as_bytes(&sigma)),
        dct_profile: client.create_from_slice(f32::as_bytes(&s.profile())),
        kaiser: client.create_from_slice(f32::as_bytes(&kaiser_window(s.kaiser_beta))),
        // Zeroed here rather than by `collab_zero_accum`, since the
        // scatter is the only thing writing them in these runs.
        accum: client.create_from_slice(i32::as_bytes(&vec![0i32; accum_len])),
        wsum: client.create_from_slice(i32::as_bytes(&vec![0i32; wsum_len])),
        group_weight: client.empty(refs * size_of::<f32>()),
        accum_len,
        wsum_len,
        refs,
        refs_x,
        refs_y,
        client,
    }
}

pub(super) fn read_back(b: Buffers, s: &Setup) -> Aggregated {
    let accum = b.client.read_one(b.accum).expect("accum readback failed");
    let wsum = b.client.read_one(b.wsum).expect("wsum readback failed");
    let group_weight = b
        .client
        .read_one(b.group_weight)
        .expect("group_weight readback failed");

    Aggregated {
        accum: i32::from_bytes(&accum)[..b.accum_len].to_vec(),
        wsum: i32::from_bytes(&wsum)[..b.wsum_len].to_vec(),
        group_weight: f32::from_bytes(&group_weight)[..b.refs].to_vec(),
        pixels: s.pixels(),
    }
}

/// Launches [`collab_fused`] on its eight-references-per-cube grid and
/// reads back what it aggregated.
///
/// The search walk is whichever one this runtime needs, so a plain run
/// covers whatever the shipping code would actually launch here.
pub(super) fn run_fused(s: &Setup) -> Aggregated {
    run_fused_walk(s, None)
}

/// [`run_fused`] with the search walk pinned rather than taken from the
/// runtime, so one test can run both and compare them.
pub(super) fn run_fused_walk(s: &Setup, warp_uniform: Option<bool>) -> Aggregated {
    let b = buffers(s);
    let profile = s.profile();
    let curve = s.noise_curve.unwrap_or([0.0f32; NOISE_CURVE_BINS]);
    let curve_buf = b.client.create_from_slice(f32::as_bytes(&curve));
    let curve_valid = u32::from(s.noise_curve.is_some());

    let (map_cols, map_rows) = strength_map_dims(s.width, s.height);
    let map_len = (map_cols * map_rows) as usize;
    let (map_values, map_mode) = match &s.strength_map {
        Some((values, mode)) => (values.clone(), *mode),
        None => (vec![1.0f32; map_len], STRENGTH_MAP_OFF),
    };
    assert_eq!(
        map_values.len(),
        map_len,
        "a strength map must cover the frame's quarter grid"
    );
    let map_buf = b.client.create_from_slice(f32::as_bytes(&map_values));
    let stored_ch = s.stored_channels();

    unsafe {
        collab_fused::launch_unchecked::<R>(
            &b.client,
            CubeCount::new_2d(fused_cubes_x(s.width), b.refs_y),
            CubeDim::new_1d(64),
            stored_ch as usize,
            ArrayArg::from_raw_parts(b.ring.clone(), s.ring.len()),
            ArrayArg::from_raw_parts(b.mv_field.clone(), s.mv_field.len()),
            ArrayArg::from_raw_parts(b.confidence.clone(), s.confidence.len()),
            ArrayArg::from_raw_parts(b.neighbour_slots.clone(), s.neighbour_slots.len()),
            ArrayArg::from_raw_parts(b.sigma.clone(), stored_ch as usize),
            ArrayArg::from_raw_parts(curve_buf, NOISE_CURVE_BINS),
            ArrayArg::from_raw_parts(map_buf, map_len),
            ArrayArg::from_raw_parts(b.dct_profile.clone(), 8),
            ArrayArg::from_raw_parts(b.kaiser.clone(), PATCH_SIZE as usize),
            ArrayArg::from_raw_parts(b.accum.clone(), b.accum_len),
            ArrayArg::from_raw_parts(b.wsum.clone(), b.wsum_len),
            ArrayArg::from_raw_parts(b.group_weight.clone(), b.refs),
            s.centre_slot,
            s.c_min,
            s.lambda_ht,
            curve_valid,
            map_mode,
            weight_scale(s.sigma, &profile),
            s.accum_scale(),
            warp_uniform.unwrap_or_else(|| needs_warp_uniform_search(&b.client)),
            s.radius,
            grid_frames(s.radius),
            s.refine,
            s.mv_stride,
            s.conf_stride,
            BLK_STEP,
            BLKSIZE,
            s.blocks_x,
            s.blocks_y,
            s.width,
            s.height,
            s.channel_mode.count(),
            s.k_max,
            stored_ch,
            s.spatial_radius,
            b.refs_x,
            map_cols,
            map_rows,
        );
    }

    read_back(b, s)
}

/// A ring of `2 * radius + 1` frames of unique content, with a motion
/// field and a confidence field that both vary by block.
///
/// The ring is laid out frame-major, exactly as `read_line` indexes it,
/// so one call to `make_unique_frame` over a `2 * radius + 1` times
/// taller image fills the whole ring with content no two 8x8 windows
/// share, across frames as well as within one.
///
/// The confidences run from below `c_min` to 1.0, so some blocks have
/// their whole window skipped and the rest are searched.
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
            let mv = (t * mv_stride + block * 2) as usize;
            // A spread of shifts in both signs, including some that push
            // the refine window off the frame so the clip matters.
            mv_field[mv] = (block % 11) as i32 - 5 + t as i32;
            mv_field[mv + 1] = 4 - (block % 9) as i32 - t as i32;
            confidence[(t * conf_stride + block) as usize] = ((block * 7 + t * 3) % 11) as f32 / 10.0;
        }
    }

    // The centre sits in the middle of the ring, and the neighbours are
    // the slots either side of it, nearest first.
    let centre_slot = radius;
    let mut neighbour_slots = Vec::new();
    for t in 0..radius {
        neighbour_slots.push(radius - 1 - t);
        neighbour_slots.push(radius + 1 + t);
    }

    Setup {
        ring: unique_frame(width, height * frames),
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
        ..Setup::spatial_only(vec![0.0f32; (width * height) as usize], width, height)
    }
}

/// A three-frame ring whose neighbours hold an exact copy of the centre
/// frame, at the position the zero motion field predicts.
///
/// An exact copy scores distance zero, which every other candidate on
/// this content loses to. At radius 1 the group is four volumes of two
/// frames, and each volume's second frame is neighbour 0, which wins
/// every tie. `refine = 0` narrows each neighbour's rectangle to that
/// one predicted position, so there is nothing else in a neighbour for
/// a volume to pick instead.
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
        ..Setup::spatial_only(vec![0.0f32; (width * height) as usize], width, height)
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
    let jitter = noisy_field_over(width, height * 4, 0.5, 0.002);
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
        ..Setup::spatial_only(vec![0.0f32; pixels], width, height)
    }
}

/// How many reference patches cover each pixel of a `width` by `height`
/// frame, on the same grid [`ref_pos`] lays out.
pub(super) fn reference_cover_counts(width: u32, height: u32) -> Vec<i64> {
    let mut counts = vec![0i64; (width * height) as usize];
    for ry in 0..refs_along(height) {
        for rx in 0..refs_along(width) {
            let px = ref_pos(rx, width);
            let py = ref_pos(ry, height);
            for row in 0..PATCH_SIZE {
                for col in 0..PATCH_SIZE {
                    counts[((py + row) * width + px + col) as usize] += 1;
                }
            }
        }
    }
    counts
}

pub(super) fn patch_pool_variance(frame: &[f32], w: u32, h: u32) -> f64 {
    let mut pool: Vec<f64> = Vec::new();
    for ry in 0..refs_along(h) {
        for rx in 0..refs_along(w) {
            let px = ref_pos(rx, w);
            let py = ref_pos(ry, h);
            for row in 0..PATCH_SIZE {
                for col in 0..PATCH_SIZE {
                    pool.push(frame[((py + row) * w + px + col) as usize] as f64);
                }
            }
        }
    }
    let mean = pool.iter().sum::<f64>() / pool.len() as f64;
    pool.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / pool.len() as f64
}

/// The variance of a run's finished pixels.
pub(super) fn output_variance(got: &Aggregated) -> f64 {
    let values: Vec<f64> = (0..got.accum.len()).map(|i| got.pixel(i)).collect();
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64
}

/// A flat field carrying nothing but noise, at the settings a real
/// caller would filter it with.
pub(super) fn flat_noise_setup(w: u32, h: u32, sigma: f32) -> Setup {
    let mut s = Setup::spatial_only(noisy_field_over(w, h, 0.5, sigma), w, h);
    s.spatial_radius = 9;
    s.sigma = sigma;
    s.lambda_ht = 2.7;
    s
}
