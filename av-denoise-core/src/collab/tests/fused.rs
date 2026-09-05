use cubecl::prelude::*;
use cubecl::server::Handle;

use super::helpers::{
    R,
    deterministic_texture,
    make_client,
    make_unique_frame,
    noisy_field_over,
    plant_patch,
};
use crate::collab::geometry::{fused_cubes_x, ref_count, ref_pos, refs_along};
use crate::collab::kernels::aggregate::{WEIGHT_GAIN, cross_frame_accum_scale, kaiser_window, weight_scale};
use crate::collab::kernels::fused::collab_fused;
use crate::collab::kernels::transforms::dct_noise_profile;
use crate::collab::{PATCH_SIZE, STEP};

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

/// Motion-block side length, the value the mismatch variance is scored
/// against.
const BLKSIZE: u32 = 16;

/// Motion-block stride. It stays at `PATCH_SIZE` so a block boundary
/// lines up with a patch boundary.
const BLK_STEP: u32 = 8;

/// `thsad(BLKSIZE, 1.0)` in normalised SAD units, the same value a real
/// caller gets at this block size and the default scale.
const THSAD: f32 = (BLKSIZE * BLKSIZE) as f32 * 0.02;

/// The noise level the filter is told to shrink against.
///
/// Small enough against content in `[0, 1]` that the threshold keeps a
/// spread of coefficients rather than everything or nothing, so both
/// sides of the keep decision are exercised.
const SIGMA: f32 = 0.02;

/// `Nl4dParams::default().lambda_ht`.
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
fn unique_frame(w: u32, h: u32) -> Vec<f32> {
    let raw = make_unique_frame(w, h);
    let peak = raw.iter().copied().fold(0.0f32, f32::max);
    raw.into_iter().map(|v| v / peak).collect()
}

/// Everything one launch of [`collab_fused`] takes, so a test reads as
/// the scenario it sets up rather than as an argument list.
struct Setup {
    ring: Vec<f32>,
    mv_field: Vec<i32>,
    confidence: Vec<f32>,
    neighbour_slots: Vec<u32>,
    centre_slot: u32,
    noise_floor: f32,
    c_min: f32,
    radius: u32,
    refine: u32,
    spatial_radius: u32,
    mv_stride: u32,
    conf_stride: u32,
    blocks_x: u32,
    blocks_y: u32,
    width: u32,
    height: u32,
    k_max: u32,
    sigma: f32,
    lambda_ht: f32,
    /// Whether a temporal member's motion-block confidence inflates its
    /// own noise variance.
    confidence_variance: bool,
    /// Residual correlation the noise profile is built for. `0.0` gives
    /// the all-ones profile most runs use.
    rho: f32,
    /// The SAD threshold a temporal member's mismatch variance is derived
    /// from. [`THSAD`] is what a real caller passes at the default block
    /// size, and raising it is how a run drives that variance far above
    /// the channel sigma.
    thsad: f32,
    /// A profile buffer supplied outright, bypassing
    /// [`dct_noise_profile`]. The weight scale still follows whatever
    /// profile is in force.
    profile_override: Option<[f32; 8]>,
    /// The aggregation window's `beta`. `0.0`, what every run here uses
    /// unless it says otherwise, is uniform aggregation.
    kaiser_beta: f32,
}

impl Setup {
    /// A single-frame ring with no neighbours, which leaves every
    /// candidate in the spatial window around the reference patch and
    /// the motion, confidence, and neighbour-slot buffers as dummies
    /// nothing reads.
    fn spatial_only(frame: Vec<f32>, width: u32, height: u32) -> Self {
        assert_eq!(frame.len(), (width * height) as usize);
        Setup {
            ring: frame,
            mv_field: vec![0i32, 0i32],
            confidence: vec![1.0f32],
            neighbour_slots: vec![0u32],
            centre_slot: 0,
            noise_floor: 0.0,
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
            confidence_variance: true,
            rho: 0.0,
            thsad: THSAD,
            profile_override: None,
            kaiser_beta: 0.0,
        }
    }

    /// Ring slots in this setup's frame ring, which is also how many
    /// regions the accumulators carry.
    fn frames(&self) -> u32 {
        self.ring.len() as u32 / (self.width * self.height)
    }

    fn pixels(&self) -> usize {
        (self.width * self.height) as usize
    }

    /// The fixed-point scale the scatter counts in.
    fn accum_scale(&self) -> f32 {
        cross_frame_accum_scale(self.spatial_radius, self.radius)
    }

    /// The correlation profile this run's threshold reads.
    fn profile(&self) -> [f32; 8] {
        self.profile_override
            .unwrap_or_else(|| dct_noise_profile(self.rho))
    }
}

/// One run's aggregated output, read back after its launch.
struct Aggregated {
    accum: Vec<i32>,
    wsum: Vec<i32>,
    group_weight: Vec<f32>,
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
    fn pixel(&self, idx: usize) -> f64 {
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
    fn frame_weight_sum(&self, slot: usize) -> i64 {
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
fn probe_index(i: usize, len: usize) -> usize {
    (i * 7919 + 1013) % len
}

/// One run's output, boiled down to numbers a test can carry as
/// literals.
struct Digest {
    /// Pixels whose weight sum is non-zero.
    covered: usize,
    /// Mean normalised pixel over every slot of the accumulator ring.
    pixel_mean: f64,
    /// Root mean square of the same pixels.
    pixel_rms: f64,
    /// Mean of the per-reference group weight.
    weight_mean: f64,
    /// Individual pixels at [`probe_index`] positions.
    probes: [f64; PROBE_COUNT],
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
fn assert_matches_recorded(label: &str, got: &Aggregated, want: &Digest) {
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
struct Buffers {
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

/// Luma always stores one channel per line, so `stored_ch` and the
/// kernel's `Size` selector are both fixed at 1 across this file.
const STORED_CH: u32 = 1;

fn buffers(s: &Setup) -> Buffers {
    let client = make_client();
    let refs_x = refs_along(s.width);
    let refs_y = refs_along(s.height);
    let refs = ref_count(s.width, s.height);
    let frames = s.frames() as usize;
    let accum_len = s.pixels() * STORED_CH as usize * frames;
    let wsum_len = s.pixels() * frames;

    Buffers {
        ring: client.create_from_slice(f32::as_bytes(&s.ring)),
        mv_field: client.create_from_slice(i32::as_bytes(&s.mv_field)),
        confidence: client.create_from_slice(f32::as_bytes(&s.confidence)),
        neighbour_slots: client.create_from_slice(u32::as_bytes(&s.neighbour_slots)),
        sigma: client.create_from_slice(f32::as_bytes(&[s.sigma])),
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

fn read_back(b: Buffers, s: &Setup) -> Aggregated {
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
fn run_fused(s: &Setup) -> Aggregated {
    let b = buffers(s);
    let profile = s.profile();

    unsafe {
        collab_fused::launch_unchecked::<R>(
            &b.client,
            CubeCount::new_2d(fused_cubes_x(s.width), b.refs_y),
            CubeDim::new_1d(64),
            STORED_CH as usize,
            ArrayArg::from_raw_parts(b.ring.clone(), s.ring.len()),
            ArrayArg::from_raw_parts(b.mv_field.clone(), s.mv_field.len()),
            ArrayArg::from_raw_parts(b.confidence.clone(), s.confidence.len()),
            ArrayArg::from_raw_parts(b.neighbour_slots.clone(), s.neighbour_slots.len()),
            ArrayArg::from_raw_parts(b.sigma.clone(), 1),
            ArrayArg::from_raw_parts(b.dct_profile.clone(), 8),
            ArrayArg::from_raw_parts(b.kaiser.clone(), PATCH_SIZE as usize),
            ArrayArg::from_raw_parts(b.accum.clone(), b.accum_len),
            ArrayArg::from_raw_parts(b.wsum.clone(), b.wsum_len),
            ArrayArg::from_raw_parts(b.group_weight.clone(), b.refs),
            s.centre_slot,
            s.noise_floor,
            s.c_min,
            s.thsad,
            s.lambda_ht,
            weight_scale(s.sigma, &profile),
            s.accum_scale(),
            s.confidence_variance,
            s.radius,
            s.refine,
            s.mv_stride,
            s.conf_stride,
            BLK_STEP,
            BLKSIZE,
            s.blocks_x,
            s.blocks_y,
            s.width,
            s.height,
            1u32,
            s.k_max,
            STORED_CH,
            s.spatial_radius,
            b.refs_x,
        );
    }

    read_back(b, s)
}

// ---------------------------------------------------------------------
// Recorded-output runs.
//
// Each of these was a differential against `collab_group_temporal` +
// `collab_filter_ht` until that pair was deleted. The scenarios are
// unchanged, and only the oracle is a set of literals now.
// ---------------------------------------------------------------------

/// Content without ties, so nothing about the result depends on how the
/// insert breaks one.
///
/// `make_unique_frame` is built so that any two distinct 8x8 windows
/// differ in most of their 64 pixels.
#[test]
fn fused_reproduces_recorded_output_on_unique_content() {
    let (w, h) = (128u32, 96u32);
    let s = Setup::spatial_only(unique_frame(w, h), w, h);
    assert_matches_recorded(
        "unique content",
        &run_fused(&s),
        &Digest {
            covered: 12288,
            pixel_mean: 0.500102660422,
            pixel_rms: 0.577471243275,
            weight_mean: 1250.000000000,
            probes: [
                0.917905456141,
                0.787302672863,
                0.650475382805,
                0.517024146186,
                0.386674649788,
                0.255359411240,
                0.120508321126,
                0.989773918601,
            ],
        },
    );
}

/// The same content with its ramp turned on its side.
///
/// `make_unique_frame` ramps along x, which makes a one-column shift far
/// costlier than a one-row shift, so every member a group keeps sits in
/// a narrow column band and the search rectangle's x extent never
/// decides anything. Transposing the frame moves that band onto the x
/// axis, so this is the run where the horizontal bounds are
/// load-bearing.
#[test]
fn fused_reproduces_recorded_output_on_a_transposed_ramp() {
    let (w, h) = (128u32, 96u32);
    let source = unique_frame(h, w);
    let mut frame = vec![0.0f32; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            frame[(y * w + x) as usize] = source[(x * h + y) as usize];
        }
    }
    let s = Setup::spatial_only(frame, w, h);
    assert_matches_recorded(
        "transposed ramp",
        &run_fused(&s),
        &Digest {
            covered: 12288,
            pixel_mean: 0.500026936557,
            pixel_rms: 0.577366054447,
            weight_mean: 1238.896681776,
            probes: [
                0.075242505755,
                0.724634047477,
                0.367116374354,
                0.014184951782,
                0.659262769363,
                0.307459000618,
                0.951007338131,
                0.589665272066,
            ],
        },
    );
}

/// A width whose reference count is not a multiple of 8 leaves the last
/// cube of each row partly out of range.
///
/// 104 pixels wide gives 25 reference patches at `STEP = 4`, so the
/// fourth cube runs one live group and seven dead ones. Those seven must
/// reach every barrier and write nothing. A dead group that scattered
/// would double the last reference's contribution, which shows up here
/// as a moved pixel rather than needing its own assertion.
#[test]
fn fused_reproduces_recorded_output_when_refs_are_not_a_multiple_of_eight() {
    let (w, h) = (104u32, 96u32);
    let s = Setup::spatial_only(unique_frame(w, h), w, h);
    assert_matches_recorded(
        "ragged reference row",
        &run_fused(&s),
        &Digest {
            covered: 9984,
            pixel_mean: 0.500121022858,
            pixel_rms: 0.577469311374,
            weight_mean: 1240.579711065,
            probes: [
                0.745903455294,
                0.891958951950,
                0.032306798299,
                0.177212221869,
                0.322843606131,
                0.468612211367,
                0.608005691977,
                0.754309082031,
            ],
        },
    );
}

/// The stack transform's shorter ladders, on a search space too small to
/// fill a group.
///
/// At `spatial_radius = 1` a corner reference sees a 2x2 rectangle, so
/// four candidates and a group of four, an edge reference sees 2x3 and
/// also keeps four, and an interior one sees 3x3 and fills to eight.
/// Every wider configuration fills every group to eight, so this is the
/// run where the 2- and 4-member ladders execute at all.
#[test]
fn fused_reproduces_recorded_output_on_a_short_search_space() {
    let (w, h) = (64u32, 64u32);
    let mut s = Setup::spatial_only(unique_frame(w, h), w, h);
    s.spatial_radius = 1;
    assert_matches_recorded(
        "short search space",
        &run_fused(&s),
        &Digest {
            covered: 4096,
            pixel_mean: 0.500333883408,
            pixel_rms: 0.577536371630,
            weight_mean: 768.518540988,
            probes: [
                0.836218530965,
                0.571090123027,
                0.304096429037,
                0.033846737369,
                0.774293684286,
                0.508250150663,
                0.242787978384,
                0.977499961853,
            ],
        },
    );
}

/// The tie-break path, on content where a great many candidates score
/// the same distance.
///
/// Noise over a flat field has no ramp to separate the candidates, so
/// this is the run where the self-match sentinel and the first-wins
/// insert decide the member set.
#[test]
fn fused_reproduces_recorded_output_on_noise() {
    let (w, h) = (64u32, 64u32);
    let s = Setup::spatial_only(noisy_field_over(w, h, 0.5, 0.05), w, h);
    assert_matches_recorded(
        "noise",
        &run_fused(&s),
        &Digest {
            covered: 4096,
            pixel_mean: 0.500598531425,
            pixel_rms: 0.500781913699,
            weight_mean: 168.165750156,
            probes: [
                0.473047106911,
                0.525827771943,
                0.505852930189,
                0.501965226326,
                0.472216666744,
                0.498205827272,
                0.493595121410,
                0.515348414403,
            ],
        },
    );
}

/// A non-zero `rho`, where the correlation profile stops being all ones.
///
/// The old filter multiplied the profile into each member's variance
/// before the variance ladder ran. The fused kernel multiplies it in at
/// the threshold instead. The ladder only averages and the profile is a
/// constant factor across the stack axis, so the two orders agree in
/// exact arithmetic, and this is the run that says so on a GPU. Every
/// other run here uses `dct_noise_profile(0.0)`, which is all ones and
/// cannot tell the two orders apart. `0.86` is the shipped table's high
/// end.
#[test]
fn fused_reproduces_recorded_output_under_correlation_shaping() {
    let (w, h) = (64u32, 64u32);
    let mut s = Setup::spatial_only(noisy_field_over(w, h, 0.5, 0.05), w, h);
    s.rho = 0.86;
    assert_matches_recorded(
        "correlation shaping",
        &run_fused(&s),
        &Digest {
            covered: 4096,
            pixel_mean: 0.500501375321,
            pixel_rms: 0.502136468679,
            weight_mean: 54.170715162,
            probes: [
                0.439152209001,
                0.572103197408,
                0.475816598569,
                0.509686441252,
                0.405882571403,
                0.498373582524,
                0.493889111273,
                0.547764034977,
            ],
        },
    );
}

/// A ring of `2 * radius + 1` frames of unique content, with a motion
/// field and a confidence field that both vary by block.
///
/// The ring is laid out frame-major, exactly as `read_line` indexes it,
/// so one call to `make_unique_frame` over a `2 * radius + 1` times
/// taller image fills the whole ring with content no two 8x8 windows
/// share, across frames as well as within one.
///
/// The confidences run from below `c_min` to 1.0, so some neighbours
/// have their whole window skipped and the rest produce a spread of
/// mismatch variances rather than one repeated value.
fn cross_frame_setup(width: u32, height: u32, radius: u32) -> Setup {
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

/// The whole temporal path at once: the `c_min` skip, the per-member
/// mismatch variance derived from a packed neighbour index, and the
/// scatter into each member's own region of the accumulator ring.
#[test]
fn fused_reproduces_recorded_output_across_frames() {
    let s = cross_frame_setup(64, 64, 2);
    assert_matches_recorded(
        "cross frame",
        &run_fused(&s),
        &Digest {
            covered: 12928,
            pixel_mean: 0.319278212107,
            pixel_rms: 0.462380832227,
            weight_mean: 1149.191924642,
            probes: [
                0.839722565729,
                0.574316714978,
                0.298724122489,
                0.000000000000,
                0.774649096602,
                0.000000000000,
                0.000000000000,
                0.000000000000,
            ],
        },
    );
}

/// The same cross-frame run with the mismatch variance off.
///
/// `use_member_sigma` is a `#[comptime]` flag, so it compiles a second
/// program, and the arm with it off is the one that checks the threshold
/// still reads a plain `sigma^2` per member.
#[test]
fn fused_reproduces_recorded_output_without_the_mismatch_variance() {
    let mut s = cross_frame_setup(64, 64, 2);
    s.confidence_variance = false;
    assert_matches_recorded(
        "cross frame, flat sigma",
        &run_fused(&s),
        &Digest {
            covered: 12928,
            pixel_mean: 0.319277061395,
            pixel_rms: 0.462378164801,
            weight_mean: 1242.592593316,
            probes: [
                0.839714050293,
                0.574348068237,
                0.298727416992,
                0.000000000000,
                0.774412972586,
                0.000000000000,
                0.000000000000,
                0.000000000000,
            ],
        },
    );
}

/// A three-frame ring whose neighbours hold an exact copy of the centre
/// frame, at the position the zero motion field predicts.
///
/// An exact copy scores distance zero, which every other candidate on
/// this content loses to, so members 1 and 2 of every group come from
/// the two neighbour slots. `refine = 0` narrows each neighbour's
/// rectangle to that one predicted position, so there is nothing else in
/// a neighbour for the group to pick instead.
fn three_frame_ring_with_a_planted_match(width: u32, height: u32) -> Setup {
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

/// A group with members in neighbour frames must scatter into those
/// frames' regions of the ring, not collapse onto the centre frame.
///
/// This is the cross-frame aggregation the temporal path exists for, and
/// it is easy to lose, because the frame a member came from is never
/// written down anywhere between the match and the scatter.
#[test]
fn fused_scatters_into_every_member_frame() {
    let (w, h) = (64u32, 64u32);
    let s = three_frame_ring_with_a_planted_match(w, h);
    let got = run_fused(&s);
    for slot in 0..3 {
        assert!(
            got.frame_weight_sum(slot) > 0,
            "ring slot {slot} received nothing"
        );
    }
    assert_matches_recorded(
        "planted cross-frame match",
        &got,
        &Digest {
            covered: 12288,
            pixel_mean: 0.500274434257,
            pixel_rms: 0.577655447440,
            weight_mean: 1246.296296658,
            probes: [
                0.836406707764,
                0.573966026306,
                0.301191602434,
                0.036788940430,
                0.774992261614,
                0.509891510010,
                0.239036560059,
                0.979254982688,
            ],
        },
    );
}

// ---------------------------------------------------------------------
// Behaviour the two-kernel pair used to be checked on directly, restated
// against the fused kernel's own outputs.
// ---------------------------------------------------------------------

/// How many reference patches cover each pixel of a `width` by `height`
/// frame, on the same grid [`ref_pos`] lays out.
fn reference_cover_counts(width: u32, height: u32) -> Vec<i64> {
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

/// At `sigma = 0` every threshold is zero, so nothing is discarded and
/// the transform chain must hand every member's own pixels back
/// unchanged.
///
/// Every contribution any pixel receives is then that pixel's own input
/// value, whatever group carried it, and the weighted mean of a set of
/// identical values is that value. `k_max = 1` exercises the `k_use = 1`
/// case, where the stack transform is a no-op and only the 2D DCT round
/// trip runs. `k_max = 8` forces a full stack over content where every
/// position differs from every other, so all three Haar levels carry
/// non-trivial detail coefficients.
#[test]
fn zero_sigma_hands_every_member_back_unchanged() {
    let (w, h) = (32u32, 32u32);
    let frame = unique_frame(w, h);

    for k_max in [1u32, 8] {
        let mut s = Setup::spatial_only(frame.clone(), w, h);
        s.k_max = k_max;
        s.sigma = 0.0;
        s.spatial_radius = 4;
        let got = run_fused(&s);

        for (idx, &want) in frame.iter().enumerate() {
            assert!(
                got.wsum[idx] > 0,
                "k_max={k_max} idx={idx}: no member covered this pixel"
            );
            let have = got.pixel(idx);
            assert!(
                (want as f64 - have).abs() < 1e-3,
                "k_max={k_max} idx={idx}: want {want} got {have}"
            );
        }
    }
}

/// A temporal member's mismatch variance has no relation to the channel
/// sigma the group weight is normalised against, so a badly matched group
/// has no lower bound on its weight (see
/// [`crate::collab::kernels::aggregate::weight_scale`]). Push that
/// variance up far enough and the weight stops being representable at
/// all, and a group that reaches the accumulators as nothing leaves a
/// covered pixel with an empty weight sum, which normalisation can only
/// render as black.
///
/// Zero confidence is the worst case `mismatch_sigma2` models, and
/// `thsad` scales the variance it implies. The radii are the shipped
/// defaults, so the run counts in the same fixed point a real cross-frame
/// pass does rather than the finer one a small search would pick. Every
/// pixel a reference covers has to keep carrying weight across all of it.
///
/// The last two rungs are past anything a caller would ask for, which is
/// the point: they run the mismatch variance so far past
/// [`crate::collab::kernels::fused::MEMBER_SIGMA2_CAP`] that only the cap
/// is holding the weight inside the fixed point at all.
#[test]
fn a_badly_matched_group_still_reaches_the_accumulators() {
    let (w, h) = (32u32, 32u32);
    let counts = reference_cover_counts(w, h);

    for scale in [1.0f32, 64.0, 1024.0, 4096.0] {
        let mut s = cross_frame_setup(w, h, 2);
        s.spatial_radius = 9;
        s.confidence.fill(0.0);
        // The confidence floor has to come down with it, or the groups
        // are skipped before they are ever scored and the run says
        // nothing about their weights.
        s.c_min = 0.0;
        s.thsad = THSAD * scale;

        let got = run_fused(&s);
        let base = s.centre_slot as usize * s.pixels();
        for (idx, &count) in counts.iter().enumerate() {
            if count == 0 {
                continue;
            }
            assert!(
                got.wsum[base + idx] > 0,
                "thsad scale {scale}: {count} references cover pixel {idx} and its weight \
                 sum is still {}",
                got.wsum[base + idx],
            );
        }
    }
}

/// A patch corner is weighted by the square of the window's end tap,
/// `0.193` at `beta = 2`, so the smallest weight the fixed point has to
/// resolve drops about fivefold against the uniform case.
#[test]
fn a_windowed_badly_matched_group_still_reaches_the_accumulators() {
    let (w, h) = (32u32, 32u32);
    let counts = reference_cover_counts(w, h);

    for scale in [1.0f32, 64.0, 1024.0, 4096.0] {
        let mut s = cross_frame_setup(w, h, 2);
        s.spatial_radius = 9;
        s.confidence.fill(0.0);
        s.c_min = 0.0;
        s.thsad = THSAD * scale;
        s.kaiser_beta = 2.0;

        let got = run_fused(&s);
        let base = s.centre_slot as usize * s.pixels();
        for (idx, &count) in counts.iter().enumerate() {
            if count == 0 {
                continue;
            }
            assert!(
                got.wsum[base + idx] > 0,
                "thsad scale {scale}: {count} references cover pixel {idx} and its weight                  sum is still {} with the window on",
                got.wsum[base + idx],
            );
        }
    }
}

/// The reference patch is always the group's first member.
///
/// At `k_max = 1` a group holds exactly one member, so the only patch it
/// scatters is whichever position slot 0 ended up holding. `sigma = 0`
/// makes every group's weight the same constant, so the weight one pixel
/// accumulates counts the patches that covered it. That count must be
/// exactly the number of reference patches covering it, which only holds
/// if every group scattered its own reference position and nothing else.
/// A group that let a search result reach slot 0 would write somewhere
/// off the reference grid and leave the counts uneven.
#[test]
fn the_reference_patch_is_always_the_first_member() {
    let (w, h) = (32u32, 32u32);
    let mut s = Setup::spatial_only(unique_frame(w, h), w, h);
    s.k_max = 1;
    s.sigma = 0.0;
    let got = run_fused(&s);

    let counts = reference_cover_counts(w, h);
    let unit = got.wsum[0] as i64 / counts[0];
    assert!(unit > 0, "the per-patch weight increment must be positive");
    for (idx, &count) in counts.iter().enumerate() {
        assert_eq!(
            got.wsum[idx] as i64,
            unit * count,
            "pixel {idx} carries {} weight, expected {} reference patches at {unit} each",
            got.wsum[idx],
            count
        );
    }
}

/// The group size is the search space size rounded down to a power of
/// two, capped at `k_max`.
///
/// At `spatial_radius = 1` the clipped rectangle holds 4 positions at a
/// corner reference, 6 at an edge one, and 9 in the interior. Rounding
/// therefore takes the edge references from 6 down to 4, and leaves the
/// interior ones at 8. Running the same frame at `k_max = 4` caps every
/// group at 4, so the two runs must agree exactly wherever rounding
/// already reached 4 and differ wherever it reached 8.
///
/// Clipping the rectangle once is what makes those counts right. Were
/// each offset clamped in turn instead, a corner would count nine
/// positions rather than four, several of them the same physical patch,
/// and the corner references would stop agreeing across the two runs.
#[test]
fn group_size_rounds_down_to_a_power_of_two() {
    let (w, h) = (64u32, 64u32);
    let frame = unique_frame(w, h);

    let mut wide = Setup::spatial_only(frame.clone(), w, h);
    wide.spatial_radius = 1;
    let mut narrow = Setup::spatial_only(frame, w, h);
    narrow.spatial_radius = 1;
    narrow.k_max = 4;

    let wide = run_fused(&wide);
    let narrow = run_fused(&narrow);

    let refs_x = refs_along(w);
    let refs_y = refs_along(h);
    let mut interior_differed = 0usize;
    for ry in 0..refs_y {
        for rx in 0..refs_x {
            let idx = (ry * refs_x + rx) as usize;
            // A clipped axis contributes 2 positions instead of 3, so a
            // reference is capped below 8 unless both of its axes are
            // interior.
            let clipped = rx == 0 || ry == 0 || rx == refs_x - 1 || ry == refs_y - 1;
            if clipped {
                assert_eq!(
                    wide.group_weight[idx], narrow.group_weight[idx],
                    "reference ({rx}, {ry}) sees fewer than 8 positions, so both runs must \
                     round it to a group of 4"
                );
            } else if wide.group_weight[idx] != narrow.group_weight[idx] {
                interior_differed += 1;
            }
        }
    }
    let interior = ((refs_x - 2) * (refs_y - 2)) as usize;
    assert!(
        interior_differed * 2 > interior,
        "expected most of the {interior} interior references to reach a group of 8 and so \
         differ from the k_max = 4 run, only {interior_differed} did"
    );
}

/// Writes a flat 8x8 block into `frame` at `(px, py)`.
fn flat_block(frame: &mut [f32], w: u32, px: u32, py: u32, value: f32) {
    for row in 0..PATCH_SIZE {
        for col in 0..PATCH_SIZE {
            frame[((py + row) * w + px + col) as usize] = value;
        }
    }
}

/// A subtracted noise floor is never clamped at zero, so it shifts every
/// candidate equally and changes nothing.
///
/// Four flat blocks sit at four widely separated distances from a flat
/// reference block, all of them far below the floor this run uses. If
/// the kernel clamped, all four would collapse onto the same `0.0`,
/// every insert would be a tie, and the first candidate raster order
/// reached would win instead of the closest one. The blocks are placed
/// so raster order reaches them worst-first, so a clamp would keep the
/// worst and drop the best.
///
/// The whole output is compared rather than a member list, because a
/// changed member set moves both the filtered pixels and the group
/// weight.
#[test]
fn a_noise_floor_shifts_every_distance_equally() {
    let (w, h) = (64u32, 64u32);
    let (rx, ry) = (40u32, 40u32);
    let ref_value = 0.7f32;

    // The background sits far from every planted block, so any 8x8
    // window carrying even one background pixel scores hundreds and
    // cannot compete for a slot.
    let mut frame = vec![0.05f32; (w * h) as usize];
    flat_block(&mut frame, w, rx, ry, ref_value);
    // Raster order reaches these worst-first, which is the order a
    // clamped distance would keep them in.
    flat_block(&mut frame, w, rx - 8, ry - 16, ref_value + 0.15);
    flat_block(&mut frame, w, rx + 8, ry - 16, ref_value + 0.01);
    flat_block(&mut frame, w, rx - 8, ry + 16, ref_value + 0.02);
    flat_block(&mut frame, w, rx + 8, ry + 16, ref_value + 0.03);

    let mut without = Setup::spatial_only(frame.clone(), w, h);
    without.spatial_radius = 16;
    without.k_max = 4;
    let mut with = Setup::spatial_only(frame, w, h);
    with.spatial_radius = 16;
    with.k_max = 4;
    // Every planted distance is under 4.4, so this floor drives all four
    // of them negative.
    with.noise_floor = 10.0;

    let without = run_fused(&without);
    let with = run_fused(&with);

    assert!(
        without.group_weight.iter().any(|&w| w != 0.0),
        "the kernel must actually have written output for this comparison to mean anything"
    );
    assert_eq!(
        without.group_weight, with.group_weight,
        "a noise floor must leave every group weight exactly where it was"
    );
    assert_eq!(
        without.accum, with.accum,
        "a noise floor must leave the accumulator exactly where it was"
    );
    assert_eq!(
        without.wsum, with.wsum,
        "a noise floor must leave the weight sum exactly where it was"
    );
}

/// A group that finds a genuine twin agrees with itself, and a group
/// that does not carries far more detail into the threshold.
///
/// One texture is planted twice over a flat background, at `(4, 4)` and
/// at `(16, 12)`. With `k_max = 2` the group at `(4, 4)` keeps the
/// self-match and exactly one other member, so the twin either is that
/// member or the matcher missed it. When it is, the two members are
/// pixel for pixel identical, the Haar difference across the pair is
/// exactly zero everywhere, and the threshold keeps nothing from that
/// level. When it is not, the second member is flat background against a
/// textured reference, the difference level carries the texture too, and
/// roughly twice as many coefficients survive, halving the weight.
///
/// `lambda_ht` sits at 1.0 so the threshold keeps nearly every
/// coefficient it is offered, which is what makes the retained count
/// track the number of levels carrying content rather than the size of
/// the coefficients in them.
///
/// The control run plants the same texture once, leaving nothing in the
/// window for the group to match.
#[test]
fn a_planted_twin_is_found() {
    let (w, h) = (32u32, 32u32);
    let texture = deterministic_texture(7);

    let mut twinned = vec![0.2f32; (w * h) as usize];
    plant_patch(&mut twinned, w, 4, 4, &texture);
    plant_patch(&mut twinned, w, 16, 12, &texture);

    let mut alone = vec![0.2f32; (w * h) as usize];
    plant_patch(&mut alone, w, 4, 4, &texture);

    let run = |frame: Vec<f32>| {
        let mut s = Setup::spatial_only(frame, w, h);
        s.spatial_radius = 12;
        s.k_max = 2;
        s.lambda_ht = 1.0;
        run_fused(&s)
    };

    let ref_idx = (4 / STEP + (4 / STEP) * refs_along(w)) as usize;
    let with_twin = run(twinned).group_weight[ref_idx];
    let without_twin = run(alone).group_weight[ref_idx];

    assert!(
        with_twin > without_twin * 1.5,
        "expected the group at (4, 4) to keep far more of its variance when its twin at \
         (16, 12) exists, got weight {with_twin} with the twin and {without_twin} without"
    );
}

/// A neighbour whose motion-block confidence sits below `c_min` is
/// skipped outright, so no member ever comes from it and its region of
/// the accumulator ring stays untouched.
///
/// The confidence field is uniform per neighbour here, so the skip is
/// the same decision for every group in the frame. A slot that received
/// even one member would show a non-zero weight sum.
#[test]
fn a_gated_neighbour_receives_no_scatter() {
    let (w, h) = (64u32, 64u32);
    let mut s = three_frame_ring_with_a_planted_match(w, h);
    // Neighbour 0 is ring slot 0 and neighbour 1 is ring slot 2, so this
    // gates the second of the two.
    let blocks = s.conf_stride as usize;
    s.confidence[..blocks].fill(1.0);
    s.confidence[blocks..].fill(0.0);
    s.c_min = 0.5;

    let got = run_fused(&s);

    assert!(
        got.frame_weight_sum(0) > 0,
        "the ungated neighbour's slot received nothing"
    );
    assert!(got.frame_weight_sum(1) > 0, "the centre slot received nothing");
    assert_eq!(
        got.frame_weight_sum(2),
        0,
        "the gated neighbour's slot must receive no scatter at all"
    );
}

/// The variance of the sample pool one plane of reference patches
/// carries.
fn patch_pool_variance(frame: &[f32], w: u32, h: u32) -> f64 {
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
fn output_variance(got: &Aggregated) -> f64 {
    let values: Vec<f64> = (0..got.accum.len()).map(|i| got.pixel(i)).collect();
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64
}

/// A flat field carrying nothing but noise, at the settings a real
/// caller would filter it with.
fn flat_noise_setup(w: u32, h: u32, sigma: f32) -> Setup {
    let mut s = Setup::spatial_only(noisy_field_over(w, h, 0.5, sigma), w, h);
    s.spatial_radius = 9;
    s.sigma = sigma;
    s.lambda_ht = 2.7;
    // The channel-scaled expected SSD between two independent noisy
    // copies of the same flat content, which is what a real match on
    // this frame costs.
    s.noise_floor = 2.0 * 3.0 * sigma * sigma * 64.0;
    s
}

#[test]
fn noise_is_suppressed_on_a_flat_field() {
    let (w, h) = (48u32, 48u32);
    let sigma = 0.04f32;
    let s = flat_noise_setup(w, h, sigma);
    let input_var = patch_pool_variance(&s.ring, w, h);
    let got = run_fused(&s);

    // A run that wrote nothing would read a variance of zero and clear
    // the bound below without filtering anything, so the output has to
    // be shown to carry the field's own brightness first.
    let output_mean: f64 = (0..got.accum.len()).map(|i| got.pixel(i)).sum::<f64>() / got.accum.len() as f64;
    assert!(
        (output_mean - 0.5).abs() < 0.01,
        "expected the filtered field to keep its 0.5 mean, got {output_mean}"
    );

    let output_var = output_variance(&got);

    assert!(
        output_var <= input_var * 0.25,
        "expected filtered variance ({output_var}) to be at most a quarter of the input \
         variance ({input_var})"
    );
}

#[test]
fn group_weight_matches_uniform_theory() {
    let (w, h) = (48u32, 48u32);
    let sigma = 0.04f32;
    let s = flat_noise_setup(w, h, sigma);
    let weights = run_fused(&s).group_weight;

    // With every member's variance equal to `sigma^2`, the ladder is a
    // fixed point, as
    // `transforms::tests::uniform_variance_is_unchanged_by_the_ladder`
    // shows. Every coefficient the threshold could keep therefore also
    // carries variance `sigma^2`, whatever level or spatial position it
    // came from.
    //
    // `group_weight` is then exactly `1 / (sigma^2 * n_ret)`, so this
    // backs out the mean retained count the run produced and checks two
    // things about it.
    //
    // It must include at least the forced group DC. And a hard threshold
    // at 2.7 standard deviations lets only about 0.7% of pure-noise
    // coefficients through by chance, so out of the up to
    // `k_max * PATCH_AREA - 1` coefficients besides the DC that a full
    // 8-member group offers, the mean false-positive count should be
    // small next to that ceiling rather than close to it.
    let sigma2 = sigma * sigma;
    let mean_weight: f64 = weights.iter().map(|&w| w as f64).sum::<f64>() / weights.len() as f64;
    let mean_n_ret = 1.0 / (mean_weight * sigma2 as f64);

    let false_positive_rate = 0.007; // ~P(|Z| >= 2.7) for a standard normal, two-tailed
    let ceiling = (8 * 64 - 1) as f64;
    let expected_n_ret = 1.0 + ceiling * false_positive_rate;

    // A run against the real kernel at this setup measures a mean
    // retained count around 6 (close to `expected_n_ret`, ~4.5, and
    // nowhere near a naive DC-only assumption of 1, which a 20% band
    // around would reject this correct result outright). The lower
    // bound below is what actually distinguishes a working threshold
    // from two ways it could be broken: forced-DC-only (would measure
    // exactly 1) and "threshold does nothing, keeps everything" (would
    // measure close to `ceiling + 1`, an order of magnitude past the
    // upper bound below).
    assert!(
        mean_n_ret > 2.0,
        "expected the mean retained count ({mean_n_ret}) to clearly exceed the forced-DC-\
         only value of 1, proving the threshold is admitting some noise-driven coefficients \
         through by chance, not just forcing the group DC"
    );
    assert!(
        mean_n_ret <= expected_n_ret * 2.0,
        "expected the mean retained count ({mean_n_ret}) to stay within 2x of the false-\
         positive-rate estimate ({expected_n_ret}), well short of the {ceiling} coefficient \
         ceiling"
    );
}

/// `rho = 0` must leave the output bit for bit identical to what it
/// would be with no noise-shaping profile in the computation at all.
///
/// This is checked two ways from the same noisy group, once through the
/// real `dct_noise_profile(0.0)` production path, and once through a
/// profile buffer built entirely by hand, `[1.0; 8]`, which is
/// mathematically the exact identity multiplier and so stands in for "no
/// profile logic at all" without needing a second copy of the kernel to
/// prove it against.
#[test]
fn dct_profile_rho_zero_matches_a_hand_built_all_ones_profile() {
    let (w, h) = (48u32, 48u32);
    let sigma = 0.04f32;

    assert_eq!(
        dct_noise_profile(0.0),
        [1.0f32; 8],
        "dct_noise_profile(0.0) must be exactly [1.0; 8], the property this comparison relies on"
    );

    let produced = flat_noise_setup(w, h, sigma);
    let mut hand_built = flat_noise_setup(w, h, sigma);
    hand_built.profile_override = Some([1.0f32; 8]);

    let produced = run_fused(&produced);
    let hand_built = run_fused(&hand_built);

    assert!(
        produced.accum.iter().any(|&v| v != 0) || produced.group_weight.iter().any(|&w| w != 0.0),
        "the kernel must actually have written output for this comparison to mean anything"
    );
    assert_eq!(
        produced.accum, hand_built.accum,
        "the accumulator at rho=0 must be identical to a hand-built all-ones profile, proving \
         correlation shaping off is exactly a no-op"
    );
    assert_eq!(
        produced.group_weight, hand_built.group_weight,
        "group_weight at rho=0 must be identical to a hand-built all-ones profile"
    );
}

/// Higher `rho` must retain more residual noise on a flat, noise-only
/// field than `rho = 0` does, at the same `lambda_ht`.
///
/// A positive `rho` moves variance out of the high frequencies and into
/// the low ones (`dct_noise_profile`'s own monotonic-decrease property),
/// so a fixed `lambda_ht` reaches a smaller threshold on most non-DC
/// coefficients than the white-noise assumption would, and more of the
/// pure noise sitting in those coefficients survives. This is the
/// documented, deliberate trade the shipped table's caveat describes. On
/// content where the true correlation is lower than the table assumes,
/// shaping under-shrinks rather than over-shrinks, trading a little
/// leftover noise for preserved detail. A flat, noise-only field
/// isolates that trade with nothing else going on.
#[test]
fn higher_rho_retains_more_noise_on_a_flat_field() {
    let (w, h) = (48u32, 48u32);
    let sigma = 0.04f32;

    let white = flat_noise_setup(w, h, sigma);
    let mut shaped = flat_noise_setup(w, h, sigma);
    shaped.rho = 0.86;

    let var_white = output_variance(&run_fused(&white));
    let var_shaped = output_variance(&run_fused(&shaped));

    assert!(
        var_shaped > var_white * 1.05,
        "expected rho=0.86 to leave meaningfully more residual variance than rho=0 at the same \
         lambda_ht, got rho=0 variance={var_white} rho=0.86 variance={var_shaped}"
    );
}

/// A centre-frame member never picks up a mismatch variance.
///
/// Every neighbour here is gated out by `c_min`, so every member of
/// every group comes from the centre frame, and the confidence buffer
/// holds `0.0`, the value that derives the largest mismatch variance
/// there is. Turning `confidence_variance` on must therefore change
/// nothing at all. A kernel that fed a centre-frame member through
/// [`crate::collab::kernels::fused::mismatch_sigma2`] would inflate
/// every threshold in the frame and move every pixel.
#[test]
fn centre_frame_members_ignore_the_confidence_field() {
    let (w, h) = (64u32, 64u32);
    let mut off = three_frame_ring_with_a_planted_match(w, h);
    off.confidence.fill(0.0);
    off.c_min = 0.5;
    off.confidence_variance = false;
    let mut on = three_frame_ring_with_a_planted_match(w, h);
    on.confidence.fill(0.0);
    on.c_min = 0.5;
    on.confidence_variance = true;

    let off = run_fused(&off);
    let on = run_fused(&on);

    assert!(
        off.group_weight.iter().any(|&w| w != 0.0),
        "the kernel must actually have written output for this comparison to mean anything"
    );
    assert_eq!(
        off.frame_weight_sum(0),
        0,
        "both neighbours must be gated for this to test centre-frame members"
    );
    assert_eq!(off.frame_weight_sum(2), 0, "both neighbours must be gated");
    assert_eq!(
        off.group_weight, on.group_weight,
        "the mismatch variance must not reach a centre-frame member"
    );
    assert_eq!(
        off.accum, on.accum,
        "the mismatch variance must not reach a centre-frame member"
    );
}
