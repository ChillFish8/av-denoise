use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::engine::{
    ChromaLayout,
    ChromaSiting,
    DevicePlane,
    Engine,
    EngineSpec,
    Geometry,
    PassSpec,
    SampleFormat,
};
use crate::motion::MotionCompensationMode;
use crate::motion::global::{MODEL_WORDS, ModelRing, PairModelProducer};
use crate::nl4d::Nl4dOptions;
use crate::nlmeans::{DenoisingMode, NlmeansAlgorithm, NlmeansOptions};
use crate::tune::EngineKernels;
use crate::tune::gpu::read_f32;
use crate::tune::tests::fixtures::{hashed_unit, noisy_field_over, upload};
use crate::tune::tests::runtime::test_client;

const WIDTH: u32 = 960;
const HEIGHT: u32 = 540;
const RING_DEPTH: u32 = 4;
const PAN_TOLERANCE: f64 = 0.1;
/// Block vectors are whole pixels, so a fractional pan fits to the nearest pixel.
const SUBPIXEL_TOLERANCE: f64 = 0.5;
const ROLL_TOLERANCE: f64 = 0.08;

/// A camera model in luma pixels, `x' = a*x - b*y + tx` and `y' = b*x + a*y + ty`.
#[derive(Debug, Clone, Copy)]
struct Similarity {
    a: f64,
    b: f64,
    tx: f64,
    ty: f64,
}

impl Similarity {
    fn identity() -> Self {
        Self::shift(0.0, 0.0)
    }

    fn shift(shift_x: f64, shift_y: f64) -> Self {
        Self {
            a: 1.0,
            b: 0.0,
            tx: shift_x,
            ty: shift_y,
        }
    }

    /// A roll about the frame centre.
    fn roll(degrees: f64) -> Self {
        let radians = degrees.to_radians();
        let a = radians.cos();
        let b = radians.sin();
        let centre_x = WIDTH as f64 / 2.0;
        let centre_y = HEIGHT as f64 / 2.0;
        let tx = centre_x - (a * centre_x - b * centre_y);
        let ty = centre_y - (b * centre_x + a * centre_y);

        Self { a, b, tx, ty }
    }

    fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        let mapped_x = self.a * x - self.b * y + self.tx;
        let mapped_y = self.b * x + self.a * y + self.ty;

        (mapped_x, mapped_y)
    }

    fn inverse(&self) -> Self {
        let norm = self.a * self.a + self.b * self.b;
        let a = self.a / norm;
        let b = -self.b / norm;
        let tx = -(a * self.tx - b * self.ty);
        let ty = -(b * self.tx + a * self.ty);

        Self { a, b, tx, ty }
    }
}

/// The five words of one ring slot.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Fitted {
    a: f32,
    b: f32,
    tx: f32,
    ty: f32,
    valid: f32,
}

impl Fitted {
    fn as_similarity(&self) -> Similarity {
        Similarity {
            a: self.a as f64,
            b: self.b as f64,
            tx: self.tx as f64,
            ty: self.ty as f64,
        }
    }

    /// Checks the model is accepted and moves the frame corners and centre within `tolerance`
    /// pixels of `expected`.
    fn assert_matches(&self, expected: Similarity, tolerance: f64, label: &str) {
        assert_eq!(
            self.valid, 1.0,
            "{label}: the fit should be accepted, got {self:?}"
        );

        let fitted = self.as_similarity();
        let width = WIDTH as f64;
        let height = HEIGHT as f64;
        let points = [
            (0.0, 0.0),
            (width, 0.0),
            (0.0, height),
            (width, height),
            (width / 2.0, height / 2.0),
        ];
        let worst = points
            .into_iter()
            .map(|(x, y)| {
                let (fitted_x, fitted_y) = fitted.apply(x, y);
                let (expected_x, expected_y) = expected.apply(x, y);
                (fitted_x - expected_x).hypot(fitted_y - expected_y)
            })
            .fold(0.0, f64::max);
        assert!(
            worst < tolerance,
            "{label}: a point lands {worst} px off, {self:?} vs {expected:?}"
        );
    }

    fn assert_identity(&self, label: &str) {
        let expected = Fitted {
            a: 1.0,
            b: 0.0,
            tx: 0.0,
            ty: 0.0,
            valid: 0.0,
        };
        assert_eq!(*self, expected, "{label}");
    }
}

fn read_slot(client: &Client, ring: &ModelRing, frame: u32) -> Fitted {
    let words = read_f32(client, ring.handle(), ring.words()).unwrap();
    let slot = ring.slot(frame);
    let base = (slot * MODEL_WORDS) as usize;

    Fitted {
        a: words[base],
        b: words[base + 1],
        tx: words[base + 2],
        ty: words[base + 3],
        valid: words[base + 4],
    }
}

fn default_mode() -> MotionCompensationMode {
    MotionCompensationMode::mvtools_default()
}

/// How far an unseeded search reaches, the coarse reach plus the fine radius.
fn search_reach(mode: MotionCompensationMode) -> u32 {
    let MotionCompensationMode::Mvtools {
        search_radius,
        pyramid_levels,
        ..
    } = mode
    else {
        panic!("the mode searches");
    };
    let coarse_scale = 1u32 << (pyramid_levels - 1);
    let coarse_reach = search_radius * coarse_scale;

    coarse_reach + search_radius
}

/// Lattice spacings of the [texture] octaves, in pixels.
const TEXTURE_CELLS: [f64; 2] = [4.0, 11.0];

/// A hashed value between 0.2 and 0.8 at lattice point (`cell_x`, `cell_y`).
fn lattice_value(cell_x: i64, cell_y: i64, seed: u32) -> f64 {
    let column = (cell_x as u32).wrapping_mul(0x8DA6_B343);
    let row = (cell_y as u32).wrapping_mul(0xD816_3841);
    let index = column ^ row;
    let value = hashed_unit(index, seed);

    value as f64
}

fn smoothstep(t: f64) -> f64 {
    t * t * (3.0 - 2.0 * t)
}

/// Value noise at spacing `cell`, smoothly interpolated between hashed lattice points.
fn value_noise(x: f64, y: f64, cell: f64, seed: u32) -> f64 {
    let scaled_x = x / cell;
    let scaled_y = y / cell;
    let cell_x = scaled_x.floor();
    let cell_y = scaled_y.floor();
    let blend_x = smoothstep(scaled_x - cell_x);
    let blend_y = smoothstep(scaled_y - cell_y);
    let left = cell_x as i64;
    let top = cell_y as i64;

    let top_left = lattice_value(left, top, seed);
    let top_right = lattice_value(left + 1, top, seed);
    let bottom_left = lattice_value(left, top + 1, seed);
    let bottom_right = lattice_value(left + 1, top + 1, seed);
    let upper = top_left + (top_right - top_left) * blend_x;
    let lower = bottom_left + (bottom_right - bottom_left) * blend_x;

    upper + (lower - upper) * blend_y
}

/// A continuous texture with no repeat, so every block has one clear match.
fn texture(x: f64, y: f64) -> f32 {
    texture_layer(x, y, 1)
}

/// The [texture] of layer `seed`, so layers never match each other.
fn texture_layer(x: f64, y: f64, seed: u32) -> f32 {
    let fine = value_noise(x, y, TEXTURE_CELLS[0], seed);
    let coarse = value_noise(x, y, TEXTURE_CELLS[1], seed + 1);
    let value = 0.6 * fine + 0.4 * coarse;

    value as f32
}

/// A `width` by `height` frame of [texture], each pixel sampled where `camera` came from.
fn render(camera: Similarity, width: u32, height: u32) -> Vec<f32> {
    let backward = camera.inverse();
    let mut frame = Vec::with_capacity((width * height) as usize);
    for y in 0..height {
        for x in 0..width {
            let (source_x, source_y) = backward.apply(x as f64, y as f64);
            frame.push(texture(source_x, source_y));
        }
    }

    frame
}

fn producer(client: &Client, mode: MotionCompensationMode) -> PairModelProducer {
    let built = PairModelProducer::new(client, WIDTH, HEIGHT, &mode, SampleFormat::F32, RING_DEPTH);

    built.expect("a searching mode builds a producer")
}

fn push_frame(client: &Client, producer: &mut PairModelProducer, camera: Similarity, frame: u32) {
    let samples = render(camera, WIDTH, HEIGHT);
    let handle = upload(client, &samples);
    let plane = DevicePlane::new(&handle, WIDTH, HEIGHT);

    producer.push(&plane, frame).unwrap();
}

/// Pushes frames `first..first + count` of a constant `velocity` pan and checks each pair's model.
fn assert_pan_recovered(
    client: &Client,
    producer: &mut PairModelProducer,
    first: u32,
    count: u32,
    velocity: f64,
    tolerance: f64,
) {
    let expected = Similarity::shift(velocity, 0.0);
    let start = Similarity::identity();
    push_frame(client, producer, start, first);

    for step in 1..count {
        let frame = first + step;
        let offset = velocity * step as f64;
        let camera = Similarity::shift(offset, 0.0);
        push_frame(client, producer, camera, frame);

        let fitted = read_slot(client, producer.ring(), frame);
        let label = format!("frame {frame} at {velocity} px");
        fitted.assert_matches(expected, tolerance, &label);
    }
}

#[test]
fn a_synthetic_pan_is_recovered_frame_by_frame() {
    let client = test_client();
    let mode = default_mode();
    let reach = search_reach(mode);
    assert!(30 > reach, "test premise: 30 px lies past the {reach} px reach");

    let mut producer = producer(&client, mode);
    assert_pan_recovered(&client, &mut producer, 0, 8, 30.0, PAN_TOLERANCE);
}

#[test]
fn a_46_px_pan_is_recovered_from_frame_0() {
    let client = test_client();
    let mut producer = producer(&client, default_mode());

    assert_pan_recovered(&client, &mut producer, 0, 6, 46.0, PAN_TOLERANCE);
}

/// How far the camera of frame `frame` has panned, speeding up by 10 px a frame to 70 px a frame.
///
/// Only a search seeded by the pairs before it could follow the top speed, which lies past the
/// unseeded reach.
fn ramp_offset(frame: u32) -> f64 {
    (1..=frame).map(|step| f64::from((10 * step).min(70))).sum()
}

/// Pushes frame `frame` of a grainy ramping pan, numbered `run_frame` in its run.
fn push_ramp_frame(client: &Client, producer: &mut PairModelProducer, frame: u32, run_frame: u32) {
    let offset = ramp_offset(frame);
    let camera = Similarity::shift(offset, 0.0);
    let clean = render(camera, WIDTH, HEIGHT);
    let samples = noisy_field_over(&clean, WIDTH, HEIGHT, 0.04, frame);
    let handle = upload(client, &samples);
    let plane = DevicePlane::new(&handle, WIDTH, HEIGHT);

    producer.push(&plane, run_frame).unwrap();
}

#[test]
fn a_pair_fits_the_same_model_in_any_run() {
    let client = test_client();
    let restart = 8u32;
    let last = 11u32;

    let mut streamed = producer(&client, default_mode());

    for frame in 0..=last {
        push_ramp_frame(&client, &mut streamed, frame, frame);
    }

    let mut restarted = producer(&client, default_mode());

    for frame in restart..=last {
        push_ramp_frame(&client, &mut restarted, frame, frame - restart);
    }

    for frame in restart + 1..=last {
        let streamed_model = read_slot(&client, streamed.ring(), frame);
        let restarted_model = read_slot(&client, restarted.ring(), frame - restart);
        assert_eq!(
            streamed_model, restarted_model,
            "frame {frame}: a pair's model must not depend on the pairs before it"
        );
    }
}

#[test]
fn a_fractional_fast_pan_is_recovered_to_the_pixel() {
    let client = test_client();
    let mut producer = producer(&client, default_mode());

    assert_pan_recovered(&client, &mut producer, 0, 6, 34.7, SUBPIXEL_TOLERANCE);
}

#[test]
fn a_fast_pan_after_a_reset_is_recovered() {
    let client = test_client();
    let mut producer = producer(&client, default_mode());
    let still = Similarity::identity();
    for frame in 0..4u32 {
        push_frame(&client, &mut producer, still, frame);
    }

    let settled = read_slot(&client, producer.ring(), 3);
    assert_eq!(settled.valid, 0.0, "a still shot fits no motion, got {settled:?}");

    producer.reset();
    assert_pan_recovered(&client, &mut producer, 0, 6, 30.0, PAN_TOLERANCE);
}

#[test]
fn a_cut_from_a_still_shot_into_a_fast_pan_is_recovered() {
    let client = test_client();
    let mut producer = producer(&client, default_mode());
    let still = Similarity::identity();
    for frame in 0..3u32 {
        push_frame(&client, &mut producer, still, frame);
    }

    assert_pan_recovered(&client, &mut producer, 3, 5, 30.0, PAN_TOLERANCE);
}

#[test]
fn a_synthetic_roll_is_recovered_with_one_refit() {
    let client = test_client();
    let mut producer = producer(&client, default_mode());
    let step = Similarity::roll(3.0);

    for frame in 0..4u32 {
        let degrees = 3.0 * frame as f64;
        let camera = Similarity::roll(degrees);
        push_frame(&client, &mut producer, camera, frame);
        if frame == 0 {
            continue;
        }

        let fitted = read_slot(&client, producer.ring(), frame);
        let label = format!("frame {frame}");
        fitted.assert_matches(step, ROLL_TOLERANCE, &label);
    }
}

#[test]
fn the_first_frame_and_after_reset_are_identity() {
    let client = test_client();
    let mut producer = producer(&client, default_mode());
    let step = Similarity::shift(4.0, 0.0);
    let camera_at = |frame: u32| Similarity::shift(4.0 * frame as f64, 0.0);

    push_frame(&client, &mut producer, camera_at(0), 0);
    let first = read_slot(&client, producer.ring(), 0);
    first.assert_identity("frame 0");

    for frame in 1..=RING_DEPTH {
        push_frame(&client, &mut producer, camera_at(frame), frame);
    }

    let wrapped = read_slot(&client, producer.ring(), RING_DEPTH);
    wrapped.assert_matches(step, PAN_TOLERANCE, "the frame reusing slot 0");

    producer.reset();
    let restart = RING_DEPTH + 1;
    push_frame(&client, &mut producer, camera_at(restart), restart);
    let after_reset = read_slot(&client, producer.ring(), restart);
    after_reset.assert_identity("the first frame after reset");

    let resumed = restart + 1;
    push_frame(&client, &mut producer, camera_at(resumed), resumed);
    let fitted = read_slot(&client, producer.ring(), resumed);
    fitted.assert_matches(step, PAN_TOLERANCE, "the frame after the restart");
}

/// The grain every scene frame carries.
const SCENE_GRAIN: f32 = 0.01;

/// The seed of the textured layer that moves on its own in a scene.
const LAYER_SEED: u32 = 5;

/// Pushes frame `frame` of a scene, each pixel given by `sample(frame, x, y)` plus `grain`.
fn push_scene(
    client: &Client,
    producer: &mut PairModelProducer,
    frame: u32,
    grain: f32,
    sample: &impl Fn(u32, f64, f64) -> f32,
) {
    let mut clean = Vec::with_capacity((WIDTH * HEIGHT) as usize);
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let value = sample(frame, x as f64, y as f64);
            clean.push(value);
        }
    }

    let samples = noisy_field_over(&clean, WIDTH, HEIGHT, grain, frame);
    let handle = upload(client, &samples);
    let plane = DevicePlane::new(&handle, WIDTH, HEIGHT);

    producer.push(&plane, frame).unwrap();
}

/// Pushes four frames of a scene and checks every pair fits no camera motion.
fn assert_still_camera(client: &Client, sample: &impl Fn(u32, f64, f64) -> f32, label: &str) {
    let mut producer = producer(client, default_mode());

    for frame in 0..4u32 {
        push_scene(client, &mut producer, frame, SCENE_GRAIN, sample);
        if frame == 0 {
            continue;
        }

        let fitted = read_slot(client, producer.ring(), frame);
        let pair_label = format!("{label}, frame {frame}");
        fitted.assert_identity(&pair_label);
    }
}

/// Pushes four frames of a scene with `grain` and checks every pair fits `expected`.
fn assert_scene_pan(
    client: &Client,
    grain: f32,
    sample: &impl Fn(u32, f64, f64) -> f32,
    expected: Similarity,
    label: &str,
) {
    let mut producer = producer(client, default_mode());

    for frame in 0..4u32 {
        push_scene(client, &mut producer, frame, grain, sample);
        if frame == 0 {
            continue;
        }

        let fitted = read_slot(client, producer.ring(), frame);
        let pair_label = format!("{label}, frame {frame}");
        fitted.assert_matches(expected, SUBPIXEL_TOLERANCE, &pair_label);
    }
}

/// Whether `value` lies between `low` and `high` shares of `extent`.
fn within_share(value: f64, extent: u32, low: f64, high: f64) -> bool {
    let start = low * extent as f64;
    let end = high * extent as f64;

    value >= start && value < end
}

#[test]
fn a_moving_subject_on_a_still_camera_fits_no_motion() {
    let client = test_client();
    let scene = |frame: u32, x: f64, y: f64| {
        let subject_x = x - 12.0 * frame as f64;
        let inside_x = within_share(subject_x, WIDTH, 0.2, 0.83);
        let inside_y = within_share(y, HEIGHT, 0.2, 0.83);
        if inside_x && inside_y {
            texture_layer(subject_x, y, LAYER_SEED)
        } else {
            texture(x, y)
        }
    };

    assert_still_camera(&client, &scene, "a subject over 40% of the frame");
}

#[test]
fn a_layer_crossing_mid_frame_on_a_still_camera_fits_no_motion() {
    let client = test_client();
    let scene = |frame: u32, x: f64, y: f64| {
        let layer_x = x - 20.0 * frame as f64;
        if within_share(y, HEIGHT, 0.32, 0.68) {
            texture_layer(layer_x, y, LAYER_SEED)
        } else {
            texture(x, y)
        }
    };

    assert_still_camera(&client, &scene, "a full-width layer");
}

#[test]
fn a_strip_moving_down_mid_frame_on_a_still_camera_fits_no_motion() {
    let client = test_client();
    let scene = |frame: u32, x: f64, y: f64| {
        let strip_y = y - 16.0 * frame as f64;
        if within_share(x, WIDTH, 0.32, 0.68) {
            texture_layer(x, strip_y, LAYER_SEED)
        } else {
            texture(x, y)
        }
    };

    assert_still_camera(&client, &scene, "a full-height strip");
}

#[test]
fn still_low_detail_grey_bands_around_a_moving_layer_fit_no_motion() {
    let client = test_client();
    let scene = |frame: u32, x: f64, y: f64| {
        let layer_x = x - 20.0 * frame as f64;
        if within_share(y, HEIGHT, 0.32, 0.68) {
            texture_layer(layer_x, y, LAYER_SEED)
        } else {
            let detail = texture_layer(x, y, 7) - 0.5;
            0.5 + 0.2 * detail
        }
    };

    assert_still_camera(&client, &scene, "grey bands");
}

#[test]
fn a_letterboxed_pan_with_grainy_black_bars_is_recovered() {
    let client = test_client();
    let bar_level = 18.0 / 255.0;
    let scene = |frame: u32, x: f64, y: f64| {
        let picture_x = x - 30.0 * frame as f64;
        if within_share(y, HEIGHT, 0.12, 0.88) {
            texture(picture_x, y)
        } else {
            bar_level
        }
    };
    let expected = Similarity::shift(30.0, 0.0);

    assert_scene_pan(&client, SCENE_GRAIN, &scene, expected, "a letterboxed pan");
}

/// How much a soft lamp glow centred on the frame lights each pixel on odd frames, up to 0.4.
fn lamp_glow(frame: u32, x: f64, y: f64) -> f32 {
    if frame.is_multiple_of(2) {
        return 0.0;
    }

    let offset_x = x - WIDTH as f64 / 2.0;
    let offset_y = y - HEIGHT as f64 / 2.0;
    let distance_sq = offset_x * offset_x + offset_y * offset_y;
    let falloff = (-distance_sq / (2.0 * 250.0 * 250.0)).exp();

    (0.4 * falloff) as f32
}

#[test]
fn a_pan_under_a_flickering_lamp_is_recovered() {
    let client = test_client();
    let scene = |frame: u32, x: f64, y: f64| {
        let picture_x = x - 30.0 * frame as f64;
        let glow = lamp_glow(frame, x, y);
        let lit = texture(picture_x, y) * (1.0 + glow);
        lit + 0.5 * glow
    };
    let expected = Similarity::shift(30.0, 0.0);

    assert_scene_pan(&client, SCENE_GRAIN, &scene, expected, "a flickering pan");
}

/// The wall has no grain, so its flat blocks match every offset equally and settle on zero.
#[test]
fn a_fast_pan_over_a_flat_wall_with_its_own_foreground_is_recovered() {
    let client = test_client();
    let scene = |frame: u32, x: f64, y: f64| {
        let wall_x = x - 35.0 * frame as f64;
        let foreground_x = x + 10.0 * frame as f64;
        let on_strip = wall_x.rem_euclid(160.0) < 24.0;
        if within_share(y, HEIGHT, 0.6, 0.65) {
            texture_layer(foreground_x, y, LAYER_SEED)
        } else if on_strip {
            texture(wall_x, y)
        } else {
            0.4
        }
    };
    let expected = Similarity::shift(35.0, 0.0);

    assert_scene_pan(&client, 0.0, &scene, expected, "a flat wall pan");
}

fn temporal_nlmeans(motion_compensation: MotionCompensationMode, radius: u32) -> NlmeansAlgorithm {
    let mode = if radius == 0 {
        DenoisingMode::Spacial
    } else {
        DenoisingMode::Temporal { radius }
    };
    let options = NlmeansOptions {
        motion_compensation,
        mode,
        ..NlmeansOptions::default()
    };

    NlmeansAlgorithm::Fast(options)
}

fn f32_geometry(width: u32, height: u32) -> Geometry {
    Geometry {
        width,
        height,
        chroma: ChromaLayout::Yuv420,
        input: SampleFormat::F32,
        output: SampleFormat::F32,
        chroma_siting: ChromaSiting::Unknown,
    }
}

fn default_kernels(_pass: &PassSpec) -> EngineKernels {
    EngineKernels::default()
}

fn models_of(client: &Client, spec: &EngineSpec) -> (Box<dyn Engine>, Option<ModelRing>) {
    let built = spec.build_with_models(client, default_kernels);

    built.expect("the spec builds")
}

/// Every plane of one 4:2:0 frame, luma rendered under `camera` and flat chroma.
fn frame_planes(client: &Client, camera: Similarity) -> Vec<Handle> {
    let luma = render(camera, WIDTH, HEIGHT);
    let chroma_pixels = (WIDTH / 2 * HEIGHT / 2) as usize;
    let chroma = vec![0.5f32; chroma_pixels];
    let luma_handle = upload(client, &luma);
    let u_handle = upload(client, &chroma);
    let v_handle = upload(client, &chroma);

    vec![luma_handle, u_handle, v_handle]
}

fn as_planes(handles: &[Handle]) -> Vec<DevicePlane<'_>> {
    let dims = [(WIDTH, HEIGHT), (WIDTH / 2, HEIGHT / 2), (WIDTH / 2, HEIGHT / 2)];

    handles
        .iter()
        .zip(dims)
        .map(|(handle, (width, height))| DevicePlane::new(handle, width, height))
        .collect()
}

/// Pushes a 4 px pan through `engine`, priming with `context` frames, and emits what it reports.
fn drive_pan(client: &Client, engine: &mut Box<dyn Engine>, context: u32, frames: u32) {
    let geometry = f32_geometry(WIDTH, HEIGHT);
    let output = geometry.empty_planes(client, SampleFormat::F32);
    let output_planes = as_planes(&output);

    for frame in 0..frames {
        let camera = Similarity::shift(4.0 * frame as f64, 0.0);
        let handles = frame_planes(client, camera);
        let planes = as_planes(&handles);
        if frame < context {
            engine.push_context(&planes).unwrap();
            continue;
        }

        let ready = engine.push(&planes).unwrap();
        for _ in 0..ready {
            engine.emit_into(&output_planes).unwrap();
        }
    }
}

#[test]
fn a_chroma_only_engine_still_fits_on_luma() {
    let client = test_client();
    let algorithm = temporal_nlmeans(default_mode(), 1);
    let spec = EngineSpec::Nlmeans {
        luma: None,
        chroma: Some(algorithm),
        geometry: f32_geometry(WIDTH, HEIGHT),
    };
    let (mut engine, ring) = models_of(&client, &spec);
    let ring = ring.expect("a chroma pass with motion compensation has a producer");

    drive_pan(&client, &mut engine, 0, 3);

    let step = Similarity::shift(4.0, 0.0);
    for frame in 1..3u32 {
        let fitted = read_slot(&client, &ring, frame);
        let label = format!("frame {frame}");
        fitted.assert_matches(step, PAN_TOLERANCE, &label);
    }
}

#[test]
fn context_pushes_feed_the_producer() {
    let client = test_client();
    let algorithm = temporal_nlmeans(default_mode(), 1);
    let spec = EngineSpec::Nlmeans {
        luma: Some(algorithm),
        chroma: None,
        geometry: f32_geometry(WIDTH, HEIGHT),
    };
    let (mut engine, ring) = models_of(&client, &spec);
    let ring = ring.expect("a luma pass with motion compensation has a producer");

    drive_pan(&client, &mut engine, 2, 3);

    let first = read_slot(&client, &ring, 0);
    first.assert_identity("the first context frame");

    let step = Similarity::shift(4.0, 0.0);
    for frame in 1..3u32 {
        let fitted = read_slot(&client, &ring, frame);
        let label = format!("frame {frame}");
        fitted.assert_matches(step, PAN_TOLERANCE, &label);
    }
}

#[test]
fn the_producer_is_absent_when_no_pass_has_motion() {
    let client = test_client();
    let geometry = f32_geometry(64, 48);
    let searching = default_mode();
    let still = MotionCompensationMode::None;
    let cases = [
        (temporal_nlmeans(still, 1), false),
        (temporal_nlmeans(searching, 0), false),
        (temporal_nlmeans(searching, 1), true),
    ];

    for (algorithm, expected) in cases {
        let spec = EngineSpec::Nlmeans {
            luma: Some(algorithm),
            chroma: Some(algorithm),
            geometry,
        };
        let (_engine, ring) = models_of(&client, &spec);
        assert_eq!(ring.is_some(), expected, "{algorithm:?}");
    }

    let nl4d_options = Nl4dOptions {
        temporal_radius: 1,
        ..Nl4dOptions::default()
    };
    let nl4d = EngineSpec::Nl4d {
        luma: None,
        chroma: Some(nl4d_options),
        geometry,
    };
    let (_engine, ring) = models_of(&client, &nl4d);
    assert!(ring.is_some(), "nl4d always tracks motion");
}

#[test]
fn a_mode_without_a_search_builds_no_producer() {
    let client = test_client();
    let mode = MotionCompensationMode::None;
    let built = PairModelProducer::new(&client, WIDTH, HEIGHT, &mode, SampleFormat::F32, RING_DEPTH);

    assert!(built.is_none());
}
