use std::time::Instant;

const WIDTH: usize = 1920;
const HEIGHT: usize = 1080;
/// 4:2:0 sample count for one frame.
const SAMPLES: usize = WIDTH * HEIGHT + 2 * ((WIDTH / 2) * (HEIGHT / 2));

const WARMUP: usize = 5;
const ITERS: usize = 200;

#[derive(clap::Parser, Debug)]
#[command(about = "Sample <-> f32 conversion benchmark", long_about = None)]
struct Cli {
    /// Swallowed, since cargo passes this when invoking the bench binary.
    #[arg(long, hide = true)]
    bench: bool,
}

fn time(label: &str, mut work: impl FnMut() -> usize) {
    for _ in 0..WARMUP {
        std::hint::black_box(work());
    }

    let start = Instant::now();
    for _ in 0..ITERS {
        std::hint::black_box(work());
    }
    let per_ms = start.elapsed().as_secs_f64() / ITERS as f64 * 1000.0;

    println!("{label:<44} {per_ms:>8.3} ms/frame");
}

fn main() {
    let _cli: Cli = clap::Parser::parse();

    let normalized: Vec<f32> = (0..SAMPLES).map(|i| (i % 1024) as f32 / 1023.0).collect();

    // Flat luma plane, the simplest read path.
    let wire8: Vec<u8> = (0..SAMPLES).map(|i| (i % 256) as u8).collect();
    let wire10: Vec<u8> = (0..SAMPLES)
        .flat_map(|i| ((i % 1024) as u16).to_le_bytes())
        .collect();

    // Equal-length YUV444 planes for the interleaving path.
    let yuv_pixels = SAMPLES / 3;
    let plane8: Vec<u8> = (0..yuv_pixels).map(|i| (i % 256) as u8).collect();

    println!("{SAMPLES} samples/frame (1080p 4:2:0), {ITERS} iters");

    println!("-- output --");
    time("f32 -> 8-bit plane", || {
        quantise_plane_narrow(&normalized, 255.0).len()
    });
    time("f32 -> 10-bit plane", || {
        quantise_plane_wide(&normalized, 1023.0).len()
    });

    println!("-- input --");
    time("8-bit plane -> f32", || read_plane_narrow(&wire8, 255.0).len());
    time("10-bit plane -> f32", || read_plane_wide(&wire10, 1023.0).len());

    // The fused YUV444 path reads three planes per pixel by index, and the bounds on the second and
    // third planes cannot be proven. These two rows show whether that stops the loop vectorising.
    println!("-- interleave (fused YUV444) --");
    time("8-bit YUV planes -> interleaved f32", || {
        interleave_yuv_narrow(&plane8, &plane8, &plane8, 255.0).len()
    });
    time("8-bit YUV planes -> interleaved f32 (sliced)", || {
        interleave_yuv_narrow_sliced(&plane8, &plane8, &plane8, 255.0).len()
    });
}

/// Mirrors `plane_to_f32`'s narrow arm.
fn read_plane_narrow(plane: &[u8], max: f32) -> Vec<f32> {
    let normalized: Vec<f32> = (0..plane.len()).map(|i| plane[i] as f32 / max).collect();
    std::hint::black_box(&normalized);
    normalized
}

/// Mirrors `plane_to_f32`'s wide arm.
fn read_plane_wide(plane: &[u8], max: f32) -> Vec<f32> {
    let samples = plane.len() / 2;
    let normalized: Vec<f32> = (0..samples)
        .map(|i| u16::from_le_bytes([plane[2 * i], plane[2 * i + 1]]) as f32 / max)
        .collect();
    std::hint::black_box(&normalized);
    normalized
}

/// Mirrors `interleave_yuv_to_f32`'s narrow arm.
fn interleave_yuv_narrow(y_plane: &[u8], u_plane: &[u8], v_plane: &[u8], max: f32) -> Vec<f32> {
    let pixels = y_plane.len();
    let mut interleaved = Vec::with_capacity(pixels * 3);

    for i in 0..pixels {
        interleaved.push(y_plane[i] as f32 / max);
        interleaved.push(u_plane[i] as f32 / max);
        interleaved.push(v_plane[i] as f32 / max);
    }

    std::hint::black_box(&interleaved);
    interleaved
}

/// The same loop with all three planes pre-sliced to `pixels`.
///
/// Slicing lets the compiler drop the per-pixel bounds checks on the U and V planes.
fn interleave_yuv_narrow_sliced(y_plane: &[u8], u_plane: &[u8], v_plane: &[u8], max: f32) -> Vec<f32> {
    let pixels = y_plane.len();
    let (y_plane, u_plane, v_plane) = (&y_plane[..pixels], &u_plane[..pixels], &v_plane[..pixels]);
    let mut interleaved = Vec::with_capacity(pixels * 3);

    for i in 0..pixels {
        interleaved.push(y_plane[i] as f32 / max);
        interleaved.push(u_plane[i] as f32 / max);
        interleaved.push(v_plane[i] as f32 / max);
    }

    std::hint::black_box(&interleaved);
    interleaved
}

/// Mirrors `f32_to_plane`'s narrow arm.
///
/// The indexed write matches `Narrow::write` rather than a `zip`, because a bounds-check-free
/// iterator idiom would measure a loop the binary never runs.
fn quantise_plane_narrow(plane: &[f32], max: f32) -> Vec<u8> {
    let mut quantised = vec![0u8; plane.len()];
    for (i, &value) in plane.iter().enumerate() {
        quantised[i] = quantise(value, max) as u8;
    }

    std::hint::black_box(&quantised);
    quantised
}

/// Mirrors `f32_to_plane`'s wide arm, indexed to match `Wide::write`.
fn quantise_plane_wide(plane: &[f32], max: f32) -> Vec<u8> {
    let mut quantised = vec![0u8; plane.len() * 2];
    for (i, &value) in plane.iter().enumerate() {
        let sample_bytes = quantise(value, max).to_le_bytes();
        quantised[2 * i..2 * i + 2].copy_from_slice(&sample_bytes);
    }

    std::hint::black_box(&quantised);
    quantised
}

#[inline(always)]
fn quantise(value: f32, max: f32) -> u16 {
    (value.clamp(0.0, 1.0) * max + 0.5) as u16
}
