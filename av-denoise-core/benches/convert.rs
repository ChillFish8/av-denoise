use std::time::Instant;

const W: usize = 1920;
const H: usize = 1080;
/// 4:2:0 sample count for one frame.
const SAMPLES: usize = W * H + 2 * ((W / 2) * (H / 2));

const WARMUP: usize = 5;
const ITERS: usize = 200;

#[derive(clap::Parser, Debug)]
#[command(about = "Sample <-> f32 conversion benchmark", long_about = None)]
struct Cli {
    /// Swallowed: cargo passes this when invoking the bench binary.
    #[arg(long, hide = true)]
    bench: bool,
}

fn time(label: &str, mut f: impl FnMut() -> usize) {
    for _ in 0..WARMUP {
        std::hint::black_box(f());
    }

    let t = Instant::now();
    for _ in 0..ITERS {
        std::hint::black_box(f());
    }
    let per_ms = t.elapsed().as_secs_f64() / ITERS as f64 * 1000.0;

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

    // The fused YUV444 path reads three planes per pixel by index. The
    // reviewer of the converter task flagged that indexed reads may not
    // vectorise as well as the zip form it replaced, because the bounds
    // on the second and third planes cannot be proven. These two rows
    // are what decides whether that concern is real.
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
    let out: Vec<f32> = (0..plane.len()).map(|i| plane[i] as f32 / max).collect();
    std::hint::black_box(&out);
    out
}

/// Mirrors `plane_to_f32`'s wide arm.
fn read_plane_wide(plane: &[u8], max: f32) -> Vec<f32> {
    let samples = plane.len() / 2;
    let out: Vec<f32> = (0..samples)
        .map(|i| u16::from_le_bytes([plane[2 * i], plane[2 * i + 1]]) as f32 / max)
        .collect();
    std::hint::black_box(&out);
    out
}

/// Mirrors `interleave_yuv_to_f32`'s narrow arm exactly as shipped.
fn interleave_yuv_narrow(y: &[u8], u: &[u8], v: &[u8], max: f32) -> Vec<f32> {
    let pixels = y.len();
    let mut out = Vec::with_capacity(pixels * 3);

    for i in 0..pixels {
        out.push(y[i] as f32 / max);
        out.push(u[i] as f32 / max);
        out.push(v[i] as f32 / max);
    }

    std::hint::black_box(&out);
    out
}

/// The same loop with all three planes pre-sliced to `pixels`, which lets
/// the compiler drop the per-pixel bounds checks on `u` and `v`.
fn interleave_yuv_narrow_sliced(y: &[u8], u: &[u8], v: &[u8], max: f32) -> Vec<f32> {
    let pixels = y.len();
    let (y, u, v) = (&y[..pixels], &u[..pixels], &v[..pixels]);
    let mut out = Vec::with_capacity(pixels * 3);

    for i in 0..pixels {
        out.push(y[i] as f32 / max);
        out.push(u[i] as f32 / max);
        out.push(v[i] as f32 / max);
    }

    std::hint::black_box(&out);
    out
}

/// Mirrors `f32_to_plane`'s narrow arm. The indexed write matches
/// `Narrow::write` rather than a `zip`, because a bounds-check-free
/// iterator idiom would measure a loop the binary never runs.
fn quantise_plane_narrow(plane: &[f32], max: f32) -> Vec<u8> {
    let mut out = vec![0u8; plane.len()];
    for (i, &v) in plane.iter().enumerate() {
        out[i] = quantise(v, max) as u8;
    }
    std::hint::black_box(&out);
    out
}

/// Mirrors `f32_to_plane`'s wide arm, indexed to match `Wide::write`.
fn quantise_plane_wide(plane: &[f32], max: f32) -> Vec<u8> {
    let mut out = vec![0u8; plane.len() * 2];
    for (i, &v) in plane.iter().enumerate() {
        out[2 * i..2 * i + 2].copy_from_slice(&quantise(v, max).to_le_bytes());
    }
    std::hint::black_box(&out);
    out
}

#[inline(always)]
fn quantise(v: f32, max: f32) -> u16 {
    (v.clamp(0.0, 1.0) * max + 0.5) as u16
}
