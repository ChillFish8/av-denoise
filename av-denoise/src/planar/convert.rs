use crate::host::Depth;

/// Reads and writes samples in one wire format.
///
/// The implementor is chosen once per conversion, which keeps the
/// per-sample path free of depth branches.
trait SampleCodec {
    const BYTES: usize;

    fn read(plane: &[u8], i: usize) -> u16;
    fn write(plane: &mut [u8], i: usize, value: u16);
}

/// One byte per sample.
struct Narrow;

impl SampleCodec for Narrow {
    const BYTES: usize = 1;

    #[inline(always)]
    fn read(plane: &[u8], i: usize) -> u16 {
        plane[i] as u16
    }

    #[inline(always)]
    fn write(plane: &mut [u8], i: usize, value: u16) {
        plane[i] = value as u8;
    }
}

/// Two bytes per sample, little-endian.
struct Wide;

impl SampleCodec for Wide {
    const BYTES: usize = 2;

    #[inline(always)]
    fn read(plane: &[u8], i: usize) -> u16 {
        u16::from_le_bytes([plane[2 * i], plane[2 * i + 1]])
    }

    #[inline(always)]
    fn write(plane: &mut [u8], i: usize, value: u16) {
        plane[2 * i..2 * i + 2].copy_from_slice(&value.to_le_bytes());
    }
}

/// Quantises a normalised value to a native-depth sample.
#[inline(always)]
fn quantise(v: f32, max: f32) -> u16 {
    (v.clamp(0.0, 1.0) * max + 0.5) as u16
}

/// Converts one wire-byte plane to normalised f32 samples.
pub fn plane_to_f32(plane: &[u8], depth: Depth) -> Vec<f32> {
    let max = depth.max_value();

    fn run<C: SampleCodec>(plane: &[u8], max: f32) -> Vec<f32> {
        let samples = plane.len() / C::BYTES;
        (0..samples).map(|i| C::read(plane, i) as f32 / max).collect()
    }

    match depth.bytes_per_sample() {
        1 => run::<Narrow>(plane, max),
        _ => run::<Wide>(plane, max),
    }
}

/// Quantises normalised f32 samples into one wire-byte plane.
pub fn f32_to_plane(plane: &[f32], depth: Depth) -> Vec<u8> {
    let max = depth.max_value();

    fn run<C: SampleCodec>(plane: &[f32], max: f32) -> Vec<u8> {
        let mut out = vec![0u8; plane.len() * C::BYTES];
        for (i, &v) in plane.iter().enumerate() {
            C::write(&mut out, i, quantise(v, max));
        }
        out
    }

    match depth.bytes_per_sample() {
        1 => run::<Narrow>(plane, max),
        _ => run::<Wide>(plane, max),
    }
}

/// Interleaves equal-length Y, U and V wire-byte planes into normalised f32 samples.
///
/// The output runs `Y0, U0, V0, Y1, U1, V1` and so on.
pub fn interleave_yuv_to_f32(y: &[u8], u: &[u8], v: &[u8], depth: Depth) -> Vec<f32> {
    debug_assert_eq!(y.len(), u.len());
    debug_assert_eq!(u.len(), v.len());

    let max = depth.max_value();

    fn run<C: SampleCodec>(y: &[u8], u: &[u8], v: &[u8], max: f32) -> Vec<f32> {
        let pixels = y.len() / C::BYTES;
        let mut out = Vec::with_capacity(pixels * 3);

        for i in 0..pixels {
            out.push(C::read(y, i) as f32 / max);
            out.push(C::read(u, i) as f32 / max);
            out.push(C::read(v, i) as f32 / max);
        }

        out
    }

    match depth.bytes_per_sample() {
        1 => run::<Narrow>(y, u, v, max),
        _ => run::<Wide>(y, u, v, max),
    }
}

/// Interleaves equal-length U and V wire-byte planes into normalised f32 samples.
///
/// The output runs `U0, V0, U1, V1` and so on.
pub fn interleave_uv_to_f32(u: &[u8], v: &[u8], depth: Depth) -> Vec<f32> {
    debug_assert_eq!(u.len(), v.len());

    let max = depth.max_value();

    fn run<C: SampleCodec>(u: &[u8], v: &[u8], max: f32) -> Vec<f32> {
        let pixels = u.len() / C::BYTES;
        let mut out = Vec::with_capacity(pixels * 2);

        for i in 0..pixels {
            out.push(C::read(u, i) as f32 / max);
            out.push(C::read(v, i) as f32 / max);
        }

        out
    }

    match depth.bytes_per_sample() {
        1 => run::<Narrow>(u, v, max),
        _ => run::<Wide>(u, v, max),
    }
}

/// Splits interleaved U and V samples into two quantised wire-byte planes.
pub fn unpack_uv_from_f32(packed: &[f32], chroma_pixels: usize, depth: Depth) -> (Vec<u8>, Vec<u8>) {
    debug_assert_eq!(packed.len(), 2 * chroma_pixels);

    let max = depth.max_value();

    fn run<C: SampleCodec>(packed: &[f32], chroma_pixels: usize, max: f32) -> (Vec<u8>, Vec<u8>) {
        let mut u = vec![0u8; chroma_pixels * C::BYTES];
        let mut v = vec![0u8; chroma_pixels * C::BYTES];

        for (i, chunk) in packed.as_chunks::<2>().0.iter().enumerate() {
            C::write(&mut u, i, quantise(chunk[0], max));
            C::write(&mut v, i, quantise(chunk[1], max));
        }

        (u, v)
    }

    match depth.bytes_per_sample() {
        1 => run::<Narrow>(packed, chroma_pixels, max),
        _ => run::<Wide>(packed, chroma_pixels, max),
    }
}

#[cfg(test)]
mod converter_tests {
    use super::*;
    use crate::planar::Planes;

    /// Reverse of [`interleave_yuv_to_f32`].
    fn unpack_yuv_from_f32(packed: &[f32], pixels: usize, depth: Depth) -> Planes {
        debug_assert_eq!(packed.len(), 3 * pixels);

        let max = depth.max_value();

        fn run<C: SampleCodec>(packed: &[f32], pixels: usize, max: f32) -> Planes {
            let mut y = vec![0u8; pixels * C::BYTES];
            let mut u = vec![0u8; pixels * C::BYTES];
            let mut v = vec![0u8; pixels * C::BYTES];

            for (i, chunk) in packed.as_chunks::<3>().0.iter().enumerate() {
                C::write(&mut y, i, quantise(chunk[0], max));
                C::write(&mut u, i, quantise(chunk[1], max));
                C::write(&mut v, i, quantise(chunk[2], max));
            }

            Planes { y, u, v }
        }

        match depth.bytes_per_sample() {
            1 => run::<Narrow>(packed, pixels, max),
            _ => run::<Wide>(packed, pixels, max),
        }
    }

    /// Encodes native-depth samples into wire bytes, the inverse of what
    /// the converters read.
    fn wire(samples: &[u16], depth: Depth) -> Vec<u8> {
        match depth.bytes_per_sample() {
            1 => samples.iter().map(|&s| s as u8).collect(),
            _ => samples.iter().flat_map(|&s| s.to_le_bytes()).collect(),
        }
    }

    #[test]
    fn plane_round_trips_boundary_codes_at_every_depth() {
        for depth in [Depth::Eight, Depth::Ten, Depth::Twelve] {
            let max = depth.max_value() as u16;
            let samples: Vec<u16> = vec![0, 1, 16, 64, 235, max / 2, max - 1, max]
                .into_iter()
                .filter(|&s| s <= max)
                .collect();

            let bytes = wire(&samples, depth);
            let restored = f32_to_plane(&plane_to_f32(&bytes, depth), depth);

            assert_eq!(restored, bytes, "plane round trip failed at {depth:?}");
        }
    }

    /// Samples above 8 bits are little-endian on the wire regardless of
    /// host endianness.
    #[test]
    fn high_depth_samples_are_little_endian() {
        // 1023 = 0x03FF -> [0xFF, 0x03]
        let bytes = wire(&[1023, 0, 512], Depth::Ten);
        assert_eq!(bytes, vec![0xFF, 0x03, 0x00, 0x00, 0x00, 0x02]);

        let f = plane_to_f32(&bytes, Depth::Ten);
        assert!(
            (f[0] - 1.0).abs() < 1e-6,
            "0x03FF should normalize to 1.0, got {}",
            f[0]
        );
        assert_eq!(f[1], 0.0);
    }

    #[test]
    fn uv_interleave_round_trips_at_every_depth() {
        for depth in [Depth::Eight, Depth::Ten, Depth::Twelve] {
            let max = depth.max_value() as u16;
            let u_samples = vec![0, max / 4, max];
            let v_samples = vec![max, max / 2, 1];

            let u_bytes = wire(&u_samples, depth);
            let v_bytes = wire(&v_samples, depth);

            let packed = interleave_uv_to_f32(&u_bytes, &v_bytes, depth);
            assert_eq!(packed.len(), 6, "packed UV length wrong at {depth:?}");

            let (ru, rv) = unpack_uv_from_f32(&packed, 3, depth);
            assert_eq!(ru, u_bytes, "U round trip failed at {depth:?}");
            assert_eq!(rv, v_bytes, "V round trip failed at {depth:?}");
        }
    }

    #[test]
    fn yuv_interleave_round_trips_at_every_depth() {
        for depth in [Depth::Eight, Depth::Ten, Depth::Twelve] {
            let max = depth.max_value() as u16;
            let y_samples = vec![0, max / 3, max];
            let u_samples = vec![max, 0, max / 2];
            let v_samples = vec![max / 4, max, 0];

            let y_bytes = wire(&y_samples, depth);
            let u_bytes = wire(&u_samples, depth);
            let v_bytes = wire(&v_samples, depth);

            let packed = interleave_yuv_to_f32(&y_bytes, &u_bytes, &v_bytes, depth);
            assert_eq!(packed.len(), 9, "packed YUV length wrong at {depth:?}");

            let out = unpack_yuv_from_f32(&packed, 3, depth);
            assert_eq!(out.y, y_bytes, "Y round trip failed at {depth:?}");
            assert_eq!(out.u, u_bytes, "U round trip failed at {depth:?}");
            assert_eq!(out.v, v_bytes, "V round trip failed at {depth:?}");
        }
    }

    #[test]
    fn quantise_matches_the_clamping_form_including_nan() {
        fn reference(v: f32, max: f32) -> u16 {
            (v.clamp(0.0, 1.0) * max + 0.5) as u16
        }

        let max = 1023.0;
        let cases = [
            -1.0,
            -0.001,
            0.0,
            0.5,
            0.999,
            1.0,
            1.001,
            2.0,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ];

        for v in cases {
            assert_eq!(quantise(v, max), reference(v, max), "mismatch at {v}");
        }
    }

    /// Limited-range codes normalise to matching values at every depth,
    /// which is the property the whole design rests on.
    ///
    /// The match is within one 8-bit code level rather than exact. ITU
    /// defines the limited-range endpoints as exact multiples, so 16
    /// becomes 64 and 235 becomes 940, but full scale is not a multiple,
    /// because 255 becomes 1023. That leaves 235/255 and 940/1023
    /// differing by 0.0027, roughly 0.69 of an 8-bit step.
    ///
    /// Agreement below one step is the real property here.
    ///
    /// `normalized_scale_is_identical_across_depths` in
    /// `src/nlmeans/mod.rs` pins the same property on the library's own
    /// normalise helper.
    #[test]
    fn limited_range_codes_agree_across_depths() {
        /// One 8-bit code level, the precision the endpoints agree to.
        const TOL: f32 = 1.0 / 255.0;

        let eight = plane_to_f32(&wire(&[16, 235], Depth::Eight), Depth::Eight);
        let ten = plane_to_f32(&wire(&[64, 940], Depth::Ten), Depth::Ten);

        for (a, b) in eight.iter().zip(ten.iter()) {
            assert!((a - b).abs() < TOL, "8-bit {a} vs 10-bit {b}");
        }
    }
}
