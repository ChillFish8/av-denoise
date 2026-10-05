use crate::host::Depth;

/// Reads and writes samples in one wire format.
///
/// The implementor is chosen once per conversion, which keeps the per-sample path free of depth
/// branches.
trait SampleCodec {
    const BYTES: usize;

    fn read(plane: &[u8], index: usize) -> u16;
    fn write(plane: &mut [u8], index: usize, value: u16);
}

/// One byte per sample.
struct Narrow;

impl SampleCodec for Narrow {
    const BYTES: usize = 1;

    #[inline(always)]
    fn read(plane: &[u8], index: usize) -> u16 {
        plane[index] as u16
    }

    #[inline(always)]
    fn write(plane: &mut [u8], index: usize, value: u16) {
        plane[index] = value as u8;
    }
}

/// Two bytes per sample, little-endian.
struct Wide;

impl SampleCodec for Wide {
    const BYTES: usize = 2;

    #[inline(always)]
    fn read(plane: &[u8], index: usize) -> u16 {
        u16::from_le_bytes([plane[2 * index], plane[2 * index + 1]])
    }

    #[inline(always)]
    fn write(plane: &mut [u8], index: usize, value: u16) {
        let bytes = value.to_le_bytes();
        plane[2 * index..2 * index + 2].copy_from_slice(&bytes);
    }
}

/// Quantises a normalised value to a native-depth sample.
#[inline(always)]
fn quantise(value: f32, max: f32) -> u16 {
    (value.clamp(0.0, 1.0) * max + 0.5) as u16
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

    fn run<C: SampleCodec>(samples: &[f32], max: f32) -> Vec<u8> {
        let mut plane_bytes = vec![0u8; samples.len() * C::BYTES];
        for (i, &sample) in samples.iter().enumerate() {
            let quantised = quantise(sample, max);
            C::write(&mut plane_bytes, i, quantised);
        }

        plane_bytes
    }

    match depth.bytes_per_sample() {
        1 => run::<Narrow>(plane, max),
        _ => run::<Wide>(plane, max),
    }
}

/// Interleaves equal-length Y, U and V wire-byte planes into normalised f32 samples.
///
/// The output runs `Y0, U0, V0, Y1, U1, V1` and so on.
pub fn interleave_yuv_to_f32(y_plane: &[u8], u_plane: &[u8], v_plane: &[u8], depth: Depth) -> Vec<f32> {
    debug_assert_eq!(y_plane.len(), u_plane.len());
    debug_assert_eq!(u_plane.len(), v_plane.len());

    let max = depth.max_value();

    fn run<C: SampleCodec>(y_plane: &[u8], u_plane: &[u8], v_plane: &[u8], max: f32) -> Vec<f32> {
        let pixels = y_plane.len() / C::BYTES;
        let mut interleaved = Vec::with_capacity(pixels * 3);

        for i in 0..pixels {
            let y_sample = C::read(y_plane, i) as f32 / max;
            let u_sample = C::read(u_plane, i) as f32 / max;
            let v_sample = C::read(v_plane, i) as f32 / max;

            interleaved.push(y_sample);
            interleaved.push(u_sample);
            interleaved.push(v_sample);
        }

        interleaved
    }

    match depth.bytes_per_sample() {
        1 => run::<Narrow>(y_plane, u_plane, v_plane, max),
        _ => run::<Wide>(y_plane, u_plane, v_plane, max),
    }
}

/// Interleaves equal-length U and V wire-byte planes into normalised f32 samples.
///
/// The output runs `U0, V0, U1, V1` and so on.
pub fn interleave_uv_to_f32(u_plane: &[u8], v_plane: &[u8], depth: Depth) -> Vec<f32> {
    debug_assert_eq!(u_plane.len(), v_plane.len());

    let max = depth.max_value();

    fn run<C: SampleCodec>(u_plane: &[u8], v_plane: &[u8], max: f32) -> Vec<f32> {
        let pixels = u_plane.len() / C::BYTES;
        let mut interleaved = Vec::with_capacity(pixels * 2);

        for i in 0..pixels {
            let u_sample = C::read(u_plane, i) as f32 / max;
            let v_sample = C::read(v_plane, i) as f32 / max;

            interleaved.push(u_sample);
            interleaved.push(v_sample);
        }

        interleaved
    }

    match depth.bytes_per_sample() {
        1 => run::<Narrow>(u_plane, v_plane, max),
        _ => run::<Wide>(u_plane, v_plane, max),
    }
}

/// Splits interleaved U and V samples into two quantised wire-byte planes.
pub fn unpack_uv_from_f32(packed: &[f32], chroma_pixels: usize, depth: Depth) -> (Vec<u8>, Vec<u8>) {
    debug_assert_eq!(packed.len(), 2 * chroma_pixels);

    let max = depth.max_value();

    fn run<C: SampleCodec>(packed: &[f32], chroma_pixels: usize, max: f32) -> (Vec<u8>, Vec<u8>) {
        let mut u_plane = vec![0u8; chroma_pixels * C::BYTES];
        let mut v_plane = vec![0u8; chroma_pixels * C::BYTES];

        let (pairs, _) = packed.as_chunks::<2>();
        for (i, pair) in pairs.iter().enumerate() {
            let u_sample = quantise(pair[0], max);
            let v_sample = quantise(pair[1], max);

            C::write(&mut u_plane, i, u_sample);
            C::write(&mut v_plane, i, v_sample);
        }

        (u_plane, v_plane)
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

    /// Reverse of [interleave_yuv_to_f32].
    fn unpack_yuv_from_f32(packed: &[f32], pixels: usize, depth: Depth) -> Planes {
        debug_assert_eq!(packed.len(), 3 * pixels);

        let max = depth.max_value();

        fn run<C: SampleCodec>(packed: &[f32], pixels: usize, max: f32) -> Planes {
            let mut y_plane = vec![0u8; pixels * C::BYTES];
            let mut u_plane = vec![0u8; pixels * C::BYTES];
            let mut v_plane = vec![0u8; pixels * C::BYTES];

            let (triples, _) = packed.as_chunks::<3>();
            for (i, triple) in triples.iter().enumerate() {
                let y_sample = quantise(triple[0], max);
                let u_sample = quantise(triple[1], max);
                let v_sample = quantise(triple[2], max);

                C::write(&mut y_plane, i, y_sample);
                C::write(&mut u_plane, i, u_sample);
                C::write(&mut v_plane, i, v_sample);
            }

            Planes {
                y: y_plane,
                u: u_plane,
                v: v_plane,
            }
        }

        match depth.bytes_per_sample() {
            1 => run::<Narrow>(packed, pixels, max),
            _ => run::<Wide>(packed, pixels, max),
        }
    }

    /// Encodes native-depth samples into wire bytes, the inverse of what the converters read.
    fn wire(samples: &[u16], depth: Depth) -> Vec<u8> {
        match depth.bytes_per_sample() {
            1 => samples.iter().map(|&sample| sample as u8).collect(),
            _ => samples.iter().flat_map(|&sample| sample.to_le_bytes()).collect(),
        }
    }

    #[test]
    fn plane_round_trips_boundary_codes_at_every_depth() {
        for depth in [Depth::Eight, Depth::Ten, Depth::Twelve] {
            let max = depth.max_value() as u16;
            let samples: Vec<u16> = vec![0, 1, 16, 64, 235, max / 2, max - 1, max]
                .into_iter()
                .filter(|&sample| sample <= max)
                .collect();

            let bytes = wire(&samples, depth);
            let normalised = plane_to_f32(&bytes, depth);
            let restored = f32_to_plane(&normalised, depth);

            assert_eq!(restored, bytes, "plane round trip failed at {depth:?}");
        }
    }

    #[test]
    fn high_depth_samples_are_little_endian() {
        // 1023 = 0x03FF -> [0xFF, 0x03]
        let bytes = wire(&[1023, 0, 512], Depth::Ten);
        assert_eq!(bytes, vec![0xFF, 0x03, 0x00, 0x00, 0x00, 0x02]);

        let normalised = plane_to_f32(&bytes, Depth::Ten);
        assert!(
            (normalised[0] - 1.0).abs() < 1e-6,
            "0x03FF should normalize to 1.0, got {}",
            normalised[0]
        );
        assert_eq!(normalised[1], 0.0);
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

            let (restored_u, restored_v) = unpack_uv_from_f32(&packed, 3, depth);
            assert_eq!(restored_u, u_bytes, "U round trip failed at {depth:?}");
            assert_eq!(restored_v, v_bytes, "V round trip failed at {depth:?}");
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

            let restored = unpack_yuv_from_f32(&packed, 3, depth);
            assert_eq!(restored.y, y_bytes, "Y round trip failed at {depth:?}");
            assert_eq!(restored.u, u_bytes, "U round trip failed at {depth:?}");
            assert_eq!(restored.v, v_bytes, "V round trip failed at {depth:?}");
        }
    }

    #[test]
    fn quantise_matches_the_clamping_form_including_nan() {
        fn reference(value: f32, max: f32) -> u16 {
            (value.clamp(0.0, 1.0) * max + 0.5) as u16
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

        for value in cases {
            let quantised = quantise(value, max);
            let expected = reference(value, max);
            assert_eq!(quantised, expected, "mismatch at {value}");
        }
    }

    /// Limited-range codes normalise to within one 8-bit code level of each other at every depth.
    ///
    /// ITU defines the limited-range endpoints as exact multiples, so 16 becomes 64 and 235 becomes
    /// 940, but full scale is not a multiple, because 255 becomes 1023. That leaves 235/255 and
    /// 940/1023 differing by 0.0027, roughly 0.69 of an 8-bit step.
    #[test]
    fn limited_range_codes_agree_across_depths() {
        /// One 8-bit code level, the precision the endpoints agree to.
        const TOLERANCE: f32 = 1.0 / 255.0;

        let eight_bit_bytes = wire(&[16, 235], Depth::Eight);
        let ten_bit_bytes = wire(&[64, 940], Depth::Ten);
        let eight = plane_to_f32(&eight_bit_bytes, Depth::Eight);
        let ten = plane_to_f32(&ten_bit_bytes, Depth::Ten);

        for (eight_bit, ten_bit) in eight.iter().zip(ten.iter()) {
            assert!(
                (eight_bit - ten_bit).abs() < TOLERANCE,
                "8-bit {eight_bit} vs 10-bit {ten_bit}"
            );
        }
    }
}
