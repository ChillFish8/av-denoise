use cubecl::prelude::*;

use super::helpers::{R, make_client};
use crate::Depth;
use crate::nlmeans::kernels::gpu_unpack_wire;
use crate::nlmeans::normalisation_table;

const BLOCK: u32 = 256;

/// Fills every destination slot before the launch, so a slot the kernel
/// never writes is visible instead of reading as a plausible zero.
const SENTINEL: f32 = 1.0;

/// Runs the kernel over `wire` and returns the `f32` frame it wrote,
/// including any padding lanes.
fn unpack(wire: &[u8], pixels: u32, channels: u32, stored_ch: u32, depth: Depth) -> Vec<f32> {
    let elements = pixels * stored_ch;
    let out = launch(wire, pixels, channels, stored_ch, depth, None, 0, elements);
    out[..elements as usize].to_vec()
}

/// The same, with the grid forced so a test can make every thread walk
/// the strided loop more than once.
fn unpack_with_grid(
    wire: &[u8],
    pixels: u32,
    channels: u32,
    stored_ch: u32,
    depth: Depth,
    grid: u32,
) -> Vec<f32> {
    let elements = pixels * stored_ch;
    let out = launch(wire, pixels, channels, stored_ch, depth, Some(grid), 0, elements);
    out[..elements as usize].to_vec()
}

/// Runs one launch and returns the whole destination buffer, sentinel
/// slots included.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors the kernel's own argument list, plus the launch geometry"
)]
fn launch(
    wire: &[u8],
    pixels: u32,
    channels: u32,
    stored_ch: u32,
    depth: Depth,
    grid_override: Option<u32>,
    dst_offset: u32,
    dst_len: u32,
) -> Vec<f32> {
    let client = make_client();

    let pack = depth.wire_pack();
    let elements = pixels * stored_ch;
    let grid = grid_override.unwrap_or_else(|| elements.div_ceil(BLOCK).max(1));
    let total_threads = grid * BLOCK;

    // The kernel reads whole words, so a plane that ends mid-word needs
    // its last word backed by real storage.
    let mut padded = wire.to_vec();
    padded.resize(wire.len().div_ceil(4) * 4, 0);

    let table = normalisation_table(depth);
    let seed = vec![SENTINEL; dst_len as usize];

    let src = client.create_from_slice(&padded);
    let lut = client.create_from_slice(f32::as_bytes(table.values()));
    let dst = client.create_from_slice(f32::as_bytes(&seed));

    unsafe {
        gpu_unpack_wire::launch_unchecked::<R>(
            &client,
            CubeCount::new_1d(grid),
            CubeDim::new_1d(BLOCK),
            ArrayArg::from_raw_parts(src, padded.len() / 4),
            ArrayArg::from_raw_parts(lut, table.values().len()),
            ArrayArg::from_raw_parts(dst.clone(), dst_len as usize),
            dst_offset,
            pixels,
            channels,
            stored_ch,
            pack.samples_per_word(),
            table.max_sample(),
            elements,
            total_threads,
        );
    }

    let out = client.read_one(dst).expect("unpack readback failed");
    f32::from_bytes(&out)[..dst_len as usize].to_vec()
}

/// Turns normalised samples into the wire bytes the kernel reads.
fn wire_bytes(samples: &[f32], depth: Depth) -> Vec<u8> {
    crate::frame::f32_to_plane(samples, depth)
}

/// A ramp over the whole range, so every test covers both ends.
fn ramp(pixels: u32) -> Vec<f32> {
    (0..pixels).map(|i| i as f32 / (pixels - 1) as f32).collect()
}

/// The only check that can catch a silently-dead kernel, since a kernel
/// compared against itself compares zeros to zeros.
#[test]
fn luma_matches_the_host_converter_at_every_depth() {
    for depth in [Depth::Eight, Depth::Ten, Depth::Twelve] {
        let pixels = 64u32;
        let wire = wire_bytes(&ramp(pixels), depth);

        let got = unpack(&wire, pixels, 1, 1, depth);
        let want = crate::frame::plane_to_f32(&wire, depth);

        assert_eq!(got, want, "depth {depth:?}");
    }
}

#[test]
fn chroma_matches_interleave_uv_from_the_host_at_every_depth() {
    for depth in [Depth::Eight, Depth::Ten, Depth::Twelve] {
        let pixels = 32u32;
        let u = ramp(pixels);
        let v: Vec<f32> = u.iter().map(|s| 1.0 - s).collect();

        let u_wire = wire_bytes(&u, depth);
        let v_wire = wire_bytes(&v, depth);
        let mut wire = u_wire.clone();
        wire.extend_from_slice(&v_wire);

        let got = unpack(&wire, pixels, 2, 2, depth);
        let want = crate::frame::interleave_uv_to_f32(&u_wire, &v_wire, depth);

        assert_eq!(got, want, "depth {depth:?}");
    }
}

/// The padding lane starts at [`SENTINEL`], so a kernel that leaves lane
/// 3 alone fails here rather than passing on a zeroed allocation.
#[test]
fn yuv_matches_the_host_and_zeroes_the_padding_lane() {
    for depth in [Depth::Eight, Depth::Ten, Depth::Twelve] {
        let pixels = 16u32;
        let make = |off: f32| -> Vec<f32> {
            (0..pixels)
                .map(|i| ((i as f32 / (pixels - 1) as f32) * 0.5 + off).clamp(0.0, 1.0))
                .collect()
        };
        let (y, u, v) = (make(0.0), make(0.25), make(0.5));
        let (yw, uw, vw) = (
            wire_bytes(&y, depth),
            wire_bytes(&u, depth),
            wire_bytes(&v, depth),
        );

        let mut wire = yw.clone();
        wire.extend_from_slice(&uw);
        wire.extend_from_slice(&vw);

        // Yuv stores 4 lanes per pixel and fills 3.
        let got = unpack(&wire, pixels, 3, 4, depth);
        let want = crate::frame::interleave_yuv_to_f32(&yw, &uw, &vw, depth);

        for p in 0..pixels as usize {
            for c in 0..3usize {
                assert_eq!(
                    got[p * 4 + c],
                    want[p * 3 + c],
                    "depth {depth:?} pixel {p} channel {c}"
                );
            }
            assert_eq!(got[p * 4 + 3], 0.0, "the padding lane must be zero");
        }
    }
}

/// A ring slot other than the first. Without this the kernel writes every
/// frame over slot 0 and every other test still passes.
#[test]
fn a_non_zero_dst_offset_writes_its_own_slot_and_leaves_the_others_alone() {
    for depth in [Depth::Eight, Depth::Ten] {
        let pixels = 64u32;
        let wire = wire_bytes(&ramp(pixels), depth);

        let got = launch(&wire, pixels, 1, 1, depth, None, pixels, pixels * 2);
        let want = crate::frame::plane_to_f32(&wire, depth);

        assert_eq!(&got[pixels as usize..], &want[..], "depth {depth:?}");
        assert_eq!(
            &got[..pixels as usize],
            &vec![SENTINEL; pixels as usize][..],
            "depth {depth:?}, the slot below must be untouched"
        );
    }
}

#[test]
fn a_sample_count_that_is_not_a_whole_number_of_words_reads_its_tail() {
    // 13 samples at 8-bit is three whole words plus one byte, and at 10
    // and 12-bit it is six whole words plus two bytes.
    let pixels = 13u32;

    for depth in [Depth::Eight, Depth::Ten, Depth::Twelve] {
        let wire = wire_bytes(&ramp(pixels), depth);

        let got = unpack(&wire, pixels, 1, 1, depth);
        let want = crate::frame::plane_to_f32(&wire, depth);

        assert_eq!(got, want, "depth {depth:?}");
    }
}

/// Forces a grid far smaller than the element count so every thread runs
/// the strided loop several times. Without this the loop body runs at most
/// once and `idx += total_threads` is never exercised.
#[test]
fn the_strided_loop_covers_every_element_when_the_grid_is_small() {
    let pixels = 1024u32;

    for depth in [Depth::Eight, Depth::Ten, Depth::Twelve] {
        let samples: Vec<f32> = (0..pixels).map(|i| (i % 251) as f32 / 250.0).collect();
        let wire = wire_bytes(&samples, depth);

        let got = unpack_with_grid(&wire, pixels, 1, 1, depth, 1);
        let want = crate::frame::plane_to_f32(&wire, depth);

        assert_eq!(got, want, "depth {depth:?}");
    }
}

/// Walks every sample value at every depth, which the GPU tests only
/// sample. Exactness rests on the table holding the host converter's own
/// results, so the two must be checked against each other directly.
#[test]
fn the_table_holds_the_host_converters_result_for_every_sample() {
    for depth in [Depth::Eight, Depth::Ten, Depth::Twelve] {
        let table = normalisation_table(depth);
        let samples = 1usize << depth.bits();

        assert_eq!(table.values().len(), samples, "depth {depth:?}");
        assert_eq!(table.max_sample() as usize, samples - 1, "depth {depth:?}");

        // One wire plane holding every sample the depth can express,
        // which the host converter then normalises.
        let mut wire = vec![0u8; samples * depth.bytes_per_sample()];
        for (s, chunk) in wire.chunks_exact_mut(depth.bytes_per_sample()).enumerate() {
            chunk.copy_from_slice(&(s as u32).to_le_bytes()[..depth.bytes_per_sample()]);
        }

        assert_eq!(
            table.values(),
            crate::frame::plane_to_f32(&wire, depth),
            "depth {depth:?}"
        );
    }
}
