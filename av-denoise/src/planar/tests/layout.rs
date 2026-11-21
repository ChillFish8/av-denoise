use super::*;

fn layout(depth: Depth) -> FrameLayout {
    FrameLayout {
        width: 4,
        height: 4,
        subsampling: Subsampling::Yuv420,
        depth,
        chroma_siting: ChromaSiting::Unknown,
    }
}

#[test]
fn byte_lengths_scale_with_depth() {
    let eight_bit = layout(Depth::Eight);
    let ten_bit = layout(Depth::Ten);

    assert_eq!(eight_bit.luma_bytes(), 16);
    assert_eq!(ten_bit.luma_bytes(), 32);
    assert_eq!(eight_bit.chroma_bytes(), 4);
    assert_eq!(ten_bit.chroma_bytes(), 8);
}

#[test]
fn float_neutral_chroma_plane_is_zero() {
    let float = layout(Depth::Float);

    assert_eq!(float.luma_bytes(), 64);
    assert_eq!(float.neutral_chroma_plane(), vec![0u8; 16]);
}

#[test]
fn neutral_chroma_fill_is_correct_at_each_depth() {
    let eight = layout(Depth::Eight).neutral_chroma_plane();
    assert_eq!(eight, vec![128u8; 4]);

    // 512 little-endian is [0x00, 0x02], repeated per sample.
    let ten = layout(Depth::Ten).neutral_chroma_plane();
    assert_eq!(ten, vec![0x00, 0x02, 0x00, 0x02, 0x00, 0x02, 0x00, 0x02]);

    // 2048 little-endian is [0x00, 0x08].
    let twelve = layout(Depth::Twelve).neutral_chroma_plane();
    assert_eq!(twelve.len(), 8);
    assert_eq!(&twelve[0..2], &[0x00, 0x08]);
}

#[test]
fn sixteen_bit_planes_pack_full_range_little_endian() {
    let depth = Depth::from_bits(16).unwrap();
    let odd = FrameLayout {
        width: 5,
        height: 3,
        ..layout(depth)
    };
    assert_eq!(odd.luma_bytes(), 30);
    assert_eq!(odd.chroma_dims(), (3, 2));
    assert_eq!(odd.neutral_chroma_plane(), [0, 128].repeat(6));

    for (sample, word) in [
        (0, [0, 0]),
        (1, [1, 0]),
        (4095, [255, 15]),
        (4096, [0, 16]),
        (32768, [0, 128]),
        (65534, [254, 255]),
        (65535, [255, 255]),
    ] {
        assert_eq!(fill_plane(3, sample, depth), word.repeat(3));
    }
}

#[cfg(feature = "vulkan")]
#[test]
fn sixteen_bit_nl4d_keeps_full_range_and_bypassed_chroma() {
    let layout = FrameLayout {
        width: 65,
        height: 65,
        ..layout(Depth::from_bits(16).unwrap())
    };
    let options = PlaneOptions {
        intent: ChannelIntent::Luma,
        algorithm: Algorithm::Nl4d(crate::Nl4dOptions {
            sigma: Some(0.03),
            ..crate::Nl4dOptions::default()
        }),
        ..super::reseed::test_plane_options(1)
    };
    let frame = Planes {
        y: fill_plane(layout.luma_pixels(), 49153, layout.depth),
        u: fill_plane(layout.chroma_pixels(), 32769, layout.depth),
        v: fill_plane(layout.chroma_pixels(), 65534, layout.depth),
    };
    let mut denoiser = PlanarDenoiser::create(&options, layout).unwrap();
    let output = denoiser
        .reseed_window(ReseedWindow {
            frames: std::slice::from_ref(&frame),
            target: 0,
            at_clip_start: true,
            at_clip_end: true,
        })
        .unwrap();
    let [output] = output.as_slice() else {
        panic!("expected one output");
    };
    assert_eq!(output.y.len(), frame.y.len());
    assert!(output.y.as_chunks::<2>().0.iter().all(|word| {
        let value = u16::from_le_bytes([word[0], word[1]]);
        value.abs_diff(49153) <= 2
    }));
    assert_eq!(output.u, frame.u);
    assert_eq!(output.v, frame.v);
}
