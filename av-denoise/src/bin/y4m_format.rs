use av_denoise::{Depth, Subsampling};

/// Maps a [Subsampling] and [Depth] onto the matching [y4m::Colorspace].
pub fn subsampling_to_y4m(subsampling: Subsampling, depth: Depth) -> y4m::Colorspace {
    match (subsampling, depth) {
        (Subsampling::Yuv420, Depth::Eight) => y4m::Colorspace::C420,
        (Subsampling::Yuv420, Depth::Ten) => y4m::Colorspace::C420p10,
        (Subsampling::Yuv420, Depth::Twelve) => y4m::Colorspace::C420p12,
        (Subsampling::Yuv422, Depth::Eight) => y4m::Colorspace::C422,
        (Subsampling::Yuv422, Depth::Ten) => y4m::Colorspace::C422p10,
        (Subsampling::Yuv422, Depth::Twelve) => y4m::Colorspace::C422p12,
        (Subsampling::Yuv444, Depth::Eight) => y4m::Colorspace::C444,
        (Subsampling::Yuv444, Depth::Ten) => y4m::Colorspace::C444p10,
        (Subsampling::Yuv444, Depth::Twelve) => y4m::Colorspace::C444p12,
    }
}

/// Maps a [y4m::Colorspace] back onto a [Subsampling] and [Depth].
///
/// Grayscale and other unsupported colorspaces are rejected with an error naming what is required.
pub fn subsampling_from_y4m(colorspace: y4m::Colorspace) -> Result<(Subsampling, Depth), anyhow::Error> {
    let subsampling = match colorspace {
        y4m::Colorspace::C420
        | y4m::Colorspace::C420jpeg
        | y4m::Colorspace::C420paldv
        | y4m::Colorspace::C420mpeg2
        | y4m::Colorspace::C420p10
        | y4m::Colorspace::C420p12 => Subsampling::Yuv420,
        y4m::Colorspace::C422 | y4m::Colorspace::C422p10 | y4m::Colorspace::C422p12 => Subsampling::Yuv422,
        y4m::Colorspace::C444 | y4m::Colorspace::C444p10 | y4m::Colorspace::C444p12 => Subsampling::Yuv444,
        other => anyhow::bail!("unsupported y4m colorspace {other:?}, need 4:2:0, 4:2:2, or 4:4:4"),
    };

    let bit_depth = colorspace.get_bit_depth();
    let depth = Depth::from_bits(bit_depth)?;

    Ok((subsampling, depth))
}

/// Pulls the `X`-prefixed vendor extension params, such as `XCOLORRANGE=LIMITED`, out of a y4m header.
///
/// The leading `X` is stripped because [y4m::EncoderBuilder::append_vendor_extension] adds it back.
/// A token that [y4m::VendorExtensionString] rejects, one containing a space, is skipped rather
/// than failing the run.
pub fn y4m_vendor_extensions(raw_params: &[u8]) -> Vec<y4m::VendorExtensionString> {
    raw_params
        .split(|&byte| byte == b' ')
        .filter(|token| token.first() == Some(&b'X'))
        .filter_map(|token| {
            let value = token[1..].to_vec();
            y4m::VendorExtensionString::new(value).ok()
        })
        .collect()
}

#[cfg(test)]
mod colorspace_tests {
    use super::*;

    #[test]
    fn colorspace_round_trips_every_supported_combination() {
        let combos = [
            (Subsampling::Yuv420, Depth::Eight),
            (Subsampling::Yuv420, Depth::Ten),
            (Subsampling::Yuv420, Depth::Twelve),
            (Subsampling::Yuv422, Depth::Eight),
            (Subsampling::Yuv422, Depth::Ten),
            (Subsampling::Yuv422, Depth::Twelve),
            (Subsampling::Yuv444, Depth::Eight),
            (Subsampling::Yuv444, Depth::Ten),
            (Subsampling::Yuv444, Depth::Twelve),
        ];

        for (subsampling, depth) in combos {
            let colorspace = subsampling_to_y4m(subsampling, depth);
            let (mapped_subsampling, mapped_depth) =
                subsampling_from_y4m(colorspace).expect("should map back");

            assert_eq!(
                mapped_subsampling, subsampling,
                "subsampling lost for {colorspace:?}"
            );
            assert_eq!(mapped_depth, depth, "depth lost for {colorspace:?}");
        }
    }

    #[test]
    fn ten_bit_420_maps_to_c420p10() {
        let colorspace = subsampling_to_y4m(Subsampling::Yuv420, Depth::Ten);

        // `y4m::Colorspace` does not derive `PartialEq`, so `assert_eq!` won't compile here.
        assert!(matches!(colorspace, y4m::Colorspace::C420p10));
    }

    #[test]
    fn eight_bit_420_variants_all_map_to_yuv420_eight() {
        for colorspace in [
            y4m::Colorspace::C420,
            y4m::Colorspace::C420jpeg,
            y4m::Colorspace::C420paldv,
            y4m::Colorspace::C420mpeg2,
        ] {
            let (subsampling, depth) = subsampling_from_y4m(colorspace).expect("should map");

            assert_eq!(subsampling, Subsampling::Yuv420);
            assert_eq!(depth, Depth::Eight);
        }
    }

    #[test]
    fn grayscale_colorspaces_are_rejected_with_a_clear_message() {
        for colorspace in [y4m::Colorspace::Cmono, y4m::Colorspace::Cmono12] {
            let err = subsampling_from_y4m(colorspace).expect_err("grayscale should be rejected");
            let message = err.to_string();
            let colorspace_name = format!("{colorspace:?}");

            assert!(
                message.contains(&colorspace_name),
                "error should name the offending colorspace, got {message}"
            );
        }
    }
}
