use std::io::{Read, Write, stdout};

use av_denoise::{FrameLayout, PlaneOptions, Planes, push_needs_retry};

use crate::warm_start::{create_denoiser, finish_warm_up};
use crate::y4m_format::{subsampling_from_y4m, subsampling_to_y4m, y4m_vendor_extensions};

/// Denoises a y4m stream frame by frame, writing y4m on stdout.
///
/// There is no scene detection here, so the temporal window slides
/// across the whole stream.
pub fn run_stream<R: Read>(opts: &PlaneOptions, reader: R) -> Result<(), anyhow::Error> {
    let mut decoder = y4m::Decoder::new(reader)?;

    let (subsampling, depth) = subsampling_from_y4m(decoder.get_colorspace())?;

    let layout = FrameLayout {
        width: decoder.get_width() as u32,
        height: decoder.get_height() as u32,
        subsampling,
        depth,
    };

    let framerate = decoder.get_framerate();
    let pixel_aspect = decoder.get_pixel_aspect();
    let colorspace = subsampling_to_y4m(layout.subsampling, layout.depth);

    let stdout = stdout();
    let mut builder = y4m::encode(layout.width as usize, layout.height as usize, framerate)
        .with_colorspace(colorspace)
        .with_pixel_aspect(pixel_aspect);
    // Forward the source's `X` extension params (e.g. `XCOLORRANGE=`)
    // verbatim instead of silently dropping them.
    for ext in y4m_vendor_extensions(decoder.get_raw_params()) {
        builder = builder.append_vendor_extension(ext);
    }
    let mut encoder = builder.write_header(stdout.lock())?;

    let (mut wd, mut warm_up) = create_denoiser(opts, layout)?;

    tracing::info!(
        accelerator = ?opts.accelerators,
        width = layout.width,
        height = layout.height,
        subsampling = ?layout.subsampling,
        depth = ?layout.depth,
        "streaming pipeline ready",
    );

    loop {
        let frame = match decoder.read_frame() {
            Ok(f) => f,
            Err(y4m::Error::EOF) => break,
            Err(e) => return Err(e.into()),
        };

        let planes = Planes {
            y: frame.get_y_plane().to_vec(),
            u: frame.get_u_plane().to_vec(),
            v: frame.get_v_plane().to_vec(),
        };

        if push_needs_retry(wd.push(&planes))? {
            if let Some(out) = wd.recv()? {
                write_planes(&mut encoder, &out)?;
                finish_warm_up(&mut warm_up);
            }

            wd.push(&planes)?;
        }

        if let Some(out) = wd.recv()? {
            write_planes(&mut encoder, &out)?;
            finish_warm_up(&mut warm_up);
        }
    }

    wd.flush(|out| {
        if let Err(e) = write_planes(&mut encoder, &out) {
            tracing::error!(error = ?e, "failed to write flushed frame");
        }
        finish_warm_up(&mut warm_up);
    })?;

    Ok(())
}

fn write_planes<W: Write>(encoder: &mut y4m::Encoder<W>, planes: &Planes) -> Result<(), anyhow::Error> {
    let frame = y4m::Frame::new([&planes.y, &planes.u, &planes.v], None);
    encoder.write_frame(&frame)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use av_denoise::{Depth, Subsampling};

    use super::*;

    #[test]
    fn write_planes_emits_a_frame_into_any_writer() {
        let mut buf = Vec::new();
        let mut encoder = y4m::encode(2, 2, y4m::Ratio::new(25, 1))
            .with_colorspace(y4m::Colorspace::C420)
            .write_header(&mut buf)
            .expect("header should write");

        let planes = Planes {
            y: vec![16; 4],
            u: vec![128; 1],
            v: vec![128; 1],
        };

        write_planes(&mut encoder, &planes).expect("frame should write");
        // Marks the end of writing before the asserts read `buf`, which
        // the encoder borrows mutably. `y4m::Encoder` implements no
        // `Drop` and a `Vec` writer needs no flush, so this documents
        // the handover rather than doing work.
        #[expect(
            clippy::drop_non_drop,
            reason = "reads as the end of the write phase, ahead of the asserts on buf"
        )]
        drop(encoder);

        assert!(
            buf.starts_with(b"YUV4MPEG2"),
            "expected a y4m header, got {buf:?}"
        );
        assert!(
            buf.windows(6).any(|w| w == b"FRAME\n"),
            "expected a FRAME marker in {buf:?}",
        );
    }

    /// Drives a synthetic 10-bit stream through the whole y4m path and
    /// checks the output header and plane sizes survive.
    #[test]
    fn ten_bit_stream_round_trips_header_and_plane_sizes() {
        let width = 4;
        let height = 4;
        let luma_bytes = width * height * 2;
        let chroma_bytes = (width / 2) * (height / 2) * 2;

        let mut input: Vec<u8> = Vec::new();
        {
            let mut enc = y4m::encode(width, height, y4m::Ratio::new(25, 1))
                .with_colorspace(y4m::Colorspace::C420p10)
                .write_header(&mut input)
                .expect("header should write");

            // 512 little-endian, mid-grey at 10-bit.
            #[expect(
                clippy::useless_vec,
                reason = "the vec! spells out that these two bytes are one 10-bit sample"
            )]
            let y = vec![0x00u8, 0x02].repeat(width * height);
            #[expect(
                clippy::useless_vec,
                reason = "the vec! spells out that these two bytes are one 10-bit sample"
            )]
            let u = vec![0x00u8, 0x02].repeat((width / 2) * (height / 2));
            let v = u.clone();

            for _ in 0..2 {
                enc.write_frame(&y4m::Frame::new([&y, &u, &v], None))
                    .expect("frame should write");
            }
        }

        let mut decoder = y4m::Decoder::new(input.as_slice()).expect("decode header");
        assert_eq!(decoder.get_bit_depth(), 10);
        assert_eq!(decoder.get_bytes_per_sample(), 2);

        let frame = decoder.read_frame().expect("read frame");
        assert_eq!(frame.get_y_plane().len(), luma_bytes);
        assert_eq!(frame.get_u_plane().len(), chroma_bytes);

        let (sub, depth) = subsampling_from_y4m(decoder.get_colorspace()).expect("colorspace should map");
        assert_eq!(sub, Subsampling::Yuv420);
        assert_eq!(depth, Depth::Ten);
    }
}
