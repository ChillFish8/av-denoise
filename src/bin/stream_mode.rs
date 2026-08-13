use std::io::{Read, Write, stdout};

use av_denoise::Depth;

use crate::ingest::{
    CliOptions,
    FrameLayout,
    Planes,
    WorkerDenoiser,
    push_needs_retry,
    subsampling_from_y4m,
    subsampling_to_y4m,
    y4m_vendor_extensions,
};

/// Denoises a y4m stream frame by frame, writing y4m on stdout.
///
/// There is no scene detection here, so the temporal window slides
/// across the whole stream.
pub fn run_stream<R: Read>(opts: &CliOptions, reader: R) -> Result<(), anyhow::Error> {
    let mut decoder = y4m::Decoder::new(reader)?;

    if decoder.get_bytes_per_sample() != 1 {
        anyhow::bail!(
            "only 8-bit y4m input is supported (got bit depth {})",
            decoder.get_bit_depth()
        );
    }

    let layout = FrameLayout {
        width: decoder.get_width() as u32,
        height: decoder.get_height() as u32,
        subsampling: subsampling_from_y4m(decoder.get_colorspace())?,
        depth: Depth::Eight,
    };

    let framerate = decoder.get_framerate();
    let pixel_aspect = decoder.get_pixel_aspect();
    let colorspace = subsampling_to_y4m(layout.subsampling);

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

    let mut wd = WorkerDenoiser::create(opts, layout)?;

    tracing::info!(
        accelerator = ?opts.accelerators,
        width = layout.width,
        height = layout.height,
        subsampling = ?layout.subsampling,
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
            }

            wd.push(&planes)?;
        }

        if let Some(out) = wd.recv()? {
            write_planes(&mut encoder, &out)?;
        }
    }

    wd.flush(|out| {
        if let Err(e) = write_planes(&mut encoder, &out) {
            tracing::error!(error = ?e, "failed to write flushed frame");
        }
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
}
