//! Single-thread y4m stdin → denoise → y4m stdout pipeline.

use std::io::{stdin, stdout};

use av_denoise::DenoiserError;

use crate::ingest::{
    CliOptions,
    FrameLayout,
    Planes,
    WorkerDenoiser,
    subsampling_from_y4m,
    subsampling_to_y4m,
};

pub fn run_stdin(opts: &CliOptions) -> Result<(), anyhow::Error> {
    let stdin = stdin();
    let mut decoder = y4m::Decoder::new(stdin.lock())?;

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
    };

    let framerate = decoder.get_framerate();
    let pixel_aspect = decoder.get_pixel_aspect();
    let colorspace = subsampling_to_y4m(layout.subsampling);

    let stdout = stdout();
    let mut encoder = y4m::encode(layout.width as usize, layout.height as usize, framerate)
        .with_colorspace(colorspace)
        .with_pixel_aspect(pixel_aspect)
        .write_header(stdout.lock())?;

    let mut wd = WorkerDenoiser::create(opts, layout)?;

    tracing::info!(
        accelerator = ?opts.accelerators,
        width = layout.width,
        height = layout.height,
        subsampling = ?layout.subsampling,
        "stdin pipeline ready",
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

        if let Err(DenoiserError::QueueFull) = wd.push(&planes) {
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

fn write_planes(
    encoder: &mut y4m::Encoder<std::io::StdoutLock<'_>>,
    planes: &Planes,
) -> Result<(), anyhow::Error> {
    let frame = y4m::Frame::new([&planes.y, &planes.u, &planes.v], None);
    encoder.write_frame(&frame)?;

    Ok(())
}
