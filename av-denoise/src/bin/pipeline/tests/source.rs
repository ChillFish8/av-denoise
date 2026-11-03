use std::path::{Path, PathBuf};
use std::process::Command;

use av_decoders::Rational32;
use av_denoise::{ChromaSiting, Depth, Subsampling};

use super::{data_clip, open_data_clip, silent_wav, y4m_reader};
use crate::pipeline::convert::SourcePixel;
use crate::pipeline::probe::nominal_frame_rate;
use crate::pipeline::source::{
    color_range_extension,
    file_frame_rate,
    open_file,
    open_y4m,
    pixel_aspect_from_sar,
};

fn y4m_bytes(colorspace: y4m::Colorspace, extension: Option<&str>, frames: usize) -> Vec<u8> {
    let mut bytes = Vec::new();
    let frame_rate = y4m::Ratio::new(25, 1);
    let pixel_aspect = y4m::Ratio::new(4, 3);
    let mut builder = y4m::encode(4, 4, frame_rate)
        .with_colorspace(colorspace)
        .with_pixel_aspect(pixel_aspect);

    if let Some(extension) = extension {
        let value = extension.as_bytes().to_vec();
        let vendor = y4m::VendorExtensionString::new(value).expect("the extension has no spaces");
        builder = builder.append_vendor_extension(vendor);
    }

    let mut encoder = builder.write_header(&mut bytes).expect("header should write");
    let sample_bytes = colorspace.get_bytes_per_sample();
    let luma = vec![0u8; 16 * sample_bytes];
    let chroma_samples = match colorspace {
        y4m::Colorspace::C422p10 => 8,
        _ => 4,
    };
    let chroma = vec![0u8; chroma_samples * sample_bytes];

    for _ in 0..frames {
        let frame = y4m::Frame::new([&luma, &chroma, &chroma], None);
        encoder.write_frame(&frame).expect("frame should write");
    }

    bytes
}

#[test]
fn a_pipe_keeps_its_vendor_extensions_and_pixel_aspect() {
    let bytes = y4m_bytes(y4m::Colorspace::C420, Some("COLORRANGE=LIMITED"), 1);
    let source = y4m_reader(bytes);
    let opened = open_y4m(source).expect("a 4:2:0 pipe opens");

    let extensions: Vec<&[u8]> = opened
        .info
        .vendor_extensions
        .iter()
        .map(|extension| extension.value())
        .collect();
    let aspect = opened.info.pixel_aspect.expect("a pipe carries its pixel aspect");

    assert_eq!(extensions, vec![b"COLORRANGE=LIMITED".as_slice()]);
    assert_eq!((aspect.num, aspect.den), (4, 3));
}

#[test]
fn a_ten_bit_pipe_reports_its_layout() {
    let bytes = y4m_bytes(y4m::Colorspace::C422p10, None, 1);
    let source = y4m_reader(bytes);
    let opened = open_y4m(source).expect("a 10-bit 4:2:2 pipe opens");

    assert_eq!(opened.info.layout.width, 4);
    assert_eq!(opened.info.layout.subsampling, Subsampling::Yuv422);
    assert_eq!(opened.info.layout.depth, Depth::Ten);
}

#[test]
fn ten_bit_stream_round_trips_header_and_plane_sizes() {
    let bytes = y4m_bytes(y4m::Colorspace::C420p10, None, 2);
    let source = y4m_reader(bytes);
    let mut opened = open_y4m(source).expect("a 10-bit 4:2:0 pipe opens");
    let layout = opened.info.layout;

    assert_eq!(layout.subsampling, Subsampling::Yuv420);
    assert_eq!(layout.depth, Depth::Ten);

    let frame = opened
        .decoder
        .read_video_frame::<u16>()
        .expect("the pipe holds a frame");
    let planes = u16::to_planes(&frame, layout).expect("the frame matches its layout");

    assert_eq!(planes.y.len(), 4 * 4 * 2);
    assert_eq!(planes.u.len(), 2 * 2 * 2);
    assert_eq!(planes.v.len(), 2 * 2 * 2);
}

#[test]
fn a_pipe_has_no_phantom_frames_or_frame_estimate() {
    let bytes = y4m_bytes(y4m::Colorspace::C420, None, 3);
    let source = y4m_reader(bytes);
    let opened = open_y4m(source).expect("a 4:2:0 pipe opens");

    assert!(opened.phantom.is_empty());
    assert_eq!(opened.info.estimated_frames, None);
}

#[test]
fn a_mono_pipe_is_rejected_without_panicking() {
    let mut bytes = Vec::new();
    let frame_rate = y4m::Ratio::new(25, 1);
    y4m::encode(4, 4, frame_rate)
        .with_colorspace(y4m::Colorspace::Cmono)
        .write_header(&mut bytes)
        .expect("header should write");

    let source = y4m_reader(bytes);
    let result = open_y4m(source);
    let err = match result {
        Ok(_) => panic!("a mono pipe must be rejected"),
        Err(err) => err,
    };

    assert!(err.to_string().contains("Cmono"), "got {err}");
}

#[test]
fn a_limited_range_maps_to_the_limited_tag() {
    let extension = color_range_extension(1).expect("MPEG range is tagged");

    assert_eq!(extension.value(), b"COLORRANGE=LIMITED");
}

#[test]
fn a_full_range_maps_to_the_full_tag() {
    let extension = color_range_extension(2).expect("JPEG range is tagged");

    assert_eq!(extension.value(), b"COLORRANGE=FULL");
}

#[test]
fn an_unspecified_or_unknown_range_adds_no_tag() {
    for range in [0, 3, -1] {
        let extension = color_range_extension(range);

        assert!(extension.is_none(), "range {range} must add no tag");
    }
}

#[test]
fn a_positive_sar_becomes_the_pixel_aspect() {
    let aspect = pixel_aspect_from_sar(32, 27).expect("32:27 is a valid SAR");

    assert_eq!((aspect.num, aspect.den), (32, 27));
}

#[test]
fn an_unset_or_invalid_sar_gives_no_pixel_aspect() {
    for (numerator, denominator) in [(0, 0), (0, 1), (1, 0), (-4, 3)] {
        let aspect = pixel_aspect_from_sar(numerator, denominator);

        assert!(aspect.is_none(), "SAR {numerator}:{denominator} must give none");
    }
}

#[test]
fn piped_y4m_siting_round_trips() {
    let cases = [
        (y4m::Colorspace::C420jpeg, ChromaSiting::Center),
        (y4m::Colorspace::C420mpeg2, ChromaSiting::Left),
        (y4m::Colorspace::C420paldv, ChromaSiting::TopLeft),
        (y4m::Colorspace::C420, ChromaSiting::Unknown),
    ];

    for (colorspace, expected) in cases {
        let bytes = y4m_bytes(colorspace, None, 1);
        let source = y4m_reader(bytes);
        let opened = open_y4m(source).expect("a 4:2:0 pipe opens");

        assert_eq!(
            opened.info.layout.chroma_siting, expected,
            "wrong siting for {colorspace:?}"
        );
    }
}

#[test]
fn a_left_sited_file_source_reports_left() {
    let Some(opened) = open_data_clip("patema-problem-clip-raw.mkv") else {
        return;
    };

    assert_eq!(opened.info.layout.chroma_siting, ChromaSiting::Left);
}

#[test]
fn a_centre_sited_file_source_reports_centre() {
    let Some(opened) = open_data_clip("food-problem-clip2-raw.mkv") else {
        return;
    };

    assert_eq!(opened.info.layout.chroma_siting, ChromaSiting::Center);
}

#[test]
fn a_file_source_takes_the_frame_rate_its_container_declares() {
    let clips = [
        "apothecary-clip.mkv",
        "food-problem-clip2-raw.mkv",
        "patema-problem-clip3-raw.mkv",
    ];
    let ntsc_film = Rational32::new(24000, 1001);

    for clip in clips {
        let Some(opened) = open_data_clip(clip) else {
            continue;
        };

        assert_eq!(opened.info.details.frame_rate, ntsc_film, "{clip}");
    }
}

#[test]
fn an_unreadable_path_has_no_nominal_frame_rate() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let missing = directory.path().join("missing.mkv");

    assert_eq!(nominal_frame_rate(&missing), None);
}

#[test]
fn a_file_with_no_video_track_has_no_nominal_frame_rate() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let audio_only = directory.path().join("silence.wav");
    let wav = silent_wav();
    std::fs::write(&audio_only, wav).expect("the WAV file writes");

    assert_eq!(nominal_frame_rate(&audio_only), None);
}

#[test]
fn a_failed_probe_keeps_the_measured_frame_rate() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let missing = directory.path().join("missing.mkv");
    let measured = Rational32::new(10849, 453);

    assert_eq!(file_frame_rate(&missing, measured), measured);
}

/// Builds a clip with ffmpeg that runs 300 frames at 29.97 fps, then 100 at 23.976 fps.
///
/// Its container declares 30 fps while it averages about 28.24 fps. The millisecond time bases keep
/// ffmpeg from snapping the 23.976 fps frames onto a 30 fps grid with frames dropped.
fn variable_rate_clip(directory: &Path) -> Option<PathBuf> {
    let path = directory.join("variable-rate.mkv");
    let timestamps = "settb=1/1000,setpts='if(lt(N,300),N/29.97,300/29.97+(N-300)/23.976)/TB'";
    let status = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x64:rate=30",
        ])
        .args(["-frames:v", "400", "-vf", timestamps, "-fps_mode", "passthrough"])
        .args(["-enc_time_base", "1/1000", "-pix_fmt", "yuv420p", "-c:v", "ffv1"])
        .arg(&path)
        .status();

    let built = matches!(status, Ok(status) if status.success());
    if !built {
        eprintln!("skipping, ffmpeg could not build the variable frame rate clip");
        return None;
    }

    Some(path)
}

#[test]
fn a_variable_rate_file_is_rejected() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let Some(path) = variable_rate_clip(directory.path()) else {
        return;
    };

    let declared = nominal_frame_rate(&path);
    let result = open_file(&path);
    let Err(error) = result else {
        panic!("a variable frame rate file must be rejected");
    };
    let message = error.to_string();

    assert_eq!(declared, Some(Rational32::new(30, 1)));
    assert!(message.contains("variable frame rate"), "got {message}");
    assert!(message.contains("averages 28.2"), "got {message}");
    assert!(message.contains("declared 30.000 fps"), "got {message}");
    assert!(message.contains("-fps_mode cfr -r 30/1"), "got {message}");
}

#[cfg(unix)]
#[test]
fn constant_rate_clips_pass_the_frame_rate_check() {
    let clips = [
        "apothecary-clip.mkv",
        "food-problem-clip2-raw.mkv",
        "colour_grain_2026_11/door_source.mkv",
        "colour_grain_2026_11/redwall_source.mkv",
        "colour_grain_2026_11/brick_source.mkv",
    ];

    // Each clip is opened through a link, so ffms2 writes its index here rather than beside the clip.
    let directory = tempfile::tempdir().expect("a temporary directory");

    for clip in clips {
        let Some(source) = data_clip(clip) else {
            continue;
        };

        let link_name = clip.replace('/', "_");
        let link = directory.path().join(link_name);
        std::os::unix::fs::symlink(&source, &link).expect("the link is created");

        let result = open_file(&link);
        if let Err(error) = result {
            panic!("{clip} was rejected: {error}");
        }
    }
}

#[test]
fn a_pipe_keeps_its_header_frame_rate() {
    let bytes = y4m_bytes(y4m::Colorspace::C420, None, 1);
    let source = y4m_reader(bytes);
    let opened = open_y4m(source).expect("a 4:2:0 pipe opens");

    assert_eq!(opened.info.details.frame_rate, Rational32::new(25, 1));
}
