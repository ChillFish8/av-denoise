use std::path::{Path, PathBuf};
use std::process::Command;

use av_decoders::Rational32;
use av_denoise::ChromaSiting;
use avtbx_bestsource::IndexCache;

use super::{data_clip, silent_wav};
use crate::pipeline::bestsource::{BestSourceReader, open_bestsource, pixel_aspect_from_sar};
use crate::pipeline::convert::planes_from_v_frame_u16;
use crate::pipeline::source::SourceInfo;

/// Builds an ffv1 clip with ffmpeg, or `None` when ffmpeg cannot.
pub(super) fn ffmpeg_clip(
    directory: &Path,
    name: &str,
    size: &str,
    pix_fmt: &str,
    extra: &[&str],
) -> Option<PathBuf> {
    let path = directory.join(name);
    let source = format!("testsrc2=size={size}:rate=25");
    let status = Command::new("ffmpeg")
        .args(["-v", "error", "-y", "-f", "lavfi", "-i", &source])
        .args(["-frames:v", "10"])
        .args(extra)
        .args(["-pix_fmt", pix_fmt, "-c:v", "ffv1"])
        .arg(&path)
        .status();

    let built = matches!(status, Ok(status) if status.success());
    if !built {
        eprintln!("skipping, ffmpeg could not build {name}");
        return None;
    }

    Some(path)
}

fn open_data(name: &str) -> Option<(BestSourceReader, SourceInfo)> {
    let path = data_clip(name)?;
    let opened = open_bestsource(&path, IndexCache::Disabled).expect("the clip opens");

    Some(opened)
}

fn open_error(path: &Path) -> String {
    let result = open_bestsource(path, IndexCache::Disabled);
    let Err(error) = result else {
        panic!("{} must be rejected", path.display());
    };

    error.to_string()
}

#[test]
fn a_left_sited_file_reports_left() {
    let Some((_, info)) = open_data("patema-problem-clip-raw.mkv") else {
        return;
    };

    assert_eq!(info.layout.chroma_siting, ChromaSiting::Left);
}

#[test]
fn a_centre_sited_file_reports_centre() {
    let Some((_, info)) = open_data("food-problem-clip2-raw.mkv") else {
        return;
    };

    assert_eq!(info.layout.chroma_siting, ChromaSiting::Center);
}

#[test]
fn a_file_takes_its_container_frame_rate() {
    let clips = [
        "apothecary-clip.mkv",
        "food-problem-clip2-raw.mkv",
        "patema-problem-clip3-raw.mkv",
    ];
    let ntsc_film = Rational32::new(24000, 1001);

    for clip in clips {
        let Some((_, info)) = open_data(clip) else {
            continue;
        };

        assert_eq!(info.details.frame_rate, ntsc_film, "{clip}");
    }
}

#[test]
fn estimated_frames_is_the_source_frame_count() {
    let Some((_, info)) = open_data("delie-sample-source.mkv") else {
        return;
    };

    assert_eq!(info.estimated_frames, Some(143));
}

#[test]
fn reading_stops_after_the_last_frame() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let Some(path) = ffmpeg_clip(directory.path(), "ten.mkv", "64x48", "yuv420p", &[]) else {
        return;
    };
    let (mut reader, _) = open_bestsource(&path, IndexCache::Disabled).expect("the clip opens");

    let mut count = 0;
    while let Some(frame) = reader.read_frame::<u8>() {
        frame.expect("the frame decodes");
        count += 1;
    }

    assert_eq!(count, 10);
}

#[test]
fn a_ten_bit_file_matches_ffmpeg_raw_bytes() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let Some(path) = ffmpeg_clip(directory.path(), "ten-bit.mkv", "64x48", "yuv420p10le", &[]) else {
        return;
    };
    let raw = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(&path)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p10le", "-frames:v", "1", "-"])
        .output()
        .expect("ffmpeg runs");
    assert!(raw.status.success(), "ffmpeg must decode the clip");

    let (mut reader, info) = open_bestsource(&path, IndexCache::Disabled).expect("the clip opens");
    let frame = reader
        .read_frame::<u16>()
        .expect("the clip holds a frame")
        .expect("the frame decodes");
    let planes = planes_from_v_frame_u16(&frame, info.layout).expect("the frame matches its layout");

    let luma_bytes = 64 * 48 * 2;
    let chroma_bytes = 32 * 24 * 2;
    let (expected_y, rest) = raw.stdout.split_at(luma_bytes);
    let (expected_u, expected_v) = rest.split_at(chroma_bytes);

    assert_eq!(planes.y, expected_y);
    assert_eq!(planes.u, expected_u);
    assert_eq!(planes.v, expected_v);
}

#[test]
fn an_odd_width_420_file_is_rejected() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let Some(path) = ffmpeg_clip(
        directory.path(),
        "odd.mkv",
        "96x64",
        "yuv420p",
        &["-vf", "scale=65:48"],
    ) else {
        return;
    };

    let message = open_error(&path);

    assert!(message.contains("subsampling"), "got {message}");
}

#[test]
fn a_gray_file_is_rejected() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let Some(path) = ffmpeg_clip(directory.path(), "gray.mkv", "64x48", "gray", &[]) else {
        return;
    };

    let message = open_error(&path);

    assert!(!message.is_empty());
}

#[test]
fn an_alpha_file_is_rejected() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let Some(path) = ffmpeg_clip(directory.path(), "alpha.mkv", "64x48", "yuva420p", &[]) else {
        return;
    };

    let message = open_error(&path);

    assert!(message.contains("alpha"), "got {message}");
}

#[test]
fn an_rgb_file_is_rejected() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let Some(path) = ffmpeg_clip(directory.path(), "rgb.mkv", "64x48", "gbrp", &[]) else {
        return;
    };

    let message = open_error(&path);

    assert!(!message.is_empty());
}

#[test]
fn a_file_with_no_video_track_is_an_error() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let audio_only = directory.path().join("silence.wav");
    let wav = silent_wav();
    std::fs::write(&audio_only, wav).expect("the WAV file writes");

    let message = open_error(&audio_only);

    assert!(!message.is_empty());
}

#[cfg(unix)]
#[test]
fn a_non_utf8_path_is_an_error() {
    use std::ffi::OsStr;
    use std::os::unix::ffi::OsStrExt;

    let name = OsStr::from_bytes(b"\xff.mkv");
    let path = Path::new(name);

    let message = open_error(path);

    assert!(message.contains("utf-8"), "got {message}");
}

#[cfg(unix)]
#[test]
fn a_read_only_source_directory_still_opens() {
    use std::os::unix::fs::PermissionsExt;

    let build_directory = tempfile::tempdir().expect("a temporary directory");
    let Some(clip) = ffmpeg_clip(build_directory.path(), "ten.mkv", "64x48", "yuv420p", &[]) else {
        return;
    };
    let read_only = tempfile::tempdir().expect("a temporary directory");
    let path = read_only.path().join("ten.mkv");
    std::fs::copy(&clip, &path).expect("the clip copies");
    let permissions = std::fs::Permissions::from_mode(0o555);
    std::fs::set_permissions(read_only.path(), permissions).expect("the directory locks");

    let result = open_bestsource(&path, IndexCache::BesideSource);

    let restored = std::fs::Permissions::from_mode(0o755);
    std::fs::set_permissions(read_only.path(), restored).expect("the directory unlocks");
    if let Err(error) = result {
        panic!("a read-only directory must still open: {error}");
    }
}

#[test]
fn the_sar_becomes_the_pixel_aspect() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let Some(path) = ffmpeg_clip(
        directory.path(),
        "sar.mkv",
        "64x48",
        "yuv420p",
        &["-vf", "setsar=4/3"],
    ) else {
        return;
    };
    let (_, info) = open_bestsource(&path, IndexCache::Disabled).expect("the clip opens");

    let aspect = info.pixel_aspect.expect("the clip declares a SAR");

    assert_eq!((aspect.num, aspect.den), (4, 3));
}

#[test]
fn a_full_range_file_tags_full_range() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let Some(path) = ffmpeg_clip(
        directory.path(),
        "full.mkv",
        "64x48",
        "yuv420p",
        &["-color_range", "pc"],
    ) else {
        return;
    };
    let (_, info) = open_bestsource(&path, IndexCache::Disabled).expect("the clip opens");

    let tags: Vec<&[u8]> = info
        .vendor_extensions
        .iter()
        .map(|extension| extension.value())
        .collect();

    assert_eq!(tags, vec![b"COLORRANGE=FULL".as_slice()]);
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

fn bsindex_files(directory: &Path) -> Vec<PathBuf> {
    let entries = std::fs::read_dir(directory).expect("the directory lists");

    entries
        .map(|entry| entry.expect("an entry reads").path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "bsindex"))
        .collect()
}

#[test]
fn an_index_is_written_beside_the_source() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let Some(path) = ffmpeg_clip(directory.path(), "ten.mkv", "64x48", "yuv420p", &[]) else {
        return;
    };

    let opened = open_bestsource(&path, IndexCache::BesideSource).expect("the clip opens");

    assert!(!bsindex_files(directory.path()).is_empty());
    drop(opened);
}

#[test]
fn no_index_is_written_when_disabled() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let Some(path) = ffmpeg_clip(directory.path(), "ten.mkv", "64x48", "yuv420p", &[]) else {
        return;
    };

    let opened = open_bestsource(&path, IndexCache::Disabled).expect("the clip opens");

    let entries = std::fs::read_dir(directory.path())
        .expect("the directory lists")
        .count();
    assert_eq!(entries, 1);
    drop(opened);
}

#[test]
fn a_replaced_source_is_reindexed() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let Some(first) = ffmpeg_clip(directory.path(), "first.mkv", "64x48", "yuv420p", &[]) else {
        return;
    };

    let Some(second) = ffmpeg_clip(
        directory.path(),
        "second.mkv",
        "64x48",
        "yuv420p",
        &["-frames:v", "20"],
    ) else {
        return;
    };

    let path = directory.path().join("clip.mkv");
    std::fs::copy(&first, &path).expect("the first clip copies");

    let (_, first_info) = open_bestsource(&path, IndexCache::BesideSource).expect("the clip opens");
    assert_eq!(first_info.estimated_frames, Some(10));

    std::fs::copy(&second, &path).expect("the second clip replaces the first");
    let (_, second_info) = open_bestsource(&path, IndexCache::BesideSource).expect("the clip reopens");

    assert_eq!(second_info.estimated_frames, Some(20));
}
