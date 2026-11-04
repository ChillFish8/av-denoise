use std::num::{NonZeroU8, NonZeroUsize};
use std::path::{Path, PathBuf};

use av_decoders::{Rational32, VideoDetails};
use avtbx_bestsource::{
    ColorFamily,
    IndexCache,
    OpenOptions,
    PlanarTarget,
    PlaneTarget,
    VideoProperties,
    VideoSource,
};
use v_frame::chroma::ChromaSubsampling;
use v_frame::frame::{Frame, FrameBuilder};
use v_frame::plane::Plane;

use super::convert::SourcePixel;
use super::source::{SourceInfo, color_range_extension, layout_from_details};
use crate::y4m_format::siting_from_chroma_location;

/// Planes in a YUV frame.
const PLANE_COUNT: usize = 3;

/// Samples of padding on each side of the luma plane, which scene detection needs.
const LUMA_PADDING: usize = 88;

/// The size and sample format of every frame in the track.
#[derive(Clone, Copy)]
struct FrameShape {
    width: NonZeroUsize,
    height: NonZeroUsize,
    subsampling: ChromaSubsampling,
    bit_depth: NonZeroU8,
    bytes_per_sample: usize,
}

impl FrameShape {
    /// The width and height in samples of plane `index`.
    fn plane_size(&self, index: usize) -> (usize, usize) {
        let width = self.width.get();
        let height = self.height.get();
        if index == 0 {
            return (width, height);
        }

        match self.subsampling {
            ChromaSubsampling::Yuv420 => (width / 2, height / 2),
            ChromaSubsampling::Yuv422 => (width / 2, height),
            _ => (width, height),
        }
    }

    fn row_bytes(&self, index: usize) -> usize {
        let (width, _) = self.plane_size(index);

        width * self.bytes_per_sample
    }
}

/// Decodes the frames of one video file in order through BestSource.
pub struct BestSourceReader {
    source: VideoSource,
    next_frame: i64,
    num_frames: i64,
    shape: FrameShape,
    scratch: Vec<Vec<u8>>,
}

/// Opens the first video track of `path`, indexing it or reading the index kept in `cache`.
pub fn open_bestsource(
    path: &Path,
    cache: IndexCache,
) -> Result<(BestSourceReader, SourceInfo), anyhow::Error> {
    let mut source = open_video_source(path, cache)?;
    let properties = *source.properties();

    let shape = frame_shape(&properties)?;
    let details = video_details(&properties, shape)?;
    let mut layout = layout_from_details(&details)?;

    let first_frame = source.frame(0)?;
    let chroma_location = first_frame.chroma_location();
    let color_range = first_frame.range();
    layout.chroma_siting = siting_from_chroma_location(chroma_location);
    let range = color_range_extension(color_range);
    let vendor_extensions = range.into_iter().collect();
    let pixel_aspect = pixel_aspect_from_sar(properties.sar.0, properties.sar.1);
    let estimated_frames = usize::try_from(properties.num_frames).ok();

    let info = SourceInfo {
        details,
        layout,
        pixel_aspect,
        vendor_extensions,
        estimated_frames,
    };

    let scratch = (0..PLANE_COUNT)
        .map(|index| {
            let (_, height) = shape.plane_size(index);
            let plane_bytes = shape.row_bytes(index) * height;

            vec![0u8; plane_bytes]
        })
        .collect();
    let reader = BestSourceReader {
        source,
        next_frame: 0,
        num_frames: properties.num_frames,
        shape,
        scratch,
    };

    Ok((reader, info))
}

/// Turns a sample aspect ratio into a y4m pixel aspect.
///
/// An unset ratio is reported as 0, so anything but a positive pair gives none.
pub fn pixel_aspect_from_sar(numerator: i32, denominator: i32) -> Option<y4m::Ratio> {
    if numerator <= 0 || denominator <= 0 {
        return None;
    }

    let ratio = y4m::Ratio::new(numerator as usize, denominator as usize);

    Some(ratio)
}

/// Opens the track, indexing without a cache when the index location is not writable.
///
/// BestSource only fails a read-only index location after indexing, so the location is checked first.
fn open_video_source(path: &Path, cache: IndexCache) -> Result<VideoSource, anyhow::Error> {
    let cache = usable_cache(path, cache);
    let options = OpenOptions {
        cache,
        ..OpenOptions::default()
    };
    let source = VideoSource::open(path, &options)?;

    Ok(source)
}

/// Returns `cache`, or a disabled cache when its directory exists but cannot be written to.
fn usable_cache(path: &Path, cache: IndexCache) -> IndexCache {
    let directory = match &cache {
        IndexCache::Disabled => return cache,
        IndexCache::BesideSource => path.parent().map(Path::to_path_buf).unwrap_or_default(),
        IndexCache::Directory(directory) => directory.clone(),
    };
    let directory = if directory.as_os_str().is_empty() {
        PathBuf::from(".")
    } else {
        directory
    };

    // A missing directory is left for BestSource to report against the right path.
    if !directory.is_dir() || directory_is_writable(&directory) {
        return cache;
    }

    tracing::warn!(
        directory = %directory.display(),
        "the frame index location is not writable, indexing without a cache",
    );

    IndexCache::Disabled
}

fn directory_is_writable(directory: &Path) -> bool {
    let name = format!(".avd-write-probe-{}", std::process::id());
    let probe = directory.join(name);
    let created = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&probe);
    if created.is_err() {
        return false;
    }

    let _ = std::fs::remove_file(&probe);

    true
}

impl BestSourceReader {
    /// Decodes the next frame, or returns `None` once every frame has been read.
    pub fn read_frame<T: SourcePixel>(&mut self) -> Option<Result<Frame<T>, anyhow::Error>> {
        if self.next_frame >= self.num_frames {
            return None;
        }

        let number = self.next_frame;
        self.next_frame += 1;

        Some(self.decode_frame(number))
    }

    fn decode_frame<T: SourcePixel>(&mut self, number: i64) -> Result<Frame<T>, anyhow::Error> {
        let decoded = self.source.frame(number)?;
        let shape = self.shape;

        let decoded_width = usize::try_from(decoded.ss_mod_width()).ok();
        let decoded_height = usize::try_from(decoded.ss_mod_height()).ok();
        let width_matches = decoded_width == Some(shape.width.get());
        let height_matches = decoded_height == Some(shape.height.get());
        if !width_matches || !height_matches {
            anyhow::bail!("frame {number} changes size part way through the file");
        }

        let planes: Vec<PlaneTarget> = self
            .scratch
            .iter_mut()
            .enumerate()
            .map(|(index, data)| PlaneTarget {
                data,
                stride: shape.row_bytes(index),
            })
            .collect();
        let mut target = PlanarTarget { planes, alpha: None };
        decoded.export_planar(&mut target)?;

        let mut frame = FrameBuilder::new(shape.width, shape.height, shape.subsampling, shape.bit_depth)
            .luma_padding_top(LUMA_PADDING)
            .luma_padding_bottom(LUMA_PADDING)
            .luma_padding_left(LUMA_PADDING)
            .luma_padding_right(LUMA_PADDING)
            .build::<T>()?;

        let y_plane = Some(&mut frame.y_plane);
        copy_plane(y_plane, &self.scratch[0], shape.row_bytes(0))?;
        let u_plane = frame.u_plane.as_mut();
        copy_plane(u_plane, &self.scratch[1], shape.row_bytes(1))?;
        let v_plane = frame.v_plane.as_mut();
        copy_plane(v_plane, &self.scratch[2], shape.row_bytes(2))?;

        Ok(frame)
    }
}

fn copy_plane<T: SourcePixel>(
    plane: Option<&mut Plane<T>>,
    bytes: &[u8],
    row_bytes: usize,
) -> Result<(), anyhow::Error> {
    let Some(plane) = plane else {
        anyhow::bail!("the frame has no chroma plane to fill");
    };
    let Some(stride) = NonZeroUsize::new(row_bytes) else {
        anyhow::bail!("the frame has an empty row");
    };

    plane.copy_from_u8_slice_with_stride(bytes, stride)?;

    Ok(())
}

/// Checks the track holds planar YUV the pipeline can take, before any frame is decoded.
fn frame_shape(properties: &VideoProperties) -> Result<FrameShape, anyhow::Error> {
    let format = properties.format;
    if format.color_family != ColorFamily::Yuv || format.float {
        anyhow::bail!(
            "only integer YUV video is supported, but the file is {:?}",
            format.color_family
        );
    }

    if format.alpha {
        anyhow::bail!("video with an alpha plane is not supported");
    }

    let subsampling = match (format.sub_sampling_w, format.sub_sampling_h) {
        (0, 0) => ChromaSubsampling::Yuv444,
        (1, 0) => ChromaSubsampling::Yuv422,
        (1, 1) => ChromaSubsampling::Yuv420,
        (width_shift, height_shift) => {
            anyhow::bail!(
                "unsupported chroma subsampling {width_shift}:{height_shift}, need 4:2:0, 4:2:2, or 4:4:4"
            )
        },
    };

    if properties.width != properties.ss_mod_width || properties.height != properties.ss_mod_height {
        anyhow::bail!(
            "a {}x{} frame is not a whole multiple of its chroma subsampling, the nearest is {}x{}",
            properties.width,
            properties.height,
            properties.ss_mod_width,
            properties.ss_mod_height,
        );
    }

    if properties.num_frames <= 0 {
        anyhow::bail!("the file holds no video frames");
    }

    let raw_width = usize::try_from(properties.width)?;
    let raw_height = usize::try_from(properties.height)?;
    let bits = u8::try_from(format.bits)?;
    let width = NonZeroUsize::new(raw_width).ok_or_else(|| anyhow::anyhow!("the video has no width"))?;
    let height = NonZeroUsize::new(raw_height).ok_or_else(|| anyhow::anyhow!("the video has no height"))?;
    let bit_depth = NonZeroU8::new(bits).ok_or_else(|| anyhow::anyhow!("the video has no bit depth"))?;
    let shape = FrameShape {
        width,
        height,
        subsampling,
        bit_depth,
        bytes_per_sample: format.bytes_per_sample(),
    };

    Ok(shape)
}

fn video_details(properties: &VideoProperties, shape: FrameShape) -> Result<VideoDetails, anyhow::Error> {
    let (rate_numerator, rate_denominator) = properties.fps;
    if rate_numerator <= 0 || rate_denominator <= 0 {
        anyhow::bail!("the video declares no frame rate");
    }

    let numerator = i32::try_from(rate_numerator)?;
    let denominator = i32::try_from(rate_denominator)?;
    let total_frames = usize::try_from(properties.num_frames)?;

    Ok(VideoDetails {
        width: shape.width.get(),
        height: shape.height.get(),
        bit_depth: usize::from(shape.bit_depth.get()),
        chroma_sampling: shape.subsampling,
        frame_rate: Rational32::new(numerator, denominator),
        total_frames: Some(total_frames),
    })
}
