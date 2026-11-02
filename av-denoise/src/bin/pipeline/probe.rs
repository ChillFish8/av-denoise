use std::ffi::{CStr, CString};
use std::path::Path;
use std::ptr;

use av_decoders::Rational32;
use ffmpeg_sys_the_third::{
    AV_DISPOSITION_ATTACHED_PIC,
    AVFormatContext,
    AVMediaType,
    AVRational,
    AVStream,
    avformat_close_input,
    avformat_find_stream_info,
    avformat_open_input,
};

/// Reads the nominal frame rate of the first video track in the file at `path`.
///
/// This is the track's `r_frame_rate`, or its `avg_frame_rate` when that is unset, which is what
/// ffprobe reports. Returns `None` when the file cannot be read, holds no video track or declares
/// no rate.
pub fn nominal_frame_rate(path: &Path) -> Option<Rational32> {
    let url = path.to_str()?;
    let url = CString::new(url).ok()?;
    let context = FormatContext::open(&url)?;

    let header_rate = context.first_video_rates().and_then(nominal_from);
    if header_rate.is_some() {
        return header_rate;
    }

    // Containers such as MPEG-TS only learn their rates by reading packets.
    context.find_stream_info()?;

    let rates = context.first_video_rates()?;
    nominal_from(rates)
}

/// Picks `r_frame_rate`, falling back to `avg_frame_rate`.
fn nominal_from((real_rate, average_rate): (AVRational, AVRational)) -> Option<Rational32> {
    let nominal = rate_from(real_rate);

    nominal.or_else(|| rate_from(average_rate))
}

fn rate_from(rational: AVRational) -> Option<Rational32> {
    if rational.num <= 0 || rational.den <= 0 {
        return None;
    }

    let rate = Rational32::new(rational.num, rational.den);

    Some(rate)
}

/// An open libavformat context, closed on drop.
struct FormatContext {
    raw: *mut AVFormatContext,
}

impl FormatContext {
    fn open(url: &CStr) -> Option<Self> {
        let mut raw = ptr::null_mut();

        // SAFETY: `raw` starts null, so libavformat allocates the context itself. `url` is a valid
        // C string that outlives the call, and a null format and options ask libavformat to detect
        // the container with its defaults. On failure it frees the context and leaves `raw` null.
        let status = unsafe { avformat_open_input(&mut raw, url.as_ptr(), ptr::null(), ptr::null_mut()) };

        if status < 0 || raw.is_null() {
            return None;
        }

        Some(Self { raw })
    }

    /// Reads packets to fill in what the container header left out.
    fn find_stream_info(&self) -> Option<()> {
        // SAFETY: `raw` is a context `avformat_open_input` opened and only `Drop` closes. Null
        // options are allowed.
        let status = unsafe { avformat_find_stream_info(self.raw, ptr::null_mut()) };

        if status < 0 {
            return None;
        }

        Some(())
    }

    /// The `r_frame_rate` and `avg_frame_rate` of the first video track, the one ffms2 decodes.
    fn first_video_rates(&self) -> Option<(AVRational, AVRational)> {
        // SAFETY: `raw` is open until `Drop`, and no other reference to the context is alive.
        let context = unsafe { &*self.raw };
        let stream_count = context.nb_streams as usize;

        for index in 0..stream_count {
            // SAFETY: `index` is below `nb_streams`, so the slot holds a valid stream pointer.
            let stream_ptr = unsafe { *context.streams.add(index) };

            // SAFETY: the stream is owned by the open context and lives until `Drop`.
            let stream = unsafe { &*stream_ptr };

            if is_video_track(stream) {
                return Some((stream.r_frame_rate, stream.avg_frame_rate));
            }
        }

        None
    }
}

/// Whether `stream` is a video track, rather than another kind or a cover image.
fn is_video_track(stream: &AVStream) -> bool {
    let attached_picture = stream.disposition & AV_DISPOSITION_ATTACHED_PIC != 0;
    if attached_picture || stream.codecpar.is_null() {
        return false;
    }

    // SAFETY: checked non-null just above, and the parameters live as long as their stream.
    let parameters = unsafe { &*stream.codecpar };

    parameters.codec_type == AVMediaType::VIDEO
}

impl Drop for FormatContext {
    fn drop(&mut self) {
        // SAFETY: `raw` came from a successful `avformat_open_input` and is closed only here.
        unsafe { avformat_close_input(&mut self.raw) };
    }
}
