use std::cell::Cell;
use std::fs::File;
use std::io::{BufReader, Read};
use std::rc::Rc;

use anyhow::{Context, bail};

/// An 8-bit 4:2:0 y4m clip with planes in 0..=1.
pub(super) struct Y4mClip {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) frames: Vec<Y4mFrame>,
}

impl Y4mClip {
    /// The chroma planes' width and height, half the luma's rounded up.
    pub(super) fn chroma_dims(&self) -> (u32, u32) {
        let chroma_width = self.width.div_ceil(2);
        let chroma_height = self.height.div_ceil(2);

        (chroma_width, chroma_height)
    }
}

/// One frame's luma plane and its chroma planes interleaved as `u, v` per chroma pixel.
pub(super) struct Y4mFrame {
    pub(super) luma: Vec<f32>,
    pub(super) chroma: Vec<f32>,
}

/// Reads the y4m file at `path`, failing on anything but 8-bit 4:2:0.
pub(super) fn read_y4m(path: &str) -> Result<Y4mClip, anyhow::Error> {
    let file = File::open(path).with_context(|| format!("failed to open {path}"))?;
    let reader = BufReader::new(file);

    decode_y4m(reader)
}

/// A reader that counts every byte it hands out.
struct CountingReader<R: Read> {
    inner: R,
    consumed: Rc<Cell<u64>>,
}

impl<R: Read> Read for CountingReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let read = self.inner.read(buf)?;
        let consumed = self.consumed.get() + read as u64;
        self.consumed.set(consumed);

        Ok(read)
    }
}

/// Decodes a y4m stream, failing on anything but 8-bit 4:2:0 or on a stream that ends mid-frame.
fn decode_y4m(reader: impl Read) -> Result<Y4mClip, anyhow::Error> {
    let consumed = Rc::new(Cell::new(0u64));
    let counting = CountingReader {
        inner: reader,
        consumed: consumed.clone(),
    };
    let mut decoder = y4m::decode(counting).context("the y4m header decodes")?;
    let colour_space = decoder.get_colorspace();
    let bit_depth = decoder.get_bit_depth();
    let is_420 = matches!(
        colour_space,
        y4m::Colorspace::C420
            | y4m::Colorspace::C420jpeg
            | y4m::Colorspace::C420paldv
            | y4m::Colorspace::C420mpeg2
    );

    if !is_420 || bit_depth != 8 {
        bail!("unsupported y4m colour space {colour_space:?} at {bit_depth} bits, need 8-bit 4:2:0");
    }

    let width = decoder.get_width() as u32;
    let height = decoder.get_height() as u32;
    let mut frames = Vec::new();

    loop {
        let before = consumed.get();
        let frame = match decoder.read_frame() {
            Ok(frame) => frame,
            Err(y4m::Error::EOF) => {
                // The crate reports a cut-off frame as the end of the stream, so any byte read
                // since the last whole frame means the stream ended mid-frame.
                let partial = consumed.get() != before;
                if partial {
                    let frame_index = frames.len();
                    bail!("the y4m stream ends partway through frame {frame_index}");
                }

                break;
            },
            Err(other) => return Err(other.into()),
        };

        let luma = frame.get_y_plane().iter().map(|&code| unit(code)).collect();
        let u_plane = frame.get_u_plane();
        let v_plane = frame.get_v_plane();
        let mut chroma = Vec::with_capacity(2 * u_plane.len());
        for (&u_code, &v_code) in u_plane.iter().zip(v_plane) {
            chroma.push(unit(u_code));
            chroma.push(unit(v_code));
        }

        frames.push(Y4mFrame { luma, chroma });
    }

    Ok(Y4mClip {
        width,
        height,
        frames,
    })
}

fn unit(code: u8) -> f32 {
    f32::from(code) / 255.0
}

/// A one-frame 4x2 8-bit 4:2:0 stream whose header names `colour_space`, with none for `None`.
fn tiny_stream(colour_space: Option<&str>) -> Vec<u8> {
    let tag = colour_space.map(|name| format!(" C{name}")).unwrap_or_default();
    let header = format!("YUV4MPEG2 W4 H2 F25:1 Ip A1:1{tag}\nFRAME\n");
    let mut bytes = header.into_bytes();
    let samples: Vec<u8> = (0..12u8).map(|index| index * 5).collect();
    bytes.extend_from_slice(&samples);

    bytes
}

#[test]
fn a_y4m_header_with_another_colour_space_is_refused() {
    for colour_space in ["444", "420p10"] {
        let bytes = tiny_stream(Some(colour_space));
        let decoded = decode_y4m(bytes.as_slice());
        let error = decoded.err().expect("the colour space is refused");
        let message = error.to_string();
        let tag = format!("C{colour_space}");
        let names_tag = message.contains(&tag);
        assert!(names_tag, "the refusal of {tag} names it, got {message:?}");
    }

    for colour_space in [None, Some("420"), Some("420mpeg2")] {
        let bytes = tiny_stream(colour_space);
        let clip = decode_y4m(bytes.as_slice()).expect("an 8-bit 4:2:0 stream decodes");
        let frame_count = clip.frames.len();
        let frame = &clip.frames[0];
        let expected_luma: Vec<f32> = (0..8u8).map(|index| unit(index * 5)).collect();
        let expected_chroma: Vec<f32> = [40u8, 50, 45, 55].map(unit).to_vec();

        assert_eq!(frame_count, 1);
        assert_eq!((clip.width, clip.height), (4, 2));
        assert_eq!(frame.luma, expected_luma);
        assert_eq!(frame.chroma, expected_chroma);
    }
}

#[test]
fn a_y4m_stream_cut_off_mid_frame_is_refused() {
    let mut cut = tiny_stream(Some("420"));
    cut.pop();

    let mut trailing = tiny_stream(Some("420"));
    trailing.extend_from_slice(b"FRAME\n");

    for bytes in [cut, trailing] {
        let decoded = decode_y4m(bytes.as_slice());
        let error = decoded.err().expect("a cut-off stream is refused");
        let message = error.to_string();
        let names_cut = message.contains("ends partway through frame");
        assert!(names_cut, "the refusal names the cut, got {message:?}");
    }
}
