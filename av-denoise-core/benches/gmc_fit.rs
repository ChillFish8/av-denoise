//! Prints the camera model fitted to each adjacent frame pair of a y4m clip
//!
//! It runs the pair model producer with the default nl4d search over the clip's luma plane. Each
//! line holds the pair index, the valid flag and the model as a row-major 2x3 matrix mapping an
//! older-frame position to the newer frame.
//!
//! ```text
//! cargo bench -p av-denoise-core --bench gmc_fit -- --device discrete:1 --input clip.y4m
//! ```

use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use av_denoise_core::bench_api::motion::{MODEL_WORDS, PairModelProducer};
use av_denoise_core::bench_api::{Device, Nl4dParams};
use av_denoise_core::{DevicePlane, SampleFormat, testing};
use clap::Parser;
use cubecl::prelude::*;

/// The model ring's depth, which only has to hold the pair just fitted.
const RING_DEPTH: u32 = 4;

/// Reads 8-bit luma planes as normalised samples, one frame per call.
struct LumaReader {
    decoder: y4m::Decoder<BufReader<File>>,
}

impl LumaReader {
    fn open(path: &Path) -> Self {
        let file = File::open(path).expect("the y4m opens");
        let reader = BufReader::new(file);
        let decoder = y4m::decode(reader).expect("the y4m header decodes");
        assert_eq!(decoder.get_bit_depth(), 8, "only 8-bit y4m is supported");

        Self { decoder }
    }

    fn dims(&self) -> (u32, u32) {
        let width = self.decoder.get_width() as u32;
        let height = self.decoder.get_height() as u32;

        (width, height)
    }

    fn next_luma(&mut self) -> Option<Vec<f32>> {
        let frame = match self.decoder.read_frame() {
            Ok(frame) => frame,
            Err(y4m::Error::EOF) => return None,
            Err(other) => panic!("the y4m frame decodes: {other:?}"),
        };

        let luma = frame
            .get_y_plane()
            .iter()
            .map(|sample| f32::from(*sample) / 255.0)
            .collect();
        Some(luma)
    }
}

/// Formats one ring slot's model as `valid a -b tx b a ty`.
fn model_line(words: &[f32]) -> String {
    let [a, b, tx, ty, valid] = words else {
        panic!("a model holds {MODEL_WORDS} words");
    };

    let flag = u32::from(*valid != 0.0);
    format!("{flag} {a:.8} {:.8} {tx:.8} {b:.8} {a:.8} {ty:.8}", -b)
}

#[derive(clap::Parser, Debug)]
#[command(about = "Prints the camera model fitted to each frame pair of a y4m clip", long_about = None)]
struct Cli {
    /// GPU device to bind to, one of `default`, `discrete[:N]`, `integrated[:N]`, `virtual[:N]` or `cpu`.
    #[arg(long, default_value = "default")]
    device: Device,

    /// The 8-bit y4m clip to fit.
    #[arg(long)]
    input: PathBuf,

    /// Swallowed, since cargo passes this when invoking the bench binary.
    #[arg(long, hide = true)]
    bench: bool,
}

fn main() {
    let cli = Cli::parse();

    let accelerator = testing::test_accelerator();
    let device = testing::device_on(accelerator, cli.device);
    eprintln!("device: {device:?}");
    let client = device.client();

    let mut reader = LumaReader::open(&cli.input);
    let (width, height) = reader.dims();
    let mode = Nl4dParams::default().nlm.motion_compensation;
    let built = PairModelProducer::new(&client, width, height, &mode, SampleFormat::F32, RING_DEPTH);
    let mut producer = built.expect("the default nl4d search builds a producer");

    println!("{width} {height}");

    let mut frame = 0u32;
    while let Some(luma) = reader.next_luma() {
        let bytes = f32::as_bytes(&luma);
        let handle = client.create_from_slice(bytes);
        let plane = DevicePlane::new(&handle, width, height);
        producer.push(&plane, frame).expect("the producer push succeeds");

        if frame > 0 {
            let ring = producer.ring();
            let ring_bytes = client
                .read_one(ring.handle().clone())
                .expect("the ring reads back");
            let ring_words = f32::from_bytes(&ring_bytes);
            let start = (ring.slot(frame) * MODEL_WORDS) as usize;
            let end = start + MODEL_WORDS as usize;
            let line = model_line(&ring_words[start..end]);
            println!("{} {line}", frame - 1);
        }

        frame += 1;
    }
}
