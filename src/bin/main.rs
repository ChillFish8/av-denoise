use av_denoise::source::stdin::StdInInput;
use clap::Parser;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(Debug, Parser)]
struct Args {
    #[command(flatten)]
    opts: av_denoise::Options,
    #[arg(long)]
    /// The width of each source frame before stacking.
    width: usize,
    #[arg(long)]
    /// The height of each source frame before stacking.
    height: usize,
    #[arg(long)]
    /// Expect HDR input as yuv444p10le instead of 8-bit yuv444p.
    hdr: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    if std::env::var("RUST_LOG").is_err() {
        unsafe { std::env::set_var("RUST_LOG", "info,wgpu=warn,wgpu_hal=warn") };
    }
    tracing_subscriber::fmt::init();

    let input = StdInInput::new(args.width, args.height, args.hdr);
    av_denoise::run(args.opts, input)?;

    Ok(())
}
