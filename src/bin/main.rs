use clap::Parser;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(Debug, Parser)]
struct Args {

}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    if std::env::var("RUST_LOG").is_err() {
        unsafe { std::env::set_var("RUST_LOG", "info,wgpu=warn,wgpu_hal=warn") };
    }
    tracing_subscriber::fmt::init();


    Ok(())
}
