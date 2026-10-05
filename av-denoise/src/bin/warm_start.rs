use av_denoise::{FrameLayout, PlanarDenoiser, PlaneOptions, WarmUp, kernel_key};

/// Builds a [PlanarDenoiser] after taking a place in the cross-process warm-up queue.
///
/// The queue stops concurrent `av-denoise` processes each compiling into a cold cache. The caller
/// finishes the returned place once the denoiser has produced its first output frame, and [WarmUp]
/// explains why it cannot be finished any earlier.
pub fn create_denoiser(
    opts: &PlaneOptions,
    layout: FrameLayout,
) -> Result<(PlanarDenoiser, Option<WarmUp>), anyhow::Error> {
    let key = kernel_key(opts, layout);
    let warm_up = WarmUp::begin(key);
    let denoiser = PlanarDenoiser::create(opts, layout)?;

    Ok((denoiser, warm_up))
}

/// Gives up a cold-cache queue place once a frame has proven its kernels are compiled and cached.
///
/// Does nothing when no place is held.
pub fn finish_warm_up(warm_up: &mut Option<WarmUp>) {
    if let Some(warm_up) = warm_up.take() {
        warm_up.finish();
    }
}

#[cfg(test)]
mod tests {
    use av_denoise::{Algorithm, ChannelIntent, DenoisingMode, Depth, Device, Subsampling};

    use super::*;

    #[test]
    fn create_denoiser_forwards_the_create_error() {
        let opts = PlaneOptions {
            accelerators: Vec::new(),
            device: Device::Default,
            intent: ChannelIntent::LumaChroma,
            mode: DenoisingMode::Spacial,
            algorithm: Algorithm::default(),
            luma_strength: None,
            chroma_strength: None,
            luma_lambda_ht: None,
            chroma_lambda_ht: None,
        };

        // Zero width collapses the 4:2:0 chroma plane to nothing, which `PlanarDenoiser::create`
        // rejects before touching the GPU.
        let layout = FrameLayout {
            width: 0,
            height: 0,
            subsampling: Subsampling::Yuv420,
            depth: Depth::Eight,
        };

        let direct = PlanarDenoiser::create(&opts, layout)
            .err()
            .expect("zero-size layout should be rejected");
        let wrapped = create_denoiser(&opts, layout)
            .err()
            .expect("wrapper should forward the same rejection");

        assert_eq!(wrapped.to_string(), direct.to_string());
    }
}
