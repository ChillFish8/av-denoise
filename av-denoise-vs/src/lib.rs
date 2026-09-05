//! VapourSynth plugin exposing av-denoise as `avd` filters.

mod filter;
pub mod frames;
pub mod params;

use anyhow::Error;
use tracing_subscriber::EnvFilter;
use vapoursynth::core::CoreRef;
use vapoursynth::plugins::{Filter, FilterArgument, Metadata};
use vapoursynth::prelude::{API, Node};
use vapoursynth::{export_vapoursynth_plugin, make_filter_function};

use crate::filter::Denoise;
use crate::params::{AlgorithmKind, RawParams};

/// Installs the tracing subscriber that writes the plugin's logs to stderr.
///
/// `RUST_LOG` picks what is printed, and without it the plugin logs at `warn` so
/// an ordinary render stays quiet.
fn init_logging() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn"));

    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .try_init();
}

/// Keeps this plugin's library mapped for the rest of the process.
///
/// VapourSynth unloads every plugin library when it frees a core, and
/// vspipe frees its core right before exiting. The GPU runtime this
/// plugin builds spawns a device thread per accelerator, plus a
/// polling thread per stream on the wgpu backends, and those threads
/// run for the rest of the process. The device thread never blocks. It
/// spins, yields, then sleeps briefly, over and over. On Windows its
/// first wake after `FreeLibrary` returns into unmapped code and the
/// process dies with an access violation, after every frame was
/// already written. The polling thread parks or waits in the driver,
/// and dies the same way once anything wakes it.
///
/// Pinning the module makes the unload a no-op, so the threads stay
/// valid until process exit terminates them. On Linux the loader
/// already refuses to unload a library that registered thread-local
/// destructors, which is what happens as soon as this plugin's threads
/// start, so nothing needs doing there. macOS is not covered and has
/// not been tested.
///
/// This runs once, on the first filter creation. That is before any
/// device thread exists, since only a filter builds a denoiser. The
/// plugin's init function runs earlier, but the export macro owns its
/// body and this plugin has no code of its own in it.
fn pin_plugin_library() {
    static PIN: std::sync::Once = std::sync::Once::new();
    PIN.call_once(|| {
        #[cfg(windows)]
        pin_plugin_library_windows();
    });
}

#[cfg(windows)]
fn pin_plugin_library_windows() {
    use std::ffi::c_void;

    const GET_MODULE_HANDLE_EX_FLAG_PIN: u32 = 0x0000_0001;
    const GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS: u32 = 0x0000_0004;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetModuleHandleExW(flags: u32, module_name: *const u16, module: *mut *mut c_void) -> i32;
    }

    let address = pin_plugin_library_windows as *const () as *const u16;
    let mut module: *mut c_void = std::ptr::null_mut();
    // SAFETY: `address` is a code address inside this library, which is
    // what `FROM_ADDRESS` asks for, and `module` is a valid out pointer.
    let ok = unsafe {
        GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_PIN | GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
            address,
            &mut module,
        )
    };
    if ok == 0 {
        tracing::warn!("could not pin the plugin library, the process may crash at exit");
    }
}

/// Reads one optional UTF-8 script argument, naming `field` in the error
/// when the bytes are not valid UTF-8.
fn opt_string(bytes: Option<&[u8]>, field: &str) -> Result<Option<String>, Error> {
    bytes
        .map(|b| String::from_utf8(b.to_vec()).map_err(|_| anyhow::anyhow!("{field} must be valid UTF-8")))
        .transpose()
}

/// Reads the optional `accelerators` script argument, a comma-separated
/// list of accelerator names, into the `Vec<String>` [`RawParams`]
/// wants.
///
/// VapourSynth script arguments have no native string array type that
/// fits cleanly into `make_filter_function!`'s generated argument
/// string, so this reuses the plain `data` type and splits it, matching
/// how `channel_mode` and `device` already take a single string.
fn opt_accelerators(bytes: Option<&[u8]>) -> Result<Option<Vec<String>>, Error> {
    let Some(joined) = opt_string(bytes, "accelerators")? else {
        return Ok(None);
    };

    let names: Vec<String> = joined
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect();

    if names.is_empty() {
        anyhow::bail!("accelerators must name at least one accelerator when set");
    }

    Ok(Some(names))
}

/// Reads the optional `motion_compensation` script argument.
///
/// VapourSynth script arguments have no native boolean type, so this
/// takes the plain `int` type every other on/off knob in the wider
/// VapourSynth ecosystem uses, and reads it the same way: `0` is off,
/// anything else is on.
fn opt_bool(value: Option<i64>) -> Option<bool> {
    value.map(|v| v != 0)
}

/// Builds a [`RawParams`] from a filter function's raw script arguments.
#[expect(clippy::too_many_arguments)]
fn raw_params(
    strength: Option<f64>,
    variant: Option<&[u8]>,
    preset: Option<&[u8]>,
    prefilter: Option<&[u8]>,
    channel_mode: Option<&[u8]>,
    luma_strength: Option<f64>,
    chroma_strength: Option<f64>,
    luma_lambda_ht: Option<f64>,
    chroma_lambda_ht: Option<f64>,
    luma_mismatch_scale: Option<f64>,
    chroma_mismatch_scale: Option<f64>,
    device: Option<&[u8]>,
    accelerators: Option<&[u8]>,
    search_radius: Option<i64>,
    patch_radius: Option<i64>,
    temporal_radius: Option<i64>,
    sigma: Option<f64>,
    sigma_scale: Option<f64>,
    motion_compensation: Option<i64>,
    lambda_ht: Option<f64>,
    lambda_ht_scale: Option<f64>,
    spatial_radius: Option<i64>,
    refine: Option<i64>,
) -> Result<RawParams, Error> {
    Ok(RawParams {
        strength,
        variant: opt_string(variant, "variant")?,
        preset: opt_string(preset, "preset")?,
        prefilter: opt_string(prefilter, "prefilter")?,
        channel_mode: opt_string(channel_mode, "channel_mode")?,
        luma_strength,
        chroma_strength,
        luma_lambda_ht,
        chroma_lambda_ht,
        luma_mismatch_scale,
        chroma_mismatch_scale,
        device: opt_string(device, "device")?,
        accelerators: opt_accelerators(accelerators)?,
        search_radius,
        patch_radius,
        temporal_radius,
        sigma,
        sigma_scale,
        motion_compensation: opt_bool(motion_compensation),
        lambda_ht,
        lambda_ht_scale,
        spatial_radius,
        refine,
    })
}

make_filter_function! {
    NlmeansFunction, "NLMeans"

    #[expect(clippy::too_many_arguments)]
    fn create_nlmeans<'core>(
        api: API,
        core: CoreRef<'core>,
        clip: Node<'core>,
        strength: Option<f64>,
        variant: Option<&[u8]>,
        preset: Option<&[u8]>,
        prefilter: Option<&[u8]>,
        channel_mode: Option<&[u8]>,
        luma_strength: Option<f64>,
        chroma_strength: Option<f64>,
        device: Option<&[u8]>,
        accelerators: Option<&[u8]>,
        search_radius: Option<i64>,
        patch_radius: Option<i64>,
        temporal_radius: Option<i64>,
        sigma: Option<f64>,
        sigma_scale: Option<f64>,
        motion_compensation: Option<i64>,
    ) -> Result<Option<Box<dyn Filter<'core> + 'core>>, Error> {
        let raw = raw_params(
            strength,
            variant,
            preset,
            prefilter,
            channel_mode,
            luma_strength,
            chroma_strength,
            None,
            None,
            None,
            None,
            device,
            accelerators,
            search_radius,
            patch_radius,
            temporal_radius,
            sigma,
            sigma_scale,
            motion_compensation,
            None,
            None,
            None,
            None,
        )?;
        let filter = Denoise::create(api, core, clip, AlgorithmKind::Nlmeans, &raw)?;
        Ok(Some(Box::new(filter)))
    }
}

make_filter_function! {
    Nl4dFunction, "NL4D"

    /// Estimates its automatic noise level fresh from each frame's own
    /// temporal window, rather than smoothing it across the whole
    /// stream, so a frame denoises to the same pixels no matter what
    /// order VapourSynth requests frames in. Passing `sigma` pins the
    /// noise level and skips that estimator entirely.
    ///
    /// The first few frames of a clip may differ slightly from the CLI's
    /// output for the same parameters. The plugin fills a clip's
    /// leading edge by repeating its first frame across the whole
    /// temporal window, while the CLI's streaming mode primes a
    /// narrower repeat before real frames start arriving. The
    /// difference is bounded, small, and confined to a clip's first
    /// `2 * temporal_radius` frames.
    #[expect(clippy::too_many_arguments)]
    fn create_nl4d<'core>(
        api: API,
        core: CoreRef<'core>,
        clip: Node<'core>,
        preset: Option<&[u8]>,
        channel_mode: Option<&[u8]>,
        luma_strength: Option<f64>,
        chroma_strength: Option<f64>,
        luma_lambda_ht: Option<f64>,
        chroma_lambda_ht: Option<f64>,
        luma_mismatch_scale: Option<f64>,
        chroma_mismatch_scale: Option<f64>,
        device: Option<&[u8]>,
        accelerators: Option<&[u8]>,
        temporal_radius: Option<i64>,
        sigma: Option<f64>,
        sigma_scale: Option<f64>,
        lambda_ht: Option<f64>,
        lambda_ht_scale: Option<f64>,
        spatial_radius: Option<i64>,
        refine: Option<i64>,
    ) -> Result<Option<Box<dyn Filter<'core> + 'core>>, Error> {
        let raw = raw_params(
            None,
            None,
            preset,
            None,
            channel_mode,
            luma_strength,
            chroma_strength,
            luma_lambda_ht,
            chroma_lambda_ht,
            luma_mismatch_scale,
            chroma_mismatch_scale,
            device,
            accelerators,
            None,
            None,
            temporal_radius,
            sigma,
            sigma_scale,
            None,
            lambda_ht,
            lambda_ht_scale,
            spatial_radius,
            refine,
        )?;
        let filter = Denoise::create(api, core, clip, AlgorithmKind::Nl4d, &raw)?;
        Ok(Some(Box::new(filter)))
    }
}

export_vapoursynth_plugin! {
    Metadata {
        identifier: "com.chillfish8.avdenoise",
        namespace: "avd",
        name: "av-denoise",
        read_only: true,
    },
    [NlmeansFunction::new(), Nl4dFunction::new()]
}
