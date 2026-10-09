//! VapourSynth plugin exposing av-denoise as `avd` filters.

mod filter;
pub mod frames;
pub mod params;
pub mod stream;

use anyhow::Error;
use tracing_subscriber::EnvFilter;
use vapoursynth::core::CoreRef;
use vapoursynth::plugins::{Filter, FilterArgument, Metadata};
use vapoursynth::prelude::{API, Node};
use vapoursynth::{export_vapoursynth_plugin, make_filter_function};

use self::filter::Denoise;
use self::params::{AlgorithmKind, RawParams};

/// Installs the tracing subscriber that writes the plugin's logs to stderr.
///
/// `RUST_LOG` picks what is printed. Without it the plugin logs at `warn` so an ordinary render stays quiet.
fn init_logging() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn"));

    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .try_init();
}

/// Keeps this plugin's library mapped for the rest of the process.
///
/// VapourSynth unloads every plugin library when it frees a core, and vspipe frees its core right
/// before exiting. The GPU runtime runs a device thread per accelerator, plus a polling thread per
/// stream on the wgpu backends, until the process exits. The device thread never blocks. It spins,
/// yields and sleeps in a loop, so on Windows its first wake after `FreeLibrary` returns into unmapped
/// code and the process dies with an access violation after every frame was written. The polling
/// thread parks or waits in the driver, and dies the same way once anything wakes it.
///
/// Pinning the module makes the unload a no-op. Linux needs nothing, since its loader refuses to
/// unload a library that registered thread-local destructors, which these threads do as soon as they
/// start. macOS is not covered and has not been tested.
///
/// This runs once, on the first filter creation, which is before any device thread exists since only
/// a filter builds a denoiser. The plugin's init function runs earlier, but the export macro owns its
/// body.
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
    let pinned = unsafe {
        GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_PIN | GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
            address,
            &mut module,
        )
    };
    if pinned == 0 {
        tracing::warn!("could not pin the plugin library, the process may crash at exit");
    }
}

/// Reads an optional UTF-8 script argument, naming `field` in the error when it is not valid UTF-8.
fn opt_string(bytes: Option<&[u8]>, field: &str) -> Result<Option<String>, Error> {
    let Some(bytes) = bytes else {
        return Ok(None);
    };

    let owned = bytes.to_vec();
    let text = String::from_utf8(owned).map_err(|_| anyhow::anyhow!("{field} must be valid UTF-8"))?;
    Ok(Some(text))
}

/// Reads the optional `accelerators` script argument, a comma-separated list of accelerator names.
///
/// VapourSynth has no string array argument type that fits `make_filter_function!`'s generated
/// argument string, so this takes the plain `data` type and splits it.
fn opt_accelerators(bytes: Option<&[u8]>) -> Result<Option<Vec<String>>, Error> {
    let Some(joined) = opt_string(bytes, "accelerators")? else {
        return Ok(None);
    };

    let names: Vec<String> = joined
        .split(',')
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(str::to_string)
        .collect();

    if names.is_empty() {
        anyhow::bail!("accelerators must name at least one accelerator when set");
    }

    Ok(Some(names))
}

fn opt_bool(value: Option<i64>) -> Option<bool> {
    value.map(|flag| flag != 0)
}

#[expect(
    clippy::too_many_arguments,
    reason = "takes one parameter per optional argument across both VapourSynth filters"
)]
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
    noise_map: Option<i64>,
    enable_psy: Option<i64>,
    psy_flat_boost: Option<f64>,
    psy_chroma_flat_boost: Option<f64>,
    psy_shadow_soften: Option<f64>,
    psy_flat_texture_cut: Option<f64>,
    psy_line_ring: Option<i64>,
    pooled_threshold: Option<i64>,
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
        noise_map: opt_bool(noise_map),
        enable_psy: opt_bool(enable_psy),
        psy_flat_boost,
        psy_chroma_flat_boost,
        psy_shadow_soften,
        psy_flat_texture_cut,
        psy_line_ring,
        pooled_threshold: opt_bool(pooled_threshold),
    })
}

make_filter_function! {
    NlmeansFunction, "NLMeans"

    #[expect(
        clippy::too_many_arguments,
        reason = "each parameter is a VapourSynth filter argument, so they cannot be grouped"
    )]
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
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )?;
        let filter = Denoise::create(api, core, clip, AlgorithmKind::Nlmeans, &raw)?;
        let boxed: Box<dyn Filter<'core> + 'core> = Box::new(filter);
        Ok(Some(boxed))
    }
}

make_filter_function! {
    Nl4dFunction, "NL4D"

    /// Creates an `avd.NL4D` filter.
    ///
    /// The automatic noise level is smoothed over the stream's history. Frames match a sequential
    /// render while VapourSynth's requests stay within twice its thread count. Frames after a seek
    /// or a wider burst of requests can differ slightly. Passing `sigma` pins the noise level and
    /// skips the estimator.
    #[expect(
        clippy::too_many_arguments,
        reason = "each parameter is a VapourSynth filter argument, so they cannot be grouped"
    )]
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
        device: Option<&[u8]>,
        accelerators: Option<&[u8]>,
        temporal_radius: Option<i64>,
        sigma: Option<f64>,
        sigma_scale: Option<f64>,
        lambda_ht: Option<f64>,
        lambda_ht_scale: Option<f64>,
        spatial_radius: Option<i64>,
        refine: Option<i64>,
        noise_map: Option<i64>,
        enable_psy: Option<i64>,
        psy_flat_boost: Option<f64>,
        psy_chroma_flat_boost: Option<f64>,
        psy_shadow_soften: Option<f64>,
        psy_flat_texture_cut: Option<f64>,
        psy_line_ring: Option<i64>,
        pooled_threshold: Option<i64>,
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
            noise_map,
            enable_psy,
            psy_flat_boost,
            psy_chroma_flat_boost,
            psy_shadow_soften,
            psy_flat_texture_cut,
            psy_line_ring,
            pooled_threshold,
        )?;
        let filter = Denoise::create(api, core, clip, AlgorithmKind::Nl4d, &raw)?;
        let boxed: Box<dyn Filter<'core> + 'core> = Box::new(filter);
        Ok(Some(boxed))
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
