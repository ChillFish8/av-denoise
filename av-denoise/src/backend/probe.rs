use std::panic::{self, AssertUnwindSafe, PanicHookInfo};
use std::sync::{Arc, Mutex};

use cubecl::client::ComputeClient;
use cubecl::prelude::*;
use cubecl::server::{Handle, ServerError};

use super::accelerate::Accelerator;

/// How many values the canary kernel writes.
const CANARY_LEN: usize = 64;

/// What the canary's output holds before the kernel runs. No value the kernel writes equals it.
const CANARY_SENTINEL: u32 = u32::MAX;

/// Backends already reported as unavailable, and the lock guarding the panic hook.
///
/// The hook is process-wide, so two probes running at once would race to restore each other's.
/// Holding this for the length of a probe keeps them in single file, and the list inside it stops a
/// backend from warning again every time it is probed.
static PROBED: Mutex<Vec<Accelerator>> = Mutex::new(Vec::new());

/// Opens a client for `accelerator` on `device`, or reports that the backend cannot run here.
///
/// The client is synchronised and then runs the canary kernel before it is handed back. A sync
/// alone does not prove kernels run. CUDA syncs fine without NVRTC, but cannot compile a kernel
/// and drops every launch without reporting an error.
///
/// Some backends report missing driver libraries by panicking. The CUDA runtime loads `libcuda` on
/// its own worker thread and panics there, which reaches the caller as a second panic when cubecl
/// unwraps the dead worker's channel. Catching that lets one binary with several backends enabled
/// run on a machine that has only one of them.
pub(crate) fn open_client<R: Runtime>(
    accelerator: Accelerator,
    device: &R::Device,
) -> Result<ComputeClient<R>, String> {
    let mut probed = PROBED.lock().unwrap_or_else(|err| err.into_inner());

    let opened = quiet_panics(|| {
        let client = R::client(device);
        let synced = client.sync();
        cubecl::future::block_on(synced)?;

        let canary_passed = run_canary(&client)?;
        Ok::<_, ServerError>(canary_passed.then_some(client))
    });

    let reason = match opened {
        Ok(Ok(Some(client))) => return Ok(client),
        Ok(Ok(None)) => "it started, but a test kernel wrote nothing".to_string(),
        Ok(Err(err)) => format!("it started, but a test kernel failed. {err}"),
        Err(panic_message) => panic_message,
    };

    warn_once(&mut probed, accelerator, &reason);
    Err(reason)
}

/// Logs that `accelerator` is skipped and why, at most once per process.
#[cfg(feature = "cuda")]
pub(crate) fn report_skipped(accelerator: Accelerator, reason: &str) {
    let mut probed = PROBED.lock().unwrap_or_else(|err| err.into_inner());
    warn_once(&mut probed, accelerator, reason);
}

/// Logs `reason` unless `accelerator` was already reported.
///
/// A denoise run probes once per denoiser it builds, and a missing driver is worth one line rather
/// than one per scene.
fn warn_once(probed: &mut Vec<Accelerator>, accelerator: Accelerator, reason: &str) {
    if probed.contains(&accelerator) {
        return;
    }

    probed.push(accelerator);
    tracing::warn!("the {accelerator} backend is skipped because {reason}");
}

/// Writes `index * 3 + 7` into each slot of `output`.
#[cube(launch_unchecked)]
fn canary(output: &mut Array<u32>) {
    let index = ABSOLUTE_POS_X;
    if (index as usize) < output.len() {
        output[index as usize] = index * 3 + 7;
    }
}

/// Runs the canary kernel on `client` and reports whether it wrote what it should.
///
/// The output starts filled with [CANARY_SENTINEL], so a launch the backend drops without an
/// error still fails the comparison.
fn run_canary<R: Runtime>(client: &ComputeClient<R>) -> Result<bool, ServerError> {
    let output = upload_canary_sentinel(client);

    // SAFETY: `output` holds exactly `CANARY_LEN` values and the kernel bounds-checks its write.
    unsafe {
        let output_arg = ArrayArg::from_raw_parts(output.clone(), CANARY_LEN);
        canary::launch_unchecked::<R>(
            client,
            CubeCount::new_1d(1),
            CubeDim::new_1d(CANARY_LEN as u32),
            output_arg,
        );
    }

    let bytes = client.read_one(output)?;
    Ok(canary_output_matches(&bytes))
}

fn upload_canary_sentinel<R: Runtime>(client: &ComputeClient<R>) -> Handle {
    let sentinel_bytes: Vec<u8> = (0..CANARY_LEN)
        .flat_map(|_| CANARY_SENTINEL.to_ne_bytes())
        .collect();

    client.create_from_slice(&sentinel_bytes)
}

fn expected_canary_bytes() -> Vec<u8> {
    (0..CANARY_LEN as u32)
        .flat_map(|index| (index * 3 + 7).to_ne_bytes())
        .collect()
}

fn canary_output_matches(bytes: &[u8]) -> bool {
    let expected = expected_canary_bytes();
    bytes.get(..expected.len()) == Some(expected.as_slice())
}

/// Runs `work`, turning a panic into an `Err` holding the first panic message seen on any thread.
///
/// A failing backend panics on its own worker thread with the real cause, and the caller then sees
/// a second panic about the dead worker. So the first message is the one kept. The hook is
/// quietened while `work` runs, sending messages to the debug log, and put back afterwards. Callers
/// hold [PROBED] across this so two probes cannot race to restore each other's hook.
fn quiet_panics<T>(work: impl FnOnce() -> T) -> Result<T, String> {
    let first_message: Arc<Mutex<Option<String>>> = Arc::default();
    let recorder = Arc::clone(&first_message);

    let previous = panic::take_hook();
    let quiet_hook = Box::new(move |info: &PanicHookInfo<'_>| {
        tracing::debug!("{info}");

        let message = info.payload_as_str().unwrap_or("a panic with no message");
        let mut recorded = recorder.lock().unwrap_or_else(|err| err.into_inner());
        recorded.get_or_insert_with(|| message.to_string());
    });
    panic::set_hook(quiet_hook);

    let unwind_safe_work = AssertUnwindSafe(work);
    let result = panic::catch_unwind(unwind_safe_work);
    panic::set_hook(previous);

    result.map_err(|_| {
        let mut recorded = first_message.lock().unwrap_or_else(|err| err.into_inner());
        recorded
            .take()
            .unwrap_or_else(|| "it panicked without a message".to_string())
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    use super::*;

    #[test]
    fn a_panic_inside_becomes_an_error() {
        let _probed = PROBED.lock().unwrap_or_else(|err| err.into_inner());

        let panicked = quiet_panics(|| panic!("the backend fell over"));
        let returned = quiet_panics(|| 7);

        assert!(panicked.is_err());
        assert_eq!(returned.unwrap(), 7);
    }

    #[test]
    fn a_panic_reports_its_message() {
        let _probed = PROBED.lock().unwrap_or_else(|err| err.into_inner());

        let panicked = quiet_panics(|| panic!("the backend fell over"));

        assert_eq!(panicked.unwrap_err(), "the backend fell over");
    }

    /// A backend's worker thread panics with the real cause, and the caller only sees a second panic.
    #[test]
    fn the_first_panic_on_any_thread_is_reported() {
        let _probed = PROBED.lock().unwrap_or_else(|err| err.into_inner());

        let panicked = quiet_panics(|| {
            let worker = std::thread::spawn(|| panic!("libcuda is missing"));
            let _ = worker.join();
            panic!("the worker's channel closed");
        });

        assert_eq!(panicked.unwrap_err(), "libcuda is missing");
    }

    /// The hook has to come back however the probe ended, or every later panic reports at debug level.
    ///
    /// Holding [PROBED], as [open_client] does, keeps a probe on another thread from swapping the hook
    /// mid-test.
    #[test]
    fn the_panic_hook_is_restored() {
        let _probed = PROBED.lock().unwrap_or_else(|err| err.into_inner());

        let marker = Arc::new(AtomicBool::new(false));
        let flag = marker.clone();
        let marking_hook = Box::new(move |_: &PanicHookInfo<'_>| flag.store(true, Ordering::SeqCst));
        panic::set_hook(marking_hook);

        let _ = quiet_panics(|| panic!("swallowed by the quiet hook"));
        assert!(
            !marker.load(Ordering::SeqCst),
            "the quiet hook did not replace the installed one",
        );

        let _ = panic::catch_unwind(|| panic!("seen by the restored hook"));
        let restored = marker.load(Ordering::SeqCst);
        let _ = panic::take_hook();

        assert!(restored, "the probe left its own panic hook installed");
    }

    #[cfg(feature = "vulkan")]
    fn vulkan_client() -> ComputeClient<cubecl::wgpu::WgpuRuntime> {
        let device = cubecl::wgpu::WgpuDevice::DefaultDevice;
        cubecl::wgpu::WgpuRuntime::client(&device)
    }

    #[cfg(feature = "vulkan")]
    #[test]
    fn the_canary_passes_on_a_working_backend() {
        let client = vulkan_client();

        let passed = run_canary(&client).expect("canary readback");

        assert!(passed);
    }

    /// A launch the backend drops leaves the sentinel in place, which is what the probe has to catch.
    #[cfg(feature = "vulkan")]
    #[test]
    fn an_unlaunched_canary_is_rejected() {
        let client = vulkan_client();

        let output = upload_canary_sentinel(&client);
        let bytes = client.read_one(output).expect("canary readback");

        assert!(!canary_output_matches(&bytes));
    }

    #[test]
    fn a_short_canary_readback_is_rejected() {
        let expected = expected_canary_bytes();
        let truncated = &expected[..expected.len() - 4];

        assert!(canary_output_matches(&expected));
        assert!(!canary_output_matches(truncated));
    }
}
