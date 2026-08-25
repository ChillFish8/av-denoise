//! Opening a backend's client without taking the process down with it.
//!
//! A build can enable a backend whose driver libraries are not
//! installed. Some backends do not report that as an error. The CUDA
//! runtime loads `libcuda` dynamically on its own worker thread and
//! panics there when the load fails, and the panic reaches the caller
//! as a second panic when cubecl unwraps the dead worker's channel.
//!
//! [`open_client`] runs that work under [`catch_unwind`], so a missing
//! driver reads as "this backend is not available here" rather than as
//! a crash. That is what makes a single binary with `cuda`, `rocm`, and
//! `vulkan` all enabled usable on a machine that has only one of them.

use std::panic::{self, AssertUnwindSafe};
use std::sync::Mutex;

use cubecl::client::ComputeClient;
use cubecl::prelude::*;

use crate::accelerate::Accelerator;

/// Backends already reported as unavailable, and the lock guarding the
/// panic hook.
///
/// The hook is process-wide, so two probes running at once would race to
/// restore each other's. Holding this for the length of a probe keeps
/// them in single file, and the list inside it keeps a backend from
/// warning again every time it is probed.
static PROBED: Mutex<Vec<Accelerator>> = Mutex::new(Vec::new());

/// Opens a client for `accelerator` on `device`, or reports that the
/// backend cannot run here.
///
/// The client is synchronised before it is handed back. cubecl kernels
/// are fully asynchronous, so a successful `sync()` is what proves the
/// backend works, and no test kernel is needed.
pub(crate) fn open_client<R: Runtime>(
    accelerator: Accelerator,
    device: &R::Device,
) -> Option<ComputeClient<R>> {
    let mut probed = PROBED.lock().unwrap_or_else(|err| err.into_inner());

    let opened = quiet_panics(|| {
        let client = R::client(device);
        cubecl::future::block_on(client.sync()).map(|()| client)
    });

    match opened {
        Ok(Ok(client)) => Some(client),
        Ok(Err(err)) => {
            tracing::debug!(err = ?err, "could not use the {accelerator} runtime");
            None
        },
        Err(_) => {
            // Only the first probe of a backend says anything. A denoise
            // run probes once per denoiser it builds, and a missing
            // driver is worth one line, not one per scene.
            if !probed.contains(&accelerator) {
                probed.push(accelerator);
                tracing::warn!(
                    "the {accelerator} backend is enabled but did not start, its driver libraries are probably missing"
                );
            }
            None
        },
    }
}

/// Runs `f`, turning a panic into an `Err` and routing the panic message
/// to the debug log rather than to stderr.
///
/// A failing backend prints its own panic from its worker thread before
/// the caller ever sees one, so the hook is quietened for as long as `f`
/// runs and put back afterwards.
///
/// The hook is process-wide. Callers hold [`PROBED`] across this so two
/// probes cannot race to restore each other's.
fn quiet_panics<T>(f: impl FnOnce() -> T) -> std::thread::Result<T> {
    let previous = panic::take_hook();
    panic::set_hook(Box::new(|info| tracing::debug!("{info}")));
    let out = panic::catch_unwind(AssertUnwindSafe(f));
    panic::set_hook(previous);
    out
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    use super::*;

    #[test]
    fn a_panic_inside_becomes_an_error() {
        let _probed = PROBED.lock().unwrap_or_else(|err| err.into_inner());

        assert!(quiet_panics(|| panic!("the backend fell over")).is_err());
        assert_eq!(quiet_panics(|| 7).unwrap(), 7);
    }

    /// The hook has to come back however the probe ended, or every later
    /// panic in the process reports at debug level.
    ///
    /// Holding [`PROBED`], the way [`open_client`] does, keeps a probe on
    /// another thread from swapping the hook mid-test.
    #[test]
    fn the_panic_hook_is_restored() {
        let _probed = PROBED.lock().unwrap_or_else(|err| err.into_inner());

        let marker = Arc::new(AtomicBool::new(false));
        let flag = marker.clone();
        panic::set_hook(Box::new(move |_| flag.store(true, Ordering::SeqCst)));

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
}
