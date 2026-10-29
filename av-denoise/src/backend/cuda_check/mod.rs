mod adapters;
mod capability;
mod kernel;
mod libdevice;

use std::collections::BTreeMap;
use std::fmt::{self, Display, Formatter};
use std::path::PathBuf;
use std::sync::Mutex;

use cubecl::prelude::Client;

use self::adapters::{NvidiaPresence, adapter_vendor_ids, nvidia_presence};
use self::capability::check_capability;
use self::kernel::run_libdevice_probe;
use self::libdevice::{CUDA_ROOTS, find_libdevice};
use super::describe::cuda_details;
use super::probe::quiet_panics;

/// Why the CUDA backend cannot be used.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CudaUnavailable {
    NoNvidiaGpu,
    Unsupported { adapter: String, capability: u32 },
    DriverUnreadable(String),
    LibdeviceMissing { searched: Vec<PathBuf> },
    RuntimeStart(String),
    TestKernel(String),
    Panicked { step: &'static str, message: String },
}

impl CudaUnavailable {
    /// Whether the failure is expected enough to skip reporting it.
    pub(crate) fn is_quiet(&self) -> bool {
        matches!(self, Self::NoNvidiaGpu)
    }
}

impl Display for CudaUnavailable {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoNvidiaGpu => write!(formatter, "no NVIDIA GPU was found"),
            Self::Unsupported { adapter, capability } => {
                let major = capability / 10;
                let minor = capability % 10;
                write!(
                    formatter,
                    "{adapter} (compute {major}.{minor}) is not supported by CUDA 13"
                )
            },
            Self::DriverUnreadable(message) => {
                write!(formatter, "the NVIDIA driver could not be read: {message}")
            },
            Self::LibdeviceMissing { searched } => {
                let roots: Vec<_> = searched.iter().map(|root| root.display().to_string()).collect();
                let roots = roots.join(", ");
                write!(formatter, "libdevice.10.bc was not found in {roots}")
            },
            Self::RuntimeStart(message) => {
                write!(formatter, "the CUDA runtime did not start: {message}")
            },
            Self::TestKernel(message) => write!(formatter, "the test kernel failed: {message}"),
            Self::Panicked { step, message } => write!(formatter, "{step} panicked: {message}"),
        }
    }
}

type Verdict = Result<(), CudaUnavailable>;

/// Verdicts already reached, so a process pays for the check once per GPU.
static VERDICTS: Mutex<BTreeMap<usize, Verdict>> = Mutex::new(BTreeMap::new());

/// Opens the client for a CUDA `device`, after checking it can really compile and run kernels.
///
/// Callers hold the probe lock across this, as the panic hook it quietens is process-wide.
pub(crate) fn open_checked(device: &cubecl::Device) -> Result<Client, CudaUnavailable> {
    let vendor_ids = visible_vendor_ids();
    let presence = nvidia_presence(&vendor_ids);
    if skips_cuda(presence) {
        return Err(CudaUnavailable::NoNvidiaGpu);
    }

    if let Some(index) = cuda_index(device) {
        cached_verdict(index, || check_device(index, device))?;
    }

    guarded("starting the CUDA runtime", || {
        let client = device.client();
        let synced = client.sync();
        let synced = cubecl::future::block_on(synced);
        if let Err(error) = synced {
            let message = format!("{error:?}");
            return Err(CudaUnavailable::RuntimeStart(message));
        }

        Ok(client)
    })
}

fn check_device(index: usize, device: &cubecl::Device) -> Verdict {
    let identity = guarded("reading the NVIDIA driver", || {
        let details = cuda_details(index);
        details.map_err(|error| {
            let message = format!("{error:#}");
            CudaUnavailable::DriverUnreadable(message)
        })
    })?;
    if let Some(capability) = identity.compute_capability {
        check_capability(&identity.adapter, capability)?;
    }

    let cuda_path = std::env::var("CUDA_PATH").ok().map(PathBuf::from);
    let defaults: Vec<PathBuf> = CUDA_ROOTS.iter().map(PathBuf::from).collect();
    find_libdevice(cuda_path, &defaults)?;

    guarded("running the test kernel", || {
        let client = device.client();
        run_libdevice_probe(&client)
    })
}

/// The adapter vendors wgpu lists, or none if enumerating them panics.
fn visible_vendor_ids() -> Vec<u32> {
    quiet_panics(adapter_vendor_ids).unwrap_or_default()
}

fn skips_cuda(presence: NvidiaPresence) -> bool {
    presence == NvidiaPresence::Absent
}

fn cuda_index(device: &cubecl::Device) -> Option<usize> {
    match device {
        cubecl::Device::Cuda(cuda_device) => Some(cuda_device.index),
        _ => None,
    }
}

/// Runs `check` once per CUDA index and replays its verdict afterwards.
fn cached_verdict(index: usize, check: impl FnOnce() -> Verdict) -> Verdict {
    {
        let verdicts = VERDICTS.lock().unwrap_or_else(|err| err.into_inner());
        if let Some(verdict) = verdicts.get(&index) {
            return verdict.clone();
        }
    }

    let verdict = check();

    let mut verdicts = VERDICTS.lock().unwrap_or_else(|err| err.into_inner());
    verdicts.insert(index, verdict.clone());
    verdict
}

/// Runs `work`, reporting a panic as [CudaUnavailable::Panicked] under `step`.
fn guarded<T>(
    step: &'static str,
    work: impl FnOnce() -> Result<T, CudaUnavailable>,
) -> Result<T, CudaUnavailable> {
    match quiet_panics(work) {
        Ok(result) => result,
        Err(payload) => {
            let message = panic_message(payload.as_ref());
            Err(CudaUnavailable::Panicked { step, message })
        },
    }
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        return message.to_string();
    }

    if let Some(message) = payload.downcast_ref::<String>() {
        return message.clone();
    }

    "unknown panic".to_string()
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::backend::probe::PROBED;

    #[test]
    fn a_panicking_step_is_reported_by_name() {
        let _probed = PROBED.lock().unwrap_or_else(|err| err.into_inner());

        let result: Result<(), _> = guarded("reading the driver", || panic!("libcuda.so missing"));

        let Err(CudaUnavailable::Panicked { step, message }) = result else {
            panic!("expected Panicked");
        };
        assert_eq!(step, "reading the driver");
        assert!(message.contains("libcuda.so missing"));
    }

    #[test]
    fn a_cached_failure_is_not_rerun() {
        let runs = AtomicUsize::new(0);
        let failing = || {
            runs.fetch_add(1, Ordering::SeqCst);
            Err(CudaUnavailable::NoNvidiaGpu)
        };

        // Index 900 keeps this test away from real devices' entries.
        let first = cached_verdict(900, failing);
        let second = cached_verdict(900, || unreachable!("the verdict was cached"));

        assert_eq!(first, second);
        assert_eq!(runs.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn cuda_index_reads_the_selected_index() {
        let device = cubecl::Device::from(cubecl::device::CudaDevice::new(1));
        assert_eq!(cuda_index(&device), Some(1));
    }

    #[test]
    fn unknown_presence_still_checks_cuda() {
        assert!(skips_cuda(NvidiaPresence::Absent));
        assert!(!skips_cuda(NvidiaPresence::Unknown));
        assert!(!skips_cuda(NvidiaPresence::Present));
    }

    /// Runs only where wgpu sees adapters and none of them is NVIDIA, such as the AMD dev machine.
    #[test]
    fn a_machine_without_nvidia_skips_cuda_quietly() {
        let _probed = PROBED.lock().unwrap_or_else(|err| err.into_inner());

        let vendor_ids = adapter_vendor_ids();
        if nvidia_presence(&vendor_ids) != NvidiaPresence::Absent {
            eprintln!("skipped: this machine has an NVIDIA GPU or no visible adapters");
            return;
        }

        let device = cubecl::Device::from(cubecl::device::CudaDevice::new(0));
        let result = open_checked(&device);
        assert_eq!(result.err(), Some(CudaUnavailable::NoNvidiaGpu));
    }

    #[test]
    fn a_runtime_start_failure_names_the_runtime() {
        let error = CudaUnavailable::RuntimeStart("boom".to_string());
        assert_eq!(error.to_string(), "the CUDA runtime did not start: boom");
    }

    #[test]
    fn only_a_missing_nvidia_gpu_is_quiet() {
        assert!(CudaUnavailable::NoNvidiaGpu.is_quiet());
        assert!(!CudaUnavailable::TestKernel("x".into()).is_quiet());
    }
}
