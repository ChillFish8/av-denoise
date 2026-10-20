use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use cudarc::nvrtc::result;

use super::accelerate::Accelerator;
use super::probe::report_skipped;

/// The toolkit headers the generated CUDA kernels include.
const REQUIRED_HEADERS: [&str; 2] = ["cuda_runtime.h", "cuda_fp16.h"];

/// A kernel that includes every header in [REQUIRED_HEADERS].
const TEST_KERNEL: &std::ffi::CStr =
    c"#include <cuda_runtime.h>\n#include <cuda_fp16.h>\nextern \"C\" __global__ void avd_toolkit_check() {}\n";

/// Why the CUDA toolkit cannot build kernels on this machine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ToolkitProblem {
    /// Neither `CUDA_PATH` nor any default install location holds a toolkit.
    NotFound,
    /// The toolkit was found, but not every header the kernels include.
    HeadersMissing {
        include_dir: PathBuf,
        missing: Vec<&'static str>,
    },
    /// The NVRTC library could not be loaded.
    NvrtcMissing,
    /// NVRTC loaded, but could not compile a kernel against the toolkit headers.
    CompileFailed { log: String },
}

impl fmt::Display for ToolkitProblem {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ToolkitProblem::NotFound => write!(
                formatter,
                "the CUDA toolkit was not found. Install it, or set CUDA_PATH to where it is installed"
            ),
            ToolkitProblem::HeadersMissing { include_dir, missing } => write!(
                formatter,
                "the CUDA toolkit headers {} are missing from {}. Set CUDA_PATH to a complete CUDA toolkit install",
                missing.join(", "),
                include_dir.display(),
            ),
            ToolkitProblem::NvrtcMissing => write!(
                formatter,
                "NVRTC, the CUDA kernel compiler, could not be loaded. It ships with the CUDA toolkit, \
                 so check the toolkit is installed and its {LIBRARY_DIR_HINT}"
            ),
            ToolkitProblem::CompileFailed { log } => {
                write!(formatter, "NVRTC could not compile a test kernel. {}", log.trim())
            },
        }
    }
}

#[cfg(windows)]
const LIBRARY_DIR_HINT: &str = "`bin` directory is on PATH";

#[cfg(not(windows))]
const LIBRARY_DIR_HINT: &str = "`lib64` directory is on LD_LIBRARY_PATH";

/// Checks the CUDA toolkit on this machine can build kernels, logging why not once per process.
pub(crate) fn check_cuda_toolkit() -> Result<(), String> {
    let Some(problem) = toolkit_problem() else {
        return Ok(());
    };

    let reason = problem.to_string();
    report_skipped(Accelerator::Cuda, &reason);
    Err(reason)
}

/// Why the CUDA toolkit on this machine cannot build kernels, or `None` when it can.
///
/// The toolkit is found the same way the CUDA runtime finds it. The answer is worked out once per
/// process.
fn toolkit_problem() -> Option<&'static ToolkitProblem> {
    static PROBLEM: OnceLock<Option<ToolkitProblem>> = OnceLock::new();

    let problem = PROBLEM.get_or_init(|| {
        let cuda_path = cubecl::cuda::install::cuda_path();
        check_toolkit(cuda_path.as_deref()).err()
    });
    problem.as_ref()
}

/// Checks the toolkit at `cuda_path` has the headers the kernels include and can compile one.
fn check_toolkit(cuda_path: Option<&Path>) -> Result<(), ToolkitProblem> {
    let cuda_path = cuda_path.ok_or(ToolkitProblem::NotFound)?;
    let include_dir = cuda_path.join("include");

    let missing: Vec<&'static str> = REQUIRED_HEADERS
        .into_iter()
        .filter(|header| !include_dir.join(header).is_file())
        .collect();
    if !missing.is_empty() {
        return Err(ToolkitProblem::HeadersMissing { include_dir, missing });
    }

    // SAFETY: this only tries to open the NVRTC library, which runs no code from it beyond its
    // initialisers.
    let nvrtc_present = unsafe { cudarc::nvrtc::sys::is_culib_present() };
    if !nvrtc_present {
        return Err(ToolkitProblem::NvrtcMissing);
    }

    compile_test_kernel(&include_dir)
}

/// Compiles [TEST_KERNEL] with the include paths the CUDA runtime passes for its own kernels.
fn compile_test_kernel(include_dir: &Path) -> Result<(), ToolkitProblem> {
    let mut options = vec![format!("--include-path={}", include_dir.display())];
    let cccl_dir = include_dir.join("cccl");
    if cccl_dir.exists() {
        options.push(format!("--include-path={}", cccl_dir.display()));
    }

    let program = result::create_program(TEST_KERNEL, None).map_err(|err| ToolkitProblem::CompileFailed {
        log: format!("{err:?}"),
    })?;

    // SAFETY: `program` was created above and is destroyed exactly once, after its log is read.
    let compiled = unsafe { result::compile_program(program, &options) };
    let log = unsafe { program_log(program) };
    let _ = unsafe { result::destroy_program(program) };

    compiled.map_err(|err| ToolkitProblem::CompileFailed {
        log: log.unwrap_or_else(|| format!("{err:?}")),
    })
}

/// Reads the compiler log of `program`, or `None` when it is empty or unreadable.
///
/// # Safety
///
/// `program` must be a live NVRTC program.
unsafe fn program_log(program: cudarc::nvrtc::sys::nvrtcProgram) -> Option<String> {
    let raw = unsafe { result::get_program_log(program) }.ok()?;
    let bytes: Vec<u8> = raw
        .iter()
        .take_while(|&&byte| byte != 0)
        .map(|&byte| byte as u8)
        .collect();

    let log = String::from_utf8_lossy(&bytes).trim().to_string();
    (!log.is_empty()).then_some(log)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    use super::*;

    fn fake_toolkit(headers: &[(&str, &str)]) -> tempfile::TempDir {
        let root = tempfile::tempdir().expect("temp dir");
        let include_dir = root.path().join("include");
        fs::create_dir(&include_dir).expect("include dir");

        for (name, contents) in headers {
            let header_path = include_dir.join(name);
            fs::write(header_path, contents).expect("header");
        }

        root
    }

    #[test]
    fn a_missing_toolkit_is_reported() {
        let checked = check_toolkit(None);

        assert_eq!(checked, Err(ToolkitProblem::NotFound));
    }

    #[test]
    fn every_header_is_missing_without_an_include_dir() {
        let root = tempfile::tempdir().expect("temp dir");

        let checked = check_toolkit(Some(root.path()));

        let Err(ToolkitProblem::HeadersMissing { missing, .. }) = checked else {
            panic!("expected missing headers, got {checked:?}");
        };
        assert_eq!(missing, REQUIRED_HEADERS.to_vec());
    }

    #[test]
    fn only_the_absent_headers_are_listed() {
        let root = fake_toolkit(&[("cuda_runtime.h", "")]);

        let checked = check_toolkit(Some(root.path()));

        let Err(ToolkitProblem::HeadersMissing { include_dir, missing }) = checked else {
            panic!("expected missing headers, got {checked:?}");
        };
        assert_eq!(include_dir, root.path().join("include"));
        assert_eq!(missing, vec!["cuda_fp16.h"]);
    }

    #[test]
    fn a_missing_headers_message_names_the_directory_and_headers() {
        let problem = ToolkitProblem::HeadersMissing {
            include_dir: Path::new("/opt/cuda/include").to_path_buf(),
            missing: vec!["cuda_fp16.h"],
        };

        let message = problem.to_string();

        assert!(message.contains("/opt/cuda/include"), "{message}");
        assert!(message.contains("cuda_fp16.h"), "{message}");
        assert!(message.contains("CUDA_PATH"), "{message}");
    }

    #[test]
    fn the_installed_toolkit_builds_the_test_kernel() {
        let cuda_path = cubecl::cuda::install::cuda_path();

        let checked = check_toolkit(cuda_path.as_deref());

        assert_eq!(checked, Ok(()));
    }

    /// Headers that exist but do not compile still have to stop the CUDA backend.
    #[test]
    fn a_broken_header_fails_with_the_compiler_log() {
        let root = fake_toolkit(&[
            ("cuda_runtime.h", "#error this toolkit is broken\n"),
            ("cuda_fp16.h", ""),
        ]);

        let checked = check_toolkit(Some(root.path()));

        let Err(ToolkitProblem::CompileFailed { log }) = checked else {
            panic!("expected a compile failure, got {checked:?}");
        };
        assert!(log.contains("this toolkit is broken"), "{log}");
    }
}
