use std::path::{Path, PathBuf};

use super::CudaUnavailable;

/// The toolkit locations cubecl checks when `CUDA_PATH` is unset.
pub(super) const CUDA_ROOTS: [&str; 4] = ["/usr/local/cuda", "/opt/cuda", "/usr/lib/cuda", "/usr"];

/// Finds `libdevice.10.bc` the way cubecl does.
///
/// A set `CUDA_PATH` is the only root checked. Otherwise the first default root with an
/// `nvvm/libdevice` directory is used, and the file must exist inside it.
pub(super) fn find_libdevice(
    cuda_path: Option<PathBuf>,
    defaults: &[PathBuf],
) -> Result<PathBuf, CudaUnavailable> {
    let root = match cuda_path {
        Some(path) => Some(path),
        None => defaults
            .iter()
            .find(|candidate| candidate.join("nvvm").join("libdevice").is_dir())
            .cloned(),
    };
    let Some(root) = root else {
        return Err(CudaUnavailable::LibdeviceMissing {
            searched: defaults.to_vec(),
        });
    };

    let file = libdevice_file(&root);
    if file.is_file() {
        return Ok(file);
    }

    Err(CudaUnavailable::LibdeviceMissing { searched: vec![root] })
}

fn libdevice_file(root: &Path) -> PathBuf {
    root.join("nvvm").join("libdevice").join("libdevice.10.bc")
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::{TempDir, tempdir};

    use super::*;

    fn toolkit_with_file(root: &Path) {
        let directory = root.join("nvvm").join("libdevice");
        fs::create_dir_all(&directory).unwrap();
        fs::write(directory.join("libdevice.10.bc"), b"bc").unwrap();
    }

    fn toolkit_without_file(root: &Path) {
        let directory = root.join("nvvm").join("libdevice");
        fs::create_dir_all(directory).unwrap();
    }

    fn roots(count: usize) -> (TempDir, Vec<PathBuf>) {
        let temp = tempdir().unwrap();
        let roots = (0..count)
            .map(|index| temp.path().join(format!("root{index}")))
            .collect();
        (temp, roots)
    }

    #[test]
    fn the_default_roots_match_cubecl() {
        assert_eq!(
            CUDA_ROOTS,
            ["/usr/local/cuda", "/opt/cuda", "/usr/lib/cuda", "/usr"]
        );
    }

    #[test]
    fn cuda_path_wins_over_defaults() {
        let (_temp, roots) = roots(2);
        toolkit_with_file(&roots[0]);
        toolkit_with_file(&roots[1]);

        let found = find_libdevice(Some(roots[0].clone()), &roots[1..]).unwrap();
        assert_eq!(found, libdevice_file(&roots[0]));
    }

    #[test]
    fn cuda_path_without_the_file_is_missing_even_when_a_default_has_it() {
        let (_temp, roots) = roots(2);
        toolkit_with_file(&roots[1]);

        let error = find_libdevice(Some(roots[0].clone()), &roots[1..]).unwrap_err();
        let expected = CudaUnavailable::LibdeviceMissing {
            searched: vec![roots[0].clone()],
        };
        assert_eq!(error, expected);
    }

    #[test]
    fn the_first_default_with_a_libdevice_dir_is_used() {
        let (_temp, roots) = roots(3);
        toolkit_with_file(&roots[1]);
        toolkit_with_file(&roots[2]);

        let found = find_libdevice(None, &roots).unwrap();
        assert_eq!(found, libdevice_file(&roots[1]));
    }

    #[test]
    fn a_default_dir_without_the_file_is_missing() {
        let (_temp, roots) = roots(2);
        toolkit_without_file(&roots[0]);
        toolkit_with_file(&roots[1]);

        let error = find_libdevice(None, &roots).unwrap_err();
        let expected = CudaUnavailable::LibdeviceMissing {
            searched: vec![roots[0].clone()],
        };
        assert_eq!(error, expected);
    }

    #[test]
    fn nothing_found_lists_every_searched_root() {
        let (_temp, roots) = roots(3);

        let error = find_libdevice(None, &roots).unwrap_err();
        let expected = CudaUnavailable::LibdeviceMissing { searched: roots };
        assert_eq!(error, expected);
    }
}
