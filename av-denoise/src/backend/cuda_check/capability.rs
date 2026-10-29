use super::CudaUnavailable;

/// The oldest compute capability CUDA 13 supports, as major and minor digits.
pub(super) const MIN_COMPUTE_CAPABILITY: u32 = 75;

pub(super) fn check_capability(adapter: &str, capability: u32) -> Result<(), CudaUnavailable> {
    if capability >= MIN_COMPUTE_CAPABILITY {
        return Ok(());
    }

    Err(CudaUnavailable::Unsupported {
        adapter: adapter.to_string(),
        capability,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn turing_and_newer_pass() {
        for capability in [75, 80, 86, 89, 90, 120] {
            assert_eq!(check_capability("card", capability), Ok(()));
        }
    }

    #[test]
    fn older_cards_are_skipped_by_name() {
        for capability in [70, 61, 52] {
            assert!(check_capability("card", capability).is_err());
        }

        let error = check_capability("GeForce GTX 1070", 61).unwrap_err();
        assert_eq!(
            error.to_string(),
            "GeForce GTX 1070 (compute 6.1) is not supported by CUDA 13"
        );
    }
}
