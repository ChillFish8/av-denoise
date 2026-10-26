pub(crate) const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Folds `bytes` into `hash` with FNV-1a.
pub(crate) fn fnv1a(mut hash: u64, bytes: &[u8]) -> u64 {
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// The key of a build's kernel cache and tune store.
///
/// It covers the kernel sources and the CubeCL version, so a CubeCL upgrade starts a fresh store.
pub(crate) fn kernel_hash(source_hash: u64, cubecl_version: &str) -> u64 {
    let separated = fnv1a(source_hash, &[0]);
    fnv1a(separated, cubecl_version.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_kernel_hash_changes_with_the_cubecl_version() {
        let source_hash = fnv1a(FNV_OFFSET_BASIS, b"kernel sources");

        let old_release = kernel_hash(source_hash, "0.10.0");
        let new_release = kernel_hash(source_hash, "0.11.3");
        let same_release = kernel_hash(source_hash, "0.11.3");

        assert_ne!(old_release, new_release);
        assert_eq!(new_release, same_release);
    }
}
