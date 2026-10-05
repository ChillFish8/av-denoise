use std::ffi::OsStr;

/// Stack bytes the kernel codegen thread needs.
pub const CODEGEN_STACK_BYTES: usize = 16 << 20;

/// Raises `RUST_MIN_STACK` to [CODEGEN_STACK_BYTES] when it is unset.
///
/// # Safety
///
/// The same safety rules as [std::env::set_var] apply.
pub unsafe fn raise_codegen_stack_limit() {
    if std::env::var_os("RUST_MIN_STACK").is_none() {
        let stack_bytes = CODEGEN_STACK_BYTES.to_string();

        // SAFETY: forwarded from this function's own precondition.
        unsafe { std::env::set_var("RUST_MIN_STACK", stack_bytes) };
    }
}

/// Whether the process's `RUST_MIN_STACK` is large enough for codegen.
pub fn codegen_stack_is_sufficient() -> bool {
    let raw = std::env::var_os("RUST_MIN_STACK");
    limit_is_sufficient(raw.as_deref())
}

/// Checks a raw `RUST_MIN_STACK` value against [CODEGEN_STACK_BYTES].
///
/// A value that is unset or fails to parse is insufficient.
fn limit_is_sufficient(raw: Option<&OsStr>) -> bool {
    raw.and_then(|value| value.to_str())
        .and_then(|text| text.parse::<usize>().ok())
        .is_some_and(|bytes| bytes >= CODEGEN_STACK_BYTES)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unset_is_insufficient() {
        assert!(!limit_is_sufficient(None));
    }

    #[test]
    fn zero_is_insufficient() {
        let raw = OsStr::new("0");
        let sufficient = limit_is_sufficient(Some(raw));

        assert!(!sufficient);
    }

    #[test]
    fn one_below_the_limit_is_insufficient() {
        let value = (CODEGEN_STACK_BYTES - 1).to_string();

        let raw = OsStr::new(&value);
        let sufficient = limit_is_sufficient(Some(raw));

        assert!(!sufficient);
    }

    #[test]
    fn exactly_the_limit_is_sufficient() {
        let value = CODEGEN_STACK_BYTES.to_string();

        let raw = OsStr::new(&value);
        let sufficient = limit_is_sufficient(Some(raw));

        assert!(sufficient);
    }

    #[test]
    fn above_the_limit_is_sufficient() {
        let value = (CODEGEN_STACK_BYTES * 2).to_string();

        let raw = OsStr::new(&value);
        let sufficient = limit_is_sufficient(Some(raw));

        assert!(sufficient);
    }

    #[test]
    fn surrounding_whitespace_is_insufficient() {
        let value = format!("  {CODEGEN_STACK_BYTES}  ");

        let raw = OsStr::new(&value);
        let sufficient = limit_is_sufficient(Some(raw));

        assert!(!sufficient);
    }

    #[test]
    fn non_numeric_is_insufficient() {
        let raw = OsStr::new("not-a-number");
        let sufficient = limit_is_sufficient(Some(raw));

        assert!(!sufficient);
    }
}
