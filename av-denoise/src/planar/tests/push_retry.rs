use super::*;

#[test]
fn ok_means_no_retry() {
    let outcome = push_needs_retry(Ok(())).expect("Ok(()) must not itself error");
    assert!(!outcome, "a landed push must not ask the caller to retry");
}

#[test]
fn queue_full_signals_retry() {
    let outcome = push_needs_retry(Err(DenoiserError::QueueFull)).expect("QueueFull must not itself error");
    assert!(outcome, "QueueFull must still trigger the retry-after-drain path");
}

#[test]
fn non_queue_full_errors_propagate_instead_of_being_swallowed() {
    let cause = anyhow::anyhow!("synthetic readback failure");
    let synthetic = DenoiserError::Other(cause);

    let outcome = push_needs_retry(Err(synthetic));

    assert!(
        outcome.is_err(),
        "a non-QueueFull push error must propagate instead of being silently treated as success"
    );
}
