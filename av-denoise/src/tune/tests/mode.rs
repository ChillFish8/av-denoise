use crate::tune::{TuningMode, resolve_mode};

#[test]
fn the_environment_overrides_the_flag() {
    assert_eq!(resolve_mode(TuningMode::Auto, Some("off")), TuningMode::Off);
    assert_eq!(resolve_mode(TuningMode::Off, Some("cached")), TuningMode::Cached);
    assert_eq!(resolve_mode(TuningMode::Cached, None), TuningMode::Cached);
}

#[test]
fn an_unknown_environment_value_keeps_the_flag() {
    assert_eq!(
        resolve_mode(TuningMode::Cached, Some("bogus")),
        TuningMode::Cached
    );
}
