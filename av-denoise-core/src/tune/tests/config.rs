use crate::tune::{BudgetRequest, TuningConfig, TuningMode};

#[test]
fn tuning_mode_parses_auto_cached_and_off() {
    for (text, mode) in [
        ("auto", TuningMode::Auto),
        ("CACHED", TuningMode::Cached),
        ("Off", TuningMode::Off),
    ] {
        let parsed: TuningMode = text.parse().expect("a known mode");
        assert_eq!(parsed, mode);

        let printed = mode.to_string();
        let reparsed: TuningMode = printed.parse().expect("a printed mode");
        assert_eq!(reparsed, mode);
    }

    assert!("sometimes".parse::<TuningMode>().is_err());
}

#[test]
fn auto_runs_as_cached_after_tuning() {
    assert_eq!(TuningMode::Auto.after_tuning(), TuningMode::Cached);
    assert_eq!(TuningMode::Cached.after_tuning(), TuningMode::Cached);
    assert_eq!(TuningMode::Off.after_tuning(), TuningMode::Off);
}

#[test]
fn the_default_config_tunes_automatically_with_no_store() {
    let config = TuningConfig::default();

    assert_eq!(config.mode, TuningMode::Auto);
    assert_eq!(config.store, None);
    assert_eq!(config.budget, BudgetRequest::Automatic { frames: None });
    assert!(!config.deterministic);
}
