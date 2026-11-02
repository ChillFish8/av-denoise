use crate::motion::MAX_SEARCH_RADIUS;

/// Speed vs quality dial.
///
/// `nlmeans` fills in its knobs from it with
/// [nlmeans_variant_for](crate::nlmeans_variant_for),
/// [nlmeans_temporal_radius_for](crate::nlmeans_temporal_radius_for), and
/// [nlmeans_search_radius_for](crate::nlmeans_search_radius_for).
/// `nl4d` runs `base`, which is its library defaults, and `slow`. Both read
/// [search_escalation_for](crate::search_escalation_for) when motion is tracked.
///
/// Both front ends parse the same names from this one type, so a preset
/// resolves to the same dials everywhere it is used.
#[derive(Debug, Copy, Clone, Default, PartialEq, Eq, strum_macros::EnumString)]
#[strum(ascii_case_insensitive)]
pub enum Preset {
    /// Fastest and lowest quality.
    Veryfast,
    /// One step up from `veryfast`.
    Fast,
    /// The default, favouring quality over speed.
    #[default]
    Base,
    /// One step down from `veryslow`.
    Slow,
    /// Slowest and highest quality.
    Veryslow,
}

/// The widest radius a motion block that fails its first search searches again with, at a preset.
///
/// It is 8 for `slow` and `veryslow`, and `None` for the other presets.
pub fn search_escalation_for(preset: Preset) -> Option<u32> {
    match preset {
        Preset::Veryfast | Preset::Fast | Preset::Base => None,
        Preset::Slow | Preset::Veryslow => Some(MAX_SEARCH_RADIUS),
    }
}
