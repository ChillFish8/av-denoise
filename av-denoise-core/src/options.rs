/// Speed vs quality dial.
///
/// `nlmeans` fills in its knobs from it with
/// [nlmeans_variant_for](crate::nlmeans_variant_for),
/// [nlmeans_temporal_radius_for](crate::nlmeans_temporal_radius_for), and
/// [nlmeans_search_radius_for](crate::nlmeans_search_radius_for).
/// `nl4d` only runs `base`, which is its library defaults.
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
