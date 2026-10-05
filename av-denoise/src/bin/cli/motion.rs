use av_denoise::MotionSearch;

/// Motion-search flags for the subcommands that track motion between frames.
#[derive(Debug, Clone, Default, clap::Args)]
pub struct MotionArgs {
    /// Size of each motion-search block, in pixels. Must be even.
    ///
    /// Larger blocks are more stable but track motion less
    /// accurately on small details.
    ///
    /// Defaults to 16 when unset.
    #[arg(long)]
    pub mc_blksize: Option<u32>,

    /// How many pixels neighbouring motion blocks may overlap.
    ///
    /// Must be less than `--mc-blksize`. Higher overlap smooths the
    /// transitions between blocks but does more work.
    ///
    /// Defaults to 8 when unset.
    #[arg(long)]
    pub mc_overlap: Option<u32>,

    /// How many pixels of motion to search for at the finest level.
    ///
    /// The coarse pyramid pass reaches further (search radius times
    /// 2 for a 2-level pyramid), so for typical content the default
    /// is fine. Raise it for very fast motion.
    ///
    /// Defaults to 4 when unset.
    #[arg(long)]
    pub mc_search: Option<u32>,

    /// How many levels the motion-search pyramid uses.
    ///
    /// `1` does a single full-resolution search (cheaper, weaker on
    /// large motion).
    ///
    /// `2` (default) does a coarse pass on a half-size image first,
    /// then refines at full resolution. This handles much larger motion
    /// at modest extra cost.
    ///
    /// Defaults to 2 when unset.
    #[arg(long)]
    pub mc_pyramid_levels: Option<u32>,
}

impl MotionArgs {
    /// Whether any of these flags was given.
    pub fn any_set(&self) -> bool {
        self.mc_blksize.is_some()
            || self.mc_overlap.is_some()
            || self.mc_search.is_some()
            || self.mc_pyramid_levels.is_some()
    }

    /// Converts these flags to a [MotionSearch], with the library default for each unset flag.
    pub fn to_motion_search(&self) -> MotionSearch {
        let defaults = MotionSearch::default();

        MotionSearch {
            blksize: self.mc_blksize.unwrap_or(defaults.blksize),
            overlap: self.mc_overlap.unwrap_or(defaults.overlap),
            search_radius: self.mc_search.unwrap_or(defaults.search_radius),
            pyramid_levels: self.mc_pyramid_levels.unwrap_or(defaults.pyramid_levels),
            estimation: defaults.estimation,
        }
    }
}
