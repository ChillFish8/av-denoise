use crate::motion::context::SearchSchedule;

/// The penalty strength the truemotion stage runs at unless a caller sets its own.
pub const DEFAULT_FIELD_LAMBDA: f32 = 1.0;

/// Propagation passes the stage runs, each reading only the field the pass before it wrote.
pub(crate) const TRUEMOTION_PASSES: u32 = 1;

/// Propagation passes the stage runs after a search that escalates.
///
/// The second pass spreads the vectors the wider windows found to the blocks around them.
pub(crate) const ESCALATING_TRUEMOTION_PASSES: u32 = 2;

/// How far around its best candidate a block searches again.
pub(crate) const TRUEMOTION_REFINE_RADIUS: u32 = 1;

/// The widest refine the stage's shared lists hold.
pub(crate) const MAX_REFINE_RADIUS: u32 = 1;

/// How much of the noise SAD floor adds to the penalty per pixel of distance.
///
/// Stronger grain lets far vectors win on flat content more often, so the penalty grows with it.
pub(crate) const TRUEMOTION_NOISE_WEIGHT: f32 = 1.0;

/// The share of its own SAD a candidate pays when it is not the neighbourhood's median.
pub(crate) const TRUEMOTION_SURCHARGE: f32 = 0.125;

/// How many `thsad` past the noise floor the median's SAD can sit before the penalty halves.
///
/// A median that matches poorly usually means occlusion or a layer moving on its own, where the
/// neighbourhood's vector is no guide.
pub(crate) const TRUEMOTION_FADE_SCALE: f32 = 2.0;

/// How the truemotion stage picks each block's vector.
///
/// Each block's cost is its SAD plus a penalty per pixel of distance from its neighbourhood's
/// median. The penalty is `field_lambda` times the sum of the block area times
/// [THSAD_PIXEL](crate::motion::THSAD_PIXEL) and `noise_weight` times the noise SAD floor. It
/// fades as the median's own match worsens, reaching half at `fade_scale` times `thsad` past the
/// floor. A `fade_scale` of 0 turns the fade off. Candidates other than the median also pay
/// `surcharge` times their SAD.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct TruemotionParams {
    pub field_lambda: f32,
    pub noise_weight: f32,
    pub surcharge: f32,
    pub fade_scale: f32,
    pub refine_radius: u32,
    pub passes: u32,
}

impl Default for TruemotionParams {
    fn default() -> Self {
        Self {
            field_lambda: DEFAULT_FIELD_LAMBDA,
            noise_weight: TRUEMOTION_NOISE_WEIGHT,
            surcharge: TRUEMOTION_SURCHARGE,
            fade_scale: TRUEMOTION_FADE_SCALE,
            refine_radius: TRUEMOTION_REFINE_RADIUS,
            passes: TRUEMOTION_PASSES,
        }
    }
}

impl TruemotionParams {
    /// One pass at `field_lambda` with no noise scaling, fade, surcharge or refine.
    #[cfg(test)]
    pub fn legacy(field_lambda: f32) -> Self {
        Self {
            field_lambda,
            noise_weight: 0.0,
            surcharge: 0.0,
            fade_scale: 0.0,
            refine_radius: 0,
            passes: 1,
        }
    }

    /// The stage that ends a search on `schedule`, which runs a second pass when the search
    /// escalates.
    pub fn for_schedule(schedule: SearchSchedule) -> Self {
        let passes = if schedule.escalates() {
            ESCALATING_TRUEMOTION_PASSES
        } else {
            TRUEMOTION_PASSES
        };

        Self {
            passes,
            ..Self::default()
        }
    }

    /// Whether the stage runs at all, which a `field_lambda` of 0 turns off.
    pub fn enabled(&self) -> bool {
        self.field_lambda > 0.0 && self.passes > 0
    }
}
