/// Perceptual tuning that filters each area by what it looks like.
///
/// Flat, grainy areas are filtered harder, textured dark areas more gently, and the areas around
/// strong lines keep the base threshold. Every knob needs the noise map on.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct PsyParams {
    /// How much harder flat, grainy 8x8 areas are filtered, as a multiplier on the luma threshold.
    ///
    /// An area counts as flat when its texture is small next to its own grain. `1.0` turns it off.
    /// Between 1.0 and 3.0, defaults to 1.75.
    pub flat_boost: f32,
    /// [Self::flat_boost] for the chroma planes, with flatness measured on the first chroma plane.
    ///
    /// Applies when chroma is denoised in its own pass. `1.0` turns it off. Between 1.0 and 3.0,
    /// defaults to 1.5.
    pub chroma_flat_boost: f32,
    /// How much more gently textured dark areas are filtered, as a multiplier on the luma
    /// threshold.
    ///
    /// It applies in full at or below luma 128 of 255 and fades back to 1.0 by 160. `1.0` turns it
    /// off. Between 0.1 and 1.0, defaults to 0.65.
    pub shadow_soften: f32,
    /// How strongly the grain in a flat area may line up before the area counts as texture instead.
    ///
    /// Grain points every which way, while faint lines and edges share one direction. A flat area
    /// whose surroundings line up at or above this cut loses the flat boost. Luma only. `1.0` turns
    /// it off. Between 0.0 and 1.0, defaults to 0.21.
    pub flat_texture_cut: f32,
    /// How far around a strong line, in pixels, shadow soften and the flat texture cut stop
    /// applying.
    ///
    /// The areas beside dark ink lines read as dark texture, so they would otherwise be filtered
    /// more gently and keep a band of grain. Luma only. `0` turns it off. Between 0 and 64,
    /// defaults to 16.
    pub line_ring: u32,
}

impl Default for PsyParams {
    fn default() -> Self {
        Self {
            flat_boost: 1.75,
            chroma_flat_boost: 1.5,
            shadow_soften: 0.65,
            flat_texture_cut: 0.21,
            line_ring: 16,
        }
    }
}

impl PsyParams {
    /// Rejects a knob outside its range.
    pub fn validate(&self) -> Result<(), String> {
        for (name, value) in [
            ("flat_boost", self.flat_boost),
            ("chroma_flat_boost", self.chroma_flat_boost),
        ] {
            if !(value.is_finite() && (1.0..=3.0).contains(&value)) {
                return Err(format!("{name} must be finite and in 1.0..=3.0, got {value}"));
            }
        }

        if !(self.shadow_soften.is_finite() && (0.1..=1.0).contains(&self.shadow_soften)) {
            return Err(format!(
                "shadow_soften must be finite and in 0.1..=1.0, got {}",
                self.shadow_soften
            ));
        }

        if !(self.flat_texture_cut.is_finite() && (0.0..=1.0).contains(&self.flat_texture_cut)) {
            return Err(format!(
                "flat_texture_cut must be finite and in 0.0..=1.0, got {}",
                self.flat_texture_cut
            ));
        }

        if self.line_ring > 64 {
            return Err(format!("line_ring must be in 0..=64, got {}", self.line_ring));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_defaults_are_the_settled_config() {
        let psy = PsyParams::default();
        assert_eq!(psy.flat_boost, 1.75);
        assert_eq!(psy.chroma_flat_boost, 1.5);
        assert_eq!(psy.shadow_soften, 0.65);
        assert_eq!(psy.flat_texture_cut, 0.21);
        assert_eq!(psy.line_ring, 16);
        assert!(psy.validate().is_ok());
    }

    #[test]
    fn validate_accepts_every_bound() {
        let low = PsyParams {
            flat_boost: 1.0,
            chroma_flat_boost: 1.0,
            shadow_soften: 0.1,
            flat_texture_cut: 0.0,
            line_ring: 0,
        };
        let high = PsyParams {
            flat_boost: 3.0,
            chroma_flat_boost: 3.0,
            shadow_soften: 1.0,
            flat_texture_cut: 1.0,
            line_ring: 64,
        };
        assert!(low.validate().is_ok());
        assert!(high.validate().is_ok());
    }

    #[test]
    fn validate_rejects_boosts_out_of_range() {
        for bad in [0.99f32, 3.01, f32::NAN, f32::INFINITY] {
            let flat = PsyParams {
                flat_boost: bad,
                ..PsyParams::default()
            };
            let error = flat.validate().expect_err("flat_boost out of range");
            assert!(error.contains("flat_boost"), "got {error}");

            let chroma = PsyParams {
                chroma_flat_boost: bad,
                ..PsyParams::default()
            };
            let error = chroma.validate().expect_err("chroma_flat_boost out of range");
            assert!(error.contains("chroma_flat_boost"), "got {error}");
        }
    }

    #[test]
    fn validate_rejects_a_shadow_soften_out_of_range() {
        for bad in [0.05f32, 1.01, f32::NAN] {
            let psy = PsyParams {
                shadow_soften: bad,
                ..PsyParams::default()
            };
            let error = psy.validate().expect_err("shadow_soften out of range");
            assert!(error.contains("shadow_soften"), "got {error}");
        }
    }

    #[test]
    fn validate_rejects_a_texture_cut_out_of_range() {
        for bad in [-0.01f32, 1.01, f32::NAN, f32::INFINITY] {
            let psy = PsyParams {
                flat_texture_cut: bad,
                ..PsyParams::default()
            };
            let error = psy.validate().expect_err("flat_texture_cut out of range");
            assert!(error.contains("flat_texture_cut"), "got {error}");
        }
    }

    #[test]
    fn validate_rejects_a_line_ring_past_64() {
        let psy = PsyParams {
            line_ring: 65,
            ..PsyParams::default()
        };
        let error = psy.validate().expect_err("line_ring out of range");
        assert!(error.contains("line_ring"), "got {error}");
    }
}
