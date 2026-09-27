use std::str::FromStr;

/// How finely nl4d aligns a temporal match between whole pixels.
#[derive(Debug, Copy, Clone, Default, Eq, PartialEq, Hash)]
pub enum SubpelPrecision {
    #[default]
    Off,
    Half,
    Quarter,
}

impl FromStr for SubpelPrecision {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "off" => Ok(SubpelPrecision::Off),
            "half" => Ok(SubpelPrecision::Half),
            "quarter" => Ok(SubpelPrecision::Quarter),
            other => Err(format!(
                "subpel must be one of off, half or quarter, got '{other}'"
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subpel_precision_parses_case_insensitively() {
        assert_eq!("off".parse::<SubpelPrecision>(), Ok(SubpelPrecision::Off));
        assert_eq!("Half".parse::<SubpelPrecision>(), Ok(SubpelPrecision::Half));
        assert_eq!("QUARTER".parse::<SubpelPrecision>(), Ok(SubpelPrecision::Quarter));
    }

    #[test]
    fn subpel_precision_rejects_unknown_names() {
        let error = "eighth".parse::<SubpelPrecision>().unwrap_err();
        assert!(error.contains("off, half or quarter"), "{error}");
    }
}
