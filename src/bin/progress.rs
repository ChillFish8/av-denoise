use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};

/// Bar style for the scene-detection pass.
const SCENE_TEMPLATE: &str = "{msg} [{bar:40}] {pos}/{len} ({per_sec}, eta {eta})";

/// Whether the scene-detection progress bar should be drawn.
///
/// True only when `no_progress` is false and the target stream is a
/// terminal. The terminal check is a parameter rather than read
/// directly here so this stays unit-testable without a real tty.
pub fn progress_visible(no_progress: bool, stream_is_terminal: bool) -> bool {
    !no_progress && stream_is_terminal
}

/// Builds the scene-detection progress bar.
///
/// Draws to stderr when `visible` is true. Otherwise returns a hidden
/// bar that never writes anything, so callers can drive it
/// unconditionally without branching on visibility.
pub fn scene_progress_bar(total_frames: Option<usize>, visible: bool) -> ProgressBar {
    if !visible {
        return ProgressBar::hidden();
    }

    let pb = match total_frames {
        Some(n) => ProgressBar::new(n as u64),
        None => ProgressBar::no_length(),
    };

    pb.set_draw_target(ProgressDrawTarget::stderr());

    if let Ok(style) = ProgressStyle::with_template(SCENE_TEMPLATE) {
        pb.set_style(style);
    }

    pb.set_message("scene detection");

    pb
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shown_when_not_disabled_and_terminal() {
        assert!(progress_visible(false, true));
    }

    #[test]
    fn hidden_when_no_progress_flag_set() {
        assert!(!progress_visible(true, true));
    }

    #[test]
    fn hidden_when_stream_is_not_a_terminal() {
        assert!(!progress_visible(false, false));
    }

    #[test]
    fn hidden_when_both_no_progress_and_not_a_terminal() {
        assert!(!progress_visible(true, false));
    }

    #[test]
    fn scene_template_parses() {
        assert!(ProgressStyle::with_template(SCENE_TEMPLATE).is_ok());
    }
}
