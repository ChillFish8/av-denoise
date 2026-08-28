use std::sync::OnceLock;

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use tracing_indicatif::IndicatifWriter;
use tracing_indicatif::writer::Stderr;

/// Bar style shared by every phase.
const BAR_TEMPLATE: &str = "{msg} [{bar:40}] {pos}/{len} ({per_sec}, eta {eta})";

/// The process-wide bar container.
static PROGRESS: OnceLock<MultiProgress> = OnceLock::new();

/// The `MultiProgress` every bar is registered on.
///
/// Bars go through this rather than drawing to stderr directly. That
/// lets tracing output routed through [`tracing_writer`] suspend them
/// while it writes.
///
/// `MultiProgress::new` draws to stderr and reports itself hidden when
/// stderr is not a terminal, so a redirected run emits nothing.
fn multi() -> &'static MultiProgress {
    PROGRESS.get_or_init(MultiProgress::new)
}

/// The writer to hand to the tracing subscriber.
///
/// Each write is wrapped in `MultiProgress::suspend`, so log lines land
/// above an intact bar instead of overwriting it.
pub fn tracing_writer() -> IndicatifWriter<Stderr> {
    IndicatifWriter::new(multi().clone())
}

/// Whether the denoising progress bar should be drawn.
///
/// That bar is opt-in. It shows only when `progress` is set and the
/// target stream is a terminal.
///
/// It runs for the whole encode, alongside whatever the consumer of our
/// output is printing, so leaving it off by default keeps a piped run
/// readable. The scene-detection bar has no such conflict and only
/// checks the terminal.
///
/// The terminal check is a parameter rather than read directly here so
/// this stays unit-testable without a real tty.
pub fn denoise_bar_visible(progress: bool, stream_is_terminal: bool) -> bool {
    progress && stream_is_terminal
}

/// Builds a bar registered on the shared [`multi`].
///
/// Returns a hidden bar when `visible` is false, so callers can drive it
/// unconditionally without branching on visibility.
fn bar(total_frames: Option<usize>, message: &str, visible: bool) -> ProgressBar {
    if !visible {
        return ProgressBar::hidden();
    }

    let pb = match total_frames {
        Some(n) => ProgressBar::new(n as u64),
        None => ProgressBar::no_length(),
    };

    if let Ok(style) = ProgressStyle::with_template(BAR_TEMPLATE) {
        pb.set_style(style);
    }

    pb.set_message(message.to_owned());

    multi().add(pb)
}

/// Builds the scene-detection progress bar.
///
/// The frame count is optional because the decoder does not always know
/// the length of the input up front.
pub fn scene_progress_bar(total_frames: Option<usize>, visible: bool) -> ProgressBar {
    bar(total_frames, "scene detection", visible)
}

/// Builds the denoising progress bar, tracking frames written to the
/// output.
pub fn denoise_progress_bar(total_frames: usize, visible: bool) -> ProgressBar {
    bar(Some(total_frames), "denoising", visible)
}

/// Clears a finished bar and drops it from the shared [`multi`], so it
/// leaves no blank line behind.
pub fn finish(pb: &ProgressBar) {
    pb.finish_and_clear();
    multi().remove(pb);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn denoise_bar_shown_when_requested_and_terminal() {
        assert!(denoise_bar_visible(true, true));
    }

    #[test]
    fn denoise_bar_hidden_when_not_requested() {
        assert!(!denoise_bar_visible(false, true));
    }

    #[test]
    fn denoise_bar_hidden_when_stream_is_not_a_terminal() {
        assert!(!denoise_bar_visible(true, false));
    }

    #[test]
    fn denoise_bar_hidden_when_neither_requested_nor_a_terminal() {
        assert!(!denoise_bar_visible(false, false));
    }

    #[test]
    fn bar_template_parses() {
        assert!(ProgressStyle::with_template(BAR_TEMPLATE).is_ok());
    }

    #[test]
    fn denoise_bar_hidden_when_not_visible() {
        assert!(denoise_progress_bar(10, false).is_hidden());
    }

    #[test]
    fn denoise_bar_uses_total_as_length() {
        assert_eq!(denoise_progress_bar(10, true).length(), Some(10));
    }

    #[test]
    fn scene_bar_without_total_has_no_length() {
        assert_eq!(scene_progress_bar(None, true).length(), None);
    }

    #[test]
    fn finish_marks_the_bar_done() {
        let pb = denoise_progress_bar(10, true);
        finish(&pb);
        assert!(pb.is_finished());
    }
}
