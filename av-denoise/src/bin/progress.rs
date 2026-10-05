use std::sync::OnceLock;

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use tracing_indicatif::IndicatifWriter;
use tracing_indicatif::writer::Stderr;

const BAR_TEMPLATE: &str = "{msg} [{bar:40}] {pos}/{len} ({per_sec}, eta {eta})";

static PROGRESS: OnceLock<MultiProgress> = OnceLock::new();

/// The `MultiProgress` every bar is registered on.
///
/// Bars draw through it so tracing output from [tracing_writer] can suspend them while it writes.
/// It reports itself hidden when stderr is not a terminal, so a redirected run emits nothing.
fn multi() -> &'static MultiProgress {
    PROGRESS.get_or_init(MultiProgress::new)
}

/// The writer for the tracing subscriber.
///
/// Each write is wrapped in `MultiProgress::suspend`, so log lines land above an intact bar
/// instead of overwriting it.
pub fn tracing_writer() -> IndicatifWriter<Stderr> {
    let progress = multi().clone();
    IndicatifWriter::new(progress)
}

/// Whether the denoising progress bar should be drawn.
///
/// The bar is opt-in because it runs for the whole encode alongside whatever the consumer of the
/// output prints, and leaving it off keeps a piped run readable. The terminal check is a parameter
/// so this stays testable without a real tty.
pub fn denoise_bar_visible(progress: bool, stream_is_terminal: bool) -> bool {
    progress && stream_is_terminal
}

/// Builds a bar registered on the shared [multi].
///
/// Returns a hidden bar when `visible` is false, so callers can drive it unconditionally.
fn bar(total_frames: Option<usize>, message: &str, visible: bool) -> ProgressBar {
    if !visible {
        return ProgressBar::hidden();
    }

    let progress_bar = match total_frames {
        Some(total) => ProgressBar::new(total as u64),
        None => ProgressBar::no_length(),
    };

    if let Ok(style) = ProgressStyle::with_template(BAR_TEMPLATE) {
        progress_bar.set_style(style);
    }

    progress_bar.set_message(message.to_owned());

    multi().add(progress_bar)
}

/// Builds the denoising progress bar, which tracks frames written to the output.
pub fn denoise_progress_bar(total_frames: Option<usize>, visible: bool) -> ProgressBar {
    bar(total_frames, "denoising", visible)
}

/// Clears a finished bar and removes it from [multi] so it leaves no blank line behind.
pub fn finish(progress_bar: &ProgressBar) {
    progress_bar.finish_and_clear();
    multi().remove(progress_bar);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn denoise_bar_shown_when_requested_and_terminal() {
        let visible = denoise_bar_visible(true, true);

        assert!(visible);
    }

    #[test]
    fn denoise_bar_hidden_when_not_requested() {
        let visible = denoise_bar_visible(false, true);

        assert!(!visible);
    }

    #[test]
    fn denoise_bar_hidden_when_stream_is_not_a_terminal() {
        let visible = denoise_bar_visible(true, false);

        assert!(!visible);
    }

    #[test]
    fn denoise_bar_hidden_when_neither_requested_nor_a_terminal() {
        let visible = denoise_bar_visible(false, false);

        assert!(!visible);
    }

    #[test]
    fn bar_template_parses() {
        let style = ProgressStyle::with_template(BAR_TEMPLATE);

        assert!(style.is_ok());
    }

    #[test]
    fn denoise_bar_hidden_when_not_visible() {
        let progress_bar = denoise_progress_bar(Some(10), false);

        assert!(progress_bar.is_hidden());
    }

    #[test]
    fn denoise_bar_uses_total_as_length() {
        let progress_bar = denoise_progress_bar(Some(10), true);

        assert_eq!(progress_bar.length(), Some(10));
    }

    #[test]
    fn denoise_bar_without_total_has_no_length() {
        let progress_bar = denoise_progress_bar(None, true);

        assert_eq!(progress_bar.length(), None);
    }

    #[test]
    fn finish_marks_the_bar_done() {
        let progress_bar = denoise_progress_bar(Some(10), true);

        finish(&progress_bar);

        assert!(progress_bar.is_finished());
    }
}
