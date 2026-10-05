use std::collections::BTreeSet;

use av_decoders::Decoder;
use ffms2_sys::{FFMS_GetFrameInfo, FFMS_GetNumFrames, FFMS_GetTrackFromVideo};

/// One entry of the ffms2 video index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexEntry {
    /// Presentation timestamp, in the video track's time base.
    pub pts: i64,
    pub keyframe: bool,
}

/// Reads the video index behind `decoder`.
///
/// Returns `None` when `decoder` is not backed by ffms2 or when ffms2 cannot describe the track.
pub fn read_index(decoder: &mut Decoder) -> Option<Vec<IndexEntry>> {
    let total = decoder.get_video_details().total_frames?;
    let source = decoder.get_ffms2_impl()?.video_source;

    // SAFETY: a live `Ffms2Decoder` holds a non-null video source, and the track belongs to that
    // source, so it stays valid for as long as the decoder does.
    let track = unsafe { FFMS_GetTrackFromVideo(source) };

    if track.is_null() {
        return None;
    }

    // SAFETY: `track` is non-null and outlives this call.
    let track_frames = unsafe { FFMS_GetNumFrames(track) };

    if track_frames < 0 {
        return None;
    }

    // `FFMS_GetFrameInfo` does not check its bound, so the read stays under the track's own count
    // in case the separately reported video properties disagree with it.
    let total = total.min(track_frames as usize);
    let mut index = Vec::with_capacity(total);

    for i in 0..total {
        // SAFETY: `track` is non-null, and `i` stays below the entry count the track reports.
        let info_ptr = unsafe { FFMS_GetFrameInfo(track, i as i32) };

        if info_ptr.is_null() {
            return None;
        }

        // SAFETY: checked non-null just above.
        let info = unsafe { &*info_ptr };

        index.push(IndexEntry {
            pts: info.PTS,
            keyframe: info.KeyFrame != 0,
        });
    }

    Some(index)
}

/// Returns the index positions that carry no picture of their own.
pub fn phantom_indices(index: &[IndexEntry]) -> BTreeSet<usize> {
    let mut phantom = BTreeSet::new();

    if index.is_empty() {
        return phantom;
    }

    // Nothing before the first keyframe can be decoded, so ffms2 answers those positions with a
    // repeat of the keyframe.
    let first_keyframe = index.iter().position(|entry| entry.keyframe).unwrap_or(0);

    // One leading picture is the ordinary case. More means the index marks no keyframe for a
    // stretch of the file, so the drop is reported rather than shortening the output quietly.
    if first_keyframe > 1 {
        tracing::warn!(
            dropped = first_keyframe,
            "the index marks no keyframe until entry {first_keyframe}, dropping every entry before it",
        );
    }

    phantom.extend(0..first_keyframe);

    let gaps: Vec<i64> = (first_keyframe + 1..index.len())
        .map(|i| index[i].pts.saturating_sub(index[i - 1].pts))
        .collect();

    if gaps.is_empty() {
        return phantom;
    }

    // The median gap is the clip's real frame spacing. A handful of phantom entries cannot move
    // it, however far apart they sit.
    let mut sorted_gaps = gaps.clone();
    sorted_gaps.sort_unstable();
    let median = sorted_gaps[sorted_gaps.len() / 2];

    // Variable frame rate pacing puts real frames closer together than the median, which is the
    // same signature a phantom leaves. Timing only tells them apart on an otherwise regular clip,
    // so an irregular clip keeps every entry.
    let regular_gaps = gaps
        .iter()
        .filter(|&&gap| gap.saturating_sub(median).saturating_abs().saturating_mul(4) <= median)
        .count();

    if regular_gaps * 10 < gaps.len() * 9 {
        tracing::debug!(
            regular = regular_gaps,
            gaps = gaps.len(),
            "frame spacing is too irregular to tell phantom entries from variable frame rate pacing",
        );

        return phantom;
    }

    // A phantom lands just ahead of the frame that follows it, so the earlier entry of a
    // too-close pair is dropped. Doubling the gap rather than halving the median keeps the
    // comparison exact when the median gap is a single time base unit.
    for (offset, &gap) in gaps.iter().enumerate() {
        if gap.saturating_mul(2) < median {
            phantom.insert(first_keyframe + offset);
        }
    }

    phantom
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds an index at a steady 42-unit spacing with the first entry as the keyframe.
    fn regular(count: usize) -> Vec<IndexEntry> {
        (0..count)
            .map(|i| IndexEntry {
                pts: i as i64 * 42,
                keyframe: i == 0,
            })
            .collect()
    }

    #[test]
    fn a_regular_index_has_no_phantoms() {
        let index = regular(20);
        let phantom = phantom_indices(&index);

        assert!(phantom.is_empty());
    }

    #[test]
    fn an_empty_index_has_no_phantoms() {
        let phantom = phantom_indices(&[]);

        assert!(phantom.is_empty());
    }

    #[test]
    fn entries_before_the_first_keyframe_are_phantom() {
        let mut index = vec![IndexEntry {
            pts: 0,
            keyframe: false,
        }];
        let following = (1..20).map(|i| IndexEntry {
            pts: 41 + (i as i64 - 1) * 42,
            keyframe: i == 1,
        });
        index.extend(following);

        let phantom = phantom_indices(&index);

        assert_eq!(phantom, BTreeSet::from([0]));
    }

    #[test]
    fn the_earlier_entry_of_a_too_close_pair_is_phantom() {
        let mut index = vec![
            IndexEntry {
                pts: 0,
                keyframe: true,
            },
            IndexEntry {
                pts: 41,
                keyframe: false,
            },
            IndexEntry {
                pts: 42,
                keyframe: false,
            },
            IndexEntry {
                pts: 82,
                keyframe: false,
            },
            IndexEntry {
                pts: 84,
                keyframe: false,
            },
        ];
        let following = (5..60).map(|i| IndexEntry {
            pts: 125 + (i as i64 - 5) * 42,
            keyframe: false,
        });
        index.extend(following);

        let phantom = phantom_indices(&index);

        assert_eq!(phantom, BTreeSet::from([1, 3]));
    }

    #[test]
    fn a_time_base_as_tight_as_the_frame_rate_still_finds_a_phantom() {
        // Every real frame is one unit apart, so a phantom shows up as a repeated timestamp rather
        // than as a fraction of a wider gap.
        let mut index: Vec<IndexEntry> = (0..40)
            .map(|i| IndexEntry {
                pts: i as i64,
                keyframe: i == 0,
            })
            .collect();

        index[4].pts = index[3].pts;

        let phantom = phantom_indices(&index);

        assert_eq!(phantom, BTreeSet::from([3]));
    }

    #[test]
    fn variable_frame_rate_pacing_keeps_every_entry() {
        // A third of these frames arrive at a third of the median spacing. They are real, and the
        // timing rule cannot tell them from phantoms.
        let mut pts = 0;
        let index: Vec<IndexEntry> = (0..30)
            .map(|i| {
                let entry = IndexEntry {
                    pts,
                    keyframe: i == 0,
                };

                pts += if i % 3 == 2 { 10 } else { 30 };
                entry
            })
            .collect();

        let phantom = phantom_indices(&index);

        assert!(phantom.is_empty());
    }

    #[test]
    fn repeated_pictures_at_regular_spacing_are_kept() {
        let index = regular(2159);
        let phantom = phantom_indices(&index);

        assert!(phantom.is_empty());
    }
}
