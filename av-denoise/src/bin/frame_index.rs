use std::collections::BTreeSet;

use av_decoders::Decoder;
use ffms2_sys::{FFMS_GetFrameInfo, FFMS_GetNumFrames, FFMS_GetTrackFromVideo};

/// One entry of the ffms2 video index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexEntry {
    /// Presentation timestamp, in the video track's time base.
    pub pts: i64,
    /// Whether ffms2 marked this entry as a keyframe.
    pub keyframe: bool,
}

/// Reads the video index behind `decoder`.
///
/// Returns `None` when `decoder` is not backed by ffms2, or when ffms2 declines to describe the
/// track, when this happens caller then keeps every frame.
pub fn read_index(decoder: &mut Decoder) -> Option<Vec<IndexEntry>> {
    let total = decoder.get_video_details().total_frames?;
    let source = decoder.get_ffms2_impl()?.video_source;

    // SAFETY: a live `Ffms2Decoder` holds a non-null video source, and
    // the track belongs to that source rather than to us, so it stays
    // valid for as long as the decoder does.
    let track = unsafe { FFMS_GetTrackFromVideo(source) };

    if track.is_null() {
        return None;
    }

    // SAFETY: `track` is non-null and outlives this call.
    let track_frames = unsafe { FFMS_GetNumFrames(track) };

    if track_frames < 0 {
        return None;
    }

    // `FFMS_GetFrameInfo` indexes the track's entries without checking the
    // bound, so the track's own count is what the read has to stay under.
    // The video properties describe the same track but are reported
    // separately, and a disagreement must not turn into a read past the end.
    let total = total.min(track_frames as usize);
    let mut index = Vec::with_capacity(total);

    for i in 0..total {
        // SAFETY: `track` is non-null, and `i` stays below the entry
        // count the track itself reports.
        let info = unsafe { FFMS_GetFrameInfo(track, i as i32) };

        if info.is_null() {
            return None;
        }

        // SAFETY: checked non-null just above.
        let info = unsafe { &*info };

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

    // Nothing before the first keyframe can be decoded, so ffms2
    // answers those positions with a repeat of the keyframe.
    let lead = index.iter().position(|e| e.keyframe).unwrap_or(0);

    // One leading picture is the ordinary case. More than that means the
    // index marks no keyframe for a stretch of the file, so say how much
    // is going rather than shortening the output quietly.
    if lead > 1 {
        tracing::warn!(
            dropped = lead,
            "the index marks no keyframe until entry {lead}, dropping every entry before it",
        );
    }

    phantom.extend(0..lead);

    let gaps: Vec<i64> = (lead + 1..index.len())
        .map(|i| index[i].pts.saturating_sub(index[i - 1].pts))
        .collect();

    if gaps.is_empty() {
        return phantom;
    }

    // The median gap is the clip's real frame spacing. A handful of
    // phantom entries cannot move it, however far apart they sit.
    let mut sorted = gaps.clone();
    sorted.sort_unstable();
    let median = sorted[sorted.len() / 2];

    // Variable frame rate pacing puts real frames closer together than the
    // median, which is the same signature a phantom leaves. Telling the two
    // apart by timing only works on a clip that is otherwise regular, so a
    // clip that is not keeps every entry.
    let regular = gaps
        .iter()
        .filter(|&&gap| gap.saturating_sub(median).saturating_abs().saturating_mul(4) <= median)
        .count();

    if regular * 10 < gaps.len() * 9 {
        tracing::debug!(
            regular,
            gaps = gaps.len(),
            "frame spacing is too irregular to tell phantom entries from variable frame rate pacing",
        );

        return phantom;
    }

    // A phantom shares a timeline slot with the frame that follows it, landing just ahead of
    // that frame's timestamp. The entry to drop is therefore the earlier of the pair.
    // Doubling the gap rather than halving the median keeps the comparison exact,
    // which matters when a clip's time base is close enough to its frame rate that the median
    // gap is a single unit.
    for (offset, &gap) in gaps.iter().enumerate() {
        if gap.saturating_mul(2) < median {
            phantom.insert(lead + offset);
        }
    }

    phantom
}

/// Rewrites scene boundaries from ffms2 index space into the output
/// frame numbering that dropping `phantom` produces.
///
/// A boundary that lands on a dropped entry moves onto the next frame that survives,
/// which can leave it equal to its neighbour. Those collapse, because a scene cannot
/// start where the one before it does.
pub fn remap_scene_starts(starts: &[usize], phantom: &BTreeSet<usize>) -> Vec<usize> {
    let mut out: Vec<usize> = starts
        .iter()
        .map(|&raw| raw - phantom.range(..raw).count())
        .collect();

    out.dedup();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds an index at a steady 42-unit spacing, with the first entry marked as the keyframe.
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
        assert!(phantom_indices(&regular(20)).is_empty());
    }

    #[test]
    fn an_empty_index_has_no_phantoms() {
        assert!(phantom_indices(&[]).is_empty());
    }

    #[test]
    fn entries_before_the_first_keyframe_are_phantom() {
        let mut index = vec![IndexEntry {
            pts: 0,
            keyframe: false,
        }];
        index.extend((1..20).map(|i| IndexEntry {
            pts: 41 + (i as i64 - 1) * 42,
            keyframe: i == 1,
        }));

        assert_eq!(phantom_indices(&index), BTreeSet::from([0]));
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
        index.extend((5..60).map(|i| IndexEntry {
            pts: 125 + (i as i64 - 5) * 42,
            keyframe: false,
        }));

        assert_eq!(phantom_indices(&index), BTreeSet::from([1, 3]));
    }

    #[test]
    fn a_time_base_as_tight_as_the_frame_rate_still_finds_a_phantom() {
        // Every real frame is one unit apart, so a phantom shows up as a
        // repeated timestamp rather than as a fraction of a wider gap.
        let mut index: Vec<IndexEntry> = (0..40)
            .map(|i| IndexEntry {
                pts: i as i64,
                keyframe: i == 0,
            })
            .collect();

        index[4].pts = index[3].pts;

        assert_eq!(phantom_indices(&index), BTreeSet::from([3]));
    }

    #[test]
    fn variable_frame_rate_pacing_keeps_every_entry() {
        // A third of these frames arrive at a third of the median spacing.
        // They are real, and the timing rule cannot tell them from phantoms.
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

        assert!(phantom_indices(&index).is_empty());
    }

    #[test]
    fn repeated_pictures_at_regular_spacing_are_kept() {
        assert!(phantom_indices(&regular(2159)).is_empty());
    }

    #[test]
    fn remap_shifts_boundaries_past_each_dropped_entry() {
        let phantom = BTreeSet::from([1, 3]);

        // Raw 0 stays put. Raw 2 loses the one phantom below it, raw 5
        // loses both.
        assert_eq!(remap_scene_starts(&[0, 2, 5], &phantom), vec![0, 1, 3]);
    }

    #[test]
    fn remap_collapses_a_boundary_that_lands_on_a_dropped_entry() {
        // Raw 1 is itself dropped, so it maps onto the same output
        // frame as raw 2. The duplicate boundary must not survive.
        assert_eq!(remap_scene_starts(&[0, 1, 2], &BTreeSet::from([1])), vec![0, 1]);
    }
}
