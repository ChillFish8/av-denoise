use std::collections::BTreeSet;

use av_decoders::Decoder;
use ffms2_sys::{FFMS_GetFrameInfo, FFMS_GetTrackFromVideo};

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

    let mut index = Vec::with_capacity(total);

    for i in 0..total {
        // SAFETY: `track` is non-null, and `i` stays below the frame
        // count ffms2 reported for this track.
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
    phantom.extend(0..lead);

    let mut gaps: Vec<i64> = (lead + 1..index.len())
        .map(|i| index[i].pts - index[i - 1].pts)
        .collect();

    if gaps.is_empty() {
        return phantom;
    }

    // The median gap is the clip's real frame spacing. A handful of
    // phantom entries cannot move it, however far apart they sit.
    gaps.sort_unstable();
    let threshold = gaps[gaps.len() / 2] / 2;

    // A phantom shares a timeline slot with the frame that follows it,
    // landing just ahead of that frame's timestamp. The entry to drop
    // is therefore the earlier of the pair.
    for i in lead + 1..index.len() {
        if index[i].pts - index[i - 1].pts < threshold {
            phantom.insert(i - 1);
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
        index.extend((5..20).map(|i| IndexEntry {
            pts: 125 + (i as i64 - 5) * 42,
            keyframe: false,
        }));

        assert_eq!(phantom_indices(&index), BTreeSet::from([1, 3]));
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
