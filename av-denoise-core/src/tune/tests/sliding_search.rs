use super::corpus::{integer_corpus, is_flat_case};
use super::fixtures::{CollabSpec, collab_site, collab_site_with_ring, integer_split_sites};
use super::integer_search::{reference_position, run_search};
use super::runtime::{TestRuntime, test_client};
use crate::collab::MAX_K;
use crate::collab::kernels::fused::{GROUP_USES_GRID, MEMBER_WORDS, Walk};
use crate::nlmeans::ChannelMode;
use crate::tune::sites::{CollabSearchSite, IntegerVariant, SearchVariant};
use crate::tune::{Fidelity, IntegerCodes, SiteCandidate, VariantSite};

const SLIDING_NAMES: [&str; 2] = ["int_sliding7x7", "int_sliding15x7"];

const CHANNEL_MODES: [ChannelMode; 3] = [ChannelMode::Luma, ChannelMode::Chroma, ChannelMode::Yuv];

fn sliding_variants() -> Vec<SearchVariant> {
    SLIDING_NAMES
        .iter()
        .map(|&name| SearchVariant::from_name(name).expect("every sliding variant is listed"))
        .collect()
}

fn scalar_clipped() -> SearchVariant {
    let variant = IntegerVariant {
        walk: Walk::Clipped,
        ..IntegerVariant::reference_for(false)
    };

    SearchVariant::Integer(variant)
}

/// Whether any group of `words` is a volume grid.
fn any_grid_group(words: &[u32]) -> bool {
    words
        .chunks(MEMBER_WORDS as usize)
        .any(|group| group[MAX_K as usize] & GROUP_USES_GRID != 0)
}

/// Checks every sliding variant against the clipped scalar search, word for word, and returns the
/// scalar words.
fn assert_sliding_matches_scalar(search: &CollabSearchSite<TestRuntime, f32>, label: &str) -> Vec<u32> {
    let (scalar_outputs, scalar_words) = run_search(search, scalar_clipped());

    for sliding in sliding_variants() {
        let name = sliding.name();
        let (sliding_outputs, sliding_words) = run_search(search, sliding);

        let verdict = search
            .compare(&scalar_outputs, &sliding_outputs, Fidelity::Exact)
            .expect("the outputs read back");
        assert!(verdict.passed(), "{label} {name}: {verdict:?}");
        assert_eq!(scalar_words, sliding_words, "{label} {name}: member words");
    }

    scalar_words
}

#[test]
fn the_sliding_centre_matches_scalar_int_bitwise() {
    let client = test_client();

    for depth in [8u32, 10] {
        let codes = IntegerCodes { depth };

        for case in integer_corpus(depth) {
            let label = format!("{} {depth}-bit", case.label);
            let fused = collab_site_with_ring(&client, &case.spec, &case.ring);
            let (search, _) = integer_split_sites(&fused, codes);

            assert_sliding_matches_scalar(&search, &label);
        }

        for channels in CHANNEL_MODES {
            let label = format!("hashed {channels:?} radius 2 {depth}-bit");
            let spec = CollabSpec::new(64, 48, channels, 2);
            let fused = collab_site(&client, &spec);
            let (search, _) = integer_split_sites(&fused, codes);

            let words = assert_sliding_matches_scalar(&search, &label);
            let built_grid = any_grid_group(&words);
            assert!(built_grid, "{label}: the neighbour pass built no grid");
        }
    }
}

#[test]
fn the_sliding_centre_matches_scalar_int_without_the_grid() {
    let client = test_client();

    for depth in [8u32, 10] {
        let codes = IntegerCodes { depth };

        for channels in CHANNEL_MODES {
            let label = format!("{channels:?} radius 0 {depth}-bit");
            let spec = CollabSpec::new(64, 48, channels, 0);
            let fused = collab_site(&client, &spec);
            let (search, _) = integer_split_sites(&fused, codes);

            let words = assert_sliding_matches_scalar(&search, &label);
            let built_grid = any_grid_group(&words);
            assert!(!built_grid, "{label}: grid groups");

            // Below MAX_K the grid never builds, so the centre's words are final at any radius.
            let label = format!("{channels:?} radius 2 k_max 4 {depth}-bit");
            let spec = CollabSpec::new(64, 48, channels, 2);
            let fused = collab_site(&client, &spec);
            let (mut search, _) = integer_split_sites(&fused, codes);
            search.shape.k_max = 4;

            let words = assert_sliding_matches_scalar(&search, &label);
            let built_grid = any_grid_group(&words);
            assert!(!built_grid, "{label}: grid groups");
        }
    }
}

#[test]
fn the_sliding_centre_matches_scalar_int_on_ragged_and_odd_frames() {
    let client = test_client();
    let frame_sizes = [(330u32, 142u32), (104, 96), (75, 53), (40, 24)];

    for depth in [8u32, 10] {
        let codes = IntegerCodes { depth };

        for (width, height) in frame_sizes {
            for channels in [ChannelMode::Luma, ChannelMode::Yuv] {
                let label = format!("{width}x{height} {channels:?} {depth}-bit");
                let spec = CollabSpec::new(width, height, channels, 2);
                let fused = collab_site(&client, &spec);
                let (search, _) = integer_split_sites(&fused, codes);

                let words = assert_sliding_matches_scalar(&search, &label);
                let built_grid = any_grid_group(&words);
                assert!(built_grid, "{label}: the neighbour pass built no grid");
            }
        }
    }
}

#[test]
fn the_sliding_centre_matches_scalar_int_on_flat_content() {
    let client = test_client();

    for depth in [8u32, 10] {
        let codes = IntegerCodes { depth };
        let corpus = integer_corpus(depth);
        let flat_cases = corpus.iter().filter(|case| is_flat_case(case.label));

        for case in flat_cases {
            let label = format!("{} {depth}-bit", case.label);
            let fused = collab_site_with_ring(&client, &case.spec, &case.ring);
            let (search, _) = integer_split_sites(&fused, codes);

            assert_sliding_matches_scalar(&search, &label);
        }
    }
}

#[test]
fn the_sliding_centre_keeps_the_self_match_first() {
    let client = test_client();

    for depth in [8u32, 10] {
        let codes = IntegerCodes { depth };

        for case in integer_corpus(depth) {
            let fused = collab_site_with_ring(&client, &case.spec, &case.ring);
            let (search, _) = integer_split_sites(&fused, codes);

            for sliding in sliding_variants() {
                let label = format!("{} {depth}-bit {}", case.label, sliding.name());
                let (_, words) = run_search(&search, sliding);
                let groups = words.chunks(MEMBER_WORDS as usize).enumerate();

                for (index, group) in groups {
                    let expected = reference_position(&case.spec, index);
                    assert_eq!(group[0], expected, "{label}: reference {index}");
                }
            }
        }
    }
}
