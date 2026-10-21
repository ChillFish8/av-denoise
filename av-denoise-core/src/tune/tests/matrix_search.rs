use super::corpus::{integer_corpus, is_flat_case};
use super::fixtures::{CollabSpec, collab_site, collab_site_with_ring, integer_split_sites};
use super::integer_search::{reference_position, reference_variant, run_search};
use super::runtime::{TestRuntime, test_client};
use crate::collab::integer::{CodeView, host_centre_members};
use crate::collab::kernels::fused::MEMBER_WORDS;
use crate::nlmeans::ChannelMode;
use crate::tune::gpu::read_u16;
use crate::tune::sites::{CollabSearchSite, CollabShape, SearchVariant};
use crate::tune::{
    DeviceLimits,
    Fidelity,
    Int8Matrix,
    IntegerCodes,
    Legality,
    SiteCandidate,
    VariantSite,
    int8_matrix_plane_width,
    matrix_plane_width,
};

const CHANNEL_MODES: [ChannelMode; 3] = [ChannelMode::Luma, ChannelMode::Chroma, ChannelMode::Yuv];

fn is_f16_centre(candidate: SearchVariant) -> bool {
    matches!(candidate, SearchVariant::Integer(variant) if variant.is_matrix())
}

fn is_int8_centre(candidate: SearchVariant) -> bool {
    matches!(candidate, SearchVariant::Integer(variant) if variant.is_int8())
}

/// One matrix candidate with the search it runs on.
struct MatrixRun {
    search: CollabSearchSite<TestRuntime, f32>,
    candidate: SearchVariant,
}

impl MatrixRun {
    /// The candidate's name with the int8 product it runs, when it runs one.
    fn label(&self) -> String {
        let name = self.candidate.name();

        if is_int8_centre(self.candidate) {
            let element = self.search.shape.int8;
            return format!("{name} on {element:?}");
        }

        name.to_string()
    }
}

/// The `wanted` candidates of `search` that are legal under `limits` and fit them, and whether any
/// was legal.
///
/// A legal candidate that does not fit prints a skip line.
fn fitting_matrix_candidates(
    search: &CollabSearchSite<TestRuntime, f32>,
    limits: &DeviceLimits,
    wanted: impl Fn(SearchVariant) -> bool,
) -> (Vec<SearchVariant>, bool) {
    let legality = Legality::default();
    let legal: Vec<SearchVariant> = SearchVariant::ALL
        .iter()
        .copied()
        .filter(|&candidate| wanted(candidate))
        .filter(|&candidate| search.legal(candidate, limits, &legality))
        .collect();
    let any_legal = !legal.is_empty();
    let mut fitting = Vec::with_capacity(legal.len());

    for candidate in legal {
        let fits = search.fits(candidate, limits, &legality);

        if fits {
            fitting.push(candidate);
        } else {
            let name = candidate.name();
            eprintln!("{name} does not fit this device's shared memory, skipping it");
        }
    }

    (fitting, any_legal)
}

/// Every matrix candidate the device runs on `search`, printing a skip line for each route it
/// offers none of.
///
/// The int8 candidates also run on a copy of `search` with the i8 product whenever the device runs
/// it and `search` names another product, so both int8 elements run where the device has them. A
/// route the device's limits offer must have a legal candidate, so a skip line never hides a wrong
/// refusal.
fn matrix_runs(search: &CollabSearchSite<TestRuntime, f32>) -> Vec<MatrixRun> {
    let limits = DeviceLimits::from_client(&search.client);
    let f16_offered = matrix_plane_width(&limits).is_some();
    let int8_offered = int8_matrix_plane_width(&limits).is_some();

    let (f16_candidates, f16_legal) = fitting_matrix_candidates(search, &limits, is_f16_centre);
    let (int8_candidates, int8_legal) = fitting_matrix_candidates(search, &limits, is_int8_centre);

    assert_eq!(
        f16_legal, f16_offered,
        "the f16 matrix centre follows the device limits"
    );
    assert_eq!(
        int8_legal, int8_offered,
        "the int8 matrix centre follows the device limits"
    );

    if !f16_legal {
        eprintln!("no f16 matrix centre on this backend, skipping its rows");
    }

    if !int8_legal {
        eprintln!("no int8 matrix centre on this backend, skipping its rows");
    }

    let mut runs: Vec<MatrixRun> = f16_candidates
        .into_iter()
        .chain(int8_candidates)
        .map(|candidate| MatrixRun {
            search: search.clone(),
            candidate,
        })
        .collect();

    let signed_differs = search.shape.int8 != Some(Int8Matrix::Signed);
    if limits.int8_signed && signed_differs {
        let signed_shape = CollabShape {
            int8: Some(Int8Matrix::Signed),
            ..search.shape
        };
        let signed_search = CollabSearchSite {
            shape: signed_shape,
            ..search.clone()
        };
        let (signed_candidates, signed_legal) =
            fitting_matrix_candidates(&signed_search, &limits, is_int8_centre);

        assert!(
            signed_legal,
            "the device runs the i8 product, so the i8 centre is legal"
        );

        for candidate in signed_candidates {
            let run = MatrixRun {
                search: signed_search.clone(),
                candidate,
            };
            runs.push(run);
        }
    }

    runs
}

/// Checks every matrix candidate writes the integer reference's members word for word.
///
/// A mismatch names the first differing word.
fn assert_matrix_matches_reference(search: &CollabSearchSite<TestRuntime, f32>, label: &str) {
    let reference = reference_variant(search);
    let (reference_outputs, reference_words) = run_search(search, reference);

    for run in matrix_runs(search) {
        let name = run.label();
        let (candidate_outputs, candidate_words) = run_search(&run.search, run.candidate);
        let reference_len = reference_words.len();
        let candidate_len = candidate_words.len();
        assert_eq!(reference_len, candidate_len, "{label} {name}");

        let word_pairs = reference_words.iter().zip(&candidate_words);
        let first_difference = word_pairs.enumerate().find(|(_, (want, got))| want != got);

        if let Some((word, (want, got))) = first_difference {
            let group = word / MEMBER_WORDS as usize;
            let slot = word % MEMBER_WORDS as usize;
            panic!(
                "{label} {name}: word {word} (reference {group}, slot {slot}) is {got:#010x}, the integer \
                 reference has {want:#010x}"
            );
        }

        let verdict = search
            .compare(&reference_outputs, &candidate_outputs, Fidelity::Exact)
            .expect("the outputs read back");
        assert!(verdict.passed(), "{label} {name}: {verdict:?}");
    }
}

/// Checks every matrix candidate's centre members match the host's exact top-8, for a search with
/// no neighbour frames.
fn assert_matrix_matches_host(search: &CollabSearchSite<TestRuntime, f32>, spec: &CollabSpec, label: &str) {
    let client = &search.client;
    let inputs = &search.inputs;
    let code_values = read_u16(client, &inputs.codes, inputs.codes_len).expect("the codes read back");
    let widened: Vec<u32> = code_values.iter().map(|&code| u32::from(code)).collect();
    let view = CodeView {
        codes: &widened,
        width: spec.width,
        height: spec.height,
        stored_ch: spec.channels.storage_count(),
        channels: spec.channels.count(),
        centre_slot: 0,
        neighbour_slots: &[],
    };

    for run in matrix_runs(search) {
        let name = run.label();
        let (_, words) = run_search(&run.search, run.candidate);
        let groups = words.chunks(MEMBER_WORDS as usize).enumerate();

        for (index, group) in groups {
            let reference_word = reference_position(spec, index);
            let ref_x = reference_word & 0x1FFF;
            let ref_y = reference_word >> 13;
            let expected = host_centre_members(&view, ref_x, ref_y, spec.spatial_radius);

            assert_eq!(
                &group[..expected.len()],
                expected.as_slice(),
                "{label} {name}: reference {index}"
            );
        }
    }
}

#[test]
fn the_matrix_centre_matches_scalar_int_bitwise() {
    let client = test_client();

    for depth in [8u32, 10] {
        let codes = IntegerCodes { depth };

        for case in integer_corpus(depth) {
            let label = format!("{} {depth}-bit", case.label);
            let fused = collab_site_with_ring(&client, &case.spec, &case.ring);
            let (search, _) = integer_split_sites(&fused, codes);

            assert_matrix_matches_reference(&search, &label);
        }
    }
}

#[test]
fn the_matrix_centre_matches_scalar_int_on_ragged_and_odd_frames() {
    let client = test_client();
    let ragged = CollabSpec::new(104, 96, ChannelMode::Luma, 1);
    let odd_specs = CHANNEL_MODES.into_iter().flat_map(|channels| {
        [
            CollabSpec::new(75, 53, channels, 1),
            CollabSpec::new(40, 24, channels, 1),
            CollabSpec::new(40, 24, channels, 0),
            CollabSpec::new(75, 53, channels, 0),
        ]
    });
    let specs: Vec<CollabSpec> = [ragged].into_iter().chain(odd_specs).collect();

    for depth in [8u32, 10] {
        let codes = IntegerCodes { depth };

        for spec in &specs {
            let label = format!(
                "{}x{} {:?} radius {} {depth}-bit",
                spec.width, spec.height, spec.channels, spec.radius
            );
            let fused = collab_site(&client, spec);
            let (search, _) = integer_split_sites(&fused, codes);

            assert_matrix_matches_reference(&search, &label);

            if spec.radius == 0 {
                assert_matrix_matches_host(&search, spec, &label);
            }
        }
    }
}

#[test]
fn the_matrix_centre_matches_scalar_int_on_flat_content() {
    let client = test_client();

    for depth in [8u32, 10] {
        let codes = IntegerCodes { depth };
        let corpus = integer_corpus(depth);
        let flat_cases = corpus.iter().filter(|case| is_flat_case(case.label));

        for case in flat_cases {
            let label = format!("{} {depth}-bit", case.label);
            let fused = collab_site_with_ring(&client, &case.spec, &case.ring);
            let (search, _) = integer_split_sites(&fused, codes);

            assert_matrix_matches_reference(&search, &label);
        }
    }
}

#[test]
fn the_matrix_centre_keeps_the_self_match_first() {
    let client = test_client();

    for depth in [8u32, 10] {
        let codes = IntegerCodes { depth };

        for case in integer_corpus(depth) {
            let label = format!("{} {depth}-bit", case.label);
            let fused = collab_site_with_ring(&client, &case.spec, &case.ring);
            let (search, _) = integer_split_sites(&fused, codes);

            for run in matrix_runs(&search) {
                let name = run.label();
                let (_, words) = run_search(&run.search, run.candidate);
                let groups = words.chunks(MEMBER_WORDS as usize).enumerate();

                for (index, group) in groups {
                    let expected = reference_position(&case.spec, index);
                    assert_eq!(group[0], expected, "{label} {name}: reference {index}");
                }
            }
        }
    }
}
