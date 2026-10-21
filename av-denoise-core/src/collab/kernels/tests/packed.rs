use crate::collab::kernels::packed::{pack_accum_host, unpack_accum_host, unpack_weight_host};

#[test]
fn packing_round_trips_signed_accumulators() {
    let cases = [
        (0i32, 0i32),
        (123_456, 789),
        (-123_456, 789),
        (i32::MAX / 2, i32::MAX / 2),
        (-(i32::MAX / 2), 1),
    ];

    for (accum, weight) in cases {
        let word = pack_accum_host(accum, weight);
        let unpacked_accum = unpack_accum_host(word);
        let unpacked_weight = unpack_weight_host(word);

        assert_eq!(unpacked_accum, accum, "{accum} {weight}");
        assert_eq!(unpacked_weight, weight, "{accum} {weight}");
    }
}

#[test]
fn sums_of_packed_words_unpack_to_sums_of_their_parts() {
    let contributions = [(-500i32, 30i32), (1_200, 0), (-7, 4_000), (300, 12)];
    let total_word: i64 = contributions
        .iter()
        .map(|&(accum, weight)| pack_accum_host(accum, weight))
        .sum();
    let total_accum: i32 = contributions.iter().map(|&(accum, _)| accum).sum();
    let total_weight: i32 = contributions.iter().map(|&(_, weight)| weight).sum();

    assert_eq!(unpack_accum_host(total_word), total_accum);
    assert_eq!(unpack_weight_host(total_word), total_weight);
}
