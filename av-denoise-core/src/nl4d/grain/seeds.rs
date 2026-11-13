use std::collections::HashMap;

use crate::nl4d::grain::consts::AR_COEFFS;
use crate::nl4d::grain::template::{TemplatePlane, seed_sigma};

/// The film grain seeds a frame can draw its template from.
///
/// The list is taken from FGSEditor (<https://github.com/PingWer/FGSEditor>).
pub(crate) const FGSE_SEEDS: [u16; 135] = [
    4, 63504, 64435, 65501, 48, 59665, 36, 64240, 63947, 59119, 42, 59846, 65466, 9, 52, 58, 65417, 59625,
    64831, 60973, 38, 43, 65464, 17, 64384, 65333, 62040, 6, 67, 47, 45, 32, 66, 50, 65506, 65476, 64274,
    62156, 59, 8, 65, 64436, 63647, 61692, 46, 63456, 13, 44, 40, 65420, 1, 60, 11, 3, 27, 64113, 64, 60619,
    53, 59637, 35, 31, 24, 39, 64093, 41, 12, 60501, 62, 34, 28, 57, 15, 59929, 30, 59669, 19, 63507, 60584,
    64921, 2, 33, 51, 61851, 60537, 62143, 37, 61476, 59991, 22, 14, 64332, 61482, 49, 65391, 60711, 25, 20,
    64189, 65260, 64448, 65345, 59724, 60878, 64176, 56, 64774, 64917, 63580, 29, 63023, 64423, 64693, 64126,
    62258, 26, 16, 10, 23, 61, 54, 59621, 63, 62518, 21, 59180, 18, 62359, 5, 65299, 7, 64285, 55, 65484,
    60123,
];

/// The seed a frame draws when none is held over from before.
pub(crate) fn fresh_seed(frame: u64) -> u16 {
    let index = frame % FGSE_SEEDS.len() as u64;

    FGSE_SEEDS[index as usize]
}

type SigmaKey = ([i32; AR_COEFFS], u32, TemplatePlane, u16);

/// A cache of the grain std each seed's template has.
#[derive(Debug, Default)]
pub(crate) struct SeedSigmas {
    sigmas: HashMap<SigmaKey, f64>,
}

impl SeedSigmas {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn sigma(
        &mut self,
        coeffs: &[i32; AR_COEFFS],
        ar_shift: u32,
        plane: TemplatePlane,
        seed: u16,
    ) -> f64 {
        let key = (*coeffs, ar_shift, plane, seed);

        *self
            .sigmas
            .entry(key)
            .or_insert_with(|| seed_sigma(coeffs, ar_shift, plane, seed))
    }

    pub(crate) fn len(&self) -> usize {
        self.sigmas.len()
    }
}
