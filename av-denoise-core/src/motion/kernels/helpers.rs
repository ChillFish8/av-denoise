use cubecl::prelude::*;

/// Clamps `value` to a position between 0 and `limit - 1`.
#[cube]
pub(super) fn clamp_i32(value: i32, limit: i32) -> i32 {
    let mut result = value;
    if value < 0 {
        result = 0;
    } else if value >= limit {
        result = limit - 1;
    }
    result
}
