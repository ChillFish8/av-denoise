mod harness;
mod legacy;
mod stage;

use cubecl::prelude::*;

pub(super) fn make_client() -> Client {
    crate::tests::compilation_cache::install_test_compilation_cache();
    crate::testing::test_client()
}
