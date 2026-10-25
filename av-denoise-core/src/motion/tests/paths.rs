#[expect(unused_imports, reason = "naming each kernel path is the check")]
use crate::bench_api::kernels::motion::{
    BLOCK_MATCH_THREADS,
    mc_block_match_coarse,
    mc_block_match_fine,
    mc_chain_compose,
    mc_downscale,
    mc_extract_luma,
    mc_pair_zero,
    mc_warp,
};
use crate::{bench_api, nlmeans};

#[test]
fn the_public_motion_paths_resolve() {
    let _search: nlmeans::MotionSearch = bench_api::motion::MotionSearch::default();
    let _mode: nlmeans::MotionCompensationMode = bench_api::motion::MotionCompensationMode::None;
    let _estimation: nlmeans::MotionEstimation = bench_api::motion::MotionEstimation::Auto;

    let _constants = [
        bench_api::motion::DEFAULT_REFINE_RADIUS,
        bench_api::motion::CHAINED_RADIUS_THRESHOLD,
        bench_api::motion::DEFAULT_BLKSIZE,
        bench_api::motion::DEFAULT_OVERLAP,
        bench_api::motion::DEFAULT_SEARCH_RADIUS,
        bench_api::motion::DEFAULT_PYRAMID_LEVELS,
        bench_api::motion::MAX_PYRAMID_LEVELS,
        bench_api::motion::MAX_SEARCH_RADIUS,
        bench_api::motion::MAX_BLKSIZE,
    ];
}
