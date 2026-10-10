use crate::nl4d::tests::helpers::make_client;
use crate::nl4d::tests::pipeline::single_pass_launch;
use crate::tune::collab::COLLAB_CANDIDATES;

#[test]
fn collab_scratch_leaves_real_buffers_untouched() {
    let client = make_client();
    let (real, read_outputs) = single_pass_launch(&client);
    let before = read_outputs(&real);
    let scratch = real.with_scratch();

    for index in 0..COLLAB_CANDIDATES.len() {
        scratch.launch_candidate(index).expect("candidate launches");
    }

    let after = read_outputs(&real);
    assert_eq!(after, before);
}
