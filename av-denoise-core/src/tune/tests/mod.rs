mod collab;
mod nlm_window;
mod regularise;

use cubecl::config::autotune::AutotuneLevel;

use crate::nlmeans::tests::helpers::make_client;
use crate::tune::{TuneId, fits, priority_for, zeroed};

#[test]
fn tune_id_is_stable_for_one_client() {
    let client = make_client();
    let first = TuneId::new(&client);
    let second = TuneId::new(&client);

    assert_eq!(first, second);

    let label = first.to_string();
    assert!(label.contains("-sm"), "unexpected tune id {label}");
}

#[test]
fn fits_accepts_within_limits() {
    let client = make_client();
    let hardware = client.properties().hardware.clone();

    assert!(fits(&hardware, 64, 1024));
}

#[test]
fn fits_rejects_too_many_units() {
    let client = make_client();
    let hardware = client.properties().hardware.clone();
    let units = hardware.max_units_per_cube + 1;

    assert!(!fits(&hardware, units, 0));
}

#[test]
fn fits_rejects_too_much_shared_memory() {
    let client = make_client();
    let hardware = client.properties().hardware.clone();
    let shared_bytes = hardware.max_shared_memory_size + 1;

    assert!(!fits(&hardware, 32, shared_bytes));
}

#[test]
fn zeroed_buffer_reads_back_as_zero() {
    let client = make_client();
    let handle = zeroed(&client, 4096);
    let bytes = client.read_one(handle).expect("zeroed readback");

    assert!(bytes.iter().all(|&byte| byte == 0));
}

#[test]
fn alternatives_are_skipped_at_the_minimal_level() {
    assert_eq!(priority_for(&AutotuneLevel::Minimal), -1);
}

#[test]
fn alternatives_are_tuned_above_the_minimal_level() {
    assert_eq!(priority_for(&AutotuneLevel::Balanced), 0);
}
