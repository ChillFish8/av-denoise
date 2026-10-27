use crate::tune::{SiteKey, bucket_dim};

#[test]
fn dimensions_round_up_to_a_power_of_two() {
    assert_eq!(bucket_dim(1920), 2048);
    assert_eq!(bucket_dim(1080), 2048);
    assert_eq!(bucket_dim(1200), 2048);
    assert_eq!(bucket_dim(1024), 1024);
    assert_eq!(bucket_dim(960), 1024);
    assert_eq!(bucket_dim(1), 1);
}

#[test]
fn nearby_resolutions_share_a_key() {
    let full_hd = SiteKey::builder("nlm_window", 1920, 1080).param("pr", 3).build();
    let wuxga = SiteKey::builder("nlm_window", 1920, 1200).param("pr", 3).build();

    assert_eq!(full_hd, wuxga);
}

#[test]
fn every_other_field_is_exact() {
    let radius_three = SiteKey::builder("nlm_window", 1920, 1080).param("pr", 3).build();
    let radius_four = SiteKey::builder("nlm_window", 1920, 1080).param("pr", 4).build();

    assert_ne!(radius_three, radius_four);
}

#[test]
fn the_text_form_is_stable() {
    let key = SiteKey::builder("collab_fused", 1920, 1080)
        .param("ch", 1)
        .param("f16", 1)
        .build();

    assert_eq!(key.as_str(), "collab_fused|w2048|h2048|ch=1|f16=1");
    assert_eq!(key.site(), "collab_fused");
}

#[test]
fn stored_text_round_trips() {
    let key = SiteKey::builder("nlm_accumulate", 640, 480)
        .param("stored", 4)
        .build();
    let text = key.as_str();
    let restored = SiteKey::from_stored(text);

    assert_eq!(restored, Some(key));
}

#[test]
fn stored_text_without_a_site_is_rejected() {
    assert_eq!(SiteKey::from_stored(""), None);
    assert_eq!(SiteKey::from_stored("|w64|h64"), None);
}
