//! `Index::find_within` on the sample index of `tests/zoo.rs`, whose
//! entries are all English tokenizers in mixed class order.

use super::*;
use turbo_zoo::Component;

#[test]
fn find_within_keeps_the_classes_asked_for_least_restrictive_first_in_index_order() {
    let index = Index::parse(sample().as_bytes()).unwrap();
    let ids = |at_most| {
        let found = index.find_within("en", Component::Tokenizer, at_most);
        found.iter().map(|a| a.id.as_str().to_owned()).collect::<Vec<_>>()
    };
    assert_eq!(ids(LicenceClass::Permissive), ["permissive"]);
    assert_eq!(ids(LicenceClass::Lgpl), ["permissive", "lgpl"]);
    let sa = ["permissive", "lgpl", "share-alike", "trained-on-share-alike", "lgpl-and-share-alike"];
    assert_eq!(ids(LicenceClass::ShareAlike), sa);
    let mut gpl = sa.to_vec();
    gpl.push("gpl");
    assert_eq!(ids(LicenceClass::Gpl), gpl);
    // Another language or component finds nothing.
    assert!(index.find_within("de", Component::Tokenizer, LicenceClass::Gpl).is_empty());
    assert!(index.find_within("en", Component::SentenceDetector, LicenceClass::Gpl).is_empty());
}
