//! `CorefAnnotator::config_hash` changes with every setting that changes
//! the output, and only with those.

use turbo_analysis::Annotator;
use turbo_coref::CorefAnnotator;

fn hash(a: &CorefAnnotator) -> u64 {
    a.config_hash()
}

fn rules(person: &[&str], neutral: &[&str]) -> CorefAnnotator {
    CorefAnnotator::rules(person, neutral).unwrap()
}

#[test]
fn entity_types_change_the_hash() {
    let a = hash(&rules(&["person"], &["organization"]));
    let b = hash(&rules(&["animal"], &["location"]));
    assert_ne!(a, 0);
    assert_ne!(a, b);
    // Swapping the classes is a different configuration.
    assert_ne!(hash(&rules(&["person"], &["organization"])), hash(&rules(&["organization"], &["person"])));
    // Order, case and repeats do not change the matching, nor the hash.
    assert_eq!(hash(&rules(&["person", "agent"], &["org"])), hash(&rules(&["AGENT", "Person", "person"], &["ORG"])));
    assert_eq!(hash(&CorefAnnotator::default()), hash(&rules(&["person"], &["organization", "location"])));
}
