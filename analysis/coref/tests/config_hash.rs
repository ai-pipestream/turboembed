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

#[test]
fn threshold_changes_the_hash() {
    let a = CorefAnnotator::new(&["person"], &["org"], None, 0.1, None, None).unwrap();
    let b = CorefAnnotator::new(&["person"], &["org"], None, 0.5, None, None).unwrap();
    assert_ne!(hash(&a), hash(&b));
}

#[test]
fn word_vectors_change_the_hash_by_content() {
    use std::sync::Arc;
    use turbo_coref::{DEFAULT_THRESHOLD, MapWordVectors, WordVectors};
    let with = |v: MapWordVectors| {
        let v: Arc<dyn WordVectors> = Arc::new(v);
        hash(&CorefAnnotator::new(&["person"], &["org"], None, DEFAULT_THRESHOLD, Some(v), None).unwrap())
    };
    let plain = hash(&CorefAnnotator::new(&["person"], &["org"], None, DEFAULT_THRESHOLD, None, None).unwrap());
    let mut a = MapWordVectors::new(2);
    a.insert("firm", vec![1.0, 0.0]).unwrap();
    a.insert("acme", vec![0.0, 1.0]).unwrap();
    let mut b = MapWordVectors::new(2);
    b.insert("acme", vec![0.0, 1.0]).unwrap();
    b.insert("firm", vec![1.0, 0.0]).unwrap();
    let mut c = MapWordVectors::new(2);
    c.insert("firm", vec![1.0, 0.5]).unwrap();
    c.insert("acme", vec![0.0, 1.0]).unwrap();
    assert_eq!(with(a.clone()), with(b), "insertion order does not matter");
    assert_ne!(with(a.clone()), with(c));
    assert_ne!(with(a), plain);
}
