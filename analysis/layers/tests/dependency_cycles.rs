use turbo_analysis::{Ix, Layer, Tag, TagSet};
use turbo_analysis_layers::{Dependencies, Tokens};

#[test]
fn dependency_forests_reject_nontrivial_cycles() {
    let relations = TagSet::new("dep", &["dep"]);
    let layer = Dependencies::<Tokens> {
        head: vec![Some(Ix::new(1)), Some(Ix::new(0)), None],
        relation: vec![Tag(0); 3],
        relations,
    };
    assert!(layer.check().is_err(), "a rooted component must not hide a cycle");
}

#[test]
fn long_chains_and_multiple_roots_are_valid_forests() {
    let n = 100_000;
    let mut heads: Vec<_> = (0..n).map(|i| if i + 1 < n { Some(Ix::new(i + 1)) } else { None }).collect();
    let mut layer = Dependencies::<Tokens> {
        head: heads.clone(),
        relation: vec![Tag(0); n as usize],
        relations: TagSet::new("dep", &["dep"]),
    };
    layer.check().unwrap();
    heads[99_999] = Some(Ix::new(0));
    layer.head = heads;
    assert!(layer.check().is_err());
    layer.head.fill(None);
    layer.check().unwrap();
}
