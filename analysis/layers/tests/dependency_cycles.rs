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

#[test]
fn all_small_head_assignments_match_a_per_node_cycle_search() {
    for n in 0..=5usize {
        for mut code in 0..(n + 1).pow(n as u32) {
            let heads: Vec<_> = (0..n)
                .map(|_| {
                    let head = code % (n + 1);
                    code /= n + 1;
                    (head < n).then_some(Ix::new(head as u32))
                })
                .collect();
            let acyclic = (0..n).all(|start| {
                let mut path = vec![false; n];
                let mut next = Some(start);
                while let Some(at) = next {
                    if path[at] {
                        return false;
                    }
                    path[at] = true;
                    next = heads[at].map(Ix::get);
                }
                true
            });
            let layer = Dependencies::<Tokens> {
                head: heads,
                relation: vec![Tag(0); n],
                relations: TagSet::new("dep", &["dep"]),
            };
            assert_eq!(layer.check().is_ok(), acyclic, "heads: {:?}", layer.head);
        }
    }
}

#[test]
fn document_insertion_refuses_a_cycle_in_a_separate_component() {
    use turbo_analysis::{Document, Key, Provenance, Span, Spans, ViewId};
    let mut doc = Document::new("a b c");
    let mut spans = Spans::new(ViewId::SOURCE);
    for start in [0, 2, 4] {
        spans.push(Span::new(start, start + 1));
    }
    doc.add(&Key::DEFAULT, Tokens { spans, sentence: None }, Provenance::by("test")).unwrap();
    let layer = Dependencies::<Tokens> {
        head: vec![None, Some(Ix::new(2)), Some(Ix::new(1))],
        relation: vec![Tag(0); 3],
        relations: TagSet::new("dep", &["dep"]),
    };
    let err = doc.add(&Key::DEFAULT, layer, Provenance::by("test")).unwrap_err();
    assert!(err.to_string().contains("cycle"), "{err}");
    assert!(doc.get::<Dependencies<Tokens>>().is_none());
}
