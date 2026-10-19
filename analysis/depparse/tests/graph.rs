use turbo_analysis::{Ix, Layer, Tag, TagSet};
use turbo_analysis_layers::{Dependencies, Tokens, Words};
use turbo_depparse::{ArcDirection, ArcStandardOracle, ArcStandardState, DependencyGraph, Transition};

fn tree(heads: &[Option<u32>]) -> DependencyGraph<Words> {
    DependencyGraph::new(heads.iter().map(|h| h.map(Ix::new)).collect(), vec!["dep".into(); heads.len()]).unwrap()
}
#[test]
fn navigation_is_typed_and_bidirectional() {
    let graph = tree(&[Some(1), None, Some(1), Some(2)]);
    assert_eq!(graph.root(), Some(Ix::new(1)));
    assert_eq!(graph.children(Ix::new(1)).unwrap(), [Ix::new(0), Ix::new(2)]);
    assert_eq!(graph.lowest_common_ancestor(Ix::new(0), Ix::new(3)).unwrap(), Some(Ix::new(1)));
    let path = graph.shortest_path(Ix::new(0), Ix::new(3)).unwrap().unwrap();
    assert_eq!(
        path.iter().map(|s| (s.from.get(), s.to.get(), s.direction)).collect::<Vec<_>>(),
        [(0, 1, ArcDirection::ToHead), (1, 2, ArcDirection::ToDependent), (2, 3, ArcDirection::ToDependent)]
    );
    assert!(graph.shortest_path(Ix::new(3), Ix::new(3)).unwrap().unwrap().is_empty());
    assert!(graph.children(Ix::new(9)).is_err());
    assert!(graph.lowest_common_ancestor(Ix::new(0), Ix::new(9)).is_err());
    let restored = DependencyGraph::from_layer(graph.to_layer()).unwrap();
    for i in 0..4 {
        assert_eq!(restored.arc(Ix::new(i)).unwrap().head, graph.arc(Ix::new(i)).unwrap().head);
    }
}
#[test]
fn forests_have_no_path_between_roots_and_preserve_tag_ids() {
    let layer = Dependencies::<Tokens> {
        head: vec![None, None],
        relation: vec![Tag(1), Tag(0)],
        relations: TagSet::new("custom", &["b", "a", "unused"]),
    };
    let graph = DependencyGraph::from_layer(layer).unwrap();
    assert!(graph.check_tree().is_err());
    assert!(graph.shortest_path(Ix::new(0), Ix::new(1)).unwrap().is_none());
    let out = graph.to_layer();
    assert_eq!(out.relation, [Tag(1), Tag(0)]);
    assert_eq!(out.relations.len(), 3);
    assert_eq!(out.relations.name, "custom");
}
#[test]
fn tree_boundaries_and_root_crossing_projectivity() {
    for heads in [vec![], vec![None, None], vec![Some(Ix::new(0))], vec![Some(Ix::new(1)), Some(Ix::new(0)), None]] {
        let rels = vec!["dep".into(); heads.len()];
        assert!(DependencyGraph::<Tokens>::new(heads, rels).is_err());
    }
    assert!(DependencyGraph::<Tokens>::new(vec![None], vec!["\u{a0}".into()]).is_err());
    // The artificial-root arc crosses 0 -> 2, even though the real arcs nest.
    let nonprojective = tree(&[Some(2), None, Some(1)]);
    assert!(!nonprojective.is_projective());
    assert!(ArcStandardOracle::transitions(&nonprojective).is_err());
}
#[test]
fn oracle_rebuilds_all_small_projective_trees() {
    for n in 1usize..=5 {
        for code in 0..(n + 1).pow(n as u32) {
            let mut c = code;
            let mut heads = Vec::new();
            for _ in 0..n {
                let h = c % (n + 1);
                c /= n + 1;
                heads.push((h < n).then(|| Ix::new(h as u32)));
            }
            let Ok(g) = DependencyGraph::<Tokens>::new(heads.clone(), vec!["dep".into(); n]) else { continue };
            let steps = ArcStandardOracle::transitions(&g);
            assert_eq!(steps.is_ok(), g.is_projective());
            if let Ok(steps) = steps {
                assert_eq!(steps.len(), 2 * n);
                let mut state = ArcStandardState::<Tokens>::new(n).unwrap();
                for t in steps {
                    assert!(state.can_apply(&t));
                    state.apply(&t).unwrap();
                }
                let out = state.to_graph().unwrap().to_layer();
                assert_eq!(out.head, heads);
                out.check().unwrap();
            }
        }
    }
}
#[test]
fn invalid_transitions_do_not_mutate_state() {
    let mut state = ArcStandardState::<Tokens>::new(1).unwrap();
    assert!(state.to_graph().is_err());
    assert!(state.apply(&Transition::LeftArc("dep".into())).is_err());
    assert!(state.apply(&Transition::RightArc(" ".into())).is_err());
    assert!(matches!(state.buffer(0), turbo_depparse::Position::Node(_)));
    assert!(matches!(state.buffer(usize::MAX), turbo_depparse::Position::Missing));
    for invalid in ["", "SHIFT:x", "LEFT_ARC:", "RIGHT_ARC: ", "UNKNOWN:x"] {
        assert!(Transition::decode(invalid).is_err());
    }
    let t = Transition::decode("LEFT_ARC:nsubj:pass").unwrap();
    assert_eq!(t.encode(), "LEFT_ARC:nsubj:pass");
}
