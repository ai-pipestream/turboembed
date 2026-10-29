use turbo_analysis::{Ix, Layer, Tag, TagSet};
use turbo_analysis_layers::{Dependencies, Tokens, Words};
use turbo_depparse::{
    ArcDirection, DependencyCrossValidator, DependencyEvaluator, DependencyGraph, DependencyParser, DependencySample,
    Result,
};

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
    let projective = tree(&[Some(1), None, Some(1)]);
    assert!(projective.is_projective());
}

#[test]
fn graph_import_rejects_a_structurally_valid_cycle() {
    let layer = Dependencies::<Tokens> {
        head: vec![None, Some(Ix::new(2)), Some(Ix::new(1))],
        relation: vec![Tag(0); 3],
        relations: TagSet::new("dep", &["dep"]),
    };
    layer.check().unwrap();
    assert!(layer.check_forest().is_err());
    assert!(DependencyGraph::from_layer(layer).is_err());
}

struct ChainParser;
impl DependencyParser<Tokens> for ChainParser {
    fn parse(&self, tokens: &[&str], _tags: &[&str]) -> Result<DependencyGraph<Tokens>> {
        let heads = (0..tokens.len()).map(|i| (i > 0).then(|| Ix::new((i - 1) as u32))).collect();
        let relations = (0..tokens.len()).map(|i| if i == 0 { "root".into() } else { "dep".into() }).collect();
        DependencyGraph::new(heads, relations)
    }
}

#[test]
fn evaluator_and_cross_validator_score_uas_and_las_with_and_without_punctuation() {
    let gold_graph = DependencyGraph::<Tokens>::new(
        vec![None, Some(Ix::new(0)), Some(Ix::new(0))],
        vec!["root".into(), "dep".into(), "punct".into()],
    )
    .unwrap();
    let sample = DependencySample::new(
        vec!["Walk".into(), "home".into(), ".".into()],
        vec!["VERB".into(), "ADV".into(), "PUNCT".into()],
        gold_graph,
    )
    .unwrap();
    let mut evaluator = DependencyEvaluator::default();
    let predicted = evaluator.evaluate(&ChainParser, &sample).unwrap();
    assert_eq!(predicted.len(), 3);
    assert_eq!(evaluator.all.words, 3);
    assert_eq!(evaluator.all.heads, 2);
    assert_eq!(evaluator.all.labels, 2);
    assert_eq!(evaluator.without_punctuation.words, 2);
    assert_eq!(evaluator.without_punctuation.heads, 2);
    assert_eq!(evaluator.without_punctuation.labels, 2);

    let mut cv = DependencyCrossValidator::default();
    let samples = [sample.clone(), sample];
    cv.evaluate(&samples, 2, |_train| Ok(ChainParser)).unwrap();
    assert_eq!(cv.scores.all.words, 6);
    assert_eq!(cv.scores.all.heads, 4);
    assert_eq!(cv.scores.without_punctuation.words, 4);
    assert_eq!(cv.scores.without_punctuation.heads, 4);
}
