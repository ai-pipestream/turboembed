use turbo_analysis::{Document, Ix, Key, Layer, Provenance, Span, Spans, Tag, TagSet, View, ViewId};
use turbo_analysis_layers::{Dependencies, PosTags, Tokens, Words};
use turbo_depparse::{
    ArcDirection, ArcStandardOracle, ArcStandardState, DependencyCrossValidator, DependencyEvaluator, DependencyGraph,
    DependencyModel, DependencyParser, DependencyParserME, DependencySample, LabelEncoding, PathSearch, Position,
    Result, Transition,
};
use turbo_train::{Key as TrainKey, TrainingParameters};

fn tree(heads: &[Option<u32>]) -> DependencyGraph<Words> {
    DependencyGraph::new(heads.iter().map(|h| h.map(Ix::new)).collect(), vec!["dep".into(); heads.len()]).unwrap()
}

#[test]
fn lca_and_shortest_path_preserve_direction_and_labels() {
    let g = DependencyGraph::<Words>::new(
        vec![Some(Ix::new(1)), None, Some(Ix::new(1)), Some(Ix::new(2))],
        vec!["nsubj".into(), "root".into(), "obj".into(), "nmod".into()],
    )
    .unwrap();
    assert_eq!(g.root(), Some(Ix::new(1)));
    assert_eq!(g.children(Ix::new(1)).unwrap(), &[Ix::new(0), Ix::new(2)]);
    assert_eq!(g.lowest_common_ancestor(Ix::new(0), Ix::new(3)).unwrap(), Some(Ix::new(1)));
    let path = g.shortest_path(Ix::new(0), Ix::new(3)).unwrap().unwrap();
    assert_eq!(path.len(), 3);
    assert_eq!(
        (path[0].from, path[0].to, path[0].direction, path[0].relation),
        (Ix::new(0), Ix::new(1), ArcDirection::ToHead, "nsubj")
    );
    assert_eq!(
        (path[1].from, path[1].to, path[1].direction, path[1].relation),
        (Ix::new(1), Ix::new(2), ArcDirection::ToDependent, "obj")
    );
    assert_eq!(
        (path[2].from, path[2].to, path[2].direction, path[2].relation),
        (Ix::new(2), Ix::new(3), ArcDirection::ToDependent, "nmod")
    );
}

#[test]
fn forest_import_supports_multiple_sentence_roots() {
    let layer = Dependencies::<Tokens> {
        head: vec![None, Some(Ix::new(0)), None, Some(Ix::new(2))],
        relation: vec![Tag(0), Tag(1), Tag(0), Tag(1)],
        relations: TagSet::new("dependency", &["root", "dep"]),
    };
    let g = DependencyGraph::from_layer(layer).unwrap();
    assert_eq!(g.roots(), &[Ix::new(0), Ix::new(2)]);
    assert_eq!(g.root(), None);
    assert!(g.check_tree().is_err());
    assert!(g.shortest_path(Ix::new(1), Ix::new(3)).unwrap().is_none());
}

#[test]
fn tree_boundaries_and_root_crossing_projectivity() {
    assert!(DependencyGraph::<Words>::new(vec![], vec![]).is_err());
    assert!(DependencyGraph::<Words>::new(vec![None], vec![" ".into()]).is_err());
    assert!(DependencyGraph::<Words>::new(vec![None, None], vec!["root".into(), "root".into()]).is_err());
    let single = tree(&[None]);
    assert_eq!(single.shortest_path(Ix::new(0), Ix::new(0)).unwrap().unwrap().len(), 0);
    assert!(single.arc(Ix::new(1)).is_err());

    // The artificial-root arc crosses 0 -> 2, even though the real arcs nest.
    let nonprojective = tree(&[Some(2), None, Some(1)]);
    assert!(!nonprojective.is_projective());
    assert!(ArcStandardOracle::transitions(&nonprojective).is_err());
    let projectivized = nonprojective.projectivize().unwrap();
    assert!(projectivized.is_projective());
    assert!(ArcStandardOracle::transitions(&projectivized).is_ok());
    let restored = projectivized.deprojectivize().unwrap();
    assert!(!restored.is_projective());
    assert_eq!(restored.to_layer().head, nonprojective.to_layer().head);
    for i in 0..3 {
        assert_eq!(restored.arc(Ix::new(i)).unwrap().relation, "dep");
    }
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
            let rels: Vec<String> = (0..n).map(|i| format!("r{i}")).collect();
            let g_labeled = DependencyGraph::<Tokens>::new(heads.clone(), rels.clone()).unwrap();
            let proj = g_labeled.projectivize().unwrap();
            assert!(proj.is_projective());
            let proj_steps = ArcStandardOracle::transitions(&proj).unwrap();
            assert_eq!(proj_steps.len(), 2 * n);
            let mut state = ArcStandardState::<Tokens>::new(n).unwrap();
            for t in proj_steps {
                assert!(state.can_apply(&t));
                state.apply(&t).unwrap();
            }
            let decoded = state.to_graph().unwrap().deprojectivize_with(LabelEncoding::EscapedV1).unwrap();
            assert_eq!(decoded.to_layer().head, heads);
            for (i, expected_rel) in rels.iter().enumerate() {
                assert_eq!(decoded.arc(Ix::new(i as u32)).unwrap().relation, expected_rel);
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
    assert!(matches!(state.buffer(0), Position::Node(_)));
    assert!(matches!(state.buffer(usize::MAX), Position::Missing));
    for invalid in ["", "SHIFT:x", "LEFT_ARC:", "RIGHT_ARC: ", "UNKNOWN:x"] {
        assert!(Transition::decode(invalid).is_err());
    }
    let t = Transition::decode("LEFT_ARC:nsubj:pass").unwrap();
    assert_eq!(t.encode(), "LEFT_ARC:nsubj:pass");
}

#[test]
fn graph_import_rejects_a_structurally_valid_cycle() {
    let layer = Dependencies::<Tokens> {
        head: vec![Some(Ix::new(1)), Some(Ix::new(0))],
        relation: vec![Tag(0), Tag(0)],
        relations: TagSet::new("dependency", &["dep"]),
    };
    assert!(layer.check().is_ok());
    assert!(DependencyGraph::from_layer(layer).is_err());
}

#[test]
fn custom_labels_with_carets_survive_projectivize_and_deprojectivize() {
    let projective = DependencyGraph::<Tokens>::new(
        vec![None, Some(Ix::new(0))],
        vec!["root^literal".into(), "dep^custom\\x".into()],
    )
    .unwrap();
    let proj = projective.projectivize().unwrap();
    let deproj = proj.deprojectivize().unwrap();
    assert_eq!(deproj.arc(Ix::new(0)).unwrap().relation, "root^literal");
    assert_eq!(deproj.arc(Ix::new(1)).unwrap().relation, "dep^custom\\x");

    let nonprojective = DependencyGraph::<Tokens>::new(
        vec![Some(Ix::new(2)), None, Some(Ix::new(1))],
        vec!["det^custom".into(), "root^literal".into(), "obj^<escaped".into()],
    )
    .unwrap();
    let lifted = nonprojective.projectivize().unwrap();
    assert!(lifted.is_projective());
    let restored = lifted.deprojectivize().unwrap();
    assert_eq!(restored.to_layer().head, nonprojective.to_layer().head);
    assert_eq!(restored.arc(Ix::new(0)).unwrap().relation, "det^custom");
    assert_eq!(restored.arc(Ix::new(1)).unwrap().relation, "root^literal");
    assert_eq!(restored.arc(Ix::new(2)).unwrap().relation, "obj^<escaped");
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

#[test]
fn linear_model_trains_roundtrips_bytes_and_resolves_bidirectional_offsets() {
    let sample = DependencySample::new(
        vec!["I".into(), "see".into(), "birds".into()],
        vec!["PRON".into(), "VERB".into(), "NOUN".into()],
        DependencyGraph::<Tokens>::new(
            vec![Some(Ix::new(1)), None, Some(Ix::new(1))],
            vec!["nsubj".into(), "root".into(), "obj".into()],
        )
        .unwrap(),
    )
    .unwrap();
    let mut params = TrainingParameters::new();
    params.put(TrainKey::CUTOFF, 0);
    params.put(TrainKey::ITERATIONS, 25);
    let model = DependencyModel::train("en", std::slice::from_ref(&sample), &params).unwrap();
    let bytes = model.to_bytes().unwrap();
    let reloaded = DependencyModel::from_bytes(&bytes).unwrap();
    assert_eq!(reloaded.language(), Some("en"));
    assert_eq!(reloaded.identity(), model.identity());
    let parser = DependencyParserME::new(reloaded);
    let parsed = parser.parse_tokens(&["I", "see", "birds"], &["PRON", "VERB", "NOUN"]).unwrap();
    assert_eq!(parsed.to_layer().head, sample.graph().to_layer().head);

    let mut doc = Document::new("I see birds");
    let mut tok_spans = Spans::new(ViewId::SOURCE);
    for (a, b) in [(0, 1), (2, 5), (6, 11)] {
        tok_spans.push(Span::new(a, b));
    }
    doc.add(&Key::DEFAULT, Tokens { spans: tok_spans, sentence: None }, Provenance::by("test")).unwrap();
    doc.add(
        &Key::DEFAULT,
        PosTags {
            tags: vec![Tag(0), Tag(1), Tag(2)],
            tagset: TagSet::new("upos", &["PRON", "VERB", "NOUN"]),
            prob: None,
        },
        Provenance::by("test"),
    )
    .unwrap();
    let view = doc.get::<View>().unwrap();
    let tokens = doc.get::<Tokens>().unwrap();
    let pos = doc.get::<PosTags>().unwrap();
    let arcs = parsed.resolve_arcs(view, tokens, tokens, Some(pos)).unwrap();
    assert_eq!(arcs.len(), 3);
    assert_eq!(arcs[0].dependent.word.form, "I");
    assert_eq!(arcs[0].dependent.pos, Some("PRON"));
    assert_eq!(arcs[0].head.as_ref().unwrap().word.form, "see");
    assert_eq!(arcs[1].children, &[Ix::new(0), Ix::new(2)]);
    assert_eq!(parsed.incoming_arcs(Ix::new(1)).unwrap().len(), 2);
    assert_eq!(parsed.subtree_span(Ix::new(1), view, tokens, tokens).unwrap(), Span::new(0, 11));
}

#[test]
fn shortest_path_into_bounds_arc_reads_by_path_length_not_tree_depth() {
    // Chain 0 <- 1 <- ... <- 3999, plus two sibling leaves 4000 and 4001 attached to 3999.
    let chain_len = 4_000u32;
    let mut heads: Vec<Option<Ix<Tokens>>> = (0..chain_len).map(|i| (i > 0).then(|| Ix::new(i - 1))).collect();
    heads.push(Some(Ix::new(chain_len - 1)));
    heads.push(Some(Ix::new(chain_len - 1)));
    let mut rels: Vec<String> = (0..chain_len).map(|i| if i == 0 { "root".into() } else { "ccomp".into() }).collect();
    rels.push("nsubj".into());
    rels.push("obj".into());
    let g = DependencyGraph::<Tokens>::new(heads, rels).unwrap();

    let (u, v) = (Ix::new(chain_len), Ix::new(chain_len + 1));
    assert_eq!(g.lowest_common_ancestor(u, v).unwrap(), Some(Ix::new(chain_len - 1)));

    let mut path = Vec::new();
    let hit = g.shortest_path_into(u, v, None, &mut path).unwrap();
    // 2 head reads to find the LCA at 3999 + 2 head reads to emit the 2 path steps.
    assert_eq!(hit, PathSearch { found: true, arc_reads: 4 });
    assert_eq!(path.len(), 2);
    assert_eq!((path[0].direction, path[0].relation), (ArcDirection::ToHead, "nsubj"));
    assert_eq!((path[1].direction, path[1].relation), (ArcDirection::ToDependent, "obj"));

    // Depth difference 4000 exceeds cap 2: rejected in O(1) with zero arc reads.
    let far = g.shortest_path_into(Ix::new(0), u, Some(2), &mut path).unwrap();
    assert_eq!(far, PathSearch { found: false, arc_reads: 0 });
    assert!(path.is_empty());

    // Self-path with cap 0 succeeds with zero steps and zero arc reads; cap 1 on 2-step path rejects.
    assert_eq!(g.shortest_path_into(u, u, Some(0), &mut path).unwrap(), PathSearch { found: true, arc_reads: 0 });
    assert!(path.is_empty());
    assert_eq!(g.shortest_path_into(u, v, Some(1), &mut path).unwrap(), PathSearch { found: false, arc_reads: 0 });

    // Equal-depth cousins (depth 2000 each) meeting only at root 0: cap 4 stops after 4 arc reads.
    let mut cousin_heads = vec![None];
    cousin_heads.extend((1..=2_000).map(|i| Some(Ix::new(if i == 1 { 0 } else { i - 1 }))));
    cousin_heads.extend((2_001..=4_000).map(|i| Some(Ix::new(if i == 2_001 { 0 } else { i - 1 }))));
    let cousins = DependencyGraph::<Tokens>::new(cousin_heads, vec!["dep".into(); 4_001]).unwrap();
    let miss = cousins.shortest_path_into(Ix::new(2_000), Ix::new(4_000), Some(4), &mut path).unwrap();
    assert_eq!(miss, PathSearch { found: false, arc_reads: 4 });

    // Out-of-range nodes fail even when equal or capped at 0.
    let oob = Ix::new(99_999);
    assert!(g.lowest_common_ancestor(oob, oob).is_err());
    assert!(g.shortest_path_into(oob, oob, Some(0), &mut path).is_err());
    assert!(g.shortest_path_into(Ix::new(0), oob, Some(0), &mut path).is_err());
}
