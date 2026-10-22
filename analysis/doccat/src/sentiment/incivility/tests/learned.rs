//! The learned head in the annotator: the three stages, the context
//! vector, neutral identity mentions against attacks, and the layer
//! through the document graph.
//!
//! The sentence vectors are given as data, each value a known property of
//! its sentence, so what the head should say about each is known without
//! an encoder.

use super::*;
use turbo_analysis_layers::export::codecs;

fn model() -> ModelRef {
    ModelRef { id: "test-encoder".into(), revision: "1".into(), sha256: "e".repeat(64) }
}

/// Sentences of `text` at `ranges`, each with the vector beside it.
fn doc(text: &str, ranges: &[(u32, u32)], vectors: &[[f32; 2]]) -> Document {
    let mut d = make_doc_with_sentences(text, ranges);
    let data = vectors.iter().flatten().copied().collect();
    d.add(&Key::DEFAULT, Vectors::<Sentences>::new(2, data, Vec::new(), model()), Provenance::by("test")).unwrap();
    d
}

/// Runs `a` and gives the layer and its provenance's model id.
fn run(a: IncivilityAnnotator, d: &mut Document) -> Result<(Vec<IncivilityScores>, String), String> {
    let p = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Vectors<Sentences>>::DEFAULT)
        .add(a)
        .build()
        .unwrap();
    if let Some(Err(e)) = p.run(std::slice::from_mut(d)).into_iter().next() {
        return Err(e.to_string());
    }
    let layer = d.get::<Incivility>().unwrap();
    let scores = (0..layer.len()).map(|i| layer.scores(Ix::new(i as u32))).collect();
    Ok((scores, d.provenance(Key::<Incivility>::DEFAULT.id()).unwrap().model.clone().unwrap().id))
}

/// Insult and identity attack read value 0 (hostility), weight 6, bias -3.
fn hostility_head() -> IncivilityVectorHead {
    let w: Vec<f32> = (0..6).flat_map(|l| if l == 0 || l == 3 { [6.0, 0.0] } else { [0.0, 0.0] }).collect();
    IncivilityVectorHead::new(2, w, [-3.0; 6]).unwrap()
}

const TEXT: &str = "You are an idiot. The weather is fine.";
const RANGES: [(u32, u32); 2] = [(0, 17), (18, 38)];

#[test]
fn each_stage_scores_with_its_own_signals() {
    // The lexicon finds "idiot" in the first sentence; the vectors say
    // the second is hostile, so the two stages disagree on purpose.
    let vectors = [[0.0, 1.0], [1.0, 0.0]];
    let head: Arc<dyn IncivilityVectorScorer> = Arc::new(hostility_head());
    let with = |stage| IncivilityAnnotator::new().with_vector_scorer(Arc::clone(&head)).with_stage(stage);
    let (lex, lex_id) = run(with(IncivilityStage::Lexicon), &mut doc(TEXT, &RANGES, &vectors)).unwrap();
    let (learned, learned_id) = run(with(IncivilityStage::Learned), &mut doc(TEXT, &RANGES, &vectors)).unwrap();
    let (fused, fused_id) = run(with(IncivilityStage::Fused), &mut doc(TEXT, &RANGES, &vectors)).unwrap();
    assert!(lex[0].insult >= 0.5 && lex[1].insult < 0.5);
    assert!(learned[0].insult < 0.5 && learned[1].insult >= 0.5);
    assert!(fused[0].insult >= 0.5 && fused[1].insult >= 0.5);
    // The learned stage gives the head's own scores.
    assert_eq!(learned[1], hostility_head().score(&vectors[1], None).unwrap());
    assert_eq!(lex_id, "oasis-incivility:lexicon:no-reply-context");
    assert_eq!(learned_id, "incivility-vector-head:learned");
    // A head with no encoder named is still in the fused stage's id.
    assert_eq!(fused_id, "incivility-vector-head+lexicon:no-reply-context");
    assert!(lex_id != learned_id && learned_id != fused_id && lex_id != fused_id);
    let preceding = with(IncivilityStage::Fused).with_vector_context(IncivilityVectorContext::Preceding);
    let (_, id) = run(preceding, &mut doc(TEXT, &RANGES, &vectors)).unwrap();
    assert_eq!(id, "incivility-vector-head+lexicon:no-reply-context:preceding-vector");
    let hashes: Vec<u64> = [IncivilityStage::Lexicon, IncivilityStage::Learned, IncivilityStage::Fused]
        .into_iter()
        .map(|s| Annotator::config_hash(&with(s)))
        .collect();
    assert!(hashes[0] != hashes[1] && hashes[1] != hashes[2] && hashes[0] != hashes[2], "{hashes:?}");
    // A positive learned label carries evidence in its sentence.
    let mut d = doc(TEXT, &RANGES, &vectors);
    run(with(IncivilityStage::Learned), &mut d).unwrap();
    assert!(!d.get::<Incivility>().unwrap().evidence_for_label(Ix::new(1), IncivilityLabel::Insult).is_empty());
}

#[test]
fn the_lexicon_stage_does_not_read_the_head() {
    // A head bound to another encoder than the vectors': the stages that
    // read it refuse the document, the lexicon stage does not.
    let other = ModelRef { id: "other-encoder".into(), ..model() };
    let head: Arc<dyn IncivilityVectorScorer> = Arc::new(hostility_head().with_model(other));
    let with = |stage| IncivilityAnnotator::new().with_vector_scorer(Arc::clone(&head)).with_stage(stage);
    let vectors = [[0.0, 1.0], [1.0, 0.0]];
    run(with(IncivilityStage::Lexicon), &mut doc(TEXT, &RANGES, &vectors)).unwrap();
    for stage in [IncivilityStage::Learned, IncivilityStage::Fused] {
        assert!(run(with(stage), &mut doc(TEXT, &RANGES, &vectors)).is_err(), "{stage:?}");
    }
    // Nor is the head in its configuration hash.
    let bare = IncivilityAnnotator::new().with_stage(IncivilityStage::Lexicon);
    assert_eq!(Annotator::config_hash(&with(IncivilityStage::Lexicon)), Annotator::config_hash(&bare));
}

#[test]
fn the_learned_stage_needs_a_head_and_vectors() {
    let learned = IncivilityAnnotator::new().with_stage(IncivilityStage::Learned);
    let e = run(learned.clone(), &mut doc(TEXT, &RANGES, &[[0.0, 1.0], [1.0, 0.0]])).unwrap_err();
    assert!(e.contains("needs a vector scorer"), "{e}");
    let mut bare = make_doc_with_sentences(TEXT, &RANGES);
    let p = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .add(learned.with_vector_scorer(Arc::new(hostility_head())))
        .build()
        .unwrap();
    let e = p.run(std::slice::from_mut(&mut bare)).remove(0).unwrap_err().to_string();
    assert!(e.contains("needs the unit layer's vectors"), "{e}");
    // A unit the vectors list as empty fails the document rather than
    // scoring 0 on every label.
    let mut d = make_doc_with_sentences(TEXT, &RANGES);
    let v = Vectors::<Sentences>::new(2, vec![0.0, 0.0, 1.0, 0.0], vec![Ix::new(0)], model());
    d.add(&Key::DEFAULT, v, Provenance::by("test")).unwrap();
    let a =
        IncivilityAnnotator::new().with_stage(IncivilityStage::Learned).with_vector_scorer(Arc::new(hostility_head()));
    let e = run(a, &mut d).unwrap_err();
    assert!(e.contains("unit 0 has no vector"), "{e}");
}

#[test]
fn the_preceding_or_parent_vector_is_read_through_the_context_matrix() {
    // Insult also reads the context's hostility, weight 5: a mild reply
    // (hostility 0.25) is an insult after a hostile sentence only.
    let ctx: Vec<f32> = (0..6).flat_map(|l| if l == 0 { [5.0, 0.0] } else { [0.0, 0.0] }).collect();
    let head: Arc<dyn IncivilityVectorScorer> = Arc::new(hostility_head().with_context(ctx).unwrap());
    let text = "That plan is a disgrace. Sure, whatever you say. Thanks for the update.";
    let ranges = [(0, 24), (25, 48), (49, 71)];
    let vectors = [[1.0, 0.0], [0.25, 0.0], [0.0, 1.0]];
    let with = |c| {
        IncivilityAnnotator::new()
            .with_vector_scorer(Arc::clone(&head))
            .with_stage(IncivilityStage::Learned)
            .with_vector_context(c)
    };
    let (none, _) = run(with(IncivilityVectorContext::None), &mut doc(text, &ranges, &vectors)).unwrap();
    let (prev, id) = run(with(IncivilityVectorContext::Preceding), &mut doc(text, &ranges, &vectors)).unwrap();
    assert!(none[1].insult < 0.5 && prev[1].insult >= 0.5, "{} {}", none[1].insult, prev[1].insult);
    // The third follows a mild sentence, and the first follows none.
    assert!(prev[2].insult < 0.5);
    assert_eq!(prev[0], none[0]);
    assert_eq!(id, "incivility-vector-head:learned:preceding-vector");
    let parent = IncivilityAnnotator::new()
        .with_vector_scorer(Arc::clone(&head))
        .with_stage(IncivilityStage::Learned)
        .with_parent_vector(&[1.0, 0.0]);
    let (by_parent, id) = run(parent, &mut doc(text, &ranges, &vectors)).unwrap();
    // A hostile parent post: every sentence reads it, the calm third one too.
    assert!(by_parent.iter().all(|s| s.insult >= 0.5), "{by_parent:?}");
    assert_eq!(id, "incivility-vector-head:learned:parent-vector");
    // A parent vector of the wrong width fails the document.
    let wrong = with(IncivilityVectorContext::Parent(vec![1.0f32].into()));
    assert!(run(wrong, &mut doc(text, &ranges, &vectors)).is_err());
}

/// The identity sentences: two neutral mentions, then two attacks.
fn identity_doc() -> Document {
    let text = "The community center hosts an LGBTQ+ book club on Tuesdays. \
                She is a Muslim software engineer from Chicago. \
                Muslims are vermin and should be thrown out. \
                Gay people are a plague on this town.";
    let starts = [0u32, 60, 108, 153];
    let ranges: Vec<(u32, u32)> =
        (0..4).map(|i| (starts[i], if i < 3 { starts[i + 1] - 1 } else { text.len() as u32 })).collect();
    // Vector values: [identity term mentioned, hostility].
    doc(text, &ranges, &[[1.0, 0.0], [1.0, 0.0], [1.0, 1.0], [1.0, 1.0]])
}

/// Each sentence's scores from `head` at `stage`.
fn identity_scores(head: IncivilityVectorHead, stage: IncivilityStage) -> Vec<IncivilityScores> {
    let a = IncivilityAnnotator::new().with_vector_scorer(Arc::new(head)).with_stage(stage);
    run(a, &mut identity_doc()).unwrap().0
}

fn neutral_below_attacks_above(scores: &[IncivilityScores]) -> bool {
    let t = DEFAULT_INCIVILITY_THRESHOLD;
    scores[..2].iter().all(|s| IncivilityLabel::ALL.into_iter().all(|l| s.get(l) < t))
        && scores[2..].iter().all(|s| s.identity_attack >= t)
}

#[test]
fn neutral_identity_mentions_stay_below_while_attacks_cross() {
    // Configured: identity attack reads hostility, not the identity term.
    let w: Vec<f32> = (0..6).flat_map(|l| if l == 3 { [0.0, 6.0] } else { [0.0, 0.0] }).collect();
    let configured = IncivilityVectorHead::new(2, w, [-3.0; 6]).unwrap();
    // Fitted, from rows where the identity term and hostility vary apart:
    // hostility without an identity term is an insult, an identity term
    // without hostility is nothing, both together is an identity attack.
    let rows: [[f32; 2]; 8] =
        [[1.0, 0.0], [1.0, 0.0], [1.0, 1.0], [1.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]];
    let insult = IncivilityScores { insult: 1.0, ..Default::default() };
    let attack = IncivilityScores { identity_attack: 1.0, insult: 1.0, ..Default::default() };
    let none = IncivilityScores::default();
    let targets = [none, none, attack, attack, insult, insult, none, none];
    let refs: Vec<&[f32]> = rows.iter().map(|r| &r[..]).collect();
    let fitted = IncivilityVectorHead::fit(&refs, None, &targets, &IncivilityFitOptions::default()).unwrap();
    for head in [configured, fitted] {
        // The head alone, then with the lexicon.
        for stage in [IncivilityStage::Learned, IncivilityStage::Fused] {
            let scores = identity_scores(head.clone(), stage);
            assert!(neutral_below_attacks_above(&scores), "{stage:?}: {scores:?}");
        }
    }
    // A head keyed on the identity term itself fails the check: the test
    // can tell such a head from the ones above.
    let w: Vec<f32> = (0..6).flat_map(|l| if l == 3 { [6.0, 0.0] } else { [0.0, 0.0] }).collect();
    let keyed = IncivilityVectorHead::new(2, w, [-3.0; 6]).unwrap();
    assert!(!neutral_below_attacks_above(&identity_scores(keyed, IncivilityStage::Learned)));
}

#[test]
fn the_learned_layer_round_trips_through_the_document_graph() {
    use turbo_analysis::export::{export, import};
    let head = Arc::new(hostility_head());
    let mut d = doc(TEXT, &RANGES, &[[0.0, 1.0], [1.0, 0.0]]);
    run(IncivilityAnnotator::new().with_vector_scorer(head).with_stage(IncivilityStage::Learned), &mut d).unwrap();
    let graph = export(&d, &codecs()).unwrap();
    let bytes = graph.to_bytes().unwrap();
    let back = import(&turbo_analysis::export::DocumentGraph::from_bytes(&bytes).unwrap(), &codecs()).unwrap();
    let (a, b) = (d.get::<Incivility>().unwrap(), back.get::<Incivility>().unwrap());
    assert_eq!(a.len(), b.len());
    for i in 0..a.len() {
        let ix = Ix::new(i as u32);
        assert_eq!(a.scores(ix), b.scores(ix));
        assert_eq!(a.evidence_of(ix).collect::<Vec<_>>(), b.evidence_of(ix).collect::<Vec<_>>());
    }
}
