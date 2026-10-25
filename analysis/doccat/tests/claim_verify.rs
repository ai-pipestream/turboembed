//! `ClaimVerificationAnnotator` with a rule-based NLI stand-in: each
//! edge kind, every candidate signal, the pair cap and token budget,
//! nested units, two-way heads and refusals.

mod claim_support;

use claim_support::{RuleNli, add_entities, add_vectors, doc, rows};
use turbo_analysis::{Annotator, Document, Error, Key, Layer, View};
use turbo_analysis_layers::{ClaimRelationKind as K, Entities, Sentences, Vectors};
use turbo_doccat::{ClaimVerificationAnnotator, ClaimVerifyConfig};

type Verifier = ClaimVerificationAnnotator<RuleNli, Sentences>;

/// The edges and how many directed pairs were scored.
type Run = (Vec<(usize, usize, K)>, usize);

fn run(verifier: Verifier, d: Document) -> Result<Run, Error> {
    let (edges,) = verifier.annotate((
        d.require(&Key::<View>::DEFAULT)?,
        d.require(&Key::<Sentences>::DEFAULT)?,
        d.get::<Entities>(),
        d.get::<Vectors<Sentences>>(),
    ))?;
    edges.check().unwrap();
    Ok((rows(&edges), verifier.scorer().seen()))
}

fn verifier() -> Verifier {
    ClaimVerificationAnnotator::new(RuleNli::three())
}

fn only_entities() -> ClaimVerifyConfig {
    ClaimVerifyConfig { min_word_overlap: None, ..ClaimVerifyConfig::default() }
}

#[test]
fn entails_contradicts_and_redundant() {
    let d = doc(&[
        "Acme bought Beta in March.",
        "Acme bought Beta.",
        "Gamma acquired Delta.",
        "Gamma never acquired Delta.",
        "Rain fell in Oslo.",
        "Rain fell in Oslo.",
    ]);
    let (edges, scored) = run(verifier(), d).unwrap();
    assert_eq!(edges, [(0, 1, K::Entails), (2, 3, K::Contradicts), (4, 5, K::Redundant)]);
    assert_eq!(scored, 6, "three candidates, both directions; unrelated pairs pruned");
}

#[test]
fn shared_entities_alone_make_candidates() {
    let sents = ["Acme grew.", "Profits at Acme fell.", "Rain fell in Oslo."];
    let (edges, scored) =
        run(ClaimVerificationAnnotator::with_config(RuleNli::three(), only_entities()), doc(&sents)).unwrap();
    assert!(edges.is_empty());
    assert_eq!(scored, 0, "no entities, no vectors, no overlap signal");
    let mut d = doc(&sents);
    add_entities(&mut d, &["Acme"]);
    let (_, scored) = run(ClaimVerificationAnnotator::with_config(RuleNli::three(), only_entities()), d).unwrap();
    assert_eq!(scored, 2);
}

#[test]
fn vector_window_keeps_related_and_drops_near_duplicates() {
    let sents = ["Acme grew.", "Rain fell.", "Beta shrank.", "Acme grew."];
    let mut d = doc(&sents);
    add_vectors(&mut d, &[&[1.0, 0.0], &[-0.6, 0.8], &[0.8, 0.6], &[1.0, 0.0]]);
    let config = ClaimVerifyConfig { max_similarity: 0.99, ..only_entities() };
    let v = ClaimVerificationAnnotator::with_config(RuleNli::three(), config);
    let (edges, scored) = run(v, d).unwrap();
    assert_eq!(scored, 4, "0-2 and 2-3 at cosine 0.8; 0-3 is a near duplicate; 1 is far from all");
    assert!(edges.is_empty());
}

#[test]
fn pair_cap_and_token_budget() {
    let sents = ["Acme sells parts.", "Acme sells parts.", "Acme sells parts.", "Acme sells parts."];
    let config = ClaimVerifyConfig { max_pairs: 2, ..ClaimVerifyConfig::default() };
    let (edges, scored) = run(ClaimVerificationAnnotator::with_config(RuleNli::three(), config), doc(&sents)).unwrap();
    assert_eq!((edges.len(), scored), (2, 4));
    let mut tight = RuleNli::three();
    tight.max_pair_tokens = 8;
    let (edges, scored) = run(ClaimVerificationAnnotator::new(tight), doc(&sents)).unwrap();
    assert_eq!((edges.len(), scored), (0, 0), "3 + 3 words + 3 specials is over 8");
}

#[test]
fn two_way_head_never_contradicts() {
    let v = ClaimVerificationAnnotator::new(RuleNli::with(&["entailment", "not_entailment"]));
    let (edges, scored) =
        run(v, doc(&["Gamma acquired Delta.", "Gamma never acquired Delta.", "Gamma acquired Delta."])).unwrap();
    assert_eq!(edges, [(0, 2, K::Redundant)]);
    assert_eq!(scored, 6);
}

#[test]
fn refuses_a_head_without_entailment() {
    let v = ClaimVerificationAnnotator::new(RuleNli::with(&["yes", "no"]));
    let err = run(v, doc(&["Acme grew.", "Acme grew."])).unwrap_err();
    assert!(format!("{err:?}").contains("no entailment outcome"), "{err:?}");
}

#[test]
fn provenance_and_config_hash() {
    let a = verifier();
    assert_eq!(a.model().unwrap().id, "rule-nli");
    let b = ClaimVerificationAnnotator::<_, Sentences>::with_config(
        RuleNli::three(),
        ClaimVerifyConfig { entail_threshold: 0.9, ..ClaimVerifyConfig::default() },
    );
    assert_ne!(a.config_hash(), b.config_hash());
    assert_eq!(a.config_hash(), verifier().config_hash());
}

#[test]
fn nested_chunks_are_never_paired() {
    use turbo_analysis::{Ix, IxRange, Provenance, Span, Spans, ViewId};
    use turbo_analysis_layers::{Chunks, Cut};
    let mut d = doc(&["Acme sells parts.", "Acme sells parts."]);
    let mut spans = Spans::new(ViewId(0));
    for s in [Span::new(0, 35), Span::new(0, 17), Span::new(18, 35)] {
        spans.push(s);
    }
    let r = |a, b| IxRange { start: Ix::new(a), end: Ix::new(b) };
    let chunks = Chunks {
        spans,
        level: vec![1, 0, 0],
        parent: vec![None, Some(Ix::new(0)), Some(Ix::new(0))],
        sentences: vec![r(0, 2), r(0, 1), r(1, 2)],
        cut: vec![Cut::DocumentStart, Cut::DocumentStart, Cut::Sentence],
        cost: vec![0; 3],
        model_tokens: vec![0; 3],
        over_budget: vec![false; 3],
        prompt_overhead: 0,
    };
    d.add(&Key::<Chunks>::DEFAULT, chunks, Provenance::by("test:chunks")).unwrap();
    let v = ClaimVerificationAnnotator::<_, Chunks>::new(RuleNli::three());
    let (edges,) = v
        .annotate((d.require(&Key::<View>::DEFAULT).unwrap(), d.require(&Key::<Chunks>::DEFAULT).unwrap(), None, None))
        .unwrap();
    let pairs: Vec<(usize, usize)> = edges.iter().map(|(_, e)| (e.from.get(), e.to.get())).collect();
    assert_eq!(pairs, [(1, 2)], "the section holds both chunks, so only the siblings pair");
    assert_eq!(v.scorer().seen(), 2);
}
