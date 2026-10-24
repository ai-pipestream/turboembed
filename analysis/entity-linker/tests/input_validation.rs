//! Regression tests for candidate ordering, malformed input layers handed
//! straight to `EntityLinker::annotate`, and `LinkedEntities` payloads whose
//! row count disagrees with the `Entities` layer they are aligned to.

use turbo_analysis::export::{ExportError, Writer};
use turbo_analysis::{
    Annotator, Document, Ix, IxRange, Key, Layer, Provenance, Span, Spans, TagSet, View, ViewId, export, import,
};
use turbo_analysis_layers::{Entities, Sentences, Tokens};
use turbo_entity_linker::{
    CatalogProvider, EntityLinker, KbEntity, KnowledgeBase, LinkedEntities, LinkerConfig, LinkerWeights, codecs,
};

fn spans_of<L>(items: impl IntoIterator<Item = Span>) -> Spans<L> {
    let mut s = Spans::new(ViewId(0));
    for sp in items {
        s.push(sp);
    }
    s
}

/// One `PERSON` entity per `(start, end)` byte span, each covering the
/// token range `tokens`.
fn entities(spans: &[(u32, u32)], tokens: IxRange<Tokens>) -> Entities {
    let kinds = TagSet::new("ner", &["PERSON"]);
    let n = spans.len();
    Entities {
        spans: spans_of(spans.iter().map(|&(s, e)| Span::new(s, e))),
        tokens: vec![tokens; n],
        kind: vec![kinds.tag("PERSON").unwrap(); n],
        kinds,
        score: vec![0.9; n],
    }
}

fn tokens(spans: &[(u32, u32)]) -> Tokens {
    Tokens { spans: spans_of(spans.iter().map(|&(s, e)| Span::new(s, e))), sentence: None }
}

fn sentences(spans: &[(u32, u32)]) -> Sentences {
    Sentences { spans: spans_of(spans.iter().map(|&(s, e)| Span::new(s, e))) }
}

fn mercury_linker(config: LinkerConfig) -> EntityLinker {
    // The person is type compatible with a `PERSON` mention but has a weak
    // prior; the untyped entry has a strong prior but a type score of 0.75.
    let kb = KnowledgeBase::from_entities(
        "test",
        vec![
            KbEntity::new("P1", "Freddie Mercury").with_alias("Mercury", 0.2).with_types(["PERSON"]).with_prior(0.05),
            KbEntity::new("U1", "Mercury").with_alias("Mercury", 4.0).with_prior(1.0),
        ],
    )
    .unwrap();
    EntityLinker::new().with_config(config).add_provider(CatalogProvider::new(kb))
}

fn run(
    linker: &EntityLinker,
    view: &View,
    ents: &Entities,
    sents: Option<&Sentences>,
    toks: Option<&Tokens>,
) -> Result<LinkedEntities, turbo_analysis::Error> {
    linker.annotate((view, ents, sents, toks, None, None, None, None)).map(|(l,)| l)
}

#[test]
fn ineligible_candidate_never_outranks_an_eligible_one_in_score() {
    // Only the prior counts for eligible candidates, so the eligible person
    // scores its weak prior (about 0.05) while the untyped entry's penalized
    // score is 0.10 * 0.97 * 0.75 (about 0.07), the inversion that used to
    // leave the candidates out of score order.
    let config = LinkerConfig {
        weights: LinkerWeights { prior_weight: 1.0, type_weight: 0.0, context_weight: 0.0, coherence_weight: 0.0 },
        min_type_compatibility: 0.80,
        nil_threshold: 0.0,
        ..LinkerConfig::default()
    };
    let linker = mercury_linker(config);
    let doc_tokens = tokens(&[(0, 7)]);
    let mut doc = Document::new("Mercury");
    doc.add(&Key::<Tokens>::DEFAULT, doc_tokens, Provenance::by("test:tokens")).unwrap();
    doc.add(&Key::<Entities>::DEFAULT, entities(&[(0, 7)], IxRange::new(0, 1)), Provenance::by("test:entities"))
        .unwrap();
    linker.annotate_document(&mut doc).expect("the written layer passes LinkedEntities::check");

    let linked = doc.require(&Key::<LinkedEntities>::DEFAULT).unwrap();
    linked.check().unwrap();
    let row = linked.get(Ix::new(0));
    assert_eq!(row.kb_id, Some("P1"), "the eligible candidate is still the link");
    let cands = &row.candidates;
    assert_eq!(cands.len(), 2);
    assert_eq!(cands[0].kb_id, "P1");
    assert_eq!(cands[1].kb_id, "U1");
    assert!(cands.windows(2).all(|w| w[0].score >= w[1].score), "{cands:?}");
    assert!(cands[1].score <= cands[0].score && cands[1].score > 0.0, "{cands:?}");
}

#[test]
fn annotate_rejects_malformed_spans_without_panicking() {
    let text = "Zoë met Ada.";
    let doc = Document::new(text);
    let view = doc.require(&Key::<View>::DEFAULT).unwrap();
    let linker = mercury_linker(LinkerConfig::default());
    let good_toks = tokens(&[(0, 4), (5, 8), (9, 12)]);
    let good_sents = sentences(&[(0, 13)]);
    let ok = entities(&[(9, 12)], IxRange::new(2, 3));
    assert!(run(&linker, view, &ok, Some(&good_sents), Some(&good_toks)).is_ok());

    let cases: Vec<(&str, Entities, Option<Sentences>, Option<Tokens>)> = vec![
        ("entity past the end", entities(&[(9, 40)], IxRange::new(2, 3)), None, None),
        ("entity mid-codepoint", entities(&[(0, 3)], IxRange::new(0, 1)), None, None),
        ("sentence past the end", ok_entity(), Some(sentences(&[(0, 99)])), None),
        ("sentence mid-codepoint", ok_entity(), Some(sentences(&[(3, 13)])), None),
        ("token past the end", ok_entity(), None, Some(tokens(&[(0, 4), (5, 8), (9, 50)]))),
        ("token mid-codepoint", ok_entity(), None, Some(tokens(&[(0, 3), (5, 8), (9, 12)]))),
        ("token range past the tokens", entities(&[(9, 12)], IxRange::new(2, 9)), None, Some(tokens(&[(0, 4)]))),
    ];
    for (what, ents, sents, toks) in cases {
        let got = run(&linker, view, &ents, sents.as_ref(), toks.as_ref());
        match got {
            Err(turbo_analysis::Error::Annotator { annotator, .. }) => assert_eq!(annotator, EntityLinker::ID),
            other => panic!("{what}: expected an annotator error, got {other:?}"),
        }
    }

    // An inverted span is refused by the layer check, before any slicing.
    let mut inverted = ok_entity();
    inverted.spans = Spans::new(ViewId(0));
    inverted.spans.push(Span { start: 12, end: 9 });
    assert!(run(&linker, view, &inverted, None, None).is_err());
}

fn ok_entity() -> Entities {
    entities(&[(9, 12)], IxRange::new(2, 3))
}

#[test]
fn import_rejects_linked_rows_that_do_not_match_entities() {
    let linker = mercury_linker(LinkerConfig::default());
    let mut doc = Document::new("Mercury met Mercury.");
    doc.add(&Key::<Tokens>::DEFAULT, tokens(&[(0, 7), (8, 11), (12, 19)]), Provenance::by("test:tokens")).unwrap();
    doc.add(
        &Key::<Entities>::DEFAULT,
        entities(&[(0, 7), (12, 19)], IxRange::new(0, 1)),
        Provenance::by("test:entities"),
    )
    .unwrap();
    linker.annotate_document(&mut doc).unwrap();
    let codec_set = codecs();
    let mut graph = export(&doc, &codec_set).unwrap();
    assert!(import(&graph, &codec_set).is_ok());

    let name = <LinkedEntities as Layer>::NAME.to_string();
    let record = graph.layers.iter_mut().find(|r| r.layer == name).expect("a LinkedEntities record");
    let mut w = Writer::new();
    w.usize(1).unwrap();
    // One NIL row: no kb_id, blank provider and name, alias, score,
    // NoCandidates, cluster 0, no candidates.
    w.u8(0);
    w.str("").unwrap();
    w.str("").unwrap();
    w.str("Mercury").unwrap();
    w.f64(0.0);
    w.u8(1);
    w.u8(1);
    w.u32(0);
    w.usize(0).unwrap();
    record.body = Some(w.finish());
    record.len = 1;
    match import(&graph, &codec_set) {
        Err(ExportError::InvalidLayer { detail, .. }) => {
            assert!(detail.contains("aligned Entities layer of 2"), "{detail}")
        }
        other => panic!("expected InvalidLayer, got {:?}", other.map(|_| ())),
    }
}
