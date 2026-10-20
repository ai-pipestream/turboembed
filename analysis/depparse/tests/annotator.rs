mod support;
use turbo_analysis::{
    Analyzer, Annotator, Document, Ix, Key, Layer, Provenance, Span, Spans, Tag, TagSet, View, ViewId,
};
use turbo_analysis_layers::{
    Dependencies, MwtLanguage, PosTags, RuleMwtExpander, Sentences, Tokens, WordPosTags, Words,
};
use turbo_depparse::DependencyAnnotator;

fn spans<L>(ranges: &[(u32, u32)]) -> Spans<L> {
    let mut out = Spans::new(ViewId::SOURCE);
    for &(start, end) in ranges {
        out.push(Span::new(start, end));
    }
    out
}
fn tokens() -> Tokens {
    Tokens {
        spans: spans(&[(0, 2), (3, 4), (5, 6), (7, 8)]),
        sentence: Some(vec![Ix::new(0), Ix::new(0), Ix::new(1), Ix::new(1)]),
    }
}
fn sentences() -> Sentences {
    Sentences { spans: spans(&[(0, 4), (5, 8)]) }
}
fn pos(n: usize) -> PosTags {
    PosTags { tags: vec![Tag(0); n], tagset: TagSet::new("upos", &["NOUN"]), prob: None }
}
#[test]
fn token_pipeline_offsets_sentence_heads_and_records_provenance() {
    let mut doc = Document::new("é a b c");
    doc.add(&Key::DEFAULT, sentences(), Provenance::by("test")).unwrap();
    doc.add(&Key::DEFAULT, tokens(), Provenance::by("test")).unwrap();
    doc.add(&Key::DEFAULT, pos(4), Provenance::by("test")).unwrap();
    let analyzer = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<PosTags>::DEFAULT)
        .add(DependencyAnnotator::<Tokens>::new(support::parser()))
        .build()
        .unwrap();
    let mut docs = vec![doc];
    analyzer.run(&mut docs)[0].as_ref().unwrap();
    let deps = docs[0].get::<Dependencies>().unwrap();
    assert_eq!(deps.head, vec![None, Some(Ix::new(0)), None, Some(Ix::new(2))]);
    let p = docs[0].provenance(Key::<Dependencies>::DEFAULT.id()).unwrap();
    assert_eq!(p.version, "1");
    assert_eq!(p.model.as_ref().unwrap().sha256.len(), 64);
    assert!(p.reads.contains(&Key::<PosTags>::DEFAULT.id()));
}
#[test]
fn word_pipeline_keeps_multiword_token_granularity() {
    let text = "Voy al mar";
    let tok = Tokens { spans: spans(&[(0, 3), (4, 6), (7, 10)]), sentence: None };
    let words = RuleMwtExpander::new(MwtLanguage::Spanish).expand(text, &tok);
    assert_eq!(words.len(), 4);
    let mut doc = Document::new(text);
    doc.add(&Key::DEFAULT, Sentences { spans: spans(&[(0, 10)]) }, Provenance::by("test")).unwrap();
    doc.add(&Key::DEFAULT, tok, Provenance::by("test")).unwrap();
    doc.add(&Key::DEFAULT, words, Provenance::by("test")).unwrap();
    doc.add(
        &Key::DEFAULT,
        WordPosTags { tags: vec![Tag(0); 4], tagset: TagSet::new("upos", &["NOUN"]), prob: None },
        Provenance::by("test"),
    )
    .unwrap();
    let analyzer = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Words>::DEFAULT)
        .given(Key::<WordPosTags>::DEFAULT)
        .add(DependencyAnnotator::<Words>::new(support::parser()))
        .build()
        .unwrap();
    let mut docs = vec![doc];
    analyzer.run(&mut docs)[0].as_ref().unwrap();
    let deps = docs[0].get::<Dependencies<Words>>().unwrap();
    assert_eq!(deps.len(), 4);
    assert!(docs[0].get::<Dependencies<Tokens>>().is_none());
    assert_eq!(Dependencies::<Words>::NAME.to_string(), "opennlp:dependencies<turbo:words>");
}
#[test]
fn direct_boundary_checks_reject_bad_offsets_views_tags_and_alignment() {
    let doc = Document::new("é a b c");
    let view = doc.get::<View>().unwrap();
    let annotator = DependencyAnnotator::<Tokens>::new(support::parser());
    let tok = tokens();
    let sent = sentences();
    let tags = pos(4);
    assert!(annotator.annotate((view, &sent, &tok, &tok, &pos(3))).is_err());
    let bad = Tokens { spans: spans(&[(0, 1), (3, 4), (5, 6), (7, 8)]), sentence: None };
    assert!(annotator.annotate((view, &sent, &bad, &bad, &tags)).is_err());
    let mut bad = tokens();
    bad.sentence = Some(vec![Ix::new(1); 4]);
    assert!(annotator.annotate((view, &sent, &bad, &bad, &tags)).is_err());
    let badsent = Sentences { spans: spans(&[(0, 3), (5, 8)]) };
    assert!(annotator.annotate((view, &badsent, &tok, &tok, &tags)).is_err());
    let badsent = Sentences { spans: Spans::new(ViewId(1)) };
    assert!(annotator.annotate((view, &badsent, &tok, &tok, &tags)).is_err());
    let mut badtags = pos(4);
    badtags.tags[0] = Tag(1);
    assert!(annotator.annotate((view, &sent, &tok, &tok, &badtags)).is_err());
    let other = Tokens { spans: spans(&[(0, 2), (3, 4), (5, 6)]), sentence: None };
    assert!(annotator.annotate((view, &sent, &tok, &other, &tags)).is_err());
}
#[test]
fn empty_document_produces_empty_forest() {
    let doc = Document::new("");
    let tok = Tokens { spans: spans(&[]), sentence: None };
    let sent = Sentences { spans: spans(&[]) };
    let (out,) = DependencyAnnotator::<Tokens>::new(support::parser())
        .annotate((doc.get::<View>().unwrap(), &sent, &tok, &tok, &pos(0)))
        .unwrap();
    assert!(out.head.is_empty());
    out.check().unwrap();
}

#[test]
fn reversed_surface_order_is_rejected_before_resolving_word_ranges() {
    use turbo_analysis::IxRange;
    use turbo_analysis_layers::WordOrTokenAnchor;
    let doc = Document::new("é a");
    let view = doc.get::<View>().unwrap();
    let tok = Tokens { spans: spans(&[(3, 4), (0, 2)]), sentence: None };
    let mut words = Words::from_tokens(&tok);
    words.tokens = vec![IxRange::new(0, 2)];
    assert!(words.resolved(view, &tok).is_err());
}
#[test]
fn unordered_tokens_cannot_be_silently_parsed_in_storage_order() {
    let doc = Document::new("é a");
    let tok = Tokens { spans: spans(&[(3, 4), (0, 2)]), sentence: None };
    let sent = Sentences { spans: spans(&[(0, 4)]) };
    assert!(
        DependencyAnnotator::<Tokens>::new(support::parser())
            .annotate((doc.get::<View>().unwrap(), &sent, &tok, &tok, &pos(2)))
            .is_err()
    );
}
