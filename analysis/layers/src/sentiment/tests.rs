use super::*;
use crate::{Block, Chunks, Cut, Sentences, Structure, Tokens};
use turbo_analysis::{
    Analyzer, Annotator, Bound, Document, Error, Ix, IxRange, Key, Layer, Provenance, Refs, Span, Spans, TagSet, ViewId,
};

fn make_sentences_and_tokens() -> (Sentences, Tokens) {
    let mut s_spans = Spans::new(ViewId::SOURCE);
    s_spans.push(Span::new(0, 49));
    s_spans.push(Span::new(50, 65));
    let sentences = Sentences { spans: s_spans };

    let word_ranges: &[(u32, u32, u32)] = &[
        (0, 3, 0),   // 0: The
        (4, 8, 0),   // 1: food
        (9, 12, 0),  // 2: was
        (13, 16, 0), // 3: not
        (17, 22, 0), // 4: great
        (22, 23, 0), // 5: ,
        (24, 27, 0), // 6: but
        (28, 31, 0), // 7: the
        (32, 39, 0), // 8: service
        (40, 43, 0), // 9: was
        (44, 48, 0), // 10: fast
        (48, 49, 0), // 11: .
        (50, 55, 1), // 12: Price
        (56, 59, 1), // 13: was
        (60, 64, 1), // 14: fair
        (64, 65, 1), // 15: .
    ];
    let mut t_spans = Spans::new(ViewId::SOURCE);
    let mut t_sent = Vec::new();
    for &(a, b, s) in word_ranges {
        t_spans.push(Span::new(a, b));
        t_sent.push(Ix::new(s));
    }
    let tokens = Tokens { spans: t_spans, sentence: Some(t_sent) };
    (sentences, tokens)
}

fn sample_doc() -> (Document, Sentences, Tokens) {
    let text = "The food was not great, but the service was fast. Price was fair.";
    let mut doc = Document::new(text);
    let (sentences, tokens) = make_sentences_and_tokens();
    doc.add(&Key::DEFAULT, sentences, Provenance::by("test")).unwrap();
    doc.add(&Key::DEFAULT, tokens, Provenance::by("test")).unwrap();
    let (sentences, tokens) = make_sentences_and_tokens();
    (doc, sentences, tokens)
}

#[test]
fn opinions_and_aspects_construct_and_link_on_document() {
    let (mut doc, sentences, tokens) = sample_doc();
    let opinions = Opinions::from_items(
        &tokens,
        [
            Opinion { tokens: IxRange::new(4, 5), polarity: 0.8, intensity: 1.25, negated: true },
            Opinion { tokens: IxRange::new(10, 11), polarity: 0.7, intensity: 1.0, negated: false },
            Opinion { tokens: IxRange::new(14, 15), polarity: 0.5, intensity: 1.0, negated: false },
        ],
    )
    .unwrap();
    assert_eq!(opinions.len(), 3);
    assert_eq!(opinions.get(Ix::new(0)).effective_polarity(), -1.0);
    assert_eq!(opinions.span(Ix::new(0), &tokens), Span::new(17, 22));
    doc.add(&Key::DEFAULT, opinions.clone(), Provenance::by("test")).unwrap();

    let cats = TagSet::new("absa", &["food#quality", "service#general", "ambience#general", "price"]);
    let aspects = Aspects::from_items(
        &tokens,
        Some(&sentences),
        Some(&opinions),
        cats.clone(),
        [
            Aspect {
                target: AspectTarget::Term { tokens: IxRange::new(1, 2), sentence: Some(Ix::new(0)) },
                category: cats.tag("food#quality"),
                polarity: PolarityScores::new(0.05, 0.90, 0.05),
                opinions: &[Ix::new(0)],
            },
            Aspect {
                target: AspectTarget::Term { tokens: IxRange::new(8, 9), sentence: Some(Ix::new(0)) },
                category: cats.tag("service#general"),
                polarity: PolarityScores::new(0.88, 0.04, 0.08),
                opinions: &[Ix::new(1), Ix::new(1)],
            },
            Aspect {
                target: AspectTarget::Implicit { sentence: Ix::new(1) },
                category: cats.tag("price"),
                polarity: PolarityScores::new(0.60, 0.10, 0.30),
                opinions: &[Ix::new(2)],
            },
        ],
    )
    .unwrap();
    assert_eq!(aspects.len(), 3);
    assert_eq!(aspects.polarity(Ix::new(0)).label(), Polarity::Negative);
    assert_eq!(aspects.polarity(Ix::new(1)).label(), Polarity::Positive);
    assert_eq!(aspects.opinions_of(Ix::new(1)), &[Ix::new(1)]);
    assert_eq!(aspects.category_label(Ix::new(2)), Some("price"));
    assert_eq!(aspects.span(Ix::new(0), &tokens, Some(&sentences)), Some(Span::new(4, 8)));
    assert_eq!(aspects.span(Ix::new(2), &tokens, Some(&sentences)), Some(Span::new(50, 65)));

    doc.add(&Key::DEFAULT, aspects, Provenance::by("test")).unwrap();
    assert_eq!(
        doc.made_on(Key::<Aspects>::DEFAULT.id()),
        [Key::<Tokens>::DEFAULT.id(), Key::<Sentences>::DEFAULT.id(), Key::<Opinions>::DEFAULT.id()]
    );
}

#[test]
fn incivility_and_pragmatics_construct_and_check_evidence_slices() {
    let (mut doc, sentences, tokens) = sample_doc();
    let make_chunks = || {
        let mut c_spans = Spans::new(ViewId::SOURCE);
        c_spans.push(Span::new(0, 65));
        Chunks {
            spans: c_spans,
            level: vec![0],
            parent: vec![None],
            sentences: vec![IxRange::new(0, 2)],
            cut: vec![Cut::DocumentStart],
            cost: vec![10],
            model_tokens: vec![16],
            over_budget: vec![false],
        }
    };
    let chunks = make_chunks();
    doc.add(&Key::DEFAULT, make_chunks(), Provenance::by("test")).unwrap();

    let make_structure = || {
        let mut st_spans = Spans::new(ViewId::SOURCE);
        st_spans.push(Span::new(0, 65));
        Structure { spans: st_spans, kind: vec![Block::Paragraph], parent: vec![None], depth: vec![0] }
    };
    let structure = make_structure();
    doc.add(&Key::DEFAULT, make_structure(), Provenance::by("test")).unwrap();

    let mut scores0 = IncivilityScores::default();
    scores0.set(IncivilityLabel::Insult, 0.75);
    scores0.set(IncivilityLabel::Profanity, 0.25);
    let incivility = Incivility::from_items(
        &tokens,
        Some(&sentences),
        Some(&chunks),
        Some(&structure),
        [
            IncivilityItem {
                target: IncivilityTarget::Sentence(Ix::new(0)),
                scores: scores0,
                evidence: &[
                    (IncivilityLabel::Profanity, IxRange::new(3, 5)),
                    (IncivilityLabel::Insult, IxRange::new(4, 5)),
                    (IncivilityLabel::Insult, IxRange::new(1, 2)),
                ],
            },
            IncivilityItem {
                target: IncivilityTarget::Chunk(Ix::new(0)),
                scores: IncivilityScores::default(),
                evidence: &[],
            },
            IncivilityItem {
                target: IncivilityTarget::Block(Ix::new(0)),
                scores: IncivilityScores::default(),
                evidence: &[],
            },
        ],
    )
    .unwrap();

    assert_eq!(
        incivility.evidence_for_label(Ix::new(0), IncivilityLabel::Insult),
        &[IxRange::new(1, 2), IxRange::new(4, 5)]
    );
    assert_eq!(incivility.evidence_for_label(Ix::new(0), IncivilityLabel::Profanity), &[IxRange::new(3, 5)]);
    assert!(incivility.evidence_for_label(Ix::new(0), IncivilityLabel::Threat).is_empty());
    assert_eq!(
        incivility.target_span(Ix::new(2), Some(&sentences), Some(&chunks), Some(&structure)),
        Some(Span::new(0, 65))
    );
    doc.add(&Key::DEFAULT, incivility, Provenance::by("test")).unwrap();

    let pragmatics = Pragmatics::from_items(
        &sentences,
        &tokens,
        [
            PragmaticsItem {
                score: 0.82,
                evidence: Some(Incongruity { positive: IxRange::new(4, 5), negative: IxRange::new(3, 4) }),
            },
            PragmaticsItem { score: 0.05, evidence: None },
        ],
    )
    .unwrap();
    assert_eq!(pragmatics.evidence_spans(Ix::new(0), &tokens), Some((Span::new(17, 22), Span::new(13, 16))));
    assert_eq!(pragmatics.evidence_spans(Ix::new(1), &tokens), None);
    doc.add(&Key::DEFAULT, pragmatics, Provenance::by("test")).unwrap();
}

#[test]
fn bounds_checking_refuses_invalid_token_sentence_and_opinion_indices() {
    let (mut doc, sentences, tokens) = sample_doc();

    // Out-of-bounds token range on Opinions.
    let err = Opinions::from_items(
        &tokens,
        [Opinion { tokens: IxRange::new(14, 17), polarity: 0.5, intensity: 1.0, negated: false }],
    )
    .unwrap_err();
    assert!(err.contains("out of bounds for 16 tokens"), "{err}");

    // Empty token range on Opinions.
    let err = Opinions::from_items(
        &tokens,
        [Opinion { tokens: IxRange::new(3, 3), polarity: 0.5, intensity: 1.0, negated: false }],
    )
    .unwrap_err();
    assert!(err.contains("empty token range"), "{err}");

    // Out-of-order token ranges on Opinions.
    let err = Opinions::from_items(
        &tokens,
        [
            Opinion { tokens: IxRange::new(4, 5), polarity: 0.5, intensity: 1.0, negated: false },
            Opinion { tokens: IxRange::new(1, 2), polarity: 0.5, intensity: 1.0, negated: false },
        ],
    )
    .unwrap_err();
    assert!(err.contains("not in token order"), "{err}");

    // Direct unchecked push with out-of-bounds index is caught by Document::add.
    let mut bad_ops = Opinions::new();
    bad_ops.push(Opinion { tokens: IxRange::new(15, 20), polarity: 0.5, intensity: 1.0, negated: false });
    let err = doc.add(&Key::DEFAULT, bad_ops, Provenance::by("test")).unwrap_err();
    assert!(matches!(err, Error::Invalid { .. }), "{err:?}");

    // Aspect whose token range crosses a sentence boundary or disagrees with its sentence index.
    let cats = TagSet::new("absa", &["price"]);
    let err = Aspects::from_items(
        &tokens,
        Some(&sentences),
        None,
        cats.clone(),
        [Aspect {
            target: AspectTarget::Term { tokens: IxRange::new(1, 2), sentence: Some(Ix::new(1)) },
            category: None,
            polarity: PolarityScores::new(0.5, 0.2, 0.3),
            opinions: &[],
        }],
    )
    .unwrap_err();
    assert!(err.contains("not sentence 1"), "{err}");

    // Aspect with out-of-bounds opinion index.
    let ops = Opinions::from_items(
        &tokens,
        [Opinion { tokens: IxRange::new(4, 5), polarity: 0.8, intensity: 1.0, negated: false }],
    )
    .unwrap();
    let err = Aspects::from_items(
        &tokens,
        Some(&sentences),
        Some(&ops),
        cats,
        [Aspect {
            target: AspectTarget::Term { tokens: IxRange::new(1, 2), sentence: Some(Ix::new(0)) },
            category: None,
            polarity: PolarityScores::new(0.5, 0.2, 0.3),
            opinions: &[Ix::new(1)],
        }],
    )
    .unwrap_err();
    assert!(err.contains("opinion index 1 is out of bounds"), "{err}");

    // Pragmatics evidence outside its own sentence.
    let err = Pragmatics::from_items(
        &sentences,
        &tokens,
        [
            PragmaticsItem {
                score: 0.9,
                evidence: Some(Incongruity { positive: IxRange::new(4, 5), negative: IxRange::new(12, 13) }),
            },
            PragmaticsItem { score: 0.1, evidence: None },
        ],
    )
    .unwrap_err();
    assert!(err.contains("outside sentence 0"), "{err}");
}

#[test]
fn score_only_layers_add_to_a_document_without_tokens() {
    // A sentence-level head can score sarcasm, incivility, or implicit
    // aspects with no token evidence; such layers index no Tokens and
    // must not require a Tokens layer on the document.
    let (sentences, _) = make_sentences_and_tokens();
    let mut doc = Document::new("The food was not great, but the service was fast. Price was fair.");
    doc.add(&Key::DEFAULT, sentences, Provenance::by("test")).unwrap();

    let mut prag = Pragmatics::new();
    prag.push(PragmaticsItem { score: 0.7, evidence: None });
    prag.push(PragmaticsItem { score: 0.1, evidence: None });
    doc.add(&Key::DEFAULT, prag, Provenance::by("test")).unwrap();

    let mut inc = Incivility::new();
    inc.push(IncivilityItem {
        target: IncivilityTarget::Sentence(Ix::new(1)),
        scores: IncivilityScores { insult: 0.2, ..Default::default() },
        evidence: &[],
    });
    doc.add(&Key::DEFAULT, inc, Provenance::by("test")).unwrap();

    let mut aspects = Aspects::default();
    aspects.push(Aspect {
        target: AspectTarget::Implicit { sentence: Ix::new(1) },
        category: None,
        polarity: PolarityScores::new(0.6, 0.1, 0.3),
        opinions: &[],
    });
    doc.add(&Key::DEFAULT, aspects, Provenance::by("test")).unwrap();

    // Evidence that does index tokens still needs the Tokens layer.
    let mut with_evidence = Incivility::on(Key::named("ev"), Key::DEFAULT, Key::DEFAULT, Key::DEFAULT);
    with_evidence.push(IncivilityItem {
        target: IncivilityTarget::Sentence(Ix::new(0)),
        scores: IncivilityScores::default(),
        evidence: &[(IncivilityLabel::Insult, IxRange::new(0, 1))],
    });
    let mut p = Provenance::by("test");
    p.reads.push(Key::<Tokens>::named("ev").id());
    let err = doc.add(&Key::named("ev"), with_evidence, p).unwrap_err();
    assert!(matches!(err, Error::Missing(_)), "{err:?}");
}

struct DummyOpinionReader;

impl Annotator for DummyOpinionReader {
    const ID: &'static str = "test:opinion-reader";
    type Reads = (Tokens, Opinions);
    type Writes = (Aspects,);

    fn annotate(&self, (_tokens, _opinions): Refs<'_, Self>) -> Result<(Aspects,), Error> {
        Ok((Aspects::default(),))
    }
}

#[test]
fn misaligned_and_unaligned_anchors_are_refused() {
    let (mut doc, _sentences, tokens) = sample_doc();

    // Pragmatics is aligned to Sentences (len 2): adding 1 item fails with Error::Misaligned.
    let mut short_prag = Pragmatics::new();
    short_prag.push(PragmaticsItem { score: 0.5, evidence: None });
    let err = doc.add(&Key::DEFAULT, short_prag, Provenance::by("test")).unwrap_err();
    assert!(matches!(err, Error::Misaligned { len: 1, anchor_len: 2, .. }), "expected Error::Misaligned, got {err:?}");

    // Opinions built on Tokens@gold cannot be read alongside the default Tokens.
    let gold_tokens_key = Key::<Tokens>::named("gold");
    let (_, gold_tokens) = make_sentences_and_tokens();
    let mut p_tok = Provenance::by("test");
    p_tok.reads.push(Key::<Sentences>::DEFAULT.id());
    doc.add(&gold_tokens_key, gold_tokens, p_tok).unwrap();
    let gold_ops = Opinions::from_items_on(
        gold_tokens_key,
        &tokens,
        [Opinion { tokens: IxRange::new(4, 5), polarity: 0.8, intensity: 1.0, negated: false }],
    )
    .unwrap();
    let mut p = Provenance::by("test");
    p.reads.push(gold_tokens_key.id());
    doc.add(&Key::DEFAULT, gold_ops, p).unwrap();

    let analyzer = Analyzer::builder()
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Opinions>::DEFAULT)
        .step(Bound::new(DummyOpinionReader))
        .build()
        .unwrap();
    let results = analyzer.run(std::slice::from_mut(&mut doc));
    assert!(
        matches!(
            results[0],
            Err(Error::Unaligned {
                layer,
                on,
                read,
            }) if layer == Key::<Opinions>::DEFAULT.id()
                && on == gold_tokens_key.id()
                && read.is_empty()
        ),
        "expected Error::Unaligned, got {:?}",
        results[0]
    );
}
