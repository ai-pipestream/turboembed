use super::*;

/// The lexicon signal matches whole tokens case-insensitively and records
/// the exact matched token range as evidence.
#[test]
fn lexicon_matching_is_case_insensitive_with_exact_spans() {
    let text = "This is FUCKING bullshit.";
    let mut doc = make_doc_with_sentences(text, &[(0, text.len() as u32)]);
    let pipe = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .add(IncivilityAnnotator::new())
        .build()
        .unwrap();
    for res in pipe.run(std::slice::from_mut(&mut doc)) {
        res.unwrap();
    }

    let inciv = doc.get::<Incivility>().unwrap();
    assert_eq!(inciv.len(), 1);
    let scores = inciv.scores(Ix::new(0));
    // Tokens: 0="This", 1="is", 2="FUCKING", 3="bullshit", 4="."
    assert!(scores.profanity >= 0.95, "profanity={}", scores.profanity);
    assert_eq!(scores.insult, 0.0);
    assert_eq!(scores.threat, 0.0);
    assert_eq!(scores.identity_attack, 0.0);
    assert_eq!(scores.intolerance, 0.0);
    assert_eq!(scores.severe_toxicity, 0.0);

    let ev: Vec<(IncivilityLabel, IxRange<Tokens>)> = inciv.evidence_of(Ix::new(0)).collect();
    assert!(ev.contains(&(IncivilityLabel::Profanity, IxRange::new(2, 3))));
    assert!(ev.contains(&(IncivilityLabel::Profanity, IxRange::new(3, 4))));
    assert_eq!(ev.len(), 2);
}

/// Matching is on token boundaries only: a profane substring inside an
/// innocent token must not fire.
#[test]
fn profane_substrings_inside_innocent_tokens_do_not_match() {
    // "shitake" carries "shit" and "bullshitty" carries "bullshit" as
    // substrings; neither token may match as itself.
    let text = "I bought shitake mushrooms, not that bullshitty tea.";
    let mut doc = make_doc_with_sentences(text, &[(0, text.len() as u32)]);
    let pipe = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .add(IncivilityAnnotator::new())
        .build()
        .unwrap();
    for res in pipe.run(std::slice::from_mut(&mut doc)) {
        res.unwrap();
    }

    let inciv = doc.get::<Incivility>().unwrap();
    assert_eq!(inciv.len(), 1);
    let scores = inciv.scores(Ix::new(0));
    assert_eq!(scores.profanity, 0.0, "profanity={}", scores.profanity);
    assert_eq!(scores.insult, 0.0);
    assert!(inciv.evidence_of(Ix::new(0)).next().is_none());
}

/// An empty document and a whitespace-only sentence score nothing and do
/// not error; labels without a signal stay at the "not scored" 0.0.
#[test]
fn empty_input_scores_nothing() {
    // 1. Empty text: no sentences, no tokens, empty output layer.
    let mut empty = make_doc_with_sentences("", &[]);
    let pipe = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .add(IncivilityAnnotator::new())
        .build()
        .unwrap();
    for res in pipe.run(std::slice::from_mut(&mut empty)) {
        res.unwrap();
    }
    assert_eq!(empty.get::<Incivility>().unwrap().len(), 0);

    // 2. Whitespace-only sentence: one item, every label unscored at 0.0,
    // no evidence spans.
    let mut blank = make_doc_with_sentences("   ", &[(0, 3)]);
    for res in pipe.run(std::slice::from_mut(&mut blank)) {
        res.unwrap();
    }
    let inciv = blank.get::<Incivility>().unwrap();
    assert_eq!(inciv.len(), 1);
    let scores = inciv.scores(Ix::new(0));
    for (label, value) in scores.iter() {
        assert_eq!(value, 0.0, "{label} must stay unscored");
    }
    assert!(inciv.evidence_of(Ix::new(0)).next().is_none());
}

/// Tokens is a required read: a document without it is refused with
/// `Error::Missing`. Sentences is an optional read the annotator itself
/// requires for its default unit, refused with `Error::Annotator`.
#[test]
fn missing_required_layers_are_reported() {
    // Sentences but no Tokens.
    let mut no_tokens = Document::new("Hello world.");
    let spans = spans_from_pairs::<Sentences>([(0, 11)]);
    no_tokens.add(&Key::<Sentences>::DEFAULT, Sentences { spans }, Provenance::by("test")).unwrap();
    let pipe = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .add(IncivilityAnnotator::new())
        .build()
        .unwrap();
    let err = pipe.run(std::slice::from_mut(&mut no_tokens)).remove(0).unwrap_err();
    assert_eq!(err, Error::Missing(Key::<Tokens>::DEFAULT.id()));

    // Tokens but no Sentences (the default unit).
    let mut no_sents = Document::new("Hello world.");
    let mut t_spans = Spans::new(ViewId::SOURCE);
    t_spans.push(Span::new(0, 5));
    t_spans.push(Span::new(6, 11));
    no_sents.add(&Key::<Tokens>::DEFAULT, Tokens { spans: t_spans, sentence: None }, Provenance::by("test")).unwrap();
    let err = pipe.run(std::slice::from_mut(&mut no_sents)).remove(0).unwrap_err();
    let Error::Annotator { annotator, detail } = &err else { panic!("unexpected error: {err}") };
    assert_eq!(*annotator, "turbo-doccat:incivility");
    assert!(detail.contains("requires a Sentences layer"), "{detail}");
}

/// A benign document carries all-zero scores and empty evidence, and the
/// provenance records that scoring was lexicon-only: no model head ran and
/// no reply context was used.
#[test]
fn unscored_document_and_lexicon_only_provenance() {
    let text = "The committee approved the budget.";
    let mut doc = make_doc_with_sentences(text, &[(0, text.len() as u32)]);
    let pipe = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .add(IncivilityAnnotator::new())
        .build()
        .unwrap();
    for res in pipe.run(std::slice::from_mut(&mut doc)) {
        res.unwrap();
    }

    let inciv = doc.get::<Incivility>().unwrap();
    assert_eq!(inciv.len(), 1);
    for (label, value) in inciv.scores(Ix::new(0)).iter() {
        assert_eq!(value, 0.0, "{label} must stay unscored");
    }
    assert!(inciv.evidence_of(Ix::new(0)).next().is_none());

    let prov = doc.provenance(Key::<Incivility>::DEFAULT.id()).unwrap();
    assert_eq!(prov.annotator, "turbo-doccat:incivility");
    let model = prov.model.as_ref().expect("lexicon-only provenance records a model");
    assert_eq!(model.id, "oasis-incivility:lexicon:no-reply-context");
    assert_eq!(model.revision, "1");
    assert_eq!(model.sha256, IncivilityLexicon::bundled().sha256());
}
