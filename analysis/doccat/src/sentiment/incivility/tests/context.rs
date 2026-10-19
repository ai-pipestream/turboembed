use super::*;

#[test]
fn reply_context_scores_conflict_against_quote_block_when_enabled() {
    let quote = "The new transit schedule adds two evening buses.";
    let reply = "You have no clue what you are talking about.";
    let text = format!("{quote} {reply}");
    let q_end = quote.len() as u32;
    let r_start = q_end + 1;
    let r_end = text.len() as u32;

    // 1. Default (`use_reply_context = false`): provenance records `:no-reply-context`,
    // and the reply is not scored against the quote.
    let mut doc_no_ctx = make_doc_with_sentences(&text, &[(0, q_end), (r_start, r_end)]);
    let structure1 = Structure {
        spans: spans_from_pairs::<Structure>([(0, q_end), (r_start, r_end)]),
        kind: vec![Block::Quote, Block::Paragraph],
        parent: vec![None, None],
        depth: vec![1, 0],
    };
    doc_no_ctx.add(&Key::<Structure>::DEFAULT, structure1, Provenance::by("test")).unwrap();

    let pipe_no_ctx = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Structure>::DEFAULT)
        .add(IncivilityAnnotator::new())
        .build()
        .unwrap();
    for res in pipe_no_ctx.run(std::slice::from_mut(&mut doc_no_ctx)) {
        res.unwrap();
    }
    let prov_no = doc_no_ctx.provenance(Key::<Incivility>::DEFAULT.id()).unwrap();
    assert_eq!(prov_no.model.as_ref().unwrap().id, "oasis-incivility:lexicon:no-reply-context");
    let inciv_no = doc_no_ctx.get::<Incivility>().unwrap();
    assert!(inciv_no.scores(Ix::new(1)).insult < 0.5);

    // 2. With `with_reply_context(true)`: provenance records `:reply-context`,
    // and the reply sentence triggers `Insult` on `"You have no clue"`.
    let mut doc_ctx = make_doc_with_sentences(&text, &[(0, q_end), (r_start, r_end)]);
    let structure2 = Structure {
        spans: spans_from_pairs::<Structure>([(0, q_end), (r_start, r_end)]),
        kind: vec![Block::Quote, Block::Paragraph],
        parent: vec![None, None],
        depth: vec![1, 0],
    };
    doc_ctx.add(&Key::<Structure>::DEFAULT, structure2, Provenance::by("test")).unwrap();

    let pipe_ctx = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Structure>::DEFAULT)
        .add(IncivilityAnnotator::new().with_reply_context(true))
        .build()
        .unwrap();
    for res in pipe_ctx.run(std::slice::from_mut(&mut doc_ctx)) {
        res.unwrap();
    }
    let prov_ctx = doc_ctx.provenance(Key::<Incivility>::DEFAULT.id()).unwrap();
    assert_eq!(prov_ctx.model.as_ref().unwrap().id, "oasis-incivility:lexicon:reply-context");
    let inciv_ctx = doc_ctx.get::<Incivility>().unwrap();
    assert!(inciv_ctx.scores(Ix::new(0)).insult < 0.5);
    assert!(inciv_ctx.scores(Ix::new(1)).insult >= 0.75);
    let ev: Vec<(IncivilityLabel, IxRange<Tokens>)> = inciv_ctx.evidence_of(Ix::new(1)).collect();
    assert!(ev.iter().any(|&(l, r)| l == IncivilityLabel::Insult && !r.is_empty()));
}

#[test]
fn regression_fixture_per_label_precision_recall_and_evidence_invariants() {
    let rows: &[(&str, &[IncivilityLabel])] = &[
        ("You are an incompetent moron and a clown.", &[IncivilityLabel::Insult]),
        ("This whole deployment is absolute bullshit.", &[IncivilityLabel::Profanity]),
        ("Watch your back, we will hunt you down.", &[IncivilityLabel::Threat]),
        ("Those people are parasites and subhuman vermin.", &[IncivilityLabel::IdentityAttack]),
        ("Ban their kind, there is no place for them here.", &[IncivilityLabel::Intolerance]),
        (
            "You fucking scumbag, we will kill you!",
            &[
                IncivilityLabel::Insult,
                IncivilityLabel::Profanity,
                IncivilityLabel::Threat,
                IncivilityLabel::SevereToxicity,
            ],
        ),
        ("You deserve to die, rot in hell.", &[IncivilityLabel::Profanity, IncivilityLabel::SevereToxicity]),
        ("The patch fixes the buffer alignment issue in the parser.", &[]),
        ("That problem is not stupid at all, thank you for checking.", &[]),
        ("We respectfully disagree with the proposed schedule change.", &[]),
    ];

    let mut full_text = String::new();
    let mut sent_spans = Vec::new();
    let mut gold = Vec::new();
    for &(s, labels) in rows {
        if !full_text.is_empty() {
            full_text.push(' ');
        }
        let start = full_text.len() as u32;
        full_text.push_str(s);
        let end = full_text.len() as u32;
        sent_spans.push((start, end));
        gold.push(GoldIncivilityUnit { positive_labels: labels.to_vec() });
    }

    let mut doc = make_doc_with_sentences(&full_text, &sent_spans);
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
    let metrics = evaluate_incivility_corpus(&gold, inciv, DEFAULT_INCIVILITY_THRESHOLD);
    assert_eq!(metrics.positive_with_evidence, metrics.total_positive_predictions);
    assert!(metrics.total_positive_predictions >= 10);
    for (lbl, m) in &metrics.per_label {
        assert_eq!(m.fp, 0, "{lbl}: unexpected false positive");
        assert_eq!(m.fn_count, 0, "{lbl}: unexpected false negative");
        assert!((m.f1 - 1.0).abs() < 1e-12, "{lbl}: expected F1 1.0, got {}", m.f1);
    }
}

#[test]
fn foreign_view_layers_are_rejected() {
    let doc = Document::new("Hello world.");
    let view = doc.get::<View>().unwrap();
    let (clean_text, align) = turbo_analysis::ViewBuilder::new(view.text()).finish();
    let child_view = View::derived(view, clean_text, align);
    let foreign = child_view.id();

    let ok_tokens = Tokens { spans: spans_from_pairs([(0, 5), (6, 11), (11, 12)]), sentence: None };
    let bad_tokens = Tokens {
        spans: {
            let mut s = Spans::new(foreign);
            s.push(Span::new(0, 5));
            s
        },
        sentence: None,
    };
    let bad_sentences = Sentences {
        spans: {
            let mut s = Spans::new(foreign);
            s.push(Span::new(0, 12));
            s
        },
    };
    let bad_chunks = Chunks {
        spans: {
            let mut s = Spans::new(foreign);
            s.push(Span::new(0, 12));
            s
        },
        level: vec![0],
        parent: vec![None],
        sentences: vec![IxRange::new(0, 1)],
        cut: vec![Cut::DocumentStart],
        cost: vec![0],
        model_tokens: vec![3],
        over_budget: vec![false],
    };
    let bad_structure = Structure {
        spans: {
            let mut s = Spans::new(foreign);
            s.push(Span::new(0, 12));
            s
        },
        kind: vec![Block::Paragraph],
        parent: vec![None],
        depth: vec![0],
    };

    let annotator = IncivilityAnnotator::new();
    let err = annotator.annotate((view, &bad_tokens, None, None, None, None, None)).unwrap_err();
    assert!(err.to_string().contains("tokens are in another view"), "{err}");

    let err = annotator.annotate((view, &ok_tokens, Some(&bad_sentences), None, None, None, None)).unwrap_err();
    assert!(err.to_string().contains("sentences are in another view"), "{err}");

    let err = annotator.annotate((view, &ok_tokens, None, Some(&bad_chunks), None, None, None)).unwrap_err();
    assert!(err.to_string().contains("chunks are in another view"), "{err}");

    let err = annotator.annotate((view, &ok_tokens, None, None, Some(&bad_structure), None, None)).unwrap_err();
    assert!(err.to_string().contains("structure is in another view"), "{err}");
}
