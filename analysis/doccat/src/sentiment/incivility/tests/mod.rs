use super::*;
use turbo_analysis::{Analyzer, Document, Key, Provenance, Span, Spans, ViewId};
use turbo_analysis_layers::Cut;

fn spans_from_pairs<L: Layer>(pairs: impl IntoIterator<Item = (u32, u32)>) -> Spans<L> {
    let mut spans = Spans::new(ViewId::SOURCE);
    for (s, e) in pairs {
        spans.push(Span::new(s, e));
    }
    spans
}

/// The pre-#507 per-pattern whole-unit scan, frozen as the reference the
/// first-token index in `scan_lexicon` must reproduce exactly. The
/// `negated_mild` rule is duplicated from `scan_lexicon` on purpose: the
/// two copies must change together.
fn reference_scan_lexicon(
    lexicon: &IncivilityLexicon,
    token_lower: &[String],
    tok_start: usize,
    tok_end: usize,
) -> (IncivilityScores, Vec<(IncivilityLabel, IxRange<Tokens>)>) {
    let mut scores = IncivilityScores::default();
    let mut evidence: Vec<(IncivilityLabel, IxRange<Tokens>)> = Vec::new();
    for (pat, label, weight) in &lexicon.entries {
        let pat_len = pat.len();
        if pat_len == 0 || tok_start + pat_len > tok_end {
            continue;
        }
        let mut i = tok_start;
        while i + pat_len <= tok_end {
            let mut matched = true;
            for k in 0..pat_len {
                if token_lower[i + k] != pat[k] {
                    matched = false;
                    break;
                }
            }
            if matched {
                let negated_mild = pat_len == 1
                    && matches!(pat[0].as_str(), "stupid" | "dumb" | "hell" | "damn")
                    && i > tok_start
                    && matches!(token_lower[i - 1].as_str(), "not" | "never" | "no");
                if !negated_mild {
                    let prev = scores.get(*label);
                    let combined =
                        if prev > 0.0 { (1.0 - (1.0 - prev) * (1.0 - *weight)).clamp(0.0, 1.0) } else { *weight };
                    scores.set(*label, combined);
                    evidence.push((*label, IxRange::new(i as u32, (i + pat_len) as u32)));
                }
                i += pat_len;
            } else {
                i += 1;
            }
        }
    }
    (scores, evidence)
}

#[test]
fn scan_lexicon_matches_the_naive_reference_scan() {
    let mut lexicon = IncivilityLexicon::empty();
    lexicon.insert("a a", IncivilityLabel::Insult, 0.5);
    lexicon.insert("a", IncivilityLabel::Profanity, 0.4);
    lexicon.insert("stupid", IncivilityLabel::Insult, 0.7);
    lexicon.insert("not", IncivilityLabel::Threat, 0.3);
    lexicon.insert("kill you", IncivilityLabel::Threat, 0.9);
    let bundled = IncivilityLexicon::bundled();

    let cases: &[&[&str]] = &[
        &["a", "a", "a"],
        &["a", "a", "a", "a"],
        &["not", "stupid"],
        &["stupid", "not", "stupid"],
        &["not"],
        &["x", "y", "z"],
        &["kill", "you", "kill", "you"],
        &["damned", "crap", "crap"],
        &["shitty", "damn", "crap", "hell"],
        &["loser", "idiot", "moron", "clown"],
        &[],
    ];
    for lex in [&lexicon, &bundled] {
        for case in cases {
            let token_lower: Vec<String> = case.iter().map(|s| s.to_string()).collect();
            for (start, end) in [(0, case.len()), (1, case.len()), (0, case.len().saturating_sub(1))] {
                if start > end || end > case.len() {
                    continue;
                }
                let (got_scores, got_evidence) = scan_lexicon(lex, &token_lower, start, end);
                let (want_scores, want_evidence) = reference_scan_lexicon(lex, &token_lower, start, end);
                for label in IncivilityLabel::ALL {
                    assert_eq!(
                        got_scores.get(label).to_bits(),
                        want_scores.get(label).to_bits(),
                        "bitwise score mismatch for {label} on {case:?} over {start}..{end}"
                    );
                }
                assert_eq!(got_evidence, want_evidence, "evidence mismatch on {case:?} over {start}..{end}");
            }
        }
    }

    // Exact IEEE-754 bit parity and threshold classification parity on the
    // 3-hit out-of-entry-order case `["damned", "crap", "crap"]` where
    // `"crap"` precedes `"damned"` in the canonical lexicon order.
    let dcc: Vec<String> = ["damned", "crap", "crap"].into_iter().map(String::from).collect();
    let (dcc_scores, _) = scan_lexicon(&bundled, &dcc, 0, 3);
    let (ref_scores, _) = reference_scan_lexicon(&bundled, &dcc, 0, 3);
    let expected_profanity = 0.9630080000000001_f64;
    assert_eq!(dcc_scores.profanity.to_bits(), expected_profanity.to_bits());
    assert_eq!(dcc_scores.profanity.to_bits(), ref_scores.profanity.to_bits());

    let ann = IncivilityAnnotator::new().with_threshold(expected_profanity);
    let (unit_scores, unit_ev) = ann.score_unit(&dcc, 0, 3, None, None).unwrap();
    assert_eq!(unit_scores.profanity.to_bits(), expected_profanity.to_bits());
    assert!(unit_scores.profanity >= ann.threshold);
    assert_eq!(unit_ev.len(), 3);
}

#[test]
fn scan_lexicon_matches_the_reference_past_the_stack_slots() {
    // A lexicon with more entries than `scan_lexicon` keeps on the stack
    // takes the heap path for its non-overlap slots; it must scan the same.
    // The bundled lexicon itself stays on the stack path.
    assert!(IncivilityLexicon::bundled().len() <= rules::NEXT_FROM_ON_STACK);
    let mut lexicon = IncivilityLexicon::bundled();
    for k in 0..rules::NEXT_FROM_ON_STACK {
        lexicon.insert(&format!("w{k} a"), IncivilityLabel::Insult, 0.2);
    }
    lexicon.insert("a a", IncivilityLabel::Threat, 0.5);
    lexicon.insert("a", IncivilityLabel::Profanity, 0.4);
    assert!(lexicon.len() > rules::NEXT_FROM_ON_STACK);

    let cases: &[&[&str]] = &[
        &["a", "a", "a", "a", "a"],
        &["w0", "a", "a", "w0", "a", "w127", "a"],
        &["damned", "crap", "crap", "w3", "a", "a"],
        &["not", "stupid", "a", "a", "w9", "a"],
    ];
    for case in cases {
        let token_lower: Vec<String> = case.iter().map(|s| s.to_string()).collect();
        for start in 0..=case.len() {
            for end in start..=case.len() {
                let (got_scores, got_evidence) = scan_lexicon(&lexicon, &token_lower, start, end);
                let (want_scores, want_evidence) = reference_scan_lexicon(&lexicon, &token_lower, start, end);
                for label in IncivilityLabel::ALL {
                    assert_eq!(
                        got_scores.get(label).to_bits(),
                        want_scores.get(label).to_bits(),
                        "bitwise score mismatch for {label} on {case:?} over {start}..{end}"
                    );
                }
                assert_eq!(got_evidence, want_evidence, "evidence mismatch on {case:?} over {start}..{end}");
            }
        }
    }
}

fn make_doc_with_sentences(text: &str, sent_ranges: &[(u32, u32)]) -> Document {
    let mut doc = Document::new(text);
    let spans = spans_from_pairs::<Sentences>(sent_ranges.iter().copied());
    let mut t_spans = Spans::new(ViewId::SOURCE);
    let mut t_sent = Vec::new();
    for (s_idx, &(s, e)) in sent_ranges.iter().enumerate() {
        let slice = &text[s as usize..e as usize];
        let mut idx = 0usize;
        while idx < slice.len() {
            let ch = slice[idx..].chars().next().unwrap();
            let ch_len = ch.len_utf8();
            if ch.is_whitespace() {
                idx += ch_len;
                continue;
            }
            if ch.is_ascii_punctuation() && ch != '\'' {
                t_spans.push(Span::new(s + idx as u32, s + (idx + ch_len) as u32));
                t_sent.push(Ix::new(s_idx as u32));
                idx += ch_len;
                continue;
            }
            let tok_start = idx;
            idx += ch_len;
            while idx < slice.len() {
                let c2 = slice[idx..].chars().next().unwrap();
                if c2.is_whitespace() || (c2.is_ascii_punctuation() && c2 != '\'') {
                    break;
                }
                idx += c2.len_utf8();
            }
            t_spans.push(Span::new(s + tok_start as u32, s + idx as u32));
            t_sent.push(Ix::new(s_idx as u32));
        }
    }
    doc.add(&Key::<Sentences>::DEFAULT, Sentences { spans }, Provenance::by("test")).unwrap();
    doc.add(&Key::<Tokens>::DEFAULT, Tokens { spans: t_spans, sentence: Some(t_sent) }, Provenance::by("test"))
        .unwrap();
    doc
}

#[test]
fn lexicon_digest_updates_on_insert_and_is_canonical() {
    let mut lex1 = IncivilityLexicon::bundled();
    let d1 = lex1.sha256().to_owned();
    assert_eq!(d1.len(), 64);

    lex1.insert("  Absolute   Clown! ", IncivilityLabel::Insult, 0.85);
    let d2 = lex1.sha256().to_owned();
    assert_ne!(d1, d2);

    let mut lex2 = IncivilityLexicon::bundled();
    lex2.insert("absolute clown", IncivilityLabel::Insult, 0.85);
    assert_eq!(d2, lex2.sha256());
}

#[test]
fn multi_label_co_occurrence_and_evidence_spans_per_label() {
    // Single sentence triggering Profanity ("fucking"), Insult ("idiot"),
    // Threat ("hunt you down"), and co-occurrence SevereToxicity.
    let text = "You fucking idiot, we will hunt you down!";
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

    let prov = doc.provenance(Key::<Incivility>::DEFAULT.id()).unwrap();
    assert_eq!(prov.annotator, "turbo-doccat:incivility");
    assert_eq!(prov.model.as_ref().unwrap().id, "oasis-incivility:lexicon:no-reply-context");

    let inciv = doc.get::<Incivility>().unwrap();
    assert_eq!(inciv.len(), 1);
    let s0 = Ix::new(0);
    let scores = inciv.scores(s0);
    assert!(scores.profanity >= 0.90, "profanity={}", scores.profanity);
    assert!(scores.insult >= 0.80, "insult={}", scores.insult);
    assert!(scores.threat >= 0.90, "threat={}", scores.threat);
    assert!(scores.severe_toxicity >= 0.85, "severe_toxicity={}", scores.severe_toxicity);

    let ev: Vec<(IncivilityLabel, IxRange<Tokens>)> = inciv.evidence_of(s0).collect();
    // Tokens: 0="You", 1="fucking", 2="idiot", 3=",", 4="we", 5="will", 6="hunt", 7="you", 8="down", 9="!"
    assert!(ev.contains(&(IncivilityLabel::Profanity, IxRange::new(1, 2))));
    assert!(ev.contains(&(IncivilityLabel::Insult, IxRange::new(2, 3))));
    assert!(ev.contains(&(IncivilityLabel::Threat, IxRange::new(6, 9))));
    assert!(ev.iter().any(|&(l, _)| l == IncivilityLabel::SevereToxicity));
}

#[test]
fn vector_scorer_and_chunk_unit_guarantee_evidence_spans() {
    struct FixedScorer;
    impl IncivilityVectorScorer for FixedScorer {
        fn score_incivility(&self, vec: &[f32]) -> Result<IncivilityScores, String> {
            let mut s = IncivilityScores::default();
            if vec.first().copied().unwrap_or(0.0) > 0.5 {
                s.intolerance = 0.88;
            }
            Ok(s)
        }

        fn model(&self) -> Option<ModelRef> {
            Some(ModelRef { id: "fixed-head".into(), revision: "1".into(), sha256: "00".into() })
        }
    }

    // Sentence with no lexicon hits, scored positive on `Intolerance` by the vector head.
    let text = "Exclude every outsider from civic life.";
    let mut doc = make_doc_with_sentences(text, &[(0, text.len() as u32)]);
    let chunks = Chunks {
        spans: spans_from_pairs::<Chunks>([(0, text.len() as u32)]),
        level: vec![0],
        parent: vec![None],
        sentences: vec![IxRange::new(0, 1)],
        cut: vec![Cut::Sentence],
        cost: vec![1000],
        model_tokens: vec![6],
        over_budget: vec![false],
    };
    doc.add(&Key::<Chunks>::DEFAULT, chunks, Provenance::by("test")).unwrap();

    let vecs = Vectors::<Chunks>::new(
        2,
        vec![0.9, 0.1],
        vec![],
        ModelRef { id: "fixed-head".into(), revision: "1".into(), sha256: "00".into() },
    );
    doc.add(&Key::<Vectors<Chunks>>::DEFAULT, vecs, Provenance::by("test")).unwrap();

    let ann = IncivilityAnnotator::new().with_unit(IncivilityUnit::Chunks).with_vector_scorer(Arc::new(FixedScorer));
    let pipe = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Chunks>::DEFAULT)
        .given(Key::<Vectors<Chunks>>::DEFAULT)
        .add(ann)
        .build()
        .unwrap();
    for res in pipe.run(std::slice::from_mut(&mut doc)) {
        res.unwrap();
    }

    let inciv = doc.get::<Incivility>().unwrap();
    assert_eq!(inciv.len(), 1);
    let c0 = Ix::new(0);
    assert_eq!(inciv.target[0], IncivilityTarget::Chunk(Ix::new(0)));
    assert!(inciv.scores(c0).intolerance >= 0.85);
    // Even though no lexicon phrase matched, the positive head label still receives a
    // non-empty salient content token range inside the chunk.
    let ev: Vec<(IncivilityLabel, IxRange<Tokens>)> = inciv.evidence_of(c0).collect();
    assert!(!ev.is_empty());
    assert!(ev.iter().any(|&(l, r)| l == IncivilityLabel::Intolerance && !r.is_empty()));
}

#[test]
fn sentence_vectors_from_another_encoder_are_refused() {
    struct NamedScorer;
    impl IncivilityVectorScorer for NamedScorer {
        fn score_incivility(&self, _vec: &[f32]) -> Result<IncivilityScores, String> {
            Ok(IncivilityScores::default())
        }

        fn model(&self) -> Option<ModelRef> {
            Some(ModelRef { id: "head-encoder".into(), revision: "1".into(), sha256: "00".into() })
        }
    }

    let text = "The patch fixes the parser.";
    let mut doc = make_doc_with_sentences(text, &[(0, text.len() as u32)]);
    let vecs = Vectors::<Sentences>::new(
        2,
        vec![0.5, 0.5],
        vec![],
        ModelRef { id: "other-encoder".into(), revision: "1".into(), sha256: "11".into() },
    );
    doc.add(&Key::<Vectors<Sentences>>::DEFAULT, vecs, Provenance::by("test")).unwrap();
    let pipe = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Vectors<Sentences>>::DEFAULT)
        .add(IncivilityAnnotator::new().with_vector_scorer(Arc::new(NamedScorer)))
        .build()
        .unwrap();
    let err = pipe.run(std::slice::from_mut(&mut doc)).remove(0).unwrap_err();
    assert!(err.to_string().contains("other-encoder"), "{err}");
    assert!(err.to_string().contains("head-encoder"), "{err}");
}

mod bias;
mod context;
mod head;
mod learned;
mod lexicon;
