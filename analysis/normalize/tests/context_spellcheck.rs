use std::sync::Arc;

use turbo_analysis::{
    Analyzer, Bound, Document, DocumentGraph, IxRange, Key, Layer, Provenance, Span, Spans, Tag, TagSet, View, ViewId,
};
use turbo_analysis_layers::{
    AbbreviationForm, Abbreviations, Currency, Date, Entities, Money, MoneyMention, Origin, Pii, PiiConfidence,
    PiiKind, PiiMention, Quantities, QuantityMention, Sentences, TemporalMention, TemporalValue, Temporals, Tokens,
};
use turbo_ngram::NGramLanguageModel;
use turbo_normalize::{
    ConfusableSet, ContextSpellAnnotator, ContextSpellCorrector, SpellCorrect, SymSpell, apply, normalize,
};
use turbo_normalize_layers::export::codecs;
use turbo_normalize_layers::{Corrections, PlaceholderKind, Placeholders, ProtectedSpans};

fn build_fixture_engine_and_lm() -> (Arc<SymSpell>, Arc<NGramLanguageModel>) {
    let mut engine = SymSpell::default();
    for (word, count) in [
        ("their", 5_000),
        ("there", 5_000),
        ("they're", 4_000),
        ("going", 4_000),
        ("to", 10_000),
        ("too", 3_000),
        ("two", 3_000),
        ("the", 15_000),
        ("store", 3_000),
        ("car", 2_500),
        ("is", 8_000),
        ("at", 5_000),
        ("i", 8_000),
        ("want", 4_000),
        ("go", 4_500),
        ("apples", 1_500),
        ("please", 3_000),
        ("fill", 2_500),
        ("out", 4_000),
        ("form", 3_500),
        ("from", 6_000),
        ("he", 5_000),
        ("walked", 2_000),
        ("send", 2_500),
        ("payment", 2_000),
        ("of", 9_000),
        ("for", 7_000),
        ("on", 6_000),
        ("by", 4_000),
        ("contact", 2_000),
        ("and", 10_000),
        ("via", 2_000),
        ("or", 5_000),
        ("café", 1_200),
        ("naïve", 1_000),
    ] {
        engine.add(word, count);
    }

    let mut lm = NGramLanguageModel::new(3).expect("valid trigram order");
    let corpus: &[&[&str]] = &[
        &["<s>", "they're", "going", "to", "the", "store", "</s>"],
        &["<s>", "they're", "going", "home", "</s>"],
        &["<s>", "their", "car", "is", "at", "the", "store", "</s>"],
        &["<s>", "their", "house", "is", "near", "the", "store", "</s>"],
        &["<s>", "i", "want", "to", "go", "to", "the", "store", "</s>"],
        &["<s>", "i", "want", "to", "go", "</s>"],
        &["<s>", "i", "want", "two", "apples", "</s>"],
        &["<s>", "please", "fill", "out", "the", "form", "</s>"],
        &["<s>", "fill", "out", "the", "form", "</s>"],
        &["<s>", "he", "walked", "from", "the", "store", "</s>"],
        &["<s>", "send", "payment", "to", "the", "store", "</s>"],
    ];
    for sent in corpus {
        for _ in 0..20 {
            lm.add(sent).expect("add training sentence");
        }
    }
    (Arc::new(engine), Arc::new(lm))
}

#[test]
fn corrects_real_word_confusables_where_isolated_symspell_makes_no_change() {
    let (engine, lm) = build_fixture_engine_and_lm();
    let isolated = SpellCorrect::new(Arc::clone(&engine));
    let contextual = ContextSpellCorrector::new(Arc::clone(&engine), Arc::clone(&lm));

    // 1. "Their going to the store" -> "They're going to the store"
    assert_eq!(apply(&isolated, "Their going to the store"), "Their going to the store");
    assert_eq!(apply(&contextual, "Their going to the store"), "They're going to the store");

    // 2. "I want two go" -> "I want to go"
    assert_eq!(apply(&isolated, "I want two go"), "I want two go");
    assert_eq!(apply(&contextual, "I want two go"), "I want to go");

    // 3. "Please fill out the from" -> "Please fill out the form"
    assert_eq!(apply(&isolated, "Please fill out the from"), "Please fill out the from");
    assert_eq!(apply(&contextual, "Please fill out the from"), "Please fill out the form");

    // Valid uses of the same words remain unchanged in context.
    assert_eq!(apply(&contextual, "Their car is at the store"), "Their car is at the store");
    assert_eq!(apply(&contextual, "I want two apples"), "I want two apples");
    assert_eq!(apply(&contextual, "He walked from the store"), "He walked from the store");
}

#[test]
fn corrects_mixed_non_word_typos_and_confusables_with_casing() {
    let (engine, lm) = build_fixture_engine_and_lm();
    let contextual = ContextSpellCorrector::new(engine, lm);

    // "Their" (real-word confusable) + "giong" (non-word typo) + "two" (real-word confusable) + "teh" (non-word typo)
    let input = "Their giong two teh store.";
    let (out, alignment) = normalize(&contextual, input);
    assert_eq!(out, "They're going to the store.");
    assert_eq!(alignment.edits().len(), 4);

    // All-uppercase contraction casing is preserved.
    assert_eq!(apply(&contextual, "THEIR going to the store"), "THEY'RE going to the store");
}

#[test]
fn protected_spans_guard_never_alters_pii_money_quantity_temporal_entity_abbreviation_or_custom() {
    let (engine, lm) = build_fixture_engine_and_lm();
    let annotator = ContextSpellAnnotator::new(engine, lm);

    // Every protected span below contains tokens that would otherwise be corrected
    // ("From" -> "Form", "Teh" -> "The", "Two" -> "To", "Giong" -> "Going", "Pleese" -> "Please", "Wnat" -> "Want"),
    // while unprotected tokens in the same sentence ("Their" -> "They're", "two" -> "to") are fixed.
    let text =
        "Their going two send $50.00 and 10Two kg on May Teh to John From at giong@example.com via Dr. Pleese or Wnat.";
    let mut doc = Document::new(text);
    let v0 = ViewId::SOURCE;

    let find_span = |needle: &str| -> Span {
        let start = text.find(needle).expect("substring in text");
        Span::new(start as u32, (start + needle.len()) as u32)
    };

    // 1. Money: "$50.00"
    let mut money = Money::new(v0);
    money.push(MoneyMention {
        span: find_span("$50.00"),
        negative: false,
        coefficient: "5000",
        scale: 2,
        currency: Currency::parse("USD").unwrap(),
    });
    doc.add(&Key::<Money>::DEFAULT, money, Provenance::by("test")).unwrap();

    // 2. Quantities: "10Two kg"
    let mut quantities = Quantities::new(v0);
    quantities.push(QuantityMention {
        span: find_span("10Two kg"),
        negative: false,
        coefficient: "10",
        scale: 0,
        unit: "kg",
    });
    doc.add(&Key::<Quantities>::DEFAULT, quantities, Provenance::by("test")).unwrap();

    // 3. Temporals: "May Teh"
    let mut temporals = Temporals::new(v0);
    temporals.push(TemporalMention {
        span: find_span("May Teh"),
        value: TemporalValue::Day(Date::new(2026, 5, 10).unwrap()),
        origin: Origin::Absolute,
    });
    doc.add(&Key::<Temporals>::DEFAULT, temporals, Provenance::by("test")).unwrap();

    // 4. Entities: "John From" (requires at least 1 token in Tokens for IxRange)
    let mut tokens_layer = Tokens { spans: Spans::new(v0), sentence: None };
    tokens_layer.spans.push(find_span("John From"));
    doc.add(&Key::<Tokens>::DEFAULT, tokens_layer, Provenance::by("test")).unwrap();

    let entity_tags = TagSet::new("ner", &["PER"]);
    let mut entity_spans = Spans::new(v0);
    entity_spans.push(find_span("John From"));
    let entities = Entities {
        spans: entity_spans,
        tokens: vec![IxRange::new(0, 1)],
        kind: vec![Tag(0)],
        kinds: entity_tags,
        score: vec![1.0],
    };
    doc.add(&Key::<Entities>::DEFAULT, entities, Provenance::by("test")).unwrap();

    // 5. Pii: "giong@example.com"
    let mut pii = Pii::new(v0);
    pii.push(PiiMention {
        span: find_span("giong@example.com"),
        kind: PiiKind::Email,
        confidence: PiiConfidence::Format,
        normalized: "giong@example.com",
    });
    doc.add(&Key::<Pii>::DEFAULT, pii, Provenance::by("test")).unwrap();

    // 6. Abbreviations: "Dr. Pleese"
    let mut abbreviations = Abbreviations::new(v0);
    abbreviations.push(find_span("Dr. Pleese"), AbbreviationForm::Short);
    doc.add(&Key::<Abbreviations>::DEFAULT, abbreviations, Provenance::by("test")).unwrap();

    // 7. Caller-supplied ProtectedSpans: "Wnat"
    let protected = ProtectedSpans::from_spans(v0, [find_span("Wnat")]);
    doc.add(&Key::<ProtectedSpans>::DEFAULT, protected, Provenance::by("test")).unwrap();

    // 8. Placeholders
    let placeholders = Placeholders::new(v0);
    doc.add(&Key::<Placeholders>::DEFAULT, placeholders, Provenance::by("test")).unwrap();

    let analyzer = Analyzer::builder()
        .given(Key::<Money>::DEFAULT)
        .given(Key::<Quantities>::DEFAULT)
        .given(Key::<Temporals>::DEFAULT)
        .given(Key::<Entities>::DEFAULT)
        .given(Key::<Pii>::DEFAULT)
        .given(Key::<Abbreviations>::DEFAULT)
        .given(Key::<ProtectedSpans>::DEFAULT)
        .given(Key::<Placeholders>::DEFAULT)
        .step(Bound::new(annotator).write(0, "corrected"))
        .build()
        .expect("valid analyzer");
    let results = analyzer.run(std::slice::from_mut(&mut doc));
    assert!(results[0].is_ok(), "{:?}", results[0]);

    let corrected = doc.get_key(&Key::<View>::named("corrected")).expect("corrected view");
    assert_eq!(
        corrected.text(),
        "They're going to send $50.00 and 10Two kg on May Teh to John From at giong@example.com via Dr. Pleese or Wnat."
    );

    let corrections = doc.get::<Corrections>().expect("corrections layer");
    assert_eq!(corrections.len(), 2);
    assert_eq!(corrections.suggestion.get(0), "They're");
    assert_eq!(corrections.suggestion.get(1), "to");

    // Verify every protected span maps identically between source and corrected view.
    for needle in ["$50.00", "10Two kg", "May Teh", "John From", "giong@example.com", "Dr. Pleese", "Wnat"] {
        let src_span = find_span(needle);
        let child_span = doc.from_source(corrected.id(), src_span);
        assert_eq!(&corrected.text()[child_span.range()], needle, "protected span {needle:?} altered");
        assert_eq!(doc.to_source(corrected.id(), child_span), src_span);
    }
}

#[test]
fn view_alignment_projects_every_downstream_span_to_source_without_splitting_codepoints() {
    let (engine, lm) = build_fixture_engine_and_lm();
    let annotator = ContextSpellAnnotator::new(engine, lm);

    // Include multi-byte UTF-8 characters ("café", "naïve", emoji) alongside length-changing
    // corrections ("Their" [5] -> "They're" [7], "two" [3] -> "to" [2], "from" [4] -> "form" [4]).
    let source_text = "Their going two the café store -- please fill out the naïve from 🌟!";
    let mut doc = Document::new(source_text);
    let analyzer =
        Analyzer::builder().step(Bound::new(annotator).write(0, "corrected")).build().expect("valid analyzer");
    let results = analyzer.run(std::slice::from_mut(&mut doc));
    assert!(results[0].is_ok(), "{:?}", results[0]);

    let corrected_view = doc.get_key(&Key::<View>::named("corrected")).expect("corrected view");
    let corrections = doc.get::<Corrections>().expect("corrections layer");

    assert_eq!(corrected_view.text(), "They're going to the café store -- please fill out the naïve form 🌟!");
    assert_eq!(corrections.len(), 3);

    let corrected_text = corrected_view.text();
    let char_boundaries: Vec<usize> =
        corrected_text.char_indices().map(|(i, _)| i).chain(std::iter::once(corrected_text.len())).collect();

    // Every single valid UTF-8 sub-span in the corrected view must project back to
    // ViewId::SOURCE on exact UTF-8 codepoint boundaries without panicking or splitting a codepoint.
    for (i, &start) in char_boundaries.iter().enumerate() {
        for &end in &char_boundaries[i..] {
            let child_span = Span::new(start as u32, end as u32);
            let source_span = doc.to_source(corrected_view.id(), child_span);
            let (s_start, s_end) = (source_span.start as usize, source_span.end as usize);
            assert!(s_start <= s_end, "inverted source span {source_span:?} for child {child_span:?}");
            assert!(s_end <= source_text.len(), "oob source span {source_span:?}");
            assert!(
                source_text.is_char_boundary(s_start),
                "source start {s_start} splits a codepoint for child span {child_span:?}"
            );
            assert!(
                source_text.is_char_boundary(s_end),
                "source end {s_end} splits a codepoint for child span {child_span:?}"
            );
        }
    }

    // Also verify whitespace-delimited downstream tokens project back to the exact original tokens.
    let expected_src_tokens =
        ["Their", "going", "two", "the", "café", "store", "--", "please", "fill", "out", "the", "naïve", "from", "🌟!"];
    let mut cursor = 0usize;
    for (tok, expected_src_word) in corrected_text.split_whitespace().zip(expected_src_tokens) {
        let rel = corrected_text[cursor..].find(tok).unwrap();
        let c_start = cursor + rel;
        let c_end = c_start + tok.len();
        cursor = c_end;
        let c_span = Span::new(c_start as u32, c_end as u32);
        let s_span = doc.to_source(corrected_view.id(), c_span);
        assert_eq!(&source_text[s_span.range()], expected_src_word);
    }
}

#[test]
fn custom_confusable_set_and_protected_spans_codec_roundtrip() {
    let mut custom = ConfusableSet::empty();
    custom.add_group([" bare ", "bear"]);
    assert_eq!(custom.alternatives("bare"), &["bear".to_string()]);
    assert_eq!(custom.alternatives("bear"), &["bare".to_string()]);

    let mut doc = Document::new("keep alpha and beta safe");
    let protected = ProtectedSpans::from_spans(
        ViewId::SOURCE,
        [Span::new(15, 19), Span::new(5, 10), Span::new(7, 10), Span::new(0, 0)],
    );
    assert_eq!(protected.len(), 2);
    assert!(protected.check().is_ok());
    doc.add(&Key::<ProtectedSpans>::DEFAULT, protected, Provenance::by("test")).unwrap();

    // Placeholder protection also works when passed via Placeholders layer.
    let mut placeholders = Placeholders::new(ViewId::SOURCE);
    placeholders.spans.push(Span::new(5, 10));
    placeholders.kind.push(PlaceholderKind::Url);
    doc.add(&Key::<Placeholders>::DEFAULT, placeholders, Provenance::by("test")).unwrap();

    // Sentence boundaries layer is respected when present.
    let mut sents = Sentences { spans: Spans::new(ViewId::SOURCE) };
    sents.spans.push(Span::new(0, 24));
    doc.add(&Key::<Sentences>::DEFAULT, sents, Provenance::by("test")).unwrap();

    let mut registry = codecs();
    turbo_analysis_layers::export::register_codecs(&mut registry);
    let bytes = turbo_analysis::export(&doc, &registry).expect("export").to_bytes().expect("to_bytes");
    let graph = DocumentGraph::from_bytes(&bytes).expect("from_bytes");
    let roundtripped = turbo_analysis::import(&graph, &registry).expect("import");

    let rt_protected = roundtripped.get::<ProtectedSpans>().expect("roundtripped ProtectedSpans");
    assert_eq!(rt_protected.len(), 2);
    let spans: Vec<Span> = rt_protected.spans.iter().map(|(_, s)| s).collect();
    assert_eq!(spans, vec![Span::new(5, 10), Span::new(15, 19)]);
}

#[test]
fn config_hash_changes_with_lm_counts_and_confusables_and_max_candidates_bounds_total_pool() {
    use turbo_analysis::Annotator;

    let mut engine = SymSpell::default();
    engine.add("form", 10);
    engine.add("from", 10);
    let engine = Arc::new(engine);

    let mut lm_a = NGramLanguageModel::new(1).unwrap();
    for _ in 0..100 {
        lm_a.add(&["form"]).unwrap();
    }
    lm_a.add(&["from"]).unwrap();

    let mut lm_b = NGramLanguageModel::new(1).unwrap();
    lm_b.add(&["form"]).unwrap();
    for _ in 0..100 {
        lm_b.add(&["from"]).unwrap();
    }

    let ann_a = ContextSpellAnnotator::new(Arc::clone(&engine), Arc::new(lm_a));
    let ann_b = ContextSpellAnnotator::new(Arc::clone(&engine), Arc::new(lm_b.clone()));
    assert_ne!(
        ann_a.config_hash(),
        ann_b.config_hash(),
        "models with identical (order, size) but swapped counts must produce distinct config_hash"
    );

    let mut conf_x = ConfusableSet::empty();
    conf_x.add_group(["alpha", "beta"]);
    let mut conf_y = ConfusableSet::empty();
    conf_y.add_group(["gamma", "delta"]);
    let ann_x = ContextSpellCorrector::new(Arc::clone(&engine), Arc::new(lm_b.clone())).with_confusables(conf_x);
    let ann_y = ContextSpellCorrector::new(Arc::clone(&engine), Arc::new(lm_b.clone())).with_confusables(conf_y);
    assert_ne!(ann_x.config_hash(), ann_y.config_hash());

    // With max_candidates(1), a known token ("form") retains only the original candidate.
    let single_cand = ContextSpellCorrector::new(engine, Arc::new(lm_b)).max_candidates(1);
    assert_eq!(apply(&single_cand, "form"), "form");
}

#[test]
fn discontinuous_entity_protects_only_fragments_and_higher_order_lm_uses_full_history() {
    let mut engine = SymSpell::default();
    for w in ["aa", "bb", "cc", "dd", "xx", "zz", "form", "from"] {
        engine.add(w, 10);
    }
    let engine = Arc::new(engine);

    let mut lm1 = NGramLanguageModel::new(1).unwrap();
    lm1.add(&["form"]).unwrap();
    for _ in 0..100 {
        lm1.add(&["from"]).unwrap();
    }
    let annotator = ContextSpellAnnotator::new(Arc::clone(&engine), Arc::new(lm1));

    // Discontinuous entity covering fragments 0..2 ("aa") and 8..10 ("zz") leaves "form" (3..7) unprotected.
    let mut doc = Document::new("aa form zz");
    let v0 = ViewId::SOURCE;
    let mut tok_spans = Spans::new(v0);
    tok_spans.push(Span::new(0, 2));
    tok_spans.push(Span::new(3, 7));
    tok_spans.push(Span::new(8, 10));
    doc.add(&Key::<Tokens>::DEFAULT, Tokens { spans: tok_spans, sentence: None }, Provenance::by("test")).unwrap();

    let mut ent_spans = Spans::new(v0);
    ent_spans.push_fragments(&[Span::new(0, 2), Span::new(8, 10)]);
    let entities = Entities {
        spans: ent_spans,
        tokens: vec![IxRange::new(0, 3)],
        kind: vec![Tag(0)],
        kinds: TagSet::new("ner", &["MISC"]),
        score: vec![1.0],
    };
    doc.add(&Key::<Entities>::DEFAULT, entities, Provenance::by("test")).unwrap();

    let analyzer = Analyzer::builder()
        .given(Key::<Entities>::DEFAULT)
        .step(Bound::new(annotator).write(0, "corrected"))
        .build()
        .unwrap();
    let res = analyzer.run(std::slice::from_mut(&mut doc));
    assert!(res[0].is_ok(), "{:?}", res[0]);
    let corrected = doc.get_key(&Key::<View>::named("corrected")).unwrap();
    assert_eq!(corrected.text(), "aa from zz");

    // 5-gram language model uses all 4 history tokens (order - 1 = 4).
    let mut lm5 = NGramLanguageModel::new(5).unwrap();
    for _ in 0..100 {
        lm5.add(&["aa", "bb", "cc", "dd", "form"]).unwrap();
    }
    lm5.add(&["xx", "bb", "cc", "dd", "from"]).unwrap();
    let corr5 = ContextSpellCorrector::new(engine, Arc::new(lm5)).edit_weight(0.0);
    assert_eq!(apply(&corr5, "xx bb cc dd form"), "xx bb cc dd from");
}

#[test]
fn viterbi_beam_pruning_scales_linearly_with_lm_order_and_candidate_width() {
    use std::time::Instant;
    use turbo_analysis::Annotator;

    let mut engine = SymSpell::default();
    let mut group = Vec::with_capacity(64);
    for c in 0..64 {
        let w = format!("w{c:02}");
        engine.add(&w, 100);
        group.push(w);
    }
    let engine = Arc::new(engine);
    let mut conf = ConfusableSet::empty();
    conf.add_group(group.iter().map(String::as_str));

    let sentence = ["w00"; 12].join(" ");
    let mut timings = Vec::new();
    for order in [3usize, 5, 6] {
        let mut lm = NGramLanguageModel::new(order).unwrap();
        lm.add(&vec!["w01"; order]).unwrap();
        let corrector = ContextSpellCorrector::new(Arc::clone(&engine), Arc::new(lm))
            .with_confusables(conf.clone())
            .max_candidates(64)
            .max_beam_width(64);
        let t0 = Instant::now();
        let out = apply(&corrector, &sentence);
        let elapsed = t0.elapsed();
        assert!(!out.is_empty());
        assert!(
            elapsed.as_millis() < 200,
            "order {order} with 64 candidates took {elapsed:?}, expected bounded beam decode < 200ms"
        );
        timings.push(elapsed);
    }

    // Also verify order-independent lm_fingerprint and distinct max_beam_width config_hash
    let mut lm_fwd = NGramLanguageModel::new(2).unwrap();
    let mut lm_rev = NGramLanguageModel::new(2).unwrap();
    for i in 0..32 {
        lm_fwd.add(&[&format!("w{i:02}"), "w00"]).unwrap();
    }
    for i in (0..32).rev() {
        lm_rev.add(&[&format!("w{i:02}"), "w00"]).unwrap();
    }
    let c_fwd = ContextSpellAnnotator::new(Arc::clone(&engine), Arc::new(lm_fwd));
    let c_rev = ContextSpellAnnotator::new(Arc::clone(&engine), Arc::new(lm_rev));
    assert_eq!(c_fwd.config_hash(), c_rev.config_hash());
    assert_ne!(c_fwd.config_hash(), c_fwd.clone().max_beam_width(16).config_hash());
}
