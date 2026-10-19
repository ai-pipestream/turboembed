use std::collections::BTreeSet;
use std::sync::Arc;

use turbo_analysis::{Analyzer, Document, Ix, Key, Provenance, Span, Spans, ViewId};
use turbo_analysis_layers::TermSlot;
use turbo_unicode::Script;

use super::*;

#[test]
fn english_german_french_spanish_and_nordic_profiles_match_opennlp() {
    let eng = NormalizationProfiles::for_language("eng").unwrap();
    assert_eq!(eng.stemmer_algorithm(), SnowballAlgorithm::English);
    assert_eq!(eng.accent_fold(), Some(AccentFoldKind::Latin));
    assert_eq!(
        eng.matching_analyzer().dimensions(),
        &[Dimension::Nfc, Dimension::CaseFold, Dimension::AccentFold, Dimension::Stem]
    );
    assert_eq!(eng.matching_analyzer().analyze("Cats").unwrap()[0].normalized(), "cat");

    let deu = NormalizationProfiles::for_language("deu").unwrap();
    assert_eq!(deu.stemmer_algorithm(), SnowballAlgorithm::German);
    assert_eq!(deu.accent_fold(), Some(AccentFoldKind::German));
    assert_eq!(
        deu.matching_analyzer().dimensions(),
        &[Dimension::Nfc, Dimension::CaseFold, Dimension::AccentFold, Dimension::Stem]
    );
    let deu_term = &deu.matching_analyzer().analyze("B\u{00FC}cher").unwrap()[0];
    assert_eq!(deu_term.peel(), "buecher");
    assert_eq!(deu_term.normalized(), "buch");

    for code in ["fra", "spa", "por", "ita", "cat"] {
        let p = NormalizationProfiles::for_language(code).unwrap();
        assert_eq!(p.accent_fold(), Some(AccentFoldKind::Latin), "{code}");
    }
    let fra = NormalizationProfiles::for_language("fr").unwrap();
    assert_eq!(fra.language(), "fra");
    assert_eq!(fra.stemmer_algorithm(), SnowballAlgorithm::French);
    let fra_term = &fra.matching_analyzer().analyze("\u{00C9}l\u{00E8}ves").unwrap()[0];
    assert_eq!(fra_term.peel(), "eleves");
    assert_eq!(fra_term.normalized(), "elev");

    let spa = NormalizationProfiles::for_language("es").unwrap();
    assert_eq!(spa.language(), "spa");
    assert_eq!(spa.stemmer_algorithm(), SnowballAlgorithm::Spanish);
    let spa_term = &spa.matching_analyzer().analyze("Canciones").unwrap()[0];
    assert_eq!(spa_term.peel(), "canciones");
    assert_eq!(spa_term.normalized(), "cancion");

    let swe = NormalizationProfiles::for_language("swe").unwrap();
    assert_eq!(swe.accent_fold(), None);
    assert_eq!(swe.matching_analyzer().dimensions(), &[Dimension::Nfc, Dimension::CaseFold, Dimension::Stem]);
}

#[test]
fn two_letter_codes_and_norwegian_standards_resolve_as_on_jvm() {
    assert_eq!(NormalizationProfiles::for_language("de").unwrap().stemmer_algorithm(), SnowballAlgorithm::German);
    assert_eq!(NormalizationProfiles::for_language("  EN ").unwrap().stemmer_algorithm(), SnowballAlgorithm::English);
    for code in ["nb", "nn", "no", "nob", "nno", "nor"] {
        assert_eq!(
            NormalizationProfiles::for_language(code).unwrap().stemmer_algorithm(),
            SnowballAlgorithm::Norwegian,
            "{code}"
        );
    }
    assert!(NormalizationProfiles::for_language("jpn").is_none());
    assert!(NormalizationProfiles::for_language("zzz").is_none());
    assert!(NormalizationProfiles::for_language("qq").is_none());
    assert_eq!(NormalizationProfile::new("   ", SnowballAlgorithm::English, None), Err(TermError::BlankLanguage));
}

#[test]
fn supported_languages_cover_every_snowball_language_except_porter() {
    let covered: BTreeSet<SnowballAlgorithm> = NormalizationProfiles::supported_languages()
        .iter()
        .map(|&code| NormalizationProfiles::for_language(code).unwrap().stemmer_algorithm())
        .collect();
    let expected: BTreeSet<SnowballAlgorithm> =
        SnowballAlgorithm::ALL.iter().copied().filter(|&alg| alg != SnowballAlgorithm::Porter).collect();
    assert_eq!(covered, expected);
    for code in ["nob", "nno", "nor"] {
        assert!(NormalizationProfiles::supported_languages().contains(&code));
    }
}

#[test]
fn matching_analyzer_is_thread_safe_for_stemming() {
    let analyzer = Arc::new(NormalizationProfiles::for_language("eng").unwrap().matching_analyzer());
    let expected = analyzer.analyze("running").unwrap()[0].at(Dimension::Stem).unwrap().to_owned();
    let mut handles = Vec::with_capacity(8);
    for _ in 0..8 {
        let a = Arc::clone(&analyzer);
        let exp = expected.clone();
        handles.push(std::thread::spawn(move || {
            for _ in 0..50 {
                assert_eq!(a.analyze("running").unwrap()[0].at(Dimension::Stem).unwrap(), exp);
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn by_language_annotator_reuses_one_analyzer_per_language() {
    let annotator = TermAnnotator::by_language(Some("deu".to_owned()));
    let mut code = Strs::default();
    code.push("en");
    let english = Language { code, score: vec![1.0] };

    let first = annotator.resolve_analyzer(Some(&english));
    let second = annotator.resolve_analyzer(Some(&english));
    assert!(first.shares_state_with(&second));
    assert_eq!(first.analyze("Cats").unwrap()[0].normalized(), "cat");

    let fallback = annotator.resolve_analyzer(None);
    assert!(fallback.shares_state_with(&annotator.resolve_analyzer(None)));
    assert!(!fallback.shares_state_with(&first));
    assert_eq!(fallback.analyze("B\u{00FC}cher").unwrap()[0].normalized(), "buch");

    let plain = TermAnnotator::default();
    let mut unknown = Strs::default();
    unknown.push("jpn");
    let japanese = Language { code: unknown, score: vec![1.0] };
    let default = plain.resolve_analyzer(Some(&japanese));
    assert!(default.shares_state_with(&plain.resolve_analyzer(None)));
    assert_eq!(default.dimensions(), &[Dimension::Nfc, Dimension::CaseFold]);
}

#[test]
fn term_annotator_reads_language_and_lemmas_and_writes_turbo_terms() {
    let text = "Caf\u{00E9}s Running";
    let mut doc = Document::new(text);
    let mut tok_spans = Spans::new(ViewId::SOURCE);
    tok_spans.push(Span::new(0, 6));
    tok_spans.push(Span::new(7, 14));
    doc.add(&Key::DEFAULT, Tokens { spans: tok_spans, sentence: None }, Provenance::by("test")).unwrap();

    let mut lang_code = Strs::default();
    lang_code.push("eng");
    doc.add(&Key::DEFAULT, Language { code: lang_code, score: vec![1.0] }, Provenance::by("test")).unwrap();

    let mut lemma_strs = Strs::default();
    lemma_strs.push("cafe");
    lemma_strs.push("run");
    doc.add(&Key::DEFAULT, Lemmas { lemma: lemma_strs }, Provenance::by("test")).unwrap();

    let pipeline = Analyzer::builder()
        .given(Key::<View>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Language>::DEFAULT)
        .given(Key::<Lemmas>::DEFAULT)
        .add(TermAnnotator::default())
        .build()
        .unwrap();
    let status = pipeline.run(std::slice::from_mut(&mut doc));
    assert_eq!(status[0], Ok(()));

    let terms = doc.get::<Terms>().unwrap();
    assert_eq!(terms.len(), 2);
    assert_eq!(
        terms.get(Ix::new(0)),
        TermSlot { surface: "Caf\u{00E9}s", folded: "cafes", stem: Some("cafe"), lemma: Some("cafe") }
    );
    assert_eq!(
        terms.get(Ix::new(1)),
        TermSlot { surface: "Running", folded: "running", stem: Some("run"), lemma: Some("run") }
    );
}

#[test]
fn term_annotator_rejects_invalid_token_spans_without_panicking() {
    let doc = Document::new("café");
    let view = doc.get::<View>().unwrap();
    let annotator = TermAnnotator::default();

    // Out-of-bounds span (end > text.len()).
    let mut oob_spans = Spans::new(ViewId::SOURCE);
    oob_spans.push(Span::new(0, 99));
    let oob_tokens = Tokens { spans: oob_spans, sentence: None };
    let err = annotator.annotate((view, &oob_tokens, None, None, None)).err().unwrap();
    assert!(err.to_string().contains("not a valid slice"), "unexpected error: {err}");

    // Mid-codepoint UTF-8 boundary ('é' is bytes 3..5 in "café").
    let mut mid_spans = Spans::new(ViewId::SOURCE);
    mid_spans.push(Span::new(0, 4));
    let mid_tokens = Tokens { spans: mid_spans, sentence: None };
    let err = annotator.annotate((view, &mid_tokens, None, None, None)).err().unwrap();
    assert!(err.to_string().contains("not a valid slice"), "unexpected error: {err}");
}

#[test]
fn term_annotator_with_lemma_dimension_uses_doc_lemmas_without_requiring_pos_or_invoking_lemmatizer() {
    let analyzer = TermAnalyzer::builder()
        .case_fold()
        .unwrap()
        .lemmatize(|_tok: &str, _tag: &str| -> Option<String> {
            panic!("analyzer lemmatizer must not be called when document Lemmas layer is present")
        })
        .build();
    let annotator = TermAnnotator::new(analyzer);

    let doc = Document::new("Was");
    let view = doc.get::<View>().unwrap();
    let mut tok_spans = Spans::new(ViewId::SOURCE);
    tok_spans.push(Span::new(0, 3));
    let tokens = Tokens { spans: tok_spans, sentence: None };

    let mut lemma_strs = Strs::default();
    lemma_strs.push("be");
    let lemmas = Lemmas { lemma: lemma_strs };

    // pos is None, but lemmas is Some(&lemmas): should succeed without MissingPosTag.
    let (terms,) = annotator.annotate((view, &tokens, None, None, Some(&lemmas))).unwrap();
    assert_eq!(terms.get(Ix::new(0)), TermSlot { surface: "Was", folded: "was", stem: None, lemma: Some("be") });
}

#[test]
fn term_annotator_config_hash_is_deterministic_and_distinguishes_custom_normalizers_and_stemmers() {
    let default_hash = TermAnnotator::default().config_hash();
    assert_eq!(default_hash, TermAnnotator::by_language(None).config_hash());
    assert_ne!(default_hash, TermAnnotator::by_language(Some("eng".to_owned())).config_hash());

    // Every distinct Snowball stemmer algorithm across the language profiles produces a distinct hash.
    let mut seen_by_alg = std::collections::BTreeMap::new();
    for &lang in NormalizationProfiles::supported_languages() {
        let profile = NormalizationProfiles::for_language(lang).unwrap();
        let hash = TermAnnotator::for_language(lang).unwrap().config_hash();
        for (&other_alg, &other_hash) in &seen_by_alg {
            if other_alg == profile.stemmer_algorithm() {
                assert_eq!(hash, other_hash, "{lang} shares algorithm {other_alg:?}");
            } else {
                assert_ne!(hash, other_hash, "{lang} ({:?}) collided with {other_alg:?}", profile.stemmer_algorithm());
            }
        }
        seen_by_alg.insert(profile.stemmer_algorithm(), hash);
    }

    let latin_accent = TermAnalyzer::builder().case_fold().unwrap().accent_fold().build();
    let german_umlaut = TermAnalyzer::builder()
        .case_fold()
        .unwrap()
        .transform(Dimension::AccentFold, GermanUmlauts::new())
        .unwrap()
        .build();
    assert_ne!(TermAnnotator::new(latin_accent).config_hash(), TermAnnotator::new(german_umlaut).config_hash());

    let preserve = TermAnalyzer::builder().accent_fold_scripts(&[Script::Latin], false).build();
    let convert = TermAnalyzer::builder().accent_fold_scripts(&[Script::Latin], true).build();
    assert_eq!(preserve.analyze_tokens(&["ø"], &[None]).unwrap()[0].normalized(), "ø");
    assert_eq!(convert.analyze_tokens(&["ø"], &[None]).unwrap()[0].normalized(), "o");
    assert_ne!(TermAnnotator::new(preserve).config_hash(), TermAnnotator::new(convert).config_hash());

    let tr_case = TermAnalyzer::builder().case_fold_locale("tr").unwrap().build();
    let en_case = TermAnalyzer::builder().case_fold_locale("en").unwrap().build();
    let root_case = TermAnalyzer::builder().case_fold().unwrap().build();
    assert_ne!(TermAnnotator::new(tr_case.clone()).config_hash(), TermAnnotator::new(en_case.clone()).config_hash());
    assert_ne!(TermAnnotator::new(tr_case).config_hash(), TermAnnotator::new(root_case).config_hash());
    assert_ne!(
        TermAnnotator::new(en_case).config_hash(),
        TermAnnotator::new(TermAnalyzer::builder().case_fold().unwrap().build()).config_hash()
    );
}
