use turbo_analysis::export::{DocumentGraph, export, import};
use turbo_analysis::{Analyzer, Bound, Document, Ix, Key, Layer, Provenance, Span, Spans, View, ViewBuilder, ViewId};
use turbo_analysis_layers::{Block, QualityFlags as F, Sentences, SpanQuality, Structure, TextQuality, Tokens};
use turbo_normalize_layers::{QualityScratch, TextQualityAnnotator};
use turbo_sentdetect::RuleSentenceDetector;
use turbo_tokenize::SimpleTokenizer;

fn analyze(text: &str) -> Document {
    let analyzer = Analyzer::builder()
        .step(Bound::new(RuleSentenceDetector::new("en")))
        .step(Bound::new(SimpleTokenizer::default()))
        .step(Bound::new(TextQualityAnnotator))
        .build()
        .unwrap();
    let mut docs = vec![Document::new(text)];
    analyzer.run(&mut docs).pop().unwrap().unwrap();
    docs.pop().unwrap()
}

fn close(got: f32, expected: f32) {
    assert!((got - expected).abs() < 1e-5, "{got} != {expected}");
}

#[test]
fn reference_passage_matches_independent_word_syllable_letter_and_sentence_counts() {
    // Twelve monosyllables, 34 letters, two sentences, fourteen input tokens.
    let doc = analyze("The cat sat on the mat. The dog ran in the sun.");
    assert_eq!(doc.get::<Tokens>().unwrap().len(), 14);
    assert_eq!(doc.get::<Sentences>().unwrap().len(), 2);
    let m = doc.get::<TextQuality>().unwrap().metrics;
    close(m.flesch_kincaid_grade, -1.45);
    close(m.automated_readability_index, -5.085);
    close(m.coleman_liau_index, -4.0733333);
    close(m.information_density, 6.0 / 14.0);
    assert_eq!(doc.get::<SpanQuality>().unwrap().metrics[0], m);
    // Seven two-vowel-group words, forty letters, one sentence.
    let doc = analyze("Many happy people enjoy reading simple stories.");
    assert_eq!(doc.get::<Tokens>().unwrap().len(), 8);
    let m = doc.get::<TextQuality>().unwrap().metrics;
    close(m.flesch_kincaid_grade, 10.74);
    close(m.automated_readability_index, 8.984286);
    close(m.coleman_liau_index, 13.571428);
}

#[test]
fn empty_short_unicode_and_numeric_inputs_are_finite_and_partition_characters() {
    for text in ["", " \n\t", "Été e\u{301}té 東京 １２３ 😀\u{a0}\u{fffd}", "1234567890"] {
        let doc = analyze(text);
        let m = doc.get::<TextQuality>().unwrap().metrics;
        m.check().unwrap();
        if !text.is_empty() {
            close(m.alpha_ratio + m.digit_ratio + m.whitespace_ratio + m.symbol_ratio, 1.0);
        }
        assert!(m.flags.contains(F::TOO_SHORT));
    }
    let empty = analyze("");
    let m = empty.get::<TextQuality>().unwrap().metrics;
    assert_eq!(m.composite_score, 0.0);
    assert_eq!(m.mtld_lexical_diversity, 0.0);
    assert_eq!(m.flesch_kincaid_grade, 0.0);
    assert!(empty.get::<SpanQuality>().unwrap().is_empty());
}

#[test]
fn synthetic_corruption_exercises_every_triage_flag() {
    for (text, flag) in [
        ("Some words \u{fffd} lost in transit.", F::MOJIBAKE_SUSPECT),
        ("Some words \u{85} lost in transit.", F::MOJIBAKE_SUSPECT),
        ("inter-\nnational con-\ntent has broken lines.", F::OCR_BROKEN_HYPHENS),
        ("!@#$%^&*[]{}~~", F::HIGH_SYMBOL_NOISE),
        ("1234567890 1234567890 text", F::HIGH_DIGIT_DENSITY),
        ("alpha beta gamma alpha beta gamma alpha beta gamma alpha beta gamma alpha beta gamma", F::REPETITIVE_NGRAM),
        ("Home | About | Contact | Privacy", F::NAV_BOILERPLATE),
    ] {
        let doc = analyze(text);
        let m = doc.get::<TextQuality>().unwrap().metrics;
        assert!(m.flags.contains(flag), "{text:?}: {:?}", m.flags);
    }
    let clean = analyze(
        "A careful reader can use this passage to learn about clear writing. Each sentence presents a useful idea and gives enough context for the reader to understand the entire explanation.",
    );
    let m = clean.get::<TextQuality>().unwrap().metrics;
    assert_eq!(m.flags, F::CLEAN);
    assert!(m.composite_score > 0.7);
}

#[test]
fn paragraph_rows_isolate_boilerplate_from_prose() {
    let doc = analyze("Home | About | Contact\r\n\r\nThe cat sat on the mat. The dog ran in the sun.");
    let rows = doc.get::<SpanQuality>().unwrap();
    assert_eq!(rows.len(), 2);
    assert!(rows.metrics[0].flags.contains(F::NAV_BOILERPLATE));
    assert!(!rows.metrics[1].flags.contains(F::NAV_BOILERPLATE));
    assert!(rows.metrics[1].composite_score > rows.metrics[0].composite_score);
    assert_eq!(
        doc.text(rows.spans.view()).get(rows.spans.get(Ix::new(1)).range()).unwrap(),
        "The cat sat on the mat. The dog ran in the sun."
    );
}

#[test]
fn nested_structure_yields_disjoint_leaf_block_rows_and_retains_uncovered_summary_text() {
    let mut doc = Document::new("heading\nThe cat sat.\nfooter");
    let mut spans = Spans::new(ViewId::SOURCE);
    for (a, b) in [(0, 27), (0, 7), (8, 20), (21, 27)] {
        spans.push(Span::new(a, b));
    }
    doc.add(
        &Key::DEFAULT,
        Structure {
            spans,
            kind: vec![Block::Section, Block::Heading, Block::Paragraph, Block::PageFooter],
            parent: vec![None, Some(Ix::new(0)), Some(Ix::new(0)), Some(Ix::new(0))],
            depth: vec![1, 1, 0, 0],
        },
        Provenance::by("fixture"),
    )
    .unwrap();
    let analyzer = Analyzer::builder()
        .given(Key::<Structure>::DEFAULT)
        .step(Bound::new(RuleSentenceDetector::new("en")))
        .step(Bound::new(SimpleTokenizer::default()))
        .step(Bound::new(TextQualityAnnotator))
        .build()
        .unwrap();
    let mut docs = [doc];
    analyzer.run(&mut docs)[0].as_ref().unwrap();
    let quality = docs[0].get::<SpanQuality>().unwrap();
    assert_eq!(quality.len(), 3);
    quality.check().unwrap();
    assert_eq!(docs[0].get::<TextQuality>().unwrap().spans.get(Ix::new(0)), Span::new(0, 27));
}

#[test]
fn arbitrary_token_order_and_overlaps_do_not_change_text_order_mtld() {
    let doc = analyze("alpha alpha beta gamma delta beta");
    let view = doc.get::<View>().unwrap();
    let tokens = doc.get::<Tokens>().unwrap();
    let sentences = doc.get::<Sentences>().unwrap();
    let mut shuffled = Tokens { spans: Spans::new(view.id()), sentence: None };
    for i in [5, 2, 0, 4, 1, 3] {
        shuffled.spans.push(tokens.spans.get(Ix::new(i)));
    }
    let mut scratch = QualityScratch::default();
    let (m, _) = TextQualityAnnotator.measure(view, &shuffled, sentences, None, &mut scratch).unwrap();
    close(m.mtld_lexical_diversity, 243.0 / 53.0);
    assert_eq!(m, doc.get::<TextQuality>().unwrap().metrics);
    // A legitimate overlapping token belongs to the region containing its
    // start, so it cannot trigger a quadratic expansion across all regions.
    shuffled.spans.push(Span::new(0, view.text().len() as u32));
    TextQualityAnnotator.measure(view, &shuffled, sentences, None, &mut scratch).unwrap().0.check().unwrap();
}

#[test]
fn malformed_and_mismatched_direct_inputs_return_errors() {
    let doc = analyze("été text.");
    let view = doc.get::<View>().unwrap();
    let sentences = doc.get::<Sentences>().unwrap();
    let mut tokens = Tokens { spans: Spans::new(ViewId::SOURCE), sentence: None };
    tokens.spans.push(Span::new(1, 3));
    let mut scratch = QualityScratch::default();
    assert!(TextQualityAnnotator.measure(view, &tokens, sentences, None, &mut scratch).is_err());
    tokens.spans = Spans::new(ViewId(999));
    assert!(TextQualityAnnotator.measure(view, &tokens, sentences, None, &mut scratch).is_err());
}

#[test]
fn named_view_pipeline_roundtrips_quality_metrics_bit_for_bit() {
    let mut doc = Document::new("old");
    let mut builder = ViewBuilder::new("old");
    builder.replace(3, "The cat sat on the mat.");
    let (text, alignment) = builder.finish();
    let view = View::derived(doc.get::<View>().unwrap(), text, alignment);
    doc.add(&Key::named("clean"), view, Provenance::by("normalize")).unwrap();
    let analyzer = Analyzer::builder()
        .given(Key::<View>::named("clean"))
        .step(Bound::new(RuleSentenceDetector::new("en")).read(0, "clean").write(0, "clean"))
        .step(Bound::new(SimpleTokenizer::default()).read(0, "clean").read(1, "clean").write(0, "clean"))
        .step(
            Bound::new(TextQualityAnnotator)
                .read(0, "clean")
                .read(1, "clean")
                .read(2, "clean")
                .write(0, "clean")
                .write(1, "clean"),
        )
        .build()
        .unwrap();
    let mut docs = [doc];
    analyzer.run(&mut docs)[0].as_ref().unwrap();
    let codecs = turbo_analysis_layers::export::codecs();
    let graph = export(&docs[0], &codecs).unwrap();
    let bytes = graph.to_bytes().unwrap();
    let rebuilt = import(&DocumentGraph::from_bytes(&bytes).unwrap(), &codecs).unwrap();
    assert_eq!(export(&rebuilt, &codecs).unwrap().to_bytes().unwrap(), bytes);
    let q = rebuilt.get_key(&Key::<TextQuality>::named("clean")).unwrap();
    assert_eq!(rebuilt.text(q.view()), "The cat sat on the mat.");
    assert_eq!(q.metrics, docs[0].get_key(&Key::<TextQuality>::named("clean")).unwrap().metrics);
}
