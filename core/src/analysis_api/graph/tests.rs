use std::sync::Arc;

use turbo_analysis::{IxRange, Provenance, Tag, TagSet, View, ViewBuilder};
use turbo_analysis_layers::{PiiKind, PiiMention, Strs};

use super::*;
use crate::analysis_api::{
    TURBO_ANALYSIS_LAYER_LEMMAS, TURBO_ANALYSIS_LAYER_POS, TURBO_PII_CONFIDENCE_FORMAT, TURBO_PII_CONFIDENCE_LABELLED,
    TURBO_PII_CONFIDENCE_LABELLED_CHECKSUM, TURBO_SPAN_LABEL, TURBO_SPAN_PROBABILITY, TURBO_SPAN_SOURCE_RANGE,
    TURBO_SPAN_TOKEN_RANGE, TURBO_SPAN_VALUE, TURBO_SPAN_VIEW_RANGE, turbo_span,
};

fn pool(result: &AnalysisResultInner, offset: u32, len: u32) -> &str {
    let start = offset as usize;
    std::str::from_utf8(&result.pool[start..start + len as usize]).unwrap()
}

fn rows<'a>(result: &'a AnalysisResultInner, kind: u32, instance: &str) -> Vec<&'a turbo_span> {
    result
        .spans
        .iter()
        .filter(|row| row.layer_kind == kind && pool(result, row.instance_offset, row.instance_len) == instance)
        .collect()
}

fn spans<L: Layer>(view: ViewId, items: &[(u32, u32)]) -> Spans<L> {
    let mut spans = Spans::new(view);
    for &(start, end) in items {
        spans.push(Span::new(start, end));
    }
    spans
}

fn fixture() -> (Document, ViewId) {
    let source = "İstanbul rocks a@b.co.";
    let mut doc = Document::new(source.to_owned());
    let source_view = doc.get::<View>().unwrap().clone();

    let mut builder = ViewBuilder::new(source);
    builder.replace(2, "i\u{0307}");
    builder.keep(source.len());
    let (lower_text, alignment) = builder.finish();
    assert_eq!(lower_text, "i\u{0307}stanbul rocks a@b.co.");

    let lower_key = Key::<View>::named("lower");
    doc.add(&lower_key, View::derived(&source_view, lower_text, alignment), Provenance::by("test:view")).unwrap();
    let lower_view = doc.get_key(&lower_key).unwrap().id();

    doc.add(
        &Key::<Sentences>::DEFAULT,
        Sentences { spans: spans(ViewId::SOURCE, &[(0, 23)]) },
        Provenance::by("test:sentences"),
    )
    .unwrap();
    doc.add(
        &Key::<Sentences>::named("lower"),
        Sentences { spans: spans(lower_view, &[(0, 24)]) },
        Provenance::by("test:sentences"),
    )
    .unwrap();

    doc.add(
        &Key::<Tokens>::DEFAULT,
        Tokens { spans: spans(ViewId::SOURCE, &[(0, 9), (10, 15), (16, 22), (22, 23)]), sentence: None },
        Provenance::by("test:tokens"),
    )
    .unwrap();
    let lower_tokens_key = Key::<Tokens>::named("lower");
    doc.add(
        &lower_tokens_key,
        Tokens { spans: spans(lower_view, &[(0, 10), (11, 16), (17, 23), (23, 24)]), sentence: None },
        Provenance::by("test:tokens"),
    )
    .unwrap();

    let mut on_lower = Provenance::by("test:aligned");
    on_lower.on.push(lower_tokens_key.id());

    let tagset = TagSet::new("test-pos", &["PROPN", "VERB", "NOUN", "PUNCT"]);
    doc.add(
        &Key::<PosTags>::named("lower"),
        PosTags {
            tagset: Arc::clone(&tagset),
            tags: vec![Tag(0), Tag(1), Tag(2), Tag(3)],
            prob: Some(vec![0.91, 0.82, 0.73, 0.99]),
        },
        on_lower.clone(),
    )
    .unwrap();

    let mut lemmas = Strs::default();
    for lemma in ["istanbul", "rock", "_", "."] {
        lemmas.push(lemma);
    }
    doc.add(&Key::<Lemmas>::named("lower"), Lemmas { lemma: lemmas }, on_lower.clone()).unwrap();

    let mut entity_spans = Spans::new(lower_view);
    entity_spans.push_fragments(&[Span::new(0, 10), Span::new(11, 16)]);
    doc.add(
        &Key::<Entities>::named("lower"),
        Entities {
            spans: entity_spans,
            tokens: vec![IxRange::new(0, 2)],
            kind: vec![Tag(0)],
            kinds: TagSet::new("test-entities", &["place-phrase"]),
            score: vec![0.87],
        },
        on_lower,
    )
    .unwrap();

    let mut pii = Pii::new(lower_view);
    pii.push(PiiMention {
        span: Span::new(17, 23),
        kind: PiiKind::Email,
        confidence: PiiConfidence::Format,
        normalized: "a@b.co",
    });
    doc.add(&Key::<Pii>::named("lower"), pii, Provenance::by("test:pii")).unwrap();

    (doc, lower_view)
}

#[test]
fn normalized_view_rows_keep_view_and_source_coordinates_separate() {
    let (doc, lower_view) = fixture();
    let result = extract_graph(&doc, b"graph".to_vec(), Origin::Analyzed).unwrap();
    assert_eq!(result.graph, b"graph");

    let lower_tokens = rows(&result, TURBO_ANALYSIS_LAYER_TOKENS, "lower");
    assert_eq!(lower_tokens.len(), 4);
    let istanbul = lower_tokens[0];
    assert_eq!(istanbul.view_id, lower_view.0 as u32);
    assert_eq!(pool(&result, istanbul.layer_offset, istanbul.layer_len), "turbo:tokens");
    assert_eq!((istanbul.view_byte_start, istanbul.view_byte_end), (0, 10));
    assert_eq!((istanbul.view_utf16_start, istanbul.view_utf16_end), (0, 9));
    assert_eq!((istanbul.view_codepoint_start, istanbul.view_codepoint_end), (0, 9));
    assert_eq!((istanbul.source_byte_start, istanbul.source_byte_end), (0, 9));
    assert_eq!((istanbul.source_utf16_start, istanbul.source_utf16_end), (0, 8));
    assert_eq!((istanbul.source_codepoint_start, istanbul.source_codepoint_end), (0, 8));
    assert_eq!(istanbul.flags, TURBO_SPAN_VIEW_RANGE | TURBO_SPAN_SOURCE_RANGE | TURBO_SPAN_TOKEN_RANGE);
    assert!(istanbul.prob.is_nan());

    let default_tokens = rows(&result, TURBO_ANALYSIS_LAYER_TOKENS, "");
    assert_eq!(default_tokens.len(), 4);
    assert_eq!(default_tokens[0].view_id, ViewId::SOURCE.0 as u32);
    assert_eq!((default_tokens[0].view_byte_start, default_tokens[0].view_byte_end), (0, 9));
    assert_eq!((default_tokens[0].source_byte_start, default_tokens[0].source_byte_end), (0, 9));

    let lower_sentence = rows(&result, TURBO_ANALYSIS_LAYER_SENTENCES, "lower");
    assert_eq!(lower_sentence.len(), 1);
    assert_eq!((lower_sentence[0].view_byte_start, lower_sentence[0].view_byte_end), (0, 24));
    assert_eq!((lower_sentence[0].source_byte_start, lower_sentence[0].source_byte_end), (0, 23));
}

#[test]
fn deleted_prefix_view_coordinates_project_to_the_source_word() {
    let source = "prefix word";
    let mut doc = Document::new(source.to_owned());
    let source_view = doc.get::<View>().unwrap().clone();

    let mut builder = ViewBuilder::new(source);
    builder.replace(7, "");
    builder.keep(source.len());
    let (trimmed_text, alignment) = builder.finish();
    assert_eq!(trimmed_text, "word");

    let trimmed_key = Key::<View>::named("trimmed");
    doc.add(&trimmed_key, View::derived(&source_view, trimmed_text, alignment), Provenance::by("test:view")).unwrap();
    let trimmed_view = doc.get_key(&trimmed_key).unwrap().id();
    doc.add(
        &Key::<Tokens>::named("trimmed"),
        Tokens { spans: spans(trimmed_view, &[(0, 4)]), sentence: None },
        Provenance::by("test:tokens"),
    )
    .unwrap();

    let result = extract_graph(&doc, Vec::new(), Origin::Analyzed).unwrap();
    let token_rows = rows(&result, TURBO_ANALYSIS_LAYER_TOKENS, "trimmed");
    assert_eq!(token_rows.len(), 1);
    assert_eq!(token_rows[0].view_id, trimmed_view.0 as u32);
    assert_eq!((token_rows[0].view_byte_start, token_rows[0].view_byte_end), (0, 4));
    assert_eq!((token_rows[0].source_byte_start, token_rows[0].source_byte_end), (7, 11));
    assert_eq!((token_rows[0].source_utf16_start, token_rows[0].source_utf16_end), (7, 11));
}

#[test]
fn aligned_layers_use_the_named_token_instance_they_were_made_on() {
    let (doc, lower_view) = fixture();
    let result = extract_graph(&doc, Vec::new(), Origin::Analyzed).unwrap();

    let pos = rows(&result, TURBO_ANALYSIS_LAYER_POS, "lower");
    assert_eq!(pos.len(), 4);
    assert!(pos.iter().all(|row| row.view_id == lower_view.0 as u32));
    assert_eq!((pos[0].view_byte_start, pos[0].view_byte_end), (0, 10));
    assert_eq!((pos[0].source_byte_start, pos[0].source_byte_end), (0, 9));
    assert_eq!(pool(&result, pos[0].label_offset, pos[0].label_len), "PROPN");
    assert_eq!(pos[0].prob, 0.91);
    assert_ne!(pos[0].flags & TURBO_SPAN_LABEL, 0);
    assert_ne!(pos[0].flags & TURBO_SPAN_PROBABILITY, 0);
    assert_eq!((pos[2].token_start, pos[2].token_end), (2, 3));

    let lemmas = rows(&result, TURBO_ANALYSIS_LAYER_LEMMAS, "lower");
    assert_eq!(lemmas.len(), 4);
    assert_eq!(pool(&result, lemmas[1].label_offset, lemmas[1].label_len), "rock");
    assert_eq!(lemmas[2].flags & TURBO_SPAN_LABEL, 0, "the unknown marker stays absent");
    assert_eq!(lemmas[2].label_len, 0);
    assert!(lemmas[2].prob.is_nan());
}

#[test]
fn discontinuous_entities_and_pii_preserve_items_and_normalized_values() {
    let (doc, _) = fixture();
    let result = extract_graph(&doc, Vec::new(), Origin::Analyzed).unwrap();

    let entities = rows(&result, TURBO_ANALYSIS_LAYER_ENTITIES, "lower");
    assert_eq!(entities.len(), 2, "one entity with two fragments");
    assert!(entities.iter().all(|row| row.item_index == 0));
    assert_eq!([entities[0].fragment_index, entities[1].fragment_index], [0, 1]);
    assert_eq!((entities[0].view_byte_start, entities[0].view_byte_end), (0, 10));
    assert_eq!((entities[0].source_byte_start, entities[0].source_byte_end), (0, 9));
    assert_eq!((entities[1].view_byte_start, entities[1].view_byte_end), (11, 16));
    assert_eq!((entities[1].source_byte_start, entities[1].source_byte_end), (10, 15));
    assert!(entities.iter().all(|row| (row.token_start, row.token_end) == (0, 2)));
    assert!(entities.iter().all(|row| row.prob == 0.87));
    assert_eq!(pool(&result, entities[0].label_offset, entities[0].label_len), "place-phrase");

    let pii = rows(&result, TURBO_ANALYSIS_LAYER_PII, "lower");
    assert_eq!(pii.len(), 1);
    assert_eq!((pii[0].view_byte_start, pii[0].view_byte_end), (17, 23));
    assert_eq!((pii[0].source_byte_start, pii[0].source_byte_end), (16, 22));
    assert_eq!(pool(&result, pii[0].label_offset, pii[0].label_len), "email");
    assert_eq!(pool(&result, pii[0].value_offset, pii[0].value_len), "a@b.co");
    assert_ne!(pii[0].flags & TURBO_SPAN_VALUE, 0);
    assert_eq!(pii[0].flags & TURBO_SPAN_PROBABILITY, 0);
    assert_eq!(pii[0].classification, TURBO_PII_CONFIDENCE_FORMAT);
}

/// `doc` as graph bytes, the way a caller would hand them back.
fn wire(doc: &Document) -> Vec<u8> {
    let mut codecs = turbo_analysis::Codecs::new();
    turbo_analysis_layers::export::register_codecs(&mut codecs);
    turbo_analysis::export(doc, &codecs).unwrap().to_bytes().unwrap()
}

/// A document of two tokens whose second part-of-speech probability is
/// `pos` and whose one entity scores `entity`: graph import takes either
/// whatever its value, NaN aside for the score.
fn scored(pos: f64, entity: f64) -> Document {
    let mut doc = Document::new("Ann sings".to_owned());
    doc.add(
        &Key::<Tokens>::DEFAULT,
        Tokens { spans: spans(ViewId::SOURCE, &[(0, 3), (4, 9)]), sentence: None },
        Provenance::by("test:tokens"),
    )
    .unwrap();
    let tagset = TagSet::new("test-pos", &["PROPN", "VERB"]);
    let layer = PosTags { tagset, tags: vec![Tag(0), Tag(1)], prob: Some(vec![0.5, pos]) };
    doc.add(&Key::<PosTags>::DEFAULT, layer, Provenance::by("test:pos")).unwrap();
    let entities = Entities {
        spans: spans(ViewId::SOURCE, &[(0, 3)]),
        tokens: vec![IxRange::new(0, 1)],
        kind: vec![Tag(0)],
        kinds: TagSet::new("test-entities", &["PER"]),
        score: vec![entity],
    };
    doc.add(&Key::<Entities>::DEFAULT, entities, Provenance::by("test:entities")).unwrap();
    doc
}

#[test]
fn a_caller_graph_holding_a_probability_outside_0_to_1_is_an_invalid_argument() {
    use crate::analysis_api::{AnalyzerConfig, AnalyzerEngine};
    use crate::status::{CAPACITY, INVALID_ARGUMENT};

    let engine = AnalyzerEngine::from_desc(AnalyzerConfig::default()).unwrap();
    let ok = engine.inspect_graph(&wire(&scored(0.5, 1.0))).unwrap();
    assert_eq!(ok.spans.iter().filter(|r| r.layer_kind == TURBO_ANALYSIS_LAYER_POS).count(), 2);
    for (pos, entity) in [(1.5, 0.5), (-0.25, 0.5), (f64::INFINITY, 0.5), (f64::NAN, 0.5), (0.5, 2.0), (0.5, -1.0)] {
        let doc = scored(pos, entity);
        let e = engine.inspect_graph(&wire(&doc)).err().unwrap_or_else(|| panic!("{pos} and {entity} decoded"));
        assert_eq!(e.code, INVALID_ARGUMENT, "{pos} and {entity}: {}", e.message);
        assert!(e.message.contains("outside 0..=1"), "{}", e.message);
        // In the analyzer's own document the same row is the analyzer's
        // fault, as the row builder refuses it.
        let own = extract_graph(&doc, Vec::new(), Origin::Analyzed).err().unwrap();
        assert_eq!(own.code, CAPACITY, "{pos} and {entity}: {}", own.message);
    }
}

#[test]
fn extract_pii_maps_labelled_to_4_and_labelled_checksum_to_3() {
    let mut doc = Document::new("SSN: 123-45-6789, SIN: 046 454 286.".to_owned());
    let mut pii = Pii::new(ViewId::SOURCE);
    pii.push(PiiMention {
        span: Span::new(5, 16),
        kind: PiiKind::UsSsn,
        confidence: PiiConfidence::Labelled,
        normalized: "123-45-6789",
    });
    pii.push(PiiMention {
        span: Span::new(23, 34),
        kind: PiiKind::CaSin,
        confidence: PiiConfidence::LabelledChecksum,
        normalized: "046454286",
    });
    doc.add(&Key::<Pii>::DEFAULT, pii, Provenance::by("test:pii")).unwrap();

    let result = extract_graph(&doc, Vec::new(), Origin::Analyzed).unwrap();
    let pii_rows = rows(&result, TURBO_ANALYSIS_LAYER_PII, "");
    assert_eq!(pii_rows.len(), 2);
    assert_eq!(pool(&result, pii_rows[0].label_offset, pii_rows[0].label_len), "us-ssn");
    assert_eq!(pii_rows[0].classification, TURBO_PII_CONFIDENCE_LABELLED);
    assert_eq!(pii_rows[0].classification, 4);
    assert_eq!(pool(&result, pii_rows[1].label_offset, pii_rows[1].label_len), "ca-sin");
    assert_eq!(pii_rows[1].classification, TURBO_PII_CONFIDENCE_LABELLED_CHECKSUM);
    assert_eq!(pii_rows[1].classification, 3);
}
