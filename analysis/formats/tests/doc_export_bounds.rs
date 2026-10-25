use std::sync::Arc;

use turbo_analysis::{Document, DocumentGraph, Key, Provenance, Span, Spans, ViewId, export, import};
use turbo_analysis_layers::{PosTags, Sentences, TagSet, Tokens};
use turbo_formats::doc_export::{
    ConlluFormatter, CsvFormatter, DocumentFormatter, ExportOffsetMode, FormatExportError, JsonLdFormatter,
    TsvFormatter, WarcFormatter, parse_warc_records,
};

fn build_doc_with_default_tokens(text: &str, spans: &[(u32, u32)]) -> Document {
    let mut doc = Document::new(text);
    let mut s_spans = Spans::<Sentences>::new(ViewId::SOURCE);
    s_spans.push(Span::new(0, text.len() as u32));
    doc.add(&Key::DEFAULT, Sentences { spans: s_spans }, Provenance::by("test")).unwrap();

    let mut t_spans = Spans::<Tokens>::new(ViewId::SOURCE);
    for &(s, e) in spans {
        t_spans.push(Span::new(s, e));
    }
    doc.add(&Key::DEFAULT, Tokens { spans: t_spans, sentence: None }, Provenance::by("test")).unwrap();
    doc
}

#[test]
fn formatters_refuse_pos_tags_bound_to_named_gold_tokens() {
    // Default tokens: 4 tokens over "é 😀 go!" (bytes 0..2, 3..7, 8..10, 10..11).
    let text = "é 😀 go!";
    let default_spans = [(0, 2), (3, 7), (8, 10), (10, 11)];

    // Control 1: named `gold` Tokens with 1 token spanning the whole text, and
    // default PosTags anchored to `gold` via `provenance.on = [gold.id()]`.
    let mut doc_short = build_doc_with_default_tokens(text, &default_spans);
    let gold_key = Key::<Tokens>::named("gold");
    let mut gold_spans_1 = Spans::<Tokens>::new(ViewId::SOURCE);
    gold_spans_1.push(Span::new(0, text.len() as u32));
    doc_short.add(&gold_key, Tokens { spans: gold_spans_1, sentence: None }, Provenance::by("gold")).unwrap();

    let upos = TagSet::new("upos", &["NOUN", "SYM", "VERB", "PUNCT"]);
    let mut prov_short = Provenance::by("gold-pos");
    prov_short.on = vec![gold_key.id()];
    doc_short
        .add(
            &Key::DEFAULT,
            PosTags { tagset: Arc::clone(&upos), tags: vec![upos.tag("NOUN").unwrap()], prob: None },
            prov_short,
        )
        .unwrap();

    let codec_reg = turbo_analysis_layers::export::codecs();
    let graph_bytes = export(&doc_short, &codec_reg).unwrap().to_bytes().unwrap();
    let imported_short = import(&DocumentGraph::from_bytes(&graph_bytes).unwrap(), &codec_reg).unwrap();

    for target_doc in [&doc_short, &imported_short] {
        for fmt in
            [&TsvFormatter as &dyn DocumentFormatter, &CsvFormatter, &JsonLdFormatter, &WarcFormatter, &ConlluFormatter]
        {
            let err = fmt.format_to_vec("gold-short", target_doc, ExportOffsetMode::Utf8).unwrap_err();
            assert!(
                matches!(&err, FormatExportError::InvalidData(msg) if msg.contains("bound to opennlp:tokens@gold")),
                "expected InvalidData naming opennlp:tokens@gold from {}, got {err:?}",
                fmt.descriptor().format_id
            );
        }
    }

    // Control 2: named `gold` Tokens with equal length (4 tokens) but different
    // segmentation, and default PosTags anchored to `gold`.
    let mut doc_equal = build_doc_with_default_tokens(text, &default_spans);
    let mut gold_spans_4 = Spans::<Tokens>::new(ViewId::SOURCE);
    for (s, e) in [(0, 2), (2, 3), (3, 7), (8, 11)] {
        gold_spans_4.push(Span::new(s, e));
    }
    doc_equal.add(&gold_key, Tokens { spans: gold_spans_4, sentence: None }, Provenance::by("gold")).unwrap();

    let mut prov_equal = Provenance::by("gold-pos");
    prov_equal.on = vec![gold_key.id()];
    doc_equal
        .add(
            &Key::DEFAULT,
            PosTags {
                tagset: Arc::clone(&upos),
                tags: vec![
                    upos.tag("NOUN").unwrap(),
                    upos.tag("SYM").unwrap(),
                    upos.tag("VERB").unwrap(),
                    upos.tag("PUNCT").unwrap(),
                ],
                prob: None,
            },
            prov_equal,
        )
        .unwrap();

    for fmt in
        [&TsvFormatter as &dyn DocumentFormatter, &CsvFormatter, &JsonLdFormatter, &WarcFormatter, &ConlluFormatter]
    {
        let err = fmt.format_to_vec("gold-equal", &doc_equal, ExportOffsetMode::Utf8).unwrap_err();
        assert!(
            matches!(&err, FormatExportError::InvalidData(msg) if msg.contains("bound to opennlp:tokens@gold")),
            "expected InvalidData naming opennlp:tokens@gold from {}, got {err:?}",
            fmt.descriptor().format_id
        );
    }
}

#[test]
fn parse_warc_records_rejects_overflowing_lengths_and_truncated_trailers() {
    for overflow_len in [usize::MAX, usize::MAX - 1, usize::MAX - 3] {
        let raw = format!("WARC/1.1\r\nContent-Length: {overflow_len}\r\n\r\n");
        let err = parse_warc_records(raw.as_bytes()).unwrap_err();
        assert!(
            matches!(err, FormatExportError::InvalidData(_)),
            "expected InvalidData for {overflow_len}, got {err:?}"
        );
    }

    let truncated_trailer = b"WARC/1.1\r\nContent-Length: 2\r\n\r\nhi\r\n";
    let err = parse_warc_records(truncated_trailer).unwrap_err();
    assert!(
        matches!(err, FormatExportError::InvalidData(_)),
        "expected InvalidData for truncated trailer, got {err:?}"
    );

    let bad_trailer = b"WARC/1.1\r\nContent-Length: 2\r\n\r\nhiXXXX";
    let err = parse_warc_records(bad_trailer).unwrap_err();
    assert!(matches!(err, FormatExportError::InvalidData(_)), "expected InvalidData for bad trailer, got {err:?}");
}
