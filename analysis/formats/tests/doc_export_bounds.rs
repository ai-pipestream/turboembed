use std::sync::Arc;

use turbo_analysis::{
    Document, DocumentGraph, Edge, Edges, Ix, IxRange, Key, Provenance, Span, Spans, ViewId, export, import,
};
use turbo_analysis_layers::{Entities, PosTags, Sentences, TagSet, Tokens};
use turbo_formats::doc_export::{
    BratFormatter, ConlluFormatter, CsvFormatter, DocumentFormatter, ExportOffsetMode, FormatExportError,
    JsonLdFormatter, TsvFormatter, WarcFormatter, parse_warc_records,
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

    let mut codec_reg = turbo_analysis_layers::export::codecs();
    codec_reg.register::<Edges<Entities, Entities>>();
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

#[test]
fn formatters_refuse_mixed_endpoint_entity_relations_and_accept_matching_default() {
    // Source "é 😀": token 0 is "é" (0..2), token 1 is "😀" (3..7).
    let text = "é 😀";
    let default_spans = [(0, 2), (3, 7)];
    let mut doc_mixed = build_doc_with_default_tokens(text, &default_spans);

    let ner_set = TagSet::new("ner", &["MISC"]);
    let misc = ner_set.tag("MISC").unwrap();

    // Default Entities[0] covers the emoji (3..7, token 1..2).
    let mut def_ent_spans = Spans::<Entities>::new(ViewId::SOURCE);
    def_ent_spans.push(Span::new(3, 7));
    doc_mixed
        .add(
            &Key::DEFAULT,
            Entities {
                spans: def_ent_spans,
                tokens: vec![IxRange::new(1, 2)],
                kind: vec![misc],
                kinds: Arc::clone(&ner_set),
                score: vec![0.9],
            },
            Provenance::by("test"),
        )
        .unwrap();

    // Named `gold` Entities[0] covers "é" (0..2, token 0..1).
    let gold_ents_key = Key::<Entities>::named("gold");
    let mut gold_ent_spans = Spans::<Entities>::new(ViewId::SOURCE);
    gold_ent_spans.push(Span::new(0, 2));
    doc_mixed
        .add(
            &gold_ents_key,
            Entities {
                spans: gold_ent_spans,
                tokens: vec![IxRange::new(0, 1)],
                kind: vec![misc],
                kinds: Arc::clone(&ner_set),
                score: vec![0.95],
            },
            Provenance::by("gold"),
        )
        .unwrap();

    // Mixed-endpoint relation: `Edges::between(&Key::DEFAULT, &gold_ents_key, ...)`
    // records bindings `[opennlp:entities, opennlp:entities@gold]`.
    let rel_set = TagSet::new("rel", &["LINK"]);
    let link = rel_set.tag("LINK").unwrap();
    let mixed_edges = Edges::<Entities, Entities>::between(
        Key::DEFAULT,
        gold_ents_key,
        Arc::clone(&rel_set),
        [Edge::new(Ix::new(0), Ix::new(0), link, 1.0)],
    );
    let mut mixed_prov = Provenance::by("mixed-rel");
    mixed_prov.reads = vec![Key::<Entities>::DEFAULT.id(), gold_ents_key.id()];
    doc_mixed.add(&Key::DEFAULT, mixed_edges, mixed_prov).unwrap();

    let mut codec_reg = turbo_analysis_layers::export::codecs();
    codec_reg.register::<Edges<Entities, Entities>>();
    let graph_bytes = export(&doc_mixed, &codec_reg).unwrap().to_bytes().unwrap();
    let imported_mixed = import(&DocumentGraph::from_bytes(&graph_bytes).unwrap(), &codec_reg).unwrap();

    for target_doc in [&doc_mixed, &imported_mixed] {
        for fmt in [&BratFormatter as &dyn DocumentFormatter, &JsonLdFormatter, &WarcFormatter] {
            let err = fmt.format_to_vec("mixed-rel", target_doc, ExportOffsetMode::Utf8).unwrap_err();
            assert!(
                matches!(&err, FormatExportError::InvalidData(msg) if msg.contains("bound to opennlp:entities@gold")),
                "expected InvalidData naming opennlp:entities@gold from {}, got {err:?}",
                fmt.descriptor().format_id
            );
        }
    }

    // Matching-default control: both endpoints bound to `Key::<Entities>::DEFAULT`.
    let mut doc_default = build_doc_with_default_tokens(text, &default_spans);
    let mut both_spans = Spans::<Entities>::new(ViewId::SOURCE);
    both_spans.push(Span::new(0, 2));
    both_spans.push(Span::new(3, 7));
    doc_default
        .add(
            &Key::DEFAULT,
            Entities {
                spans: both_spans,
                tokens: vec![IxRange::new(0, 1), IxRange::new(1, 2)],
                kind: vec![misc, misc],
                kinds: ner_set,
                score: vec![0.9, 0.95],
            },
            Provenance::by("test"),
        )
        .unwrap();
    let default_edges = Edges::<Entities, Entities>::between(
        Key::DEFAULT,
        Key::DEFAULT,
        rel_set,
        [Edge::new(Ix::new(0), Ix::new(1), link, 1.0)],
    );
    let mut default_prov = Provenance::by("default-rel");
    default_prov.reads = vec![Key::<Entities>::DEFAULT.id(), Key::<Entities>::DEFAULT.id()];
    doc_default.add(&Key::DEFAULT, default_edges, default_prov).unwrap();

    let brat_out = BratFormatter.format_to_string("default-rel", &doc_default, ExportOffsetMode::Utf8).unwrap();
    assert!(brat_out.contains("R1\tLINK Arg1:T1 Arg2:T2\n"), "unexpected brat output: {brat_out}");
    let jsonld_out = JsonLdFormatter.format_to_string("default-rel", &doc_default, ExportOffsetMode::Utf8).unwrap();
    assert!(jsonld_out.contains("\"kind\": \"LINK\""), "unexpected jsonld output: {jsonld_out}");
    let warc_out = WarcFormatter.format_to_vec("default-rel", &doc_default, ExportOffsetMode::Utf8).unwrap();
    assert!(!warc_out.is_empty());
}
