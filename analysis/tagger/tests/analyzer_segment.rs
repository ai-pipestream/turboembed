//! The in-process C analyzer over a bundle with a sentence boundary head,
//! a token split head and a span classifier entity head writes what the
//! annotators of this crate write over the same tokens: the sentences of
//! [`SentenceTagger`], the syntactic words of [`WordTagger`] and the
//! entities of [`SpanTagger`], from the graph and from the direct tasks
//! alike, and the graph names the heads that wrote them.

mod common;

use std::ffi::CStr;
use std::path::Path;

use common::*;
use serde_json::json;
use turbo::{
    TURBO_ANALYSIS_LAYER_ENTITIES, TURBO_ANALYSIS_LAYER_SENTENCES, TURBO_ANALYSIS_LAYER_TOKENS,
    TURBO_ANALYSIS_LAYER_WORDS, TURBO_ERROR_MESSAGE_LEN, TURBO_SPAN_LABEL, status, turbo_analysis_result,
    turbo_analysis_result_graph, turbo_analysis_result_release, turbo_analysis_result_spans,
    turbo_analysis_result_strings, turbo_analyzer, turbo_analyzer_analyze_graph, turbo_analyzer_create,
    turbo_analyzer_desc, turbo_analyzer_detect_sentences, turbo_analyzer_find_text, turbo_analyzer_find_tokens,
    turbo_analyzer_inspect_graph, turbo_analyzer_release, turbo_analyzer_tokenize, turbo_error, turbo_span, turbo_text,
};
use turbo_analysis::{Codecs, Document, DocumentGraph, Ix, Key, Layer, Span, Spans, View, ViewId, import};
use turbo_analysis_embed::EncoderOptions;
use turbo_analysis_layers::export::register_codecs;
use turbo_analysis_layers::{Entities, Sentences, Tokens, Words};
use turbo_tagger::{SentenceTagger, SpanTagger, WordTagger};

const TEXT: &str = "Ann met Bob del Paris , and the quiet old dog slept by the river . Bob can't walk the old dog . \
                    Ann watched del río while the dog slept .";

fn t(s: &str) -> turbo_text {
    turbo_text { ptr: s.as_ptr().cast(), len: s.len() as u64 }
}

fn new_err() -> turbo_error {
    turbo_error {
        struct_size: size_of::<turbo_error>() as u32,
        code: -1,
        field: 0,
        message: [0; TURBO_ERROR_MESSAGE_LEN],
    }
}

/// A row with its label read from the pool, empty when it has none.
#[derive(Clone, Debug)]
struct Row {
    span: turbo_span,
    label: String,
}

/// The tiny bundle with a sentence head, a split head that splits `del`
/// and `can't` whenever their split scores lead, and a flat span head.
fn segment_bundle() -> std::path::PathBuf {
    bundle("analyzer-segment", |m| {
        let del = split_label("del", &["de", "el"]);
        let cant = split_label("can't", &["ca", "n't"]);
        m["heads"] = json!([
            sentence_head(),
            {
                "name": "words", "layer": "turboembed:words", "kind": "HEAD_TOKEN_SPLIT",
                "labels": [del, cant, "_"], "scheme": "SCHEME_TAGS", "word_pooling": "WORD_POOLING_FIRST",
                "decode": "DECODE_EDIT_TREE", "artifact": "weights-f32", "tensor_prefix": "heads.words"
            },
            span_head("DECODE_SPANS_FLAT"),
        ]);
    })
}

struct Analyzer(*mut turbo_analyzer);

impl Drop for Analyzer {
    fn drop(&mut self) {
        unsafe { turbo_analyzer_release(self.0) }
    }
}

impl Analyzer {
    fn open(bundle: &Path) -> Analyzer {
        let path = bundle.to_str().unwrap();
        let mut err = new_err();
        unsafe {
            let mut desc: turbo_analyzer_desc = std::mem::zeroed();
            desc.struct_size = size_of::<turbo_analyzer_desc>() as u32;
            desc.bundle_path = t(path);
            let mut a: *mut turbo_analyzer = std::ptr::null_mut();
            let rc = turbo_analyzer_create(&desc, &mut a, &mut err);
            assert_eq!(rc, status::OK, "{}", CStr::from_ptr(err.message.as_ptr()).to_str().unwrap());
            Analyzer(a)
        }
    }

    /// The rows of one call's result, and its graph bytes.
    fn rows(
        &self,
        call: impl FnOnce(*mut turbo_analyzer, *mut *mut turbo_analysis_result, *mut turbo_error) -> i32,
    ) -> (Vec<Row>, Vec<u8>) {
        let mut err = new_err();
        unsafe {
            let mut res: *mut turbo_analysis_result = std::ptr::null_mut();
            let rc = call(self.0, &mut res, &mut err);
            assert_eq!(rc, status::OK, "{}", CStr::from_ptr(err.message.as_ptr()).to_str().unwrap());
            let mut spans_ptr: *const turbo_span = std::ptr::null();
            let mut count: u32 = 0;
            assert_eq!(turbo_analysis_result_spans(res, &mut spans_ptr, &mut count, &mut err), status::OK);
            let mut pool_ptr: *const std::ffi::c_char = std::ptr::null();
            let mut pool_len: u64 = 0;
            assert_eq!(turbo_analysis_result_strings(res, &mut pool_ptr, &mut pool_len, &mut err), status::OK);
            let spans = if count == 0 { &[][..] } else { std::slice::from_raw_parts(spans_ptr, count as usize) };
            let pool = if pool_len == 0 {
                &[][..]
            } else {
                std::slice::from_raw_parts(pool_ptr.cast::<u8>(), pool_len as usize)
            };
            let rows = spans
                .iter()
                .map(|&span| {
                    let (a, b) = (span.label_offset as usize, (span.label_offset + span.label_len) as usize);
                    Row { span, label: std::str::from_utf8(&pool[a..b]).unwrap().to_owned() }
                })
                .collect();
            let (mut ptr, mut len) = (std::ptr::null(), 0u64);
            assert_eq!(turbo_analysis_result_graph(res, &mut ptr, &mut len, &mut err), status::OK);
            let graph = if len == 0 { Vec::new() } else { std::slice::from_raw_parts(ptr, len as usize).to_vec() };
            turbo_analysis_result_release(res);
            (rows, graph)
        }
    }
}

fn of(rows: &[Row], kind: u32) -> Vec<Row> {
    rows.iter().filter(|r| r.span.layer_kind == kind).cloned().collect()
}

fn spans_of<L>(rows: &[Row]) -> Spans<L> {
    let mut spans = Spans::new(ViewId::SOURCE);
    for r in rows {
        spans.push(Span::new(r.span.source_byte_start, r.span.source_byte_end));
    }
    spans
}

fn bytes(rows: &[Row]) -> Vec<(u32, u32)> {
    rows.iter().map(|r| (r.span.source_byte_start, r.span.source_byte_end)).collect()
}

/// A word as (token range, byte range, explicit form).
type WordRow = ((u32, u32), (u32, u32), Option<String>);

/// Each word row as a [`WordRow`].
fn word_rows(rows: &[Row]) -> Vec<WordRow> {
    rows.iter()
        .map(|r| {
            let label = (r.span.flags & TURBO_SPAN_LABEL != 0).then(|| r.label.clone());
            ((r.span.token_start, r.span.token_end), (r.span.source_byte_start, r.span.source_byte_end), label)
        })
        .collect()
}

/// The words layer as word rows over `tokens`.
fn word_layer_rows(words: &Words, tokens: &Tokens) -> Vec<WordRow> {
    (0..words.len() as u32)
        .map(|i| {
            let w = Ix::new(i);
            let r = words.tokens[i as usize];
            let span = words.subspan_of(w).unwrap_or_else(|| words.token_span(w, tokens));
            let form = words.form.as_ref().and_then(|f| f[i as usize].as_deref()).map(str::to_owned);
            ((r.start.get() as u32, r.end.get() as u32), (span.start, span.end), form)
        })
        .collect()
}

/// Each entity row as (token range, kind, probability).
fn entity_rows(rows: &[Row]) -> Vec<((u32, u32), String, f64)> {
    rows.iter().map(|r| ((r.span.token_start, r.span.token_end), r.label.clone(), r.span.prob)).collect()
}

fn entity_layer_rows(e: &Entities) -> Vec<((u32, u32), String, f64)> {
    (0..e.tokens.len())
        .map(|i| {
            let r = e.tokens[i];
            ((r.start.get() as u32, r.end.get() as u32), e.kinds.label(e.kind[i]).to_owned(), e.score[i])
        })
        .collect()
}

#[test]
fn the_c_analyzer_reads_the_sentence_split_and_span_heads_as_the_annotators_do() {
    let dir = segment_bundle();
    let a = Analyzer::open(&dir);
    let (rows, graph) = a.rows(|a, out, err| unsafe { turbo_analyzer_analyze_graph(a, t(TEXT), out, err) });
    let (token_rows, sentence_rows) =
        (of(&rows, TURBO_ANALYSIS_LAYER_TOKENS), of(&rows, TURBO_ANALYSIS_LAYER_SENTENCES));
    let (graph_words, graph_entities) =
        (of(&rows, TURBO_ANALYSIS_LAYER_WORDS), of(&rows, TURBO_ANALYSIS_LAYER_ENTITIES));
    assert!(token_rows.len() > 30 && !sentence_rows.is_empty(), "{} tokens", token_rows.len());

    // The annotators over the C analyzer's own tokens.
    let doc = Document::new(TEXT);
    let view = doc.require(&Key::<View>::DEFAULT).unwrap();
    let toks = Tokens { spans: spans_of(&token_rows), sentence: None };
    let o = EncoderOptions::default();
    let sents = SentenceTagger::open(&dir, None, &o).unwrap().sentences(view, &toks, None).unwrap();
    let want_sents: Vec<(u32, u32)> = sents.spans.iter().map(|(_, s)| (s.start, s.end)).collect();
    assert_eq!(bytes(&sentence_rows), want_sents);
    assert!(want_sents.len() > 1, "the seeded sentence head ends some sentence early");

    // The sentence task reads the head, not the rule splitter.
    let (direct, _) = a.rows(|a, out, err| unsafe { turbo_analyzer_detect_sentences(a, t(TEXT), out, err) });
    assert_eq!(bytes(&direct), want_sents);

    // Syntactic words over those sentences, from the graph and from the
    // tokenizer task, which also writes the tokens it always wrote.
    let words = WordTagger::open(&dir, None, &o).unwrap().words(view, &toks, Some(&sents), None).unwrap();
    let want_words = word_layer_rows(&words, &toks);
    assert_eq!(word_rows(&graph_words), want_words);
    assert!(graph_words.len() > token_rows.len(), "the seeded split head splits some token");
    let (tokenized, _) = a.rows(|a, out, err| unsafe { turbo_analyzer_tokenize(a, t(TEXT), out, err) });
    assert_eq!(bytes(&of(&tokenized, TURBO_ANALYSIS_LAYER_TOKENS)), bytes(&token_rows));
    assert_eq!(word_rows(&of(&tokenized, TURBO_ANALYSIS_LAYER_WORDS)), want_words);
    for r in &graph_words {
        assert!(r.span.prob.is_nan());
        if r.span.flags & TURBO_SPAN_LABEL == 0 {
            assert_eq!(r.span.label_len, 0, "a word in the text puts no copy of it in the pool");
        }
    }

    // Entities from the span classifier, a sentence at a time, from the
    // graph and from the text task.
    let ents = SpanTagger::open(&dir, None, &o).unwrap().entities(view, &toks, Some(&sents), None).unwrap();
    assert!(!ents.tokens.is_empty(), "the seeded span head keeps some span");
    assert_eq!(entity_rows(&graph_entities), entity_layer_rows(&ents));
    let (found, _) = a.rows(|a, out, err| unsafe { turbo_analyzer_find_text(a, t(TEXT), out, err) });
    assert_eq!(entity_rows(&found), entity_rows(&graph_entities));
    assert_eq!(bytes(&found), bytes(&graph_entities));

    // The pre-tokenized task on the first sentence's tokens.
    let first = sents.spans.get(Ix::new(0));
    let first_rows: Vec<&Row> = token_rows.iter().take_while(|r| r.span.source_byte_end <= first.end).collect();
    let first_words: Vec<turbo_text> = first_rows
        .iter()
        .map(|r| t(&TEXT[r.span.source_byte_start as usize..r.span.source_byte_end as usize]))
        .collect();
    let (seq, _) = a.rows(|a, out, err| unsafe {
        turbo_analyzer_find_tokens(a, first_words.as_ptr(), first_words.len() as u32, out, err)
    });
    let in_first: Vec<_> = entity_rows(&graph_entities)
        .into_iter()
        .filter(|((_, end), _, _)| (*end as usize) <= first_rows.len())
        .collect();
    assert_eq!(entity_rows(&seq), in_first);

    // A caller's graph holding the words gives the same rows back.
    let (inspected, _) =
        a.rows(|a, out, err| unsafe { turbo_analyzer_inspect_graph(a, graph.as_ptr(), graph.len() as u64, out, err) });
    assert_eq!(word_rows(&of(&inspected, TURBO_ANALYSIS_LAYER_WORDS)), want_words);

    // The graph names the heads that wrote each layer.
    let mut codecs = Codecs::new();
    register_codecs(&mut codecs);
    let doc = import(&DocumentGraph::from_bytes(&graph).unwrap(), &codecs).unwrap();
    let by = |id| doc.provenance(id).map(|p| p.annotator);
    assert_eq!(by(Key::<Sentences>::DEFAULT.id()), Some("turbo-tagger-sentences"));
    assert_eq!(by(Key::<Words>::DEFAULT.id()), Some("turbo-tagger-words"));
    assert_eq!(by(Key::<Entities>::DEFAULT.id()), Some("turbo-tagger-spans"));
    assert_eq!(by(Key::<Tokens>::DEFAULT.id()), Some("turbo:tokenize"));
}

#[test]
fn a_text_of_no_tokens_has_no_rows_and_a_head_of_another_kind_on_the_sentence_layer_is_refused() {
    let dir = segment_bundle();
    let a = Analyzer::open(&dir);
    for text in ["", "   "] {
        let (rows, _) = a.rows(|a, out, err| unsafe { turbo_analyzer_detect_sentences(a, t(text), out, err) });
        assert!(rows.is_empty(), "{text:?}");
        let (rows, _) = a.rows(|a, out, err| unsafe { turbo_analyzer_tokenize(a, t(text), out, err) });
        assert!(rows.is_empty(), "{text:?}");
    }

    // A tagger writing the sentence layer: no rule splitter stands in.
    let wrong = bundle("analyzer-segment-wrong", |m| {
        let mut h = heads()[0].clone();
        h["name"] = json!("sents");
        h["layer"] = json!("turboembed:sentences");
        h["tensor_prefix"] = json!("heads.sents");
        m["heads"] = json!([h]);
    });
    let path = wrong.to_str().unwrap();
    let mut err = new_err();
    unsafe {
        let mut desc: turbo_analyzer_desc = std::mem::zeroed();
        desc.struct_size = size_of::<turbo_analyzer_desc>() as u32;
        desc.bundle_path = t(path);
        let mut out: *mut turbo_analyzer = std::ptr::null_mut();
        assert_eq!(turbo_analyzer_create(&desc, &mut out, &mut err), status::UNSUPPORTED_TASK);
        assert!(out.is_null());
        assert_eq!(err.field, 7);
        let message = CStr::from_ptr(err.message.as_ptr()).to_str().unwrap();
        assert!(message.contains("\"sents\"") && message.contains("sentences"), "{message}");
    }
}
