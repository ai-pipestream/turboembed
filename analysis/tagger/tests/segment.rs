//! The sentence and span annotators end to end on the small BERT in
//! testdata, with a sentence boundary head and a span classifier head of
//! seeded weights: the core heads' decodes on the view's tokens, exact
//! offsets, determinism, and refusals.

mod common;

use common::*;
use serde_json::{Value, json};
use turbo::head::boundary::sentences;
use turbo::head::span::SpanWork;
use turbo::head::tagger::Work;
use turbo::manifest::WordPooling;
use turbo_analysis::{Annotator, Document, Ix, Key, Layer, View};
use turbo_analysis_embed::EncoderOptions;
use turbo_tagger::{EntityTagger, SentenceTagger, SpanTagger, WordEncoder};

fn sentence_head() -> Value {
    json!({
        "name": "sents", "layer": "turboembed:sentences", "kind": "HEAD_SENTENCE_BOUNDARY",
        "labels": ["inside", "last"], "scheme": "SCHEME_TAGS", "word_pooling": "WORD_POOLING_FIRST",
        "decode": "DECODE_ARGMAX", "artifact": "weights-f32", "tensor_prefix": "heads.sents"
    })
}

fn span_head(decode: &str) -> Value {
    json!({
        "name": "mentions", "layer": "turboembed:entities", "kind": "HEAD_SPAN_CLASSIFIER",
        "labels": ["O", "PER", "LOC"], "scheme": "SCHEME_TAGS", "word_pooling": "WORD_POOLING_FIRST",
        "decode": decode, "span": {"max_words": 3, "width_dim": 4},
        "artifact": "weights-f32", "tensor_prefix": "heads.mentions"
    })
}

/// The tiny bundle carrying only the given heads.
fn with_heads(name: &str, heads: Value) -> std::path::PathBuf {
    bundle(name, |m| m["heads"] = heads)
}

/// The text's words and their first-piece features, as the annotators
/// read them.
fn features(dir: &std::path::Path, text: &str) -> (Vec<String>, Vec<f32>) {
    let words: Vec<&str> = text.split(' ').collect();
    let mut enc = WordEncoder::open(dir, &EncoderOptions::default()).unwrap();
    let f = enc.features(std::slice::from_ref(&words), WordPooling::First).unwrap().remove(0);
    (words.iter().map(|w| w.to_string()).collect(), f)
}

#[test]
fn the_sentence_tagger_writes_the_heads_sentences_on_token_bounds() {
    let dir = with_heads("sents", json!([sentence_head()]));
    let doc = Document::new(TEXT);
    let view = doc.require(&Key::<View>::DEFAULT).unwrap();
    let toks = tokens(TEXT);
    let tagger = SentenceTagger::open(&dir, None, &EncoderOptions::default()).unwrap();
    let (layer,) = tagger.annotate((view, &toks, None)).unwrap();
    layer.check().unwrap();
    layer.check_contiguous_in(view).unwrap();

    // The core head's decode of the same features, as token ranges.
    let (words, f) = features(&dir, TEXT);
    let ids: Vec<u32> = (0..words.len() as u32).collect();
    let (mut scores, mut ends, mut work) = (Vec::new(), Vec::new(), Work::default());
    tagger.head().scores(&f, &ids, None, &mut scores, &mut work).unwrap();
    tagger.head().decode(&scores, words.len(), &mut ends).unwrap();
    let want: Vec<(u32, u32)> = sentences(&ends)
        .unwrap()
        .into_iter()
        .map(|r| (toks.spans.get(Ix::new(r.start as u32)).start, toks.spans.get(Ix::new(r.end as u32 - 1)).end))
        .collect();
    let got: Vec<(u32, u32)> = layer.spans.iter().map(|(_, s)| (s.start, s.end)).collect();
    assert_eq!(got, want);
    assert_eq!(got.first().map(|s| s.0), Some(0));
    assert_eq!(got.last().map(|s| s.1), Some(TEXT.len() as u32));

    // A sentence column on the tokens does not change the reading.
    let mut split = tokens(TEXT);
    split.sentence = Some((0..split.spans.len()).map(|i| Ix::new(u32::from(i >= 5))).collect());
    let (again,) = tagger.annotate((view, &split, None)).unwrap();
    assert_eq!(again.spans.iter().map(|(_, s)| (s.start, s.end)).collect::<Vec<_>>(), got);

    let empty = Document::new("");
    let ev = empty.require(&Key::<View>::DEFAULT).unwrap();
    let none = turbo_analysis_layers::Tokens { spans: turbo_analysis::Spans::new(ev.id()), sentence: None };
    assert!(tagger.annotate((ev, &none, None)).unwrap().0.spans.is_empty());
}

#[test]
fn the_span_tagger_writes_the_heads_spans_as_entities() {
    for decode in ["DECODE_SPANS_NESTED", "DECODE_SPANS_FLAT"] {
        let dir = with_heads(&format!("spans-{decode}"), json!([span_head(decode)]));
        let doc = Document::new(TEXT);
        let view = doc.require(&Key::<View>::DEFAULT).unwrap();
        let toks = tokens(TEXT);
        let tagger = SpanTagger::open(&dir, None, &EncoderOptions::default()).unwrap();
        assert_eq!(tagger.kinds().len(), 2);
        let (layer,) = tagger.annotate((view, &toks, None, None)).unwrap();
        layer.check().unwrap();

        let (words, f) = features(&dir, TEXT);
        let ids: Vec<u32> = (0..words.len() as u32).collect();
        let (mut scores, mut kept, mut w, mut sw) = (Vec::new(), Vec::new(), Work::default(), SpanWork::default());
        tagger.head().scores(&f, &ids, None, &mut scores, &mut w, &mut sw).unwrap();
        tagger.head().decode(&scores, words.len(), &mut kept).unwrap();
        assert!(!kept.is_empty(), "{decode}: the seeded head keeps some span");
        let want: Vec<(u32, u32, u16)> = kept.iter().map(|s| (s.first, s.last + 1, s.label as u16 - 1)).collect();
        let got: Vec<(u32, u32, u16)> = layer
            .tokens
            .iter()
            .zip(&layer.kind)
            .map(|(r, k)| (r.start.get() as u32, r.end.get() as u32, k.0))
            .collect();
        assert_eq!(got, want, "{decode}");
        for ((_, s), r) in layer.spans.iter().zip(&layer.tokens) {
            let first = toks.spans.get(Ix::new(r.start.get() as u32));
            let last = toks.spans.get(Ix::new(r.end.get() as u32 - 1));
            assert_eq!((s.start, s.end), (first.start, last.end));
        }
        assert!(layer.score.iter().all(|p| *p > 0.0 && *p <= 1.0));
        if decode == "DECODE_SPANS_FLAT" {
            assert!(layer.tokens.windows(2).all(|p| p[0].end <= p[1].start), "flat spans do not overlap");
        }
        let (again,) = tagger.annotate((view, &toks, None, None)).unwrap();
        assert_eq!((again.tokens, again.score), (layer.tokens, layer.score));
    }
}

#[test]
fn each_annotator_opens_only_its_own_kind() {
    let dir = with_heads("kinds", json!([sentence_head(), span_head("DECODE_SPANS_FLAT"), heads()[1].clone()]));
    let o = EncoderOptions::default();
    assert!(SentenceTagger::open(&dir, Some("mentions"), &o).is_err());
    assert!(SpanTagger::open(&dir, Some("sents"), &o).is_err());
    // Two heads write the entity layer, so the span tagger needs a name.
    assert!(SpanTagger::open(&dir, None, &o).is_err());
    assert!(SpanTagger::open(&dir, Some("mentions"), &o).is_ok());
    assert!(EntityTagger::open(&dir, Some("mentions"), &o).is_err());
}
