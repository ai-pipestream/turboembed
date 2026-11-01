//! The sentence, word-split and span annotators end to end on the small
//! BERT in testdata, with a sentence boundary head, a token split head
//! and a span classifier head of seeded weights: the core heads' decodes
//! on the view's tokens, exact offsets, determinism, and refusals.

mod common;

use common::*;
use serde_json::{Value, json};
use turbo::head::boundary::sentences;
use turbo::head::edit_tree::EditTree;
use turbo::head::span::SpanWork;
use turbo::head::tagger::Work;
use turbo::manifest::WordPooling;
use turbo_analysis::{Annotator, Document, Ix, Key, Layer, Span, View};
use turbo_analysis_embed::EncoderOptions;
use turbo_tagger::{EntityTagger, SentenceTagger, SpanTagger, WordEncoder, WordTagger};

fn sentence_head() -> Value {
    json!({
        "name": "sents", "layer": "turboembed:sentences", "kind": "HEAD_SENTENCE_BOUNDARY",
        "labels": ["inside", "last"], "scheme": "SCHEME_TAGS", "word_pooling": "WORD_POOLING_FIRST",
        "decode": "DECODE_ARGMAX", "artifact": "weights-f32", "tensor_prefix": "heads.sents"
    })
}

fn split_label(form: &str, words: &[&str]) -> String {
    words.iter().map(|w| EditTree::mine(form, w).label()).collect::<Vec<_>>().join(" ")
}

fn split_head() -> Value {
    let del = split_label("del", &["de", "el"]);
    let cant = split_label("can't", &["ca", "n't"]);
    let and = split_label("and", &["an", "d"]);
    json!({
        "name": "words", "layer": "turboembed:words", "kind": "HEAD_TOKEN_SPLIT",
        "labels": ["_", del, cant, and], "scheme": "SCHEME_TAGS", "word_pooling": "WORD_POOLING_FIRST",
        "decode": "DECODE_EDIT_TREE", "artifact": "weights-f32", "tensor_prefix": "heads.words"
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
fn the_word_tagger_writes_the_heads_splits_with_subspans_and_forms() {
    let dir = with_heads("words", json!([split_head()]));
    let text = "El perro del vecino can't dormir and ladra .";
    let doc = Document::new(text);
    let view = doc.require(&Key::<View>::DEFAULT).unwrap();
    let toks = tokens(text);
    let tagger = WordTagger::open(&dir, None, &EncoderOptions::default()).unwrap();
    let (layer,) = tagger.annotate((view, &toks, None, None)).unwrap();
    layer.check_against(text, &toks).unwrap();

    // The core head's decode of the same features and forms.
    let (words, f) = features(&dir, text);
    let forms: Vec<&str> = words.iter().map(String::as_str).collect();
    let ids: Vec<u32> = (0..words.len() as u32).collect();
    let (mut scores, mut labels, mut work) = (Vec::new(), Vec::new(), Work::default());
    tagger.head().scores(&f, &ids, None, &mut scores, &mut work).unwrap();
    tagger.head().decode(&scores, &forms, &mut labels).unwrap();
    let want: Vec<String> = tagger.head().words(&labels, &forms).unwrap().into_iter().flatten().collect();
    let got: Vec<String> = layer.iter(text, &toks).map(|w| w.form.to_owned()).collect();
    assert_eq!(got, want);

    // Every word's token_span matches its surface token, and any subspan
    // slices its surface token directly while fusional forms have subspan None.
    for w in layer.iter(text, &toks) {
        let tok_span = toks.spans.get(w.tokens.start);
        assert_eq!(w.token_span, tok_span);
        if let Some(sub) = w.subspan {
            assert_eq!(&text[sub.range()], w.form);
        }
    }

    // Exercise explicit fusional forms ("del" -> "de" + "el") and contiguous
    // byte subspans ("can't" -> "ca" + "n't") directly when the head picks
    // their split labels (high bias on the two split labels, low on "_").
    let forced = bundle("words-forced", |m| {
        let del = split_label("del", &["de", "el"]);
        let cant = split_label("can't", &["ca", "n't"]);
        m["heads"] = json!([{
            "name": "words", "layer": "turboembed:words", "kind": "HEAD_TOKEN_SPLIT",
            "labels": [del, cant, "_"], "scheme": "SCHEME_TAGS", "word_pooling": "WORD_POOLING_FIRST",
            "decode": "DECODE_EDIT_TREE", "artifact": "weights-f32", "tensor_prefix": "heads.words"
        }]);
    });
    let ft = WordTagger::open(&forced, None, &EncoderOptions::default()).unwrap();
    // With "_" last, any positive score on del or can't splits them, while
    // other tokens do not fit either tree and fall back to "_".
    let (fl,) = ft.annotate((view, &toks, None, None)).unwrap();
    fl.check_against(text, &toks).unwrap();
    let del_words: Vec<_> = fl.of_token(Ix::new(2)).iter().map(|w| fl.get(w, text, &toks)).collect();
    if del_words.len() == 2 {
        assert_eq!((del_words[0].form, del_words[1].form), ("de", "el"));
        assert_eq!((del_words[0].subspan, del_words[1].subspan), (None, None));
    }
    let cant_span = toks.spans.get(Ix::new(4));
    let cant_words: Vec<_> = fl.of_token(Ix::new(4)).iter().map(|w| fl.get(w, text, &toks)).collect();
    if cant_words.len() == 2 {
        assert_eq!((cant_words[0].form, cant_words[1].form), ("ca", "n't"));
        assert_eq!(cant_words[0].subspan, Some(Span::new(cant_span.start, cant_span.start + 2)));
        assert_eq!(cant_words[1].subspan, Some(Span::new(cant_span.start + 2, cant_span.end)));
    }

    // When no token in the text fits any split tree, the layer is 1-to-1
    // with neither explicit forms nor subspans allocated.
    let plain_text = "Ann met Bob .";
    let plain_doc = Document::new(plain_text);
    let pv = plain_doc.require(&Key::<View>::DEFAULT).unwrap();
    let pt = tokens(plain_text);
    let (plain_words,) = tagger.annotate((pv, &pt, None, None)).unwrap();
    assert!(plain_words.is_identity());

    let (again,) = tagger.annotate((view, &toks, None, None)).unwrap();
    let pairs = |l: &turbo_analysis_layers::Words| {
        l.iter(text, &toks).map(|w| (w.tokens, w.subspan, w.form.to_owned())).collect::<Vec<_>>()
    };
    assert_eq!(pairs(&again), pairs(&layer));

    let empty = Document::new("");
    let ev = empty.require(&Key::<View>::DEFAULT).unwrap();
    let none = turbo_analysis_layers::Tokens { spans: turbo_analysis::Spans::new(ev.id()), sentence: None };
    assert!(tagger.annotate((ev, &none, None, None)).unwrap().0.is_empty());
}

#[test]
fn each_annotator_opens_only_its_own_kind() {
    let dir =
        with_heads("kinds", json!([sentence_head(), split_head(), span_head("DECODE_SPANS_FLAT"), heads()[1].clone()]));
    let o = EncoderOptions::default();
    assert!(SentenceTagger::open(&dir, Some("mentions"), &o).is_err());
    assert!(WordTagger::open(&dir, Some("sents"), &o).is_err());
    assert!(WordTagger::open(&dir, None, &o).is_ok());
    assert!(SpanTagger::open(&dir, Some("sents"), &o).is_err());
    // Two heads write the entity layer, so the span tagger needs a name.
    assert!(SpanTagger::open(&dir, None, &o).is_err());
    assert!(SpanTagger::open(&dir, Some("mentions"), &o).is_ok());
    assert!(EntityTagger::open(&dir, Some("mentions"), &o).is_err());
}
