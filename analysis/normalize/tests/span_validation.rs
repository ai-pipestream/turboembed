use std::sync::Arc;

use turbo_analysis::{Annotator, Document, Error, Span, Spans, View, ViewId};
use turbo_analysis_layers::{Sentences, Tokens};
use turbo_ngram::NGramLanguageModel;
use turbo_normalize::{ContextSpellCorrector, SymSpell, SymbolJoiner};
use turbo_normalize_layers::Corrections;

fn corrector() -> ContextSpellCorrector {
    let mut dictionary = SymSpell::default();
    dictionary.add("the", 1000);
    let mut lm = NGramLanguageModel::new(1).unwrap();
    for _ in 0..100 {
        lm.add(&["the"]).unwrap();
    }
    ContextSpellCorrector::new(Arc::new(dictionary), Arc::new(lm)).edit_weight(0.0)
}

fn correct(view: &View, sentences: &Sentences) -> Result<(View, Corrections), Error> {
    corrector().annotate((view, None, None, None, None, None, None, None, None, Some(sentences)))
}

#[test]
fn spelling_processes_prefix_inter_sentence_and_trailing_tokens_once() {
    let doc = Document::new("teh the teh the teh");
    let view = doc.get::<View>().unwrap();
    let mut sentences = Sentences { spans: Spans::new(view.id()) };
    for (start, end) in [(4, 7), (12, 15)] {
        sentences.spans.push(Span::new(start, end));
    }
    let (out, corrections) = correct(view, &sentences).unwrap();
    assert_eq!(out.text(), "the the the the the");
    assert_eq!(
        corrections.spans.iter().map(|(_, s)| s).collect::<Vec<_>>(),
        vec![Span::new(0, 3), Span::new(8, 11), Span::new(16, 19)]
    );
    assert!(corrections.suggestion.iter().all(|s| s == "the"));
}

#[test]
fn empty_sentence_layer_preserves_fallback_correction() {
    let doc = Document::new("teh");
    let view = doc.get::<View>().unwrap();
    let sentences = Sentences { spans: Spans::new(view.id()) };
    assert_eq!(correct(view, &sentences).unwrap().0.text(), "the");
}

#[test]
fn both_annotators_refuse_invalid_spans_without_panicking() {
    let doc = Document::new("é & +");
    let view = doc.get::<View>().unwrap();
    for spans in [
        vec![Span { start: 3, end: 2 }],
        vec![Span::new(0, 99)],
        vec![Span::new(1, 2)],
        vec![Span::new(0, 1)],
        vec![Span::new(3, 4), Span::new(3, 4)],
    ] {
        let mut sentences = Sentences { spans: Spans::new(view.id()) };
        let mut tokens = Tokens { spans: Spans::new(view.id()), sentence: None };
        for span in spans {
            sentences.spans.push(span);
            tokens.spans.push(span);
        }
        assert!(correct(view, &sentences).is_err());
        assert!(SymbolJoiner::new().annotate((view, &tokens)).is_err());
    }
}

#[test]
fn both_annotators_refuse_foreign_views_and_discontinuous_items() {
    let doc = Document::new("& +");
    let view = doc.get::<View>().unwrap();
    let sentences = Sentences { spans: Spans::new(ViewId(7)) };
    let tokens = Tokens { spans: Spans::new(ViewId(7)), sentence: None };
    assert!(correct(view, &sentences).is_err());
    assert!(SymbolJoiner::new().annotate((view, &tokens)).is_err());
    let mut sentences = Sentences { spans: Spans::new(view.id()) };
    let mut tokens = Tokens { spans: Spans::new(view.id()), sentence: None };
    sentences.spans.push_fragments(&[Span::new(0, 1), Span::new(2, 3)]);
    tokens.spans.push_fragments(&[Span::new(0, 1), Span::new(2, 3)]);
    assert!(correct(view, &sentences).is_err());
    assert!(SymbolJoiner::new().annotate((view, &tokens)).is_err());
}

#[test]
fn symbol_joiner_preserves_valid_cut_emoji_tokens() {
    use turbo_analysis::Layer;
    let doc = Document::new("&\u{1f600}.");
    let view = doc.get::<View>().unwrap();
    let mut spans = Spans::new(view.id());
    spans.push(Span::new(0, 5));
    spans.push(Span::new(1, 6));
    let tokens = Tokens { spans, sentence: None };
    tokens.check().unwrap();
    let (out, mapped) = SymbolJoiner::new().annotate((view, &tokens)).unwrap();
    assert_eq!(out.text(), view.text());
    assert_eq!(mapped.spans.iter().map(|(_, s)| s).collect::<Vec<_>>(), vec![Span::new(0, 5), Span::new(1, 6)]);
    for (_, span) in mapped.spans.iter() {
        assert_eq!(out.alignment().span_to_parent(span), span);
    }
}

#[test]
fn symbol_joiner_sorts_writes_without_reordering_token_spans() {
    let doc = Document::new("& +");
    let view = doc.get::<View>().unwrap();
    let mut spans = Spans::new(view.id());
    spans.push(Span::new(2, 3));
    spans.push(Span::new(0, 1));
    let (out, mapped) = SymbolJoiner::new().annotate((view, &Tokens { spans, sentence: None })).unwrap();
    assert_eq!(out.text(), "and plus");
    assert_eq!(mapped.spans.iter().map(|(_, s)| s).collect::<Vec<_>>(), vec![Span::new(4, 8), Span::new(0, 3)]);
    assert_eq!(out.alignment().span_to_parent(Span::new(4, 8)), Span::new(2, 3));
    assert_eq!(out.alignment().span_to_parent(Span::new(0, 3)), Span::new(0, 1));
}

#[test]
fn sentence_order_validation_remains_stricter_than_token_order() {
    let doc = Document::new("é & +");
    let view = doc.get::<View>().unwrap();
    for raw in [[Span::new(0, 4), Span::new(3, 6)], [Span::new(5, 6), Span::new(3, 4)]] {
        let mut spans = Spans::new(view.id());
        for span in raw {
            spans.push(span);
        }
        assert!(correct(view, &Sentences { spans }).is_err());
    }
}
