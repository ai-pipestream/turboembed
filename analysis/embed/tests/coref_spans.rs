//! The layer-aware TokenVectors path uses existing token and subword spans.

use turbo::bundle::Bundle;
use turbo::manifest::{PromptRole, Truncation};
use turbo::tokenizer::{Encode, Tokenizer};
use turbo_analysis::{Span, Spans, ViewId};
use turbo_analysis_embed::{BundleTokenVectors, VectorOptions};
use turbo_analysis_layers::{Sentences, Subwords, Tokens};
use turbo_coref::TokenVectors;

#[test]
fn span_vectors_match_the_document_encoder_without_joining_words() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-bert-bundle");
    let tokenizer = Tokenizer::load(&Bundle::open(&dir).unwrap()).unwrap();
    let text = "hello world";
    let bare = Encode {
        add_special_tokens: false,
        truncation: Truncation::None,
        max_tokens: u32::MAX,
        prompt: PromptRole::None,
    };
    let (mut ids, mut raw) = (Vec::new(), Vec::new());
    tokenizer.encode_spans(text, bare, &mut ids, &mut raw).unwrap();
    let tv = BundleTokenVectors::open(&dir, &VectorOptions::default()).unwrap();
    let mut sub_spans = Spans::new(ViewId::SOURCE);
    for &[start, end] in &raw {
        sub_spans.push(Span::new(start, end));
    }
    let subwords = Subwords { spans: sub_spans, ids, model: tv.model().clone(), token: None };
    let mut token_spans = Spans::new(ViewId::SOURCE);
    token_spans.push(Span::new(0, 5));
    token_spans.push(Span::new(6, 11));
    let tokens = Tokens { spans: token_spans, sentence: None };
    let mut sentence_spans = Spans::new(ViewId::SOURCE);
    sentence_spans.push(Span::new(0, text.len() as u32));
    let sentences = Sentences { spans: sentence_spans };

    let direct = tv.vectors_spans(ViewId::SOURCE, &tokens, Some(&subwords), &sentences).unwrap().unwrap();
    let expected = tv.embed_document_tokens(&tokens, &subwords, Some(&sentences.spans)).unwrap();
    assert_eq!(direct, expected);
    assert_eq!(direct.len(), 2);
    assert!(direct.iter().flatten().all(|v| v.is_finite()));
}
