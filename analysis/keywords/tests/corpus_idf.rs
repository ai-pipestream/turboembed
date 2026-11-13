//! Collection-wide BM25 IDF calibration (`#2346`) for YAKE and TextRank.

use std::sync::Arc;

use turbo_analysis::{CorpusIdf, Document, Key, Provenance, Span, Spans, Tag, TagSet, ViewId};
use turbo_analysis_layers::{PosTags, Sentences, Strs, Terms, Tokens};
use turbo_keywords::{Stopwords, TextRankExtractor, YakeExtractor};

fn fixture_doc(sentences: &[Vec<(&str, &str)>]) -> Document {
    let mut text = String::new();
    let mut spans = Spans::new(ViewId::SOURCE);
    let mut surface = Strs::default();
    let mut folded = Strs::default();
    let mut s_spans = Spans::new(ViewId::SOURCE);
    let mut cursor = 0usize;
    for sent in sentences {
        let start = cursor;
        for (token, _) in sent {
            if !text.is_empty() {
                text.push(' ');
            }
            let s = text.len() as u32;
            text.push_str(token);
            spans.push(Span::new(s, text.len() as u32));
            surface.push(token);
            folded.push(&token.to_lowercase());
            cursor = text.len();
        }
        s_spans.push(Span::new(start as u32, cursor as u32));
    }
    let mut doc = Document::new(text);
    let by = Provenance::by("test");
    doc.add(&Key::DEFAULT, Sentences { spans: s_spans }, by.clone()).unwrap();
    doc.add(&Key::DEFAULT, Tokens { spans, sentence: None }, by.clone()).unwrap();
    let labels = ["ADJ", "NOUN", "PROPN", "VERB"];
    let tags: Vec<Tag> = sentences
        .iter()
        .flat_map(|s| s.iter())
        .map(|(_, pos)| Tag(labels.iter().position(|l| l == pos).unwrap() as u16))
        .collect();
    doc.add(&Key::DEFAULT, PosTags { tagset: TagSet::new("universal", &labels), tags, prob: None }, by.clone())
        .unwrap();
    doc.add(&Key::DEFAULT, Terms { surface, folded, stem: None, lemma: None }, by).unwrap();
    doc
}

#[test]
fn corpus_idf_promotes_rare_domain_terms_in_yake_and_textrank() {
    let doc = fixture_doc(&[
        vec![("system", "NOUN"), ("processes", "VERB"), ("quasar", "NOUN")],
        vec![("system", "NOUN"), ("monitors", "VERB"), ("quasar", "NOUN")],
    ]);
    let sentences = doc.get::<Sentences>().unwrap();
    let tokens = doc.get::<Tokens>().unwrap();
    let pos_tags = doc.get::<PosTags>().unwrap();
    let terms = doc.get::<Terms>().unwrap();

    let idf = Arc::new(CorpusIdf::from_entries(10_000, [("system", 9_900), ("quasar", 3)]).unwrap());

    let sw = Stopwords::custom(["the"]).unwrap();
    let yake = YakeExtractor::new(sw).with_ngrams(1, 1).with_corpus_idf(Arc::clone(&idf));
    let y_ranked = yake.extract(sentences, tokens, terms).unwrap();
    assert_eq!(y_ranked[0].phrase, "quasar");

    let tr = TextRankExtractor::new().with_corpus_idf(idf);
    let tr_ranked = tr.extract(sentences, tokens, pos_tags, terms).unwrap();
    assert_eq!(tr_ranked[0].phrase, "quasar");
}
