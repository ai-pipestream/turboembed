//! Helpers the `heads_*` tests share: the tiny BERT bundle, its
//! subword and vector annotators, and documents with given sentences.

#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use turbo_analysis::*;
use turbo_analysis_embed::*;
use turbo_analysis_layers::{Sentences, Vectors};

pub mod masc_polarity;

pub fn testdata() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata")
}

pub fn bert() -> PathBuf {
    testdata().join("tiny-bert-bundle")
}

pub fn open_subwords() -> SubwordAnnotator {
    SubwordAnnotator::open(&bert()).unwrap_or_else(|e| panic!("{e}"))
}

pub fn open_cpu(max_seq: u32) -> (SubwordAnnotator, Arc<VectorAnnotator>) {
    let sub = open_subwords();
    let o =
        VectorOptions { encoder: EncoderOptions { max_seq, ..EncoderOptions::default() }, ..VectorOptions::default() };
    let vec = Arc::new(VectorAnnotator::open(&bert(), &o).unwrap_or_else(|e| panic!("{e}")));
    (sub, vec)
}

pub fn make_doc_with_sentences(text: &str, sentences: &[(u32, u32)]) -> Document {
    let mut doc = Document::new(text.to_string());
    let mut spans = Spans::new(ViewId::SOURCE);
    for &(start, end) in sentences {
        spans.push(Span::new(start, end));
    }
    doc.add(&Key::DEFAULT, Sentences { spans }, Provenance::by("test")).unwrap();
    doc
}

pub fn embed_single_sentence(vec_ann: &Arc<VectorAnnotator>, text: &str) -> Vec<f32> {
    let a = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .add(open_subwords())
        .add(SentenceVectors(Arc::clone(vec_ann)))
        .build()
        .unwrap();
    let mut docs = [make_doc_with_sentences(text, &[(0, text.len() as u32)])];
    a.run(&mut docs).remove(0).unwrap();
    docs[0].get::<Vectors<Sentences>>().unwrap().row(Ix::new(0)).to_vec()
}
