#![allow(dead_code)]
use std::path::Path;
use turbo_analysis::{Analyzer, Bound, Document, Key};
use turbo_analysis_embed::SubwordAnnotator;
use turbo_analysis_layers::{Boundaries, Chunks, Sentences, Subwords, Tables};
use turbo_chunker::{Budget, Chunker, TableBoundaryAnnotator, TableHeaderAnnotator};
use turbo_sentdetect::NewlineDetector;
use turbo_structure::TableGridAnnotator;
use turbo_tokenize::{SimpleTokenizer, TokenizerOptions};

pub fn table(rows: usize) -> String {
    let mut text = "| Metric | Q1 2025 | Q2 2025 |\n| :--- | :---: | ---: |\n".to_owned();
    for _ in 0..rows {
        text.push_str("| Revenue | 12 | 34 |\n");
    }
    text
}
pub fn prepared(text: &str) -> Document {
    let pipeline = Analyzer::builder()
        .step(Bound::new(NewlineDetector::default()))
        .step(Bound::new(SimpleTokenizer::new(TokenizerOptions::default())))
        .step(Bound::new(
            SubwordAnnotator::open(
                Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-bert-bundle").as_path(),
            )
            .unwrap(),
        ))
        .step(Bound::new(TableGridAnnotator::default()))
        .step(Bound::new(TableBoundaryAnnotator::default()))
        .build()
        .unwrap();
    let mut docs = [Document::new(text)];
    assert_eq!(pipeline.run(&mut docs)[0], Ok(()));
    docs.into_iter().next().unwrap()
}
pub fn four_chunks(doc: &mut Document) -> Budget {
    let total = doc.get::<Subwords>().unwrap().spans.len() as u32;
    let budget = Budget { max: total * 5 / 16, target: total / 4, size_weight: 1_000_000, overhead: 0 };
    chunk(doc, budget);
    budget
}
pub fn chunk(doc: &mut Document, budget: Budget) {
    let pipeline = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Subwords>::DEFAULT)
        .given(Key::<Boundaries>::DEFAULT)
        .given(Key::<Tables>::DEFAULT)
        .step(Bound::new(Chunker { budget }))
        .build()
        .unwrap();
    assert_eq!(pipeline.run(std::slice::from_mut(doc))[0], Ok(()));
}
pub fn headers(doc: &mut Document) {
    let pipeline = Analyzer::builder()
        .given(Key::<Chunks>::DEFAULT)
        .given(Key::<Tables>::DEFAULT)
        .step(Bound::new(TableHeaderAnnotator))
        .build()
        .unwrap();
    assert_eq!(pipeline.run(std::slice::from_mut(doc))[0], Ok(()));
}
