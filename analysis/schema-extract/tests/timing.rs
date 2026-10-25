//! Time per document on the path that fits one pass, on the reduced
//! model, so a change to the long-text path can show it costs the short
//! one nothing. `cargo test --release -p turbo-schema-extract --test
//! timing -- --ignored --nocapture`

mod common;

use std::time::Instant;

use turbo_analysis::{Analyzer, Document, Key, Provenance, Span, Spans, ViewId};
use turbo_analysis_layers::Tokens;
use turbo_schema_extract::{Cardinality, EntityType, ExtractionSchema, Limits, SchemaAnnotator};

#[test]
#[ignore = "timing"]
fn one_pass_documents() {
    let text = "Zoë ordered two crêpes from Café São Paulo for €12.50; Ana paid and Bruno left for Porto.";
    let schema = ExtractionSchema {
        entities: ["person", "place", "food"]
            .iter()
            .map(|n| EntityType {
                name: (*n).into(),
                description: None,
                threshold: None,
                cardinality: Cardinality::Repeated,
                checks: Vec::new(),
            })
            .collect(),
        classifications: Vec::new(),
        structures: Vec::new(),
    };
    let annotator = SchemaAnnotator::new(common::offline(Limits::default()), schema, 0.5).unwrap();
    let analyzer = Analyzer::builder().given(Key::<Tokens>::DEFAULT).add(annotator).build().unwrap();
    let doc = || {
        let mut d = Document::new(text);
        let mut spans = Spans::new(ViewId::SOURCE);
        let mut at = 0;
        for w in text.split_whitespace() {
            let s = text[at..].find(w).unwrap() + at;
            spans.push(Span::new(s as u32, (s + w.len()) as u32));
            at = s + w.len();
        }
        d.add(&Key::DEFAULT, Tokens { spans, sentence: None }, Provenance::by("test")).unwrap();
        d
    };
    let mut best = f64::MAX;
    for _ in 0..7 {
        let mut docs: Vec<Document> = (0..64).map(|_| doc()).collect();
        let start = Instant::now();
        for r in analyzer.run(&mut docs) {
            r.unwrap();
        }
        best = best.min(start.elapsed().as_secs_f64() / 64.0);
    }
    println!("one-pass document: {:.1} us", best * 1e6);
}
