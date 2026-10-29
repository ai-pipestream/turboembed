//! Evidence for a document categorizer: one record naming the model that
//! scored the document, the document's text as its span and the best
//! category's probability as its raw score; a record naming a model the
//! document did not run is refused.

use turbo_analysis::export::{export, import};
use turbo_analysis::{
    Analyzer, Annotator, Document, Evidence, EvidenceError, EvidenceStep, Key, Provenance, Span, Spans, ViewId,
};
use turbo_analysis_layers::{Categories, Tokens};
use turbo_doccat::sentiment::PolarityAnnotator;

fn categorizer() -> PolarityAnnotator {
    PolarityAnnotator::new()
}

fn document(text: &str) -> Document {
    let mut spans = Spans::new(ViewId::SOURCE);
    let mut start = 0;
    for piece in text.split(' ') {
        spans.push(Span::new(start as u32, (start + piece.len()) as u32));
        start += piece.len() + 1;
    }
    let mut doc = Document::new(text);
    doc.add(&Key::DEFAULT, Tokens { spans, sentence: None }, Provenance::by("test")).unwrap();
    doc
}

fn run(texts: &[&str]) -> Vec<Document> {
    let analyzer = Analyzer::builder()
        .given(Key::<Tokens>::DEFAULT)
        .add(categorizer())
        .step(EvidenceStep::new(Key::<Categories>::DEFAULT))
        .build()
        .unwrap();
    let mut docs: Vec<Document> = texts.iter().map(|t| document(t)).collect();
    for r in analyzer.run(&mut docs) {
        r.unwrap();
    }
    docs
}

#[test]
fn the_document_category_has_one_record_naming_its_model() {
    let model = Annotator::model(&categorizer()).unwrap();
    for doc in run(&["wonderful great", "terrible awful"]) {
        let ev = doc.get::<Evidence<Categories>>().unwrap();
        assert_eq!(ev.records(), 1);
        let p = ev.producer_of(0);
        assert_eq!((p.annotator.as_str(), p.model.as_ref()), ("turbo-doccat:polarity", Some(&model)));
        assert_eq!(ev.spans_of(0), [Span::new(0, doc.source().len() as u32)]);
        let best = doc.get::<Categories>().unwrap().best().unwrap().1;
        assert_eq!((ev.raw_of(0), ev.calibrated_of(0), ev.abstention_of(0)), (Some(best), None, None));
        assert!(ev.support_of(0).is_empty());
    }
}

#[test]
fn a_record_naming_a_model_that_did_not_run_is_refused() {
    let doc = run(&["wonderful great"]).pop().unwrap();
    let mut ev = doc.get::<Evidence<Categories>>().unwrap().clone();
    let real = ev.producers[0].model.clone().unwrap();
    let other = turbo_analysis::ModelRef { sha256: "0".repeat(64), ..real };
    std::sync::Arc::make_mut(&mut ev.producers)[0].model = Some(other.clone());
    assert_eq!(ev.verify(&doc), Err(EvidenceError::UnknownModel { record: Some(0), model: other }));

    let mut forged = doc.clone();
    let reads = vec![Key::<Categories>::DEFAULT.id()];
    forged.add(&Key::named("forged"), ev, Provenance { reads, ..Provenance::by("test:forge") }).unwrap();
    let codecs = turbo_analysis_layers::export::codecs();
    let graph = export(&forged, &codecs).unwrap();
    let Err(err) = import(&graph, &codecs) else { panic!("a record naming a model that did not run imported") };
    assert!(err.to_string().contains("did not run here"), "{err}");
}
