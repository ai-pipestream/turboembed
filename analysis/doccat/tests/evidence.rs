//! Evidence for a document categorizer: one record naming the model that
//! scored the document, the document's text as its span and the best
//! category's probability as its raw score; a record naming a model the
//! document did not run is refused.

use turbo_analysis::export::{export, import};
use turbo_analysis::{Analyzer, Document, Evidence, EvidenceError, EvidenceStep, Key, Provenance, Span, Spans, ViewId};
use turbo_analysis_layers::{Categories, Tokens};
use turbo_doccat::svm::{SvmCategorizer, SvmDoccatModel};

/// Two categories, a linear kernel: `a` scores `bow=x` and `bow=y`, `b`
/// scores `bow=p`.
const MODEL: &str = "svm_type c_svc
kernel_type linear
nr_class 2
total_sv 2
rho 0.25
label 0 1
probA -2.0
probB 0.0
nr_sv 1 1
SV
1.0 1:1.0 2:1.0
-1.0 3:1.0
";

const SIDECAR: &str = "svm-doccat 1
language\teng
weighting\tBINARY
selection\tNONE\t-1
scaling\tnone
generator\tbag-of-words
category\t0\ta
category\t1\tb
feature\t1\t0.6931471805599453\t0.0\t1.0\tbow=x
feature\t2\t1.3862943611198906\t0.0\t1.0\tbow=y
feature\t3\t0.6931471805599453\t0.0\t1.0\tbow=p
";

fn categorizer() -> SvmCategorizer {
    SvmCategorizer::new(SvmDoccatModel::from_text(MODEL, SIDECAR).unwrap()).unwrap()
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
    let model = categorizer().model().model_ref().clone();
    for doc in run(&["x y", "p zzz q"]) {
        let ev = doc.get::<Evidence<Categories>>().unwrap();
        assert_eq!(ev.records(), 1);
        let p = ev.producer_of(0);
        assert_eq!((p.annotator.as_str(), p.model.as_ref()), ("turbo-doccat:svm", Some(&model)));
        assert_eq!(ev.spans_of(0), [Span::new(0, doc.source().len() as u32)]);
        let best = doc.get::<Categories>().unwrap().best().unwrap().1;
        assert_eq!((ev.raw_of(0), ev.calibrated_of(0), ev.abstention_of(0)), (Some(best), None, None));
        assert!(ev.support_of(0).is_empty());
    }
}

#[test]
fn a_record_naming_a_model_that_did_not_run_is_refused() {
    let doc = run(&["x y"]).pop().unwrap();
    let mut ev = doc.get::<Evidence<Categories>>().unwrap().clone();
    let real = ev.producers[0].model.clone().unwrap();
    let other = turbo_analysis::ModelRef { sha256: "0".repeat(64), ..real };
    std::sync::Arc::make_mut(&mut ev.producers)[0].model = Some(other.clone());
    assert_eq!(ev.verify(&doc), Err(EvidenceError::UnknownModel { record: Some(0), model: other }));

    // Added under another instance, the record crosses out and is refused
    // on the way back in.
    let mut forged = doc.clone();
    let reads = vec![Key::<Categories>::DEFAULT.id()];
    forged.add(&Key::named("forged"), ev, Provenance { reads, ..Provenance::by("test:forge") }).unwrap();
    let codecs = turbo_analysis_layers::export::codecs();
    let graph = export(&forged, &codecs).unwrap();
    let Err(err) = import(&graph, &codecs) else { panic!("a record naming a model that did not run imported") };
    assert!(err.to_string().contains("did not run here"), "{err}");
}
