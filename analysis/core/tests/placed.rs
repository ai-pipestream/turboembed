//! A step that says where each call ran (`Annotator::annotate_batch_placed`):
//! each document's layer records that execution in place of the step's
//! own, and a document run again alone after a panic records where it ran
//! the second time.

use turbo_analysis::*;
use turbo_analysis_layers::{Language, Strs};

fn english() -> Language {
    let mut code = Strs::default();
    code.push("en");
    Language { code, score: vec![1.0] }
}

fn execution(backend: &str) -> Execution {
    Execution { backend: backend.into(), device: "test".into(), precision: "f32".into() }
}

/// Sends a call of one document to "small" and any larger one to "large",
/// and panics on a text that says so.
struct BySize;

impl Annotator for BySize {
    const ID: &'static str = "test:by-size";
    type Reads = (View,);
    type Writes = (Language,);

    fn annotate(&self, input: Refs<'_, Self>) -> Result<(Language,), Error> {
        self.annotate_batch(vec![input]).remove(0)
    }

    fn annotate_batch<'d>(&self, inputs: Vec<Refs<'d, Self>>) -> Vec<Result<(Language,), Error>> {
        self.annotate_batch_placed(inputs).into_iter().map(|r| r.map(|(w, _)| w)).collect()
    }

    fn annotate_batch_placed<'d>(&self, inputs: Vec<Refs<'d, Self>>) -> Vec<Placed<(Language,)>> {
        let backend = if inputs.len() == 1 { "small" } else { "large" };
        assert!(inputs.iter().all(|(v,)| !v.text().contains("PANIC")), "cannot handle this one");
        inputs.iter().map(|_| Ok(((english(),), Some(execution(backend))))).collect()
    }

    fn execution(&self) -> Option<Execution> {
        Some(execution("declared"))
    }
}

/// Says nothing about where a call ran, so its layers record `execution`.
struct Plain;

impl Annotator for Plain {
    const ID: &'static str = "test:plain";
    type Reads = (View,);
    type Writes = (Language,);

    fn annotate(&self, _: Refs<'_, Self>) -> Result<(Language,), Error> {
        Ok((english(),))
    }

    fn execution(&self) -> Option<Execution> {
        Some(execution("declared"))
    }
}

fn backend(d: &Document) -> String {
    d.provenance(Key::<Language>::DEFAULT.id()).unwrap().execution.as_ref().unwrap().backend.clone()
}

#[test]
fn each_document_records_where_its_call_ran() {
    let a = Analyzer::builder().add(BySize).build().unwrap();
    let mut many: Vec<Document> = (0..3).map(|i| Document::new(format!("Text {i}."))).collect();
    assert!(a.run(&mut many).iter().all(|s| s.is_ok()));
    assert!(many.iter().all(|d| backend(d) == "large"));
    let mut one = vec![Document::new("Alone.")];
    assert!(a.run(&mut one)[0].is_ok());
    assert_eq!(backend(&one[0]), "small");
}

#[test]
fn a_step_that_does_not_say_records_its_own_execution() {
    let a = Analyzer::builder().add(Plain).build().unwrap();
    let mut docs = vec![Document::new("One."), Document::new("Two.")];
    assert!(a.run(&mut docs).iter().all(|s| s.is_ok()));
    assert!(docs.iter().all(|d| backend(d) == "declared"));
}

#[test]
fn a_document_run_again_alone_records_where_it_ran_then() {
    let a = Analyzer::builder().add(BySize).build().unwrap();
    let mut docs = vec![Document::new("Fine."), Document::new("PANIC now."), Document::new("Fine too.")];
    let status = a.run(&mut docs);
    assert!(status[0].is_ok() && status[1].is_err() && status[2].is_ok(), "{status:?}");
    // The batch panicked, so each document ran again in a call of its own.
    assert_eq!(backend(&docs[0]), "small");
    assert_eq!(backend(&docs[2]), "small");
}
